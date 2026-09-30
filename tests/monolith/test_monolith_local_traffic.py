"""Local-LLM traffic control + re-prime after eviction (2026-09-29, r6).

WHY (live, v2.0.118, one Ollama slot, ~12.5k-token prompt)
==========================================================
Every owner turn re-evaluated the whole prompt (prompt_eval 2.0-2.5 s) even
with ~300 chars of new context, because ANY local request between two turns
replaced the one-slot cache:

  _call_llm                        pe=12439/2157   <- turn
  get_followup_response            pe=12945/225    <- extends it: warm
  _llm_quick<_worker   (learn)     pe=2682/393     <- evicts
  _call_llm                        pe=12549/2189   <- cold again
  [reprime] 2588 pe=12399                          <- idle re-prime ...
  [local-vision] served via ...    (Teams nudger)  <- ... evicted 9 s later
  _call_llm                        pe=12639/2181   <- cold

WHAT THESE TESTS PIN
====================
1. A TAGGED background local call (learn_from_turn, the ambient extractor /
   judge / screen observer, the Teams nudger, the session checkpoint, the LTM
   reflector) waits while the owner is in a conversation on the local route,
   bounded by LOCAL_BACKGROUND_MAX_DEFER_S (0 = off); an owner call and the
   main thread never wait; the cloud route never waits.
2. learn_from_turn queues turns and ONE worker extracts the turns that
   arrived during the wait in ONE call (a single turn keeps the old prompt).
3. After a NON-owner local request (chat or vision) completes outside a turn,
   the idle re-prime is scheduled when the owner spoke within
   LOCAL_REPRIME_AFTER_BACKGROUND_WINDOW_S (0 = off); the owner's own calls
   never schedule it. The live sequence (prime -> Teams vision -> turn) now
   ends in "[reprime] hit" instead of a silent full re-evaluation.
4. The owner turn's stale check logs hit / evicted / stale with the prime's
   age, and typed turns take it (they run _call_llm like spoken ones).
5. Every local POST (chat, vision, re-prime) is tracked as JARVIS's own
   inference for the system-pulse GPU check; the re-prime is not counted as
   an eviction.
6. A vision call on the shared chat model keeps the chat's keep_alive.

Every class fails on the pre-r6 tree (no core.local_traffic, no gate, no
queue, no trigger, no hit/evicted lines, no keep_alive on vision).
"""
from __future__ import annotations

import contextlib
import inspect
import io
import os
import sys
import threading
import time
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


def _wait_until(pred, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.01)
    return bool(pred())


def _pairs(n, tag="t"):
    out = []
    for i in range(n):
        out.append({"role": "user", "content": f"{tag}-u{i}"})
        out.append({"role": "assistant", "content": f"{tag}-a{i}"})
    return out


@requires_monolith
class _Base(MonolithGlobalsTestCase):
    # The setup is deliberately tolerant of the pre-r6 tree (no _lt, no new
    # knobs / cells): run against it, the behavioural tests reach their own
    # assertions and fail THERE (a background POST mid-conversation, three
    # extractions instead of one, no re-prime, no hit line) rather than on a
    # missing attribute.
    def setUp(self):
        super().setUp()
        bc = self.bc
        lt = getattr(bc, "_lt", None)
        if lt is not None:
            # Fast polling for the shared gate; restored after the test.
            lt.GATE.configure(poll_s=0.02)
            self.addCleanup(lt.GATE.configure, poll_s=lt.DEFAULT_POLL_S)
        self._p(bc, "PROMPT_FREEZE_QUIET_S", 30.0)
        self._p(bc, "LOCAL_BACKGROUND_MAX_DEFER_S", 30.0, create=True)
        self._p(bc, "LOCAL_REPRIME_AFTER_BACKGROUND_WINDOW_S", 600.0,
                create=True)
        bc._turn_in_progress[0] = False
        bc._utterance_in_progress[0] = False
        bc._last_convo_activity[0] = 0.0
        if getattr(bc, "_last_owner_turn_at", None) is None:
            self._p(bc, "_last_owner_turn_at", [0.0], create=True)
        bc._last_owner_turn_at[0] = 0.0

    def _p(self, *args, **kwargs):
        patcher = mock.patch.object(*args, **kwargs)
        m = patcher.start()
        self.addCleanup(patcher.stop)
        return m

    def _route(self, route):
        import core.config as cfg
        self._p(cfg, "model_route", return_value=route)

    def _talking(self):
        """The owner is in a conversation (a reply 1 s ago)."""
        self.bc._last_convo_activity[0] = time.monotonic() - 1.0

    def _quiet(self):
        self.bc._last_convo_activity[0] = 0.0

    def _local_llm(self, body=None):
        """Route every /api/chat POST to a recorder; returns the payloads."""
        bc = self.bc
        posted = []
        seen_inflight = []
        fake_req = mock.Mock()

        def _post(url, json=None, timeout=None, **_k):
            lt = getattr(bc, "_lt", None)
            seen_inflight.append(lt.TRACKER.inflight if lt else None)
            posted.append(json)
            return _Resp(body or _CHAT_BODY)
        fake_req.post.side_effect = _post
        fake_req.RequestException = Exception
        fake_req.Timeout = TimeoutError
        self._p(bc, "LOCAL_LLM_FALLBACK", True)
        self._p(bc, "_ollama_alive", return_value=True)
        self._p(bc, "_ollama_has_model", return_value=True)
        self._p(bc, "_get_local_llm_model", return_value="gemma-test")
        self._p(bc, "_next_local_llm_fallback", return_value=None)
        self._p(bc, "requests", fake_req)
        self.seen_inflight = seen_inflight
        return posted

    def _stable_layout(self):
        bc = self.bc
        self._p(bc, "_DYNAMIC_LOCAL_PROMPT", True)
        self._p(bc, "_STABLE_LOCAL_PREFIX", True)
        self._p(bc, "_system_prompt",
                "BASE IDENTITY\n" + bc.PC_CONTROL_PROMPT
                + "\n\nWhat you know about your owner:\n- likes tea")

    def _quiet_turn_helpers(self):
        bc = self.bc
        self._p(bc, "_ltm_context", return_value="")
        self._p(bc, "_ltm_enqueue")
        self._p(bc, "load_memory", return_value=bc._empty_memory())
        self._p(bc, "save_memory")
        self._p(bc, "_voice_mood_response", None)
        bc._phrase_rotation_last[0] = {}

    def _vision_ready(self):
        bc = self.bc
        self._p(bc, "LOCAL_VISION_MODEL", "gemma-test")
        self._p(bc, "_local_vision_usable", return_value=True)
        self._p(bc, "_local_vision_model_already_resident", return_value=True)
        self._p(bc, "_log_gpu_state")
        bc._RESOLVED_LOCAL_LLM_MODEL[0] = "gemma-test"

    def _in_thread(self, fn, tag=None):
        """Run fn on a daemon thread (tagged as background work when `tag`)."""
        bc = self.bc
        out = {}

        def _t():
            try:
                work = getattr(bc, "background_local_work", None)
                if tag and work is not None:
                    with work(tag):
                        out["value"] = fn()
                else:
                    out["value"] = fn()
            except BaseException as e:  # pragma: no cover - surfaced below
                out["error"] = e
        th = threading.Thread(target=_t, daemon=True)
        th.start()
        self.addCleanup(th.join, 5)
        return th, out


# ════════════════════════════════════════════════════════════════════════════
#  1. The gate: who waits, when, and for how long
# ════════════════════════════════════════════════════════════════════════════
class DeferReasonTests(_Base):
    def test_reasons(self):
        bc = self.bc
        self._route("local")
        self.assertIsNone(bc._background_defer_reason())
        self._talking()
        self.assertEqual(bc._background_defer_reason(), "conversation")
        bc._turn_in_progress[0] = True
        self.assertEqual(bc._background_defer_reason(), "turn")
        bc._utterance_in_progress[0] = True
        self.assertEqual(bc._background_defer_reason(), "utterance")

    def test_cloud_route_and_zero_cap_never_defer(self):
        bc = self.bc
        self._talking()
        bc._turn_in_progress[0] = True
        self._route("cloud")
        self.assertIsNone(bc._background_defer_reason())
        self._route("local")
        self._p(bc, "LOCAL_BACKGROUND_MAX_DEFER_S", 0.0)
        self.assertIsNone(bc._background_defer_reason())

    def test_the_monolith_installed_its_predicate_on_the_shared_gate(self):
        bc = self.bc
        self.assertIs(bc._lt.GATE._defer_reason, bc._background_defer_reason)
        self.assertIs(bc._lt.GATE._max_defer_s, bc._local_bg_max_defer_s)


class BackgroundLlmQuickTests(_Base):
    def setUp(self):
        super().setUp()
        self._route("local")                  # chat AND ambient local
        self.posted = self._local_llm()

    def test_tagged_call_waits_while_the_owner_talks_then_runs(self):
        bc = self.bc
        self._talking()
        with contextlib.redirect_stdout(io.StringIO()) as out:
            th, res = self._in_thread(
                lambda: bc._llm_quick("sys", "extract this"),
                tag="learn_from_turn")
            time.sleep(0.3)
            self.assertEqual(self.posted, [],
                             "a background call evicted the prefix mid-chat")
            self._quiet()
            th.join(5)
        self.assertFalse(th.is_alive())
        self.assertEqual(res.get("value"), "ok")
        self.assertEqual(len(self.posted), 1)
        log = out.getvalue()
        self.assertIn("[bg-local] defer learn_from_turn (conversation)", log)
        self.assertRegex(log, r"\[bg-local\] run learn_from_turn after \d+ ms "
                              r"\(released\)")
        # privacy: the gate's lines carry no prompt text
        for line in log.splitlines():
            if "[bg-local]" in line:
                self.assertNotIn("extract this", line)

    def test_owner_call_on_another_thread_never_waits(self):
        bc = self.bc
        self._talking()
        bc._turn_in_progress[0] = True
        th, res = self._in_thread(lambda: bc._llm_quick("sys", "recall"))
        th.join(5)
        self.assertFalse(th.is_alive(), "an owner action waited")
        self.assertEqual(len(self.posted), 1)

    def test_main_thread_never_waits_even_when_tagged(self):
        bc = self.bc
        self._talking()
        bc._turn_in_progress[0] = True
        t0 = time.monotonic()
        with bc.background_local_work("proactive"):
            self.assertEqual(bc._llm_quick("sys", "x"), "ok")
        self.assertLess(time.monotonic() - t0, 2.0)
        self.assertEqual(len(self.posted), 1)

    def test_max_defer_bounds_the_wait(self):
        # Consumer test: LOCAL_BACKGROUND_MAX_DEFER_S is the soft cap.
        bc = self.bc
        self._p(bc, "LOCAL_BACKGROUND_MAX_DEFER_S", 0.25)
        self._talking()                      # and it never goes quiet
        t0 = time.monotonic()
        with contextlib.redirect_stdout(io.StringIO()) as out:
            th, _ = self._in_thread(lambda: bc._llm_quick("sys", "x"),
                                    tag="ambient-extract")
            th.join(10)
        self.assertFalse(th.is_alive())
        self.assertEqual(len(self.posted), 1)
        self.assertLess(time.monotonic() - t0, 5.0)
        self.assertIn("(forced)", out.getvalue())

    def test_zero_max_defer_turns_the_wait_off(self):
        bc = self.bc
        self._p(bc, "LOCAL_BACKGROUND_MAX_DEFER_S", 0.0)
        self._talking()
        th, _ = self._in_thread(lambda: bc._llm_quick("sys", "x"),
                                tag="ambient-extract")
        th.join(5)
        self.assertFalse(th.is_alive())
        self.assertEqual(len(self.posted), 1)

    def test_cloud_chat_route_never_waits(self):
        bc = self.bc
        import core.config as cfg
        self._p(cfg, "model_route",
                side_effect=lambda f: "cloud" if f == "chat" else "local")
        self._talking()
        th, _ = self._in_thread(lambda: bc._llm_quick("sys", "x"),
                                tag="learn_from_turn")
        th.join(5)
        self.assertFalse(th.is_alive())
        self.assertEqual(len(self.posted), 1)


class BackgroundCallerTagTests(_Base):
    """The monolith's own non-urgent callers are tagged."""

    def test_ambient_content_judge_is_background_work(self):
        bc = self.bc
        seen = []

        def _fake(system, messages, max_tokens=500):
            job = bc._lt.current_job()
            seen.append(job.tag if job else None)
            return "PERSON"
        self._p(bc, "_call_local_llm", side_effect=_fake)
        self.assertFalse(bc._ambient_content_is_media("so I said to him"))
        self.assertEqual(seen, ["ambient-learn"])
        self.assertIsNone(bc._lt.current_job())

    def test_session_checkpoint_and_ltm_reflector_are_background_work(self):
        bc = self.bc
        self.assertIn('background_work("session-checkpoint")',
                      inspect.getsource(bc._session_summary_checkpoint_thread))
        self.assertIn('background_work("ltm-reflect")',
                      inspect.getsource(bc._ltm_boot_warm))

    def test_skill_facing_helpers_are_exported(self):
        bc = self.bc
        self.assertIs(bc.background_local_work, bc._lt.background_work)
        self.assertTrue(callable(bc.wait_for_local_quiet))


class WaitForLocalQuietTests(_Base):
    def test_cloud_vision_route_does_not_wait(self):
        bc = self.bc
        self._p(bc, "_vision_goes_local", return_value=False)
        self._talking()
        self.assertEqual(bc.wait_for_local_quiet("vision"), "cloud")

    def test_local_vision_waits_for_quiet_before_the_capture(self):
        bc = self.bc
        self._route("local")
        self._p(bc, "_vision_goes_local", return_value=True)
        self._talking()
        with contextlib.redirect_stdout(io.StringIO()):
            th, res = self._in_thread(lambda: bc.wait_for_local_quiet("vision"),
                                      tag="teams-nudge")
            time.sleep(0.2)
            self.assertTrue(th.is_alive())
            self._quiet()
            th.join(5)
        self.assertEqual(res.get("value"), "released")
        self.assertFalse(bc._lt.GATE.busy(), "the pre-wait kept the slot")


# ════════════════════════════════════════════════════════════════════════════
#  2. learn_from_turn: queued, coalesced, never dropped
# ════════════════════════════════════════════════════════════════════════════
class LearnCoalescingTests(_Base):
    def setUp(self):
        super().setUp()
        bc = self.bc
        self._route("local")
        self._p(bc, "LEARN_EVERY_TURN", True)
        self._p(bc, "load_memory", return_value=bc._empty_memory())
        self.merge = self._p(bc, "merge_memory", return_value=([], []))
        self.rebuild = self._p(bc, "_request_prompt_rebuild",
                               return_value="deferred")
        self.posted = self._local_llm()

    def _drained(self):
        bc = self.bc
        self.assertTrue(_wait_until(lambda: not bc._learn_worker_live[0]),
                        "learn worker never finished")

    @staticmethod
    def _user(payload):
        return payload["messages"][-1]["content"]

    def test_turns_during_a_conversation_become_one_extraction(self):
        bc = self.bc
        mem = bc._empty_memory()
        self._talking()
        with contextlib.redirect_stdout(io.StringIO()) as out:
            for i in range(3):
                bc.learn_from_turn(f"owner says {i}", f"reply {i}", mem)
            time.sleep(0.3)
            self.assertEqual(self.posted, [],
                             "learn_from_turn evicted the prefix mid-chat")
            self.assertEqual(len(bc._learn_pending), 3)
            self._quiet()
            self._drained()
        self.assertEqual(len(self.posted), 1, "one call for the whole batch")
        user = self._user(self.posted[0])
        order = [user.index(f"owner says {i}") for i in range(3)]
        self.assertEqual(order, sorted(order))
        self.assertIn("Turn 1:\nUser said: owner says 0\nAssistant said: "
                      "reply 0", user)
        self.assertIn("Turn 3:", user)
        self.assertIn("several consecutive turns",
                      self.posted[0]["messages"][0]["content"])
        self.assertEqual(self.posted[0]["options"]["num_predict"], 400)
        self.assertIn("[learn] extracting 3 queued turns in one call",
                      out.getvalue())
        self.assertEqual(bc._learn_pending, [])

    def test_a_single_turn_keeps_the_original_prompt(self):
        bc = self.bc
        bc.learn_from_turn("I have a cat", "Noted.", bc._empty_memory())
        self._drained()
        self.assertEqual(len(self.posted), 1)
        self.assertEqual(self._user(self.posted[0]),
                         "User said: I have a cat\nAssistant said: Noted.")
        self.assertEqual(self.posted[0]["options"]["num_predict"], 250)
        self.assertNotIn("several consecutive turns",
                         self.posted[0]["messages"][0]["content"])

    _JSON_BODY = {"model": "gemma-test",
                  "message": {"role": "assistant",
                              "content": '{"new_facts": [], "new_projects": ["p"], '
                                         '"topic": "t"}'},
                  "done": True, "prompt_eval_count": 1234, "eval_count": 1}

    def _provenances(self):
        return [c.kwargs.get("provenance") for c in self.merge.call_args_list]

    def test_overheard_and_owner_turns_never_share_an_extraction(self):
        # v2.0.129 merge of the batching queue (R6) with topic hygiene (Q4):
        # a batch must be homogeneous, or overheard speech would ride an
        # owner-directed extraction and be allowed to teach projects.
        bc = self.bc
        self.posted = self._local_llm(body=self._JSON_BODY)
        mem = bc._empty_memory()
        self._talking()
        with contextlib.redirect_stdout(io.StringIO()):
            bc.learn_from_turn("owner one", "r1", mem)
            bc.learn_from_turn("tv line", "", mem, owner_directed=False)
            bc.learn_from_turn("owner two", "r2", mem)
            time.sleep(0.2)
            self._quiet()
            self._drained()
        self.assertEqual(len(self.posted), 3, "a mixed backlog must split into runs")
        self.assertEqual([p["owner_directed"] for p in self._provenances()],
                         [True, False, True])
        self.assertEqual([p["turn_text"] for p in self._provenances()],
                         ["owner one", "tv line", "owner two"])

    def test_a_batch_passes_the_strictest_confidence_of_its_turns(self):
        bc = self.bc
        self.posted = self._local_llm(body=self._JSON_BODY)
        mem = bc._empty_memory()
        self._talking()
        with contextlib.redirect_stdout(io.StringIO()):
            bc.learn_from_turn("clear one", "r1", mem, conf={
                "avg_logprob": -0.2, "no_speech_prob": 0.1, "compression_ratio": 1.2})
            bc.learn_from_turn("murky two", "r2", mem, conf={
                "avg_logprob": -0.9, "no_speech_prob": 0.5, "compression_ratio": 2.0})
            time.sleep(0.2)
            self._quiet()
            self._drained()
        self.assertEqual(len(self.posted), 1)
        (prov,) = self._provenances()
        self.assertEqual(prov["conf"], {"no_speech_prob": 0.5, "avg_logprob": -0.9,
                                        "compression_ratio": 2.0})
        self.assertTrue(prov["owner_directed"])
        self.assertEqual(prov["turn_text"], "murky two")
        self.assertEqual(prov["source"], "owner turn")

    def test_a_long_backlog_is_split_into_capped_batches(self):
        bc = self.bc
        self._talking()
        with contextlib.redirect_stdout(io.StringIO()):
            for i in range(bc._LEARN_BATCH_MAX + 2):
                bc.learn_from_turn(f"u{i}", f"a{i}", {})
            time.sleep(0.1)
            self._quiet()
            self._drained()
        self.assertEqual(len(self.posted), 2)
        self.assertIn(f"Turn {bc._LEARN_BATCH_MAX}:", self._user(self.posted[0]))
        self.assertNotIn(f"Turn {bc._LEARN_BATCH_MAX + 1}:",
                         self._user(self.posted[0]))
        self.assertIn(f"u{bc._LEARN_BATCH_MAX + 1}", self._user(self.posted[1]))

    def test_new_facts_are_folded_into_the_prompt_after_the_merge(self):
        bc = self.bc
        body = dict(_CHAT_BODY, message={
            "role": "assistant",
            "content": '{"new_facts": ["User has a cat"], "new_projects": [],'
                       ' "topic": "pets"}'})
        self.posted = self._local_llm(body)
        order = []
        self.merge.side_effect = lambda **k: (order.append("merge"),
                                              (["User has a cat"], []))[1]
        self.rebuild.side_effect = lambda: (order.append("rebuild"),
                                            "applied")[1]
        again = self._p(bc, "_reprime_after_background",
                        side_effect=lambda tag: order.append(f"again:{tag}"))
        with contextlib.redirect_stdout(io.StringIO()):
            bc.learn_from_turn("I have a cat", "Noted.", {})
            self._drained()
        self.merge.assert_called_once()
        self.rebuild.assert_called_once_with()
        # The POST's own completion asks for a re-prime (coalesced by the
        # single-flight worker); the LAST ask comes after the rebuild, so the
        # prime is built from the prompt that holds the new fact.
        self.assertEqual(order[-3:], ["merge", "rebuild",
                                      "again:learn_from_turn"])
        again.assert_called_with("learn_from_turn")

    def test_a_dead_worker_never_strands_the_queue(self):
        bc = self.bc
        self._p(bc, "_learn_prompt", side_effect=RuntimeError("boom"))
        with contextlib.redirect_stdout(io.StringIO()) as out:
            bc.learn_from_turn("x", "y", {})
            self._drained()
        self.assertIn("[learn] failed: RuntimeError", out.getvalue())
        self.assertEqual(bc._learn_pending, [])
        self.assertFalse(bc._learn_worker_live[0])


# ════════════════════════════════════════════════════════════════════════════
#  3. Re-prime after a non-owner local request
# ════════════════════════════════════════════════════════════════════════════
class ReprimeAfterBackgroundTriggerTests(_Base):
    def setUp(self):
        super().setUp()
        bc = self.bc
        self._route("local")
        self._p(bc, "LOCAL_PREFIX_REPRIME", True)
        self.posted = self._local_llm()
        self.sched = self._p(bc, "_schedule_local_reprime", return_value=True)
        bc._last_owner_turn_at[0] = time.monotonic() - 5.0

    def _bg_chat(self, tag="learn_from_turn"):
        bc = self.bc
        with contextlib.redirect_stdout(io.StringIO()) as out:
            th, _ = self._in_thread(
                lambda: bc._call_local_llm("sys", [{"role": "user",
                                                    "content": "x"}]),
                tag=tag)
            th.join(5)
        return out.getvalue()

    def test_background_chat_call_schedules_the_reprime(self):
        log = self._bg_chat("learn_from_turn")
        self.sched.assert_called_once_with()
        self.assertIn("[reprime] after learn_from_turn", log)

    def test_untagged_non_owner_call_outside_a_turn_also_schedules(self):
        # e.g. a proactive comment or a notification classifier: it evicted
        # the prefix just the same.
        bc = self.bc
        th, _ = self._in_thread(lambda: bc._call_local_llm(
            "sys", [{"role": "user", "content": "x"}]))
        th.join(5)
        self.sched.assert_called_once_with()

    def test_outside_the_window_nothing_is_scheduled(self):
        bc = self.bc
        bc._last_owner_turn_at[0] = time.monotonic() - 700.0
        self._bg_chat()
        self.sched.assert_not_called()

    def test_window_knob_is_honoured(self):
        # Consumer test: LOCAL_REPRIME_AFTER_BACKGROUND_WINDOW_S.
        bc = self.bc
        bc._last_owner_turn_at[0] = time.monotonic() - 700.0
        self._p(bc, "LOCAL_REPRIME_AFTER_BACKGROUND_WINDOW_S", 1000.0)
        self._bg_chat()
        self.sched.assert_called_once_with()
        self.sched.reset_mock()
        bc._last_owner_turn_at[0] = time.monotonic() - 1.0
        self._p(bc, "LOCAL_REPRIME_AFTER_BACKGROUND_WINDOW_S", 0.0)
        self._bg_chat()
        self.sched.assert_not_called()

    def test_no_owner_turn_yet_nothing_is_scheduled(self):
        self.bc._last_owner_turn_at[0] = 0.0
        self._bg_chat()
        self.sched.assert_not_called()

    def test_calls_inside_a_turn_never_schedule(self):
        # The turn's follow-ups / owner actions re-warm the prefix themselves.
        # The second call is an UNTAGGED one on another thread (an owner
        # action): owner work never waits, so it really POSTs inside the
        # turn. It used to be _bg_chat() - a TAGGED background job, which the
        # gate DEFERS for the whole turn: it never ran here (so this passed
        # without testing it), outlived the test, was released ~10 s later
        # once the harness cleared _turn_in_progress, and its POST scheduled
        # a re-prime inside whichever test was running then -
        # test_untagged_non_owner_call_outside_a_turn_also_schedules failed
        # intermittently on "Called 2 times" (2026-09-29/30).
        bc = self.bc
        bc._turn_in_progress[0] = True
        bc._call_local_llm("sys", [{"role": "user", "content": "x"}])
        th, _ = self._in_thread(lambda: bc._call_local_llm(
            "sys", [{"role": "user", "content": "y"}]))
        th.join(5)
        self.assertFalse(th.is_alive(), "an owner-side call waited")
        self.assertEqual(len(self.posted), 2, "both calls must really POST")
        self.sched.assert_not_called()

    def test_the_owner_turns_primary_call_never_schedules(self):
        bc = self.bc
        self._stable_layout()
        self._quiet_turn_helpers()
        bc._call_llm("what time is it")
        self.assertEqual(len(self.posted), 1)
        self.sched.assert_not_called()

    def test_cloud_route_and_disabled_reprime_never_schedule(self):
        bc = self.bc
        import core.config as cfg
        self._p(cfg, "model_route",
                side_effect=lambda f: "cloud" if f == "chat" else "local")
        self._bg_chat()
        self._route("local")
        self._p(bc, "LOCAL_PREFIX_REPRIME", False)
        self._bg_chat()
        self.sched.assert_not_called()

    def test_no_post_no_trigger(self):
        bc = self.bc
        self._p(bc, "_ollama_alive", return_value=False)
        self._p(bc, "_ollama_selfheal_async")
        self._p(bc, "_ollama_install_async")
        self._bg_chat()
        self.sched.assert_not_called()


class VisionTrafficTests(_Base):
    def setUp(self):
        super().setUp()
        bc = self.bc
        self._route("local")
        self._p(bc, "LOCAL_PREFIX_REPRIME", True)
        self.posted = self._local_llm(
            {"message": {"role": "assistant", "content": "UNREAD: 2 | NONE"}})
        self._vision_ready()
        self.sched = self._p(bc, "_schedule_local_reprime", return_value=True)
        bc._last_owner_turn_at[0] = time.monotonic() - 5.0

    def test_tagged_vision_waits_for_quiet_then_schedules_the_reprime(self):
        bc = self.bc
        self._talking()
        with contextlib.redirect_stdout(io.StringIO()) as out:
            th, res = self._in_thread(
                lambda: bc._call_local_vision("q", [b"png"]),
                tag="teams-nudge")
            time.sleep(0.3)
            self.assertEqual(self.posted, [],
                             "the nudger's screenshot read evicted the prefix "
                             "mid-chat")
            self._quiet()
            th.join(5)
        self.assertEqual(res.get("value"), "UNREAD: 2 | NONE")
        self.assertEqual(len(self.posted), 1)
        self.sched.assert_called_once_with()
        self.assertIn("[reprime] after teams-nudge", out.getvalue())

    def test_owner_vision_mid_turn_neither_waits_nor_schedules(self):
        bc = self.bc
        self._talking()
        bc._turn_in_progress[0] = True
        with contextlib.redirect_stdout(io.StringIO()):
            out = bc._call_local_vision("what's on screen", [b"png"])
        self.assertEqual(out, "UNREAD: 2 | NONE")
        self.sched.assert_not_called()

    def test_shared_model_keeps_the_chat_keep_alive(self):
        bc = self.bc
        with contextlib.redirect_stdout(io.StringIO()):
            bc._call_local_vision("q", [b"png"])
        self.assertEqual(self.posted[0].get("keep_alive"), "20m",
                         "a vision call cut the shared brain's residency to "
                         "Ollama's 5 min default")

    def test_a_separate_vlm_keeps_ollamas_default(self):
        bc = self.bc
        self._p(bc, "LOCAL_VISION_MODEL", "some-vlm:7b")
        with contextlib.redirect_stdout(io.StringIO()):
            bc._call_local_vision("q", [b"png"])
        self.assertNotIn("keep_alive", self.posted[0])


# ════════════════════════════════════════════════════════════════════════════
#  4. The live sequence end to end, and the stale-check diagnostics
# ════════════════════════════════════════════════════════════════════════════
class ReprimeDiagnosticsTests(_Base):
    def setUp(self):
        super().setUp()
        bc = self.bc
        self._route("local")
        self._stable_layout()
        self.posted = self._local_llm()
        self._quiet_turn_helpers()
        self._vision_ready()
        self._p(bc, "LOCAL_PREFIX_REPRIME", True)
        import core.ollama_opts as oo
        self._p(oo, "model_resident", return_value=True)
        self._p(bc, "_REPRIME_DEBOUNCE_S", 0.05)
        bc._reprime_running[0] = False
        bc._reprime_again[0] = False
        bc._prompt_rebuild_pending[0] = False
        bc.conversation_history[:] = _pairs(2)
        bc._last_owner_turn_at[0] = time.monotonic() - 60.0

    def _prime(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(self.bc._reprime_once(), "primed")

    def _owner_turn(self, text="next question"):
        bc = self.bc
        bc._turn_in_progress[0] = True
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            bc._call_llm(text)
        bc._turn_in_progress[0] = False
        return out.getvalue()

    def _teams_vision(self):
        bc = self.bc
        with contextlib.redirect_stdout(io.StringIO()):
            th, _ = self._in_thread(
                lambda: bc._call_local_vision("q", [b"png"]),
                tag="teams-nudge")
            th.join(5)
            self.assertTrue(_wait_until(lambda: not bc._reprime_running[0]),
                            "re-prime worker never finished")

    def test_live_sequence_prime_then_teams_vision_then_turn_is_warm(self):
        # 15:23:07 [reprime] -> 15:23:16 [local-vision] (nudger) -> 15:24:06
        # turn. The vision call now re-primes, so the turn hits.
        self._prime()
        self._teams_vision()
        kinds = ["prime" if (p.get("options") or {}).get("num_predict") == 1
                 else "vision" if "images" in p["messages"][-1] else "turn"
                 for p in self.posted]
        self.assertEqual(kinds, ["prime", "vision", "prime"])
        log = self._owner_turn()
        self.assertRegex(log, r"\[reprime\] hit age=\d+s")
        self.assertNotIn("evicted", log)
        self.assertNotIn("stale", log)

    def test_without_the_trigger_the_turn_reports_the_eviction(self):
        self._p(self.bc, "LOCAL_REPRIME_AFTER_BACKGROUND_WINDOW_S", 0.0)
        self._prime()
        self._teams_vision()
        log = self._owner_turn()
        self.assertRegex(log, r"\[reprime\] evicted age=\d+s by=1\b")

    def test_hit_line_on_a_clean_prime(self):
        self._prime()
        self.assertRegex(self._owner_turn(), r"\[reprime\] hit age=0s")
        # consumed: the next turn logs nothing
        self.assertNotIn("[reprime]", self._owner_turn("and another"))

    def test_stale_line_carries_the_age(self):
        bc = self.bc
        self._prime()
        bc._reprime_primed_at[0] = time.monotonic() - 42.0
        bc.conversation_history.insert(0, {"role": "user", "content": "x"})
        bc.conversation_history.insert(1, {"role": "assistant",
                                           "content": "y"})
        self.assertRegex(self._owner_turn(), r"\[reprime\] stale age=4[12]s")

    def test_the_reprime_post_is_own_inference_but_not_an_eviction(self):
        bc = self.bc
        before = bc._lt.TRACKER.posts
        self._prime()
        self.assertEqual(self.seen_inflight, [1],
                         "the re-prime POST is not tracked as own inference")
        self.assertEqual(bc._lt.TRACKER.posts, before)
        self.assertRegex(self._owner_turn(), r"\[reprime\] hit")

    def test_a_typed_turn_takes_the_stale_check(self):
        # Typed / injected turns run the same get_response_with_animation ->
        # _call_llm path as spoken ones (main() differs only in voice=).
        bc = self.bc
        self._prime()
        self._p(bc, "pause_face_tracking")
        self._p(bc, "set_state")
        self._p(bc, "_thinking_loop")
        bc._turn_in_progress[0] = True
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            bc.get_response_with_animation("typed from the web page")
        self.assertRegex(out.getvalue(), r"\[reprime\] hit")
        src = inspect.getsource(bc.main)
        self.assertIn("reply = _run_llm_dispatch(text, voice=_injected_text "
                      "is None)", src)
        self.assertIn("reply = get_response_with_animation(text)",
                      inspect.getsource(bc._run_llm_dispatch_body))


# ════════════════════════════════════════════════════════════════════════════
#  5. Own-inference tracking for the system-pulse GPU check
# ════════════════════════════════════════════════════════════════════════════
class OwnInferenceTrackingTests(_Base):
    def test_chat_and_vision_posts_are_in_flight_while_they_run(self):
        bc = self.bc
        self.posted = self._local_llm()
        self._vision_ready()
        before = bc._lt.TRACKER.posts
        with contextlib.redirect_stdout(io.StringIO()):
            bc._call_local_llm("sys", [{"role": "user", "content": "x"}])
            bc._call_local_vision("q", [b"png"])
        self.assertEqual(self.seen_inflight, [1, 1])
        self.assertEqual(bc._lt.TRACKER.posts, before + 2)
        self.assertEqual(bc._lt.TRACKER.inflight, 0)
        self.assertTrue(bc._lt.own_inference_recent(10.0))


@requires_monolith
class ConfigDefaultsTests(MonolithGlobalsTestCase):
    def test_new_settings_reach_the_monolith_with_their_types(self):
        import core.config as cfg
        self.assertIsInstance(cfg.LOCAL_BACKGROUND_MAX_DEFER_S, float)
        self.assertEqual(cfg.LOCAL_BACKGROUND_MAX_DEFER_S, 120.0)
        self.assertIsInstance(cfg.LOCAL_REPRIME_AFTER_BACKGROUND_WINDOW_S,
                              float)
        self.assertEqual(cfg.LOCAL_REPRIME_AFTER_BACKGROUND_WINDOW_S, 600.0)
        # the monolith's globals come from `from core.config import *`
        self.assertEqual(self.bc.LOCAL_BACKGROUND_MAX_DEFER_S, 120.0)
        self.assertEqual(self.bc.LOCAL_REPRIME_AFTER_BACKGROUND_WINDOW_S,
                         600.0)


if __name__ == "__main__":
    unittest.main()
