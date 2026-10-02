"""After-reply hooks (2026-10-01): skill_utils["register_after_reply"](fn).

A skill can follow up an owner turn once its reply has been spoken. For every
owner turn the LLM path answers (mic or typed), each hook is called twice with
one ctx dict: stage "ready" right BEFORE the reply is spoken (so slow work can
start in parallel), then stage "spoken" once the whole turn has been said,
where it may return ONE callable, an encore, that the main loop runs once,
right after the turn, on a bounded thread of its own.

Pinned here:
  * registration: True, idempotent, a reloaded skill's hook replaces its old
    copy, bounded, non-callables refused, skill_utils + JarvisServices wired;
  * the call order (ready -> speech -> spoken -> encore) and the ctx keys and
    values (reply_text is the prose heard: tags stripped, the streamed lead put
    back; actions; question; barged; failed; typed);
  * owner turns only: nothing unless the main loop armed the turn, never in
    staging, for a test harness's inject, asleep / in standby / for an ambient
    answer-then-quiet turn, or for a dispatch on another thread;
  * isolation: a raising hook or encore never reaches the turn and the log
    carries no text; a slow hook never holds up the reply (time-boxed, run in
    parallel), is never called twice at once and is dropped after three late
    calls in a row;
  * the encore: once, after the turn, never after a barged / failed / raised
    turn, one per turn, skipped asleep / staging, bounded, and a device
    dialogue it opens owns the microphone (the main loop's capture yields).

GENERIC fixtures only ("desk device"). No real audio, no LLM, no network.

    python -m unittest tests.monolith.test_monolith_after_reply
"""
from __future__ import annotations

import contextlib
import inspect
import io
import threading
import time
import unittest
from unittest import mock

from tests._monolith_harness import requires_monolith
from tests.monolith.test_monolith_claim_validation import _Base

USER = "what a lovely day"
REPLY = "[intent:wry] It is, sir. Sunny."
HEARD = "It is, sir. Sunny."
# Every ctx key a consumer may read (the device-skill contract reads all but
# "typed").
CTX_KEYS = {"stage", "user_text", "reply_text", "actions", "question",
            "barged", "failed", "typed"}
OWNER_WORDS = "the owner's private words"


class _Hook:
    """A recording hook: (stage, ctx copy, speech so far, thread name) per
    call. Returns ``encore`` at "spoken"; can raise or block at a stage."""

    def __init__(self, test, encore=None, raise_at=(), block_at=(),
                 sleep_s=None):
        self.test = test
        self.encore = encore
        self.raise_at = tuple(raise_at)
        self.block_at = tuple(block_at)
        self.sleep_s = dict(sleep_s or {})
        self.gate = threading.Event()
        self.entered = threading.Event()
        self.calls: list = []

    def __call__(self, ctx):
        stage = ctx.get("stage")
        self.calls.append((stage, dict(ctx), list(self.test.spoken),
                           threading.current_thread().name))
        self.entered.set()
        if stage in self.block_at:
            self.gate.wait(5.0)
        if stage in self.sleep_s:
            time.sleep(self.sleep_s[stage])
        if stage in self.raise_at:
            raise RuntimeError(OWNER_WORDS)
        return self.encore if stage == "spoken" else None

    def stages(self):
        return [c[0] for c in self.calls]


class _Encore:
    def __init__(self, raise_exc=None, block=False):
        self.runs: list = []
        self.raise_exc = raise_exc
        self.block = block
        self.gate = threading.Event()
        self.entered = threading.Event()

    def __call__(self):
        self.runs.append(threading.current_thread().name)
        self.entered.set()
        if self.block:
            self.gate.wait(5.0)
        if self.raise_exc is not None:
            raise self.raise_exc
        return "encore done"


@requires_monolith
class _AfterReplyBase(_Base):
    def setUp(self):
        super().setUp()
        bc = self.bc
        self._p(bc, "_AFTER_REPLY_HOOKS", [])
        self._p(bc, "_after_reply_turn", [None])
        self._p(bc, "_after_reply_encore_thread", [None])
        self._p(bc, "_UTTERANCE_ROUTES", [])
        # Tests import the monolith in staging (JARVIS_STAGING=1): an awake,
        # non-staging owner session here.
        self._p(bc, "_is_staging", return_value=False)
        self._p(bc, "_sleep_mode", [False])
        self._p(bc, "_standby_mode", [False])
        self._p(bc, "_resume_to_ambient", [False])
        self._p(bc, "_last_inject_source", [None])
        self.history: list = []
        self._p(bc, "conversation_history", self.history)
        self._threads: list = []
        self.addCleanup(self._release)

    def _out(self, fn, *a, **k):
        """(fn's return value, what it printed)."""
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            r = fn(*a, **k)
        return r, buf.getvalue()

    def _release(self):
        for t in self._threads:
            for obj in (getattr(t, "gate", None),):
                if obj is not None:
                    obj.set()
        t = self.bc._after_reply_encore_thread[0]
        if t is not None:
            t.join(5.0)
        for rec in list(self.bc._AFTER_REPLY_HOOKS):
            b = rec.get("busy")
            if b is not None:
                b.join(5.0)

    def hook(self, **kw):
        h = _Hook(self, **kw)
        self._threads.append(h)
        self.assertTrue(self.bc.register_after_reply(h))
        return h

    def encore(self, **kw):
        e = _Encore(**kw)
        self._threads.append(e)
        return e

    def turn(self, text=USER, reply=REPLY, followups=(), typed=False,
             arm=True, run_encore=True):
        """One owner turn the way main() runs it: arm, dispatch, encore.
        Returns the captured stdout."""
        bc = self.bc
        self._p(bc, "get_response_with_animation", return_value=reply)
        self.gfr = self._p(bc, "get_followup_response",
                           side_effect=list(followups) + [None] * 8)

        def run():
            if arm:
                bc._after_reply_begin(text, typed)
            bc._run_llm_dispatch(text)
            if run_encore:
                bc._after_reply_run_encore()

        return self._out(run)[1]


# ════════════════════════════════════════════════════════════════════════════
#  registration
# ════════════════════════════════════════════════════════════════════════════
class RegistrationTests(_AfterReplyBase):
    def test_true_and_idempotent(self):
        fn = lambda ctx: None  # noqa: E731
        self.assertTrue(self._quiet(self.bc.register_after_reply, fn))
        self.assertTrue(self._quiet(self.bc.register_after_reply, fn))
        self.assertEqual(len(self.bc._AFTER_REPLY_HOOKS), 1)

    def test_non_callable_is_refused(self):
        for bad in (None, "hook", 3):
            self.assertFalse(self.bc.register_after_reply(bad))
        self.assertEqual(self.bc._AFTER_REPLY_HOOKS, [])

    def test_the_list_is_bounded(self):
        bc = self.bc
        for _ in range(bc._AFTER_REPLY_MAX_HOOKS):
            self.assertTrue(self._quiet(bc.register_after_reply,
                                        lambda ctx: None))
        out = []
        with mock.patch("builtins.print", side_effect=out.append):
            self.assertFalse(bc.register_after_reply(lambda ctx: None))
        self.assertEqual(len(bc._AFTER_REPLY_HOOKS), bc._AFTER_REPLY_MAX_HOOKS)
        self.assertTrue(any("REFUSED" in str(o) for o in out))

    def test_a_reloaded_skill_replaces_its_hook(self):
        def make():
            def hook(ctx):
                return None
            hook.__qualname__ = "hook"
            hook.__module__ = "skill_desk_device"
            return hook
        old, new = make(), make()
        self._quiet(self.bc.register_after_reply, old)
        self._quiet(self.bc.register_after_reply, new)
        self.assertEqual([r["fn"] for r in self.bc._AFTER_REPLY_HOOKS], [new])
        self.assertEqual(self.bc._AFTER_REPLY_HOOKS[0]["label"],
                         "skill_desk_device.hook")

    def test_skill_utils_and_services_are_wired(self):
        from core.services import JarvisServices
        su = self.bc.skill_utils
        self.assertIn("register_after_reply", su)
        f1 = lambda ctx: None  # noqa: E731
        f2 = lambda ctx: None  # noqa: E731
        self.assertTrue(self._quiet(su["register_after_reply"], f1))
        svc = JarvisServices.from_skill_utils(su)
        self.assertTrue(self._quiet(svc.register_after_reply, f2))
        self.assertEqual([r["fn"] for r in self.bc._AFTER_REPLY_HOOKS],
                         [f1, f2])
        self.assertFalse(JarvisServices.from_skill_utils({})
                         .register_after_reply(f1))


# ════════════════════════════════════════════════════════════════════════════
#  call order + ctx
# ════════════════════════════════════════════════════════════════════════════
class CallOrderTests(_AfterReplyBase):
    def test_ready_before_the_reply_spoken_after_then_the_encore_once(self):
        enc = self.encore()
        h = self.hook(encore=enc)
        bc = self.bc
        self._p(bc, "get_response_with_animation", return_value=REPLY)
        self._p(bc, "get_followup_response", side_effect=[None] * 8)
        self._quiet(bc._after_reply_begin, USER, False)
        self._quiet(bc._run_llm_dispatch, USER)
        self.assertEqual(h.stages(), ["ready", "spoken"])
        self.assertEqual(h.calls[0][2], [], "ready must come BEFORE the speech")
        self.assertEqual(h.calls[1][2], [REPLY], "spoken comes after it")
        self.assertEqual(h.calls[0][3], "after-reply-hook")
        self.assertEqual(enc.runs, [], "the encore runs after the turn only")
        _, out = self._out(bc._after_reply_run_encore)
        self.assertEqual(enc.runs, ["after-reply-encore"])
        self.assertIn("encore done", out)
        self._quiet(bc._after_reply_run_encore)          # no second run
        self.assertEqual(enc.runs, ["after-reply-encore"])
        bc._heartbeat.assert_called()                    # stall watchdog fed

    def test_ctx_keys_and_values_for_a_chit_chat_turn(self):
        h = self.hook()
        self.turn()
        ready, spoken = h.calls[0][1], h.calls[1][1]
        for ctx, stage in ((ready, "ready"), (spoken, "spoken")):
            self.assertEqual(set(ctx), CTX_KEYS)
            self.assertEqual(ctx["stage"], stage)
            self.assertEqual(ctx["user_text"], USER)
            self.assertEqual(ctx["reply_text"], HEARD)   # [intent:..] stripped
            self.assertEqual(ctx["actions"], ())
            self.assertIs(ctx["question"], False)
            self.assertIs(ctx["barged"], False)
            self.assertIs(ctx["failed"], False)
            self.assertIs(ctx["typed"], False)

    def test_reply_text_puts_the_streamed_lead_back(self):
        # The flush buffer voiced "It is, sir." while the reply streamed; the
        # body speaks only the rest, but the owner heard both.
        h = self.hook()
        self.bc._stream_spoken_prefix[0] = "[intent:wry] It is, sir."
        self.turn()
        self.assertEqual(self.spoken, ["Sunny."])
        self.assertEqual(h.calls[0][1]["reply_text"], HEARD)

    def test_question_owner_or_reply(self):
        h = self.hook()
        self.turn(text="what's the capital of France", reply="Paris, sir.")
        self.assertIs(h.calls[0][1]["question"], True)
        h.calls.clear()
        self.turn(text="I'm a bit tired", reply="Shall I dim the lights, sir?")
        self.assertIs(h.calls[0][1]["question"], True)
        h.calls.clear()
        self.turn(text="I'm a bit tired", reply="Understandable, sir.")
        self.assertIs(h.calls[0][1]["question"], False)

    def test_an_exclamation_is_not_a_question(self):
        q = self.bc._after_reply_question
        for text in ("what a lovely day", "What a game!", "Jarvis, what an idea.",
                     "how nice", "Oh, how lovely."):
            self.assertIs(q(text, "Indeed, sir."), False, text)
        for text in ("what a lovely day?", "what is the time",
                     "how are you", "how long until dinner", "how about tea",
                     "what's the weather like"):
            self.assertIs(q(text, "Indeed, sir."), True, text)
        self.assertIs(q("lovely day", "Isn't it, sir?"), True)

    def test_actions_and_a_failed_action(self):
        h = self.hook(encore=self.encore())
        self._stub("desk_light", "I couldn't reach the desk light.")
        self._stub("desk_lamp", "Desk lamp on.")
        out = self.turn(text="turn on the desk light",
                        reply="[ACTION: desk_light, on] Right away, sir.",
                        followups=["[ACTION: desk_lamp, on] The lamp, then."])
        ready, spoken = h.calls[0][1], h.calls[1][1]
        self.assertEqual(ready["actions"], ("desk_light",))
        self.assertIs(ready["failed"], True)
        self.assertEqual(spoken["actions"], ("desk_light", "desk_lamp"))
        self.assertIs(spoken["failed"], True)
        self.assertIn("encore dropped (the turn failed)", out)
        self.assertEqual(h.encore.runs, [])

    def test_a_reply_that_could_not_be_played_is_failed(self):
        h = self.hook()
        self.bc._speak.side_effect = lambda t, *a, **k: False
        self.turn()
        self.assertIs(h.calls[0][1]["failed"], False)
        self.assertIs(h.calls[1][1]["failed"], True)

    def test_a_typed_turn_is_followed_and_says_so(self):
        h = self.hook()
        self.bc._last_inject_source[0] = "web"
        self.turn(typed=True)
        self.assertEqual(h.stages(), ["ready", "spoken"])
        self.assertIs(h.calls[0][1]["typed"], True)

    def test_each_hook_gets_its_own_ctx(self):
        def vandal(ctx):
            ctx["reply_text"] = "changed"
            ctx["actions"] = ("x",)
        self._quiet(self.bc.register_after_reply, vandal)
        h = self.hook()
        self.turn()
        self.assertEqual(h.calls[0][1]["reply_text"], HEARD)
        self.assertEqual(h.calls[1][1]["actions"], ())


# ════════════════════════════════════════════════════════════════════════════
#  barge / raise
# ════════════════════════════════════════════════════════════════════════════
class BargeTests(_AfterReplyBase):
    def test_a_barge_during_the_reply_drops_the_encore(self):
        h = self.hook(encore=self.encore())

        def speak(t, *a, **k):
            self.spoken.append(t)
            self.bc._tts_interrupt_seq[0] += 1           # accepted barge-in
        self.bc._speak.side_effect = speak
        out = self.turn()
        self.assertIs(h.calls[0][1]["barged"], False)
        self.assertIs(h.calls[1][1]["barged"], True)
        self.assertIn("encore dropped (the turn was barged)", out)
        self.assertEqual(h.encore.runs, [])

    def test_a_barge_while_streaming_silences_the_reply(self):
        h = self.hook()

        def llm(text):
            self.bc._tts_interrupt_seq[0] += 1           # barged mid-stream
            return REPLY
        self._p(self.bc, "get_response_with_animation", side_effect=llm)
        self._p(self.bc, "get_followup_response", side_effect=[None] * 8)
        self._quiet(self.bc._after_reply_begin, USER, False)
        self._quiet(self.bc._run_llm_dispatch, USER)
        self.assertEqual(self.spoken, [])
        self.assertIs(h.calls[0][1]["barged"], True)
        self.assertEqual(h.calls[0][1]["reply_text"], "")

    def test_a_raising_turn_reports_failed_and_disarms(self):
        h = self.hook(encore=self.encore())
        bc = self.bc
        self._p(bc, "get_response_with_animation", return_value=REPLY)
        self._p(bc, "get_followup_response", side_effect=[None] * 8)
        bc._speak.side_effect = RuntimeError("audio device gone")
        self._quiet(bc._after_reply_begin, USER, False)
        with self.assertRaises(RuntimeError):
            self._quiet(bc._run_llm_dispatch, USER)
        self.assertEqual(h.stages(), ["ready", "spoken"])
        self.assertIs(h.calls[0][1]["failed"], False)
        self.assertIs(h.calls[1][1]["failed"], True)
        self.assertIsNone(bc._after_reply_turn[0])
        self._quiet(bc._after_reply_run_encore)
        self.assertEqual(h.encore.runs, [])

    def test_a_turn_that_raised_before_its_reply_calls_nothing(self):
        h = self.hook()
        bc = self.bc
        self._p(bc, "get_response_with_animation",
                side_effect=RuntimeError("llm down"))
        self._quiet(bc._after_reply_begin, USER, False)
        with self.assertRaises(RuntimeError):
            self._quiet(bc._run_llm_dispatch, USER)
        self.assertEqual(h.calls, [])
        self.assertIsNone(bc._after_reply_turn[0])


# ════════════════════════════════════════════════════════════════════════════
#  owner turns only
# ════════════════════════════════════════════════════════════════════════════
class OwnerTurnOnlyTests(_AfterReplyBase):
    def test_an_unarmed_dispatch_calls_nothing(self):
        # Proactive lines, the web / tray paths and every other caller of
        # _run_llm_dispatch: only main() arms a turn.
        h = self.hook()
        self.turn(arm=False)
        self.assertEqual(h.calls, [])
        self.assertEqual(self.spoken, [REPLY])

    def test_staging_is_never_followed(self):
        h = self.hook(encore=self.encore())
        self.bc._is_staging.return_value = True
        self.assertIsNone(self._quiet(self.bc._after_reply_begin, USER))
        self.turn()
        self.assertEqual(h.calls, [])

    def test_a_test_harness_inject_is_not_the_owner(self):
        h = self.hook()
        self.bc._last_inject_source[0] = "test"
        self.turn(typed=True)
        self.assertEqual(h.calls, [])

    def test_sleep_standby_and_answer_then_quiet(self):
        h = self.hook()
        for cell in ("_sleep_mode", "_standby_mode", "_resume_to_ambient"):
            with self.subTest(cell=cell):
                getattr(self.bc, cell)[0] = True
                try:
                    self.turn()
                finally:
                    getattr(self.bc, cell)[0] = False
                self.assertEqual(h.calls, [])

    def test_no_hooks_arms_nothing(self):
        self.assertIsNone(self._quiet(self.bc._after_reply_begin, USER))
        self.assertIsNone(self.bc._after_reply_turn[0])

    def test_a_dispatch_on_another_thread_is_not_the_armed_turn(self):
        h = self.hook()
        bc = self.bc
        self._p(bc, "get_response_with_animation", return_value=REPLY)
        self._p(bc, "get_followup_response", side_effect=[None] * 8)
        turn = self._quiet(bc._after_reply_begin, USER, False)
        self.assertIsNotNone(turn)
        t = threading.Thread(target=lambda: self._quiet(bc._run_llm_dispatch,
                                                        USER), daemon=True)
        t.start()
        t.join(5.0)
        self.assertEqual(h.calls, [])
        self.assertIs(bc._after_reply_turn[0], turn)      # still armed
        self.assertEqual(turn.stage, "armed")

    def test_another_utterance_is_not_the_armed_turn(self):
        h = self.hook()
        bc = self.bc
        self._p(bc, "get_response_with_animation", return_value=REPLY)
        self._p(bc, "get_followup_response", side_effect=[None] * 8)
        self._quiet(bc._after_reply_begin, "something else", False)
        self._quiet(bc._run_llm_dispatch, USER)
        self.assertEqual(h.calls, [])

    def test_main_arms_only_answered_turns_and_runs_the_encore_after(self):
        src = inspect.getsource(self.bc.main)
        arm = src.index("_after_reply_begin(text, _injected_text is not None)")
        disp = src.index("reply = _run_llm_dispatch(text, voice=_injected_text "
                         "is None)")
        self.assertLess(arm, disp)
        self.assertEqual(src[arm:disp].count("\n"), 1, "arm right before it")
        for earlier in ("_handle_sleep_triggers(text)",
                        "handle_confirmation_response(text)",
                        "_run_voice_shortcuts(text)", "_maybe_orchestrate(text)",
                        "_device_speech_ignored(", "_self_echo_ignored("):
            self.assertLess(src.index(earlier), arm, earlier)
        enc = src.index("_after_reply_run_encore()")
        self.assertLess(src.index("learn_from_turn(text, reply"), enc)
        self.assertLess(src.index("_resume_to_ambient[0] = False"), enc)
        self.assertLess(enc, src.index("except Exception as _loop_exc"))
        body = inspect.getsource(self.bc._run_llm_dispatch_body)
        self.assertLess(body.index("_after_reply_ready(text, spoken_text"),
                        body.index("_after_reply_note(spoke=_speak(spoken_text))"))
        wrap = inspect.getsource(self.bc._run_llm_dispatch)
        self.assertLess(wrap.index("_filler_end_turn(_pf_turn)"),
                        wrap.index("_after_reply_spoken(text"))


# ════════════════════════════════════════════════════════════════════════════
#  isolation + time-box
# ════════════════════════════════════════════════════════════════════════════
class IsolationTests(_AfterReplyBase):
    def test_a_raising_hook_never_reaches_the_turn_and_logs_no_text(self):
        h = self.hook(raise_at=("ready", "spoken"))
        out = self.turn()
        self.assertEqual(h.stages(), ["ready", "spoken"])
        self.assertEqual(self.spoken, [REPLY])
        self.assertIn("ready failed: RuntimeError", out)
        self.assertIn("spoken failed: RuntimeError", out)
        self.assertNotIn(OWNER_WORDS, out)
        self.assertEqual(self.bc._AFTER_REPLY_HOOKS[0]["late"], 0)

    def test_a_raising_encore_never_reaches_the_loop(self):
        enc = self.encore(raise_exc=ValueError(OWNER_WORDS))
        self.hook(encore=enc)
        out = self.turn()
        self.assertEqual(enc.runs, ["after-reply-encore"])
        self.assertIn("encore failed: ValueError", out)
        self.assertNotIn(OWNER_WORDS, out)

    def test_a_non_callable_return_is_ignored(self):
        self.hook(encore="not callable")
        out = self.turn()
        self.assertIn("returned a non-callable; ignored", out)
        self.assertIsNone(self.bc._after_reply_turn[0])

    def test_a_slow_ready_never_holds_up_the_reply(self):
        self._p(self.bc, "_AFTER_REPLY_BUDGET_S", 0.02)
        h = self.hook(block_at=("ready",), encore=self.encore())
        seen = []

        def speak(t, *a, **k):
            seen.append(h.entered.is_set() and not h.gate.is_set())
            self.spoken.append(t)
        self.bc._speak.side_effect = speak
        t0 = time.monotonic()
        out = self.turn()
        self.assertLess(time.monotonic() - t0, 2.0)
        self.assertEqual(seen, [True], "spoken while the hook still ran")
        self.assertEqual(h.stages(), ["ready"], "never called twice at once")
        self.assertIn("ready: overran 20 ms (1/3)", out)
        self.assertIn("spoken: still busy with its last call (2/3)", out)
        self.assertEqual(h.encore.runs, [])

    def test_hooks_run_in_parallel_within_one_budget(self):
        self._p(self.bc, "_AFTER_REPLY_BUDGET_S", 0.25)
        a = self.hook(block_at=("ready",))
        b = self.hook(block_at=("ready",))
        fast = self.hook()
        at = []
        self.bc._speak.side_effect = (
            lambda t, *x, **k: at.append(time.monotonic()) or self.spoken.append(t))
        t0 = time.monotonic()
        self.turn()
        # One shared 0.25 s budget, not one per slow hook (that would be 0.5).
        self.assertLess(at[0] - t0, 0.45, "two slow hooks must share one budget")
        self.assertEqual(fast.stages(), ["ready", "spoken"])
        self.assertEqual(a.stages(), ["ready"])
        self.assertEqual(b.stages(), ["ready"])

    def test_a_late_ready_that_ends_during_the_reply_still_gets_spoken(self):
        self._p(self.bc, "_AFTER_REPLY_BUDGET_S", 0.03)
        enc = self.encore()
        h = self.hook(sleep_s={"ready": 0.08}, encore=enc)

        def speak(t, *a, **k):
            time.sleep(0.3)                              # the reply plays
            self.spoken.append(t)
        self.bc._speak.side_effect = speak
        out = self.turn()
        self.assertEqual(h.stages(), ["ready", "spoken"])
        self.assertIn("overran 30 ms (1/3)", out)
        self.assertEqual(enc.runs, ["after-reply-encore"])
        self.assertEqual(self.bc._AFTER_REPLY_HOOKS[0]["late"], 0,
                         "an on-time call resets the count")

    def test_three_late_calls_in_a_row_drop_the_hook(self):
        self._p(self.bc, "_AFTER_REPLY_BUDGET_S", 0.01)
        h = self.hook(sleep_s={"ready": 0.1, "spoken": 0.1})
        out = self.turn() + self.turn()
        self.assertIn("dropped: 3 late calls in a row", out)
        self.assertEqual(self.bc._AFTER_REPLY_HOOKS, [])
        n = len(h.calls)
        self.turn()
        self.assertEqual(len(h.calls), n, "a dropped hook is never called")
        self.assertEqual(self.spoken, [REPLY] * 3)


# ════════════════════════════════════════════════════════════════════════════
#  the encore
# ════════════════════════════════════════════════════════════════════════════
class EncoreTests(_AfterReplyBase):
    def test_one_encore_per_turn(self):
        first, second = self.encore(), self.encore()
        self.hook(encore=first)
        self.hook(encore=second)
        out = self.turn()
        self.assertEqual(first.runs, ["after-reply-encore"])
        self.assertEqual(second.runs, [])
        self.assertIn("encore dropped (one per turn)", out)

    def test_skipped_once_asleep_or_staging(self):
        for why, cell in (("asleep", "_standby_mode"), ("asleep", "_sleep_mode"),
                          ("staging", None)):
            with self.subTest(cell=cell):
                enc = self.encore()
                self.bc._AFTER_REPLY_HOOKS[:] = []
                self.hook(encore=enc)
                self.turn(run_encore=False)
                if cell:
                    getattr(self.bc, cell)[0] = True
                else:
                    self.bc._is_staging.return_value = True
                try:
                    _, out = self._out(self.bc._after_reply_run_encore)
                finally:
                    if cell:
                        getattr(self.bc, cell)[0] = False
                    else:
                        self.bc._is_staging.return_value = False
                self.assertIn(f"encore skipped ({why})", out)
                self.assertEqual(enc.runs, [])

    def test_the_wait_is_bounded_and_one_encore_runs_at_a_time(self):
        self._p(self.bc, "_AFTER_REPLY_ENCORE_MAX_S", 0.1)
        slow = self.encore(block=True)
        self.hook(encore=slow)
        t0 = time.monotonic()
        out = self.turn()
        self.assertLess(time.monotonic() - t0, 2.0)
        self.assertTrue(slow.entered.is_set())
        self.assertIn("encore still running after 0s; listening again", out)
        nxt = self.encore()
        self.bc._AFTER_REPLY_HOOKS[:] = []
        self.hook(encore=nxt)
        out = self.turn()
        self.assertIn("encore skipped (the last one is still running)", out)
        self.assertEqual(nxt.runs, [])
        slow.gate.set()
        self.bc._after_reply_encore_thread[0].join(5.0)
        self.turn()
        self.assertEqual(nxt.runs, ["after-reply-encore"])

    def test_a_dialogue_the_encore_opens_owns_the_microphone(self):
        bc = self.bc
        for name, value in (("_mic_input_disabled", mock.Mock(return_value=False)),
                            ("_session_start_time", time.time() - 1000.0),
                            ("DIALOGUE_ENABLED", True),
                            ("_tts_muted", [False]), ("_mic_muted", [False]),
                            ("_realtime_session", [None]),
                            ("_dialogue_active", [False]),
                            ("_dialogue_current", [None]),
                            ("_speech_hold_until", [0.0]),
                            ("_turn_hold_until", [0.0]),
                            ("_reprime_after_background", mock.Mock())):
            self._p(bc, name, value)
        self._p(bc, "_AFTER_REPLY_ENCORE_MAX_S", 0.2)
        from core import device_speech_filter as dsf
        dsf._reset_cache_for_tests()
        self.addCleanup(dsf._reset_cache_for_tests)
        inside = threading.Event()
        leave = threading.Event()

        def encore():
            with bc._dialogue_session("desk device", max_s=10):
                inside.set()
                leave.wait(5.0)
        self.addCleanup(leave.set)
        self.hook(encore=encore)
        self.assertFalse(bc._capture_holds_mic())
        self.turn()
        self.assertTrue(inside.is_set())
        # The main loop is back, and its next capture yields to the dialogue.
        self.assertTrue(bc._dialogue_holds_mic())
        self.assertTrue(bc._capture_holds_mic())
        leave.set()
        bc._after_reply_encore_thread[0].join(5.0)
        self.assertFalse(bc._capture_holds_mic())


if __name__ == "__main__":   # pragma: no cover
    unittest.main()
