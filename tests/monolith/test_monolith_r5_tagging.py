"""Speed plan R5 (2026-10-02): the background brain callers found untagged.

WHY
===
Ollama serves the local brain from ONE slot, and core/local_traffic's gate can
only hold a background call that is TAGGED (background_work). v2.0.126 tagged
learn_from_turn, the ambient extractor / judge / screen observer, the Teams
nudger, the session checkpoint and the LTM reflector. The R5 inventory found
four background callers still untagged -- the notification classifier's local
fallback (listener thread), Chappie's daemon passes, the hourly credits check
(a vision read) and the scheduled evening briefing's headline summaries (live
2026-10-01 22:12: three of them reached the model while the owner's mic was
recording) -- plus a bounded path (_ollama_chat_bounded) that sent no runner
options at all.

WHAT THESE TESTS PIN
====================
1. With BACKGROUND_TAG_STRICT on, each newly tagged caller -- driven through
   its REAL function and the real _call_local_llm / gate, with only the HTTP
   layer faked -- makes no POST while the owner is mid-turn or mid-sentence,
   then runs once he is done.
2. With the switch off (the shipped default) the same callers run at once,
   exactly as before, and the log says once that they would have waited;
   the first-tagged jobs keep waiting either way.
3. The owner's own calls never wait: the voice turn (main thread) and an
   untagged owner call on another thread POST at once with the switch on.
4. Every wait is bounded: a turn that never ends still releases the job at
   the cap + hard grace, and the notification classifier's own short cap
   ends a quiet-window wait early.
5. An untagged off-thread local call made mid-turn is named in the log once
   per caller (chat, vision, complete) and is not delayed.
6. _ollama_chat_bounded sends num_ctx (the shared resolver) and keep_alive.

Mutation: drop any one tag and its class-1 test goes red (a POST lands while
the turn is active); drop the strict wiring and class 2 goes red.
"""
from __future__ import annotations

import contextlib
import io
import os
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
from tests._skill_harness import load_skill_isolated  # noqa: E402


class _Resp:
    def __init__(self, body, ok=True, status_code=200):
        self._body = body
        self.ok = ok
        self.status_code = status_code
        self.text = "body"

    def json(self):
        return self._body


def _chat_body(content="urgent"):
    return {"model": "gemma-test",
            "message": {"role": "assistant", "content": content},
            "done": True, "prompt_eval_count": 600, "eval_count": 2}


@requires_monolith
class _Base(MonolithGlobalsTestCase):
    def setUp(self):
        super().setUp()
        bc = self.bc
        lt = bc._lt
        lt.GATE.configure(poll_s=0.02)
        self.addCleanup(lt.GATE.configure, poll_s=lt.DEFAULT_POLL_S)
        self.addCleanup(lt.GATE.reset)
        self._p(bc, "PROMPT_FREEZE_QUIET_S", 30.0)
        self._p(bc, "LOCAL_BACKGROUND_MAX_DEFER_S", 30.0)
        # No re-prime worker after a background POST: it would add its own
        # (fake) POST to the counts below.
        self._p(bc, "LOCAL_REPRIME_AFTER_BACKGROUND_WINDOW_S", 0.0)
        self._p(bc, "BACKGROUND_TAG_STRICT", True)
        import core.config as cfg
        self._p(cfg, "model_route", return_value="local")
        for cell in (bc._turn_in_progress, bc._utterance_in_progress):
            cell[0] = False
        bc._last_convo_activity[0] = 0.0
        self.addCleanup(self._idle)
        bc._untagged_local_seen.clear()
        self.addCleanup(bc._untagged_local_seen.clear)
        self.posted = self._local_llm()

    def _idle(self):
        bc = self.bc
        bc._turn_in_progress[0] = False
        bc._utterance_in_progress[0] = False

    def _p(self, *args, **kwargs):
        patcher = mock.patch.object(*args, **kwargs)
        m = patcher.start()
        self.addCleanup(patcher.stop)
        return m

    def _local_llm(self, content="urgent"):
        """Route every /api/chat POST to a recorder; returns the payloads."""
        bc = self.bc
        posted = []
        fake_req = mock.Mock()

        def _post(url, json=None, timeout=None, **_k):
            posted.append(json)
            return _Resp(_chat_body(content))
        fake_req.post.side_effect = _post
        fake_req.RequestException = Exception
        fake_req.Timeout = TimeoutError
        self._p(bc, "LOCAL_LLM_FALLBACK", True)
        self._p(bc, "_ollama_alive", return_value=True)
        self._p(bc, "_ollama_has_model", return_value=True)
        self._p(bc, "_get_local_llm_model", return_value="gemma-test")
        self._p(bc, "_next_local_llm_fallback", return_value=None)
        self._p(bc, "requests", fake_req)
        return posted

    def _thread(self, fn):
        out = {}

        def _t():
            try:
                out["value"] = fn()
            except BaseException as e:  # pragma: no cover - surfaced below
                out["error"] = e
        th = threading.Thread(target=_t, daemon=True)
        th.start()
        self.addCleanup(th.join, 5)
        return th, out

    def _held_during(self, cell, fn, *, done=None):
        """Run fn on a background thread while `cell` (the owner's turn or
        utterance) is set; assert nothing reached the model meanwhile, then
        clear it and let the job finish. Returns (result, log)."""
        cell[0] = True
        with contextlib.redirect_stdout(io.StringIO()) as log:
            th, out = self._thread(fn)
            time.sleep(0.35)
            self.assertTrue(th.is_alive(), "the job did not wait at all")
            self.assertEqual(self.posted, [],
                             "a background call reached the model mid-turn")
            if done is not None:
                self.assertFalse(done(), "the job's result landed mid-turn")
            cell[0] = False
            th.join(5)
        self.assertFalse(th.is_alive(), "the job never ran after the turn")
        self.assertNotIn("error", out, out.get("error"))
        return out.get("value"), log.getvalue()

    def _skill(self, name):
        mod, _ = load_skill_isolated(name, register=False)
        return mod


# ════════════════════════════════════════════════════════════════════════════
#  1. Switch on: each newly tagged caller defers, then runs
# ════════════════════════════════════════════════════════════════════════════
class NewlyTaggedCallersDeferTests(_Base):
    def test_notification_classifier_waits_out_the_turn(self):
        mod = self._skill("notification_triage")
        self._p(mod, "_cloud_allowed", return_value=False)
        verdict, log = self._held_during(
            self.bc._turn_in_progress,
            lambda: mod._classify_with_llm("App", "Title", "Body"))
        self.assertEqual(verdict, "urgent")
        self.assertEqual(len(self.posted), 1)
        self.assertIn("[bg-local] defer notification-triage (turn)", log)
        self.assertRegex(log, r"\[bg-local\] run notification-triage after "
                              r"\d+ ms \(released\)")
        # privacy: the gate's lines carry no toast text
        for line in log.splitlines():
            if "[bg-local]" in line:
                self.assertNotIn("Title", line)
                self.assertNotIn("Body", line)

    def test_chappie_pass_waits_out_the_sentence(self):
        mod = self._skill("chappie_consciousness")
        out, log = self._held_during(
            self.bc._utterance_in_progress, lambda: mod._llm("sys", "episode"))
        self.assertEqual(out, "urgent")
        self.assertEqual(len(self.posted), 1)
        self.assertIn("[bg-local] defer chappie (utterance)", log)

    def test_scheduled_evening_briefing_waits_then_speaks(self):
        mod = self._skill("evening_briefing")
        bc = self.bc
        spoken = []

        def _build():   # stands in for the headline summaries
            return bc._call_local_llm(
                "sys", [{"role": "user", "content": "Headline: x"}],
                max_tokens=20) or ""
        self._p(mod, "_build_briefing", side_effect=_build)
        self._p(mod, "_enqueue_speech", side_effect=spoken.append)
        self._p(mod, "_show_card_safe")
        self._p(mod, "_save_last_fired_date")
        out, log = self._held_during(
            bc._turn_in_progress, lambda: mod._fire_briefing("timed-out"),
            done=lambda: bool(spoken))
        self.assertEqual(out, "urgent")
        self.assertEqual(spoken, ["urgent"])
        self.assertIn("[bg-local] defer evening-briefing (turn)", log)

    def test_credits_check_waits_before_it_captures(self):
        mod = self._skill("credits_monitor")
        bc = self.bc
        self._p(bc, "_vision_goes_local", return_value=True)
        reads = []

        def _read():
            reads.append(bc._turn_in_progress[0])
            return (50.0, "BALANCE: $50.00")
        self._p(mod, "_read_credits_via_vision", side_effect=_read)
        self._p(mod, "_save_state")
        self._p(mod, "_enqueue_speech")
        _, log = self._held_during(
            bc._turn_in_progress, mod._check_and_maybe_alert,
            done=lambda: bool(reads))
        self.assertEqual(reads, [False], "the capture ran mid-turn")
        self.assertIn("[bg-local] defer credits-monitor (turn)", log)
        self.assertFalse(mod._check_lock.locked())


# ════════════════════════════════════════════════════════════════════════════
#  2. Switch off (the shipped default): today's timing, plus one shadow line
# ════════════════════════════════════════════════════════════════════════════
class SwitchOffShadowTests(_Base):
    def setUp(self):
        super().setUp()
        self._p(self.bc, "BACKGROUND_TAG_STRICT", False)

    def test_newly_tagged_caller_runs_at_once_and_logs_once(self):
        mod = self._skill("chappie_consciousness")
        self.bc._turn_in_progress[0] = True
        with contextlib.redirect_stdout(io.StringIO()) as log:
            th, out = self._thread(lambda: (mod._llm("s", "a"),
                                            mod._llm("s", "b")))
            th.join(5)
        self.assertFalse(th.is_alive(), "a shadow job waited")
        self.assertEqual(len(self.posted), 2)
        text = log.getvalue()
        self.assertEqual(text.count("[bg-local] shadow chappie would defer "
                                    "(turn)"), 2,
                         "one line per job: each _llm call is its own job")
        self.assertNotIn("[bg-local] defer chappie", text)

    def test_first_tagged_jobs_still_wait(self):
        bc = self.bc

        def _learn():
            with bc.background_local_work("learn_from_turn"):
                return bc._llm_quick("sys", "extract")
        out, log = self._held_during(bc._turn_in_progress, _learn)
        self.assertEqual(out, "urgent")
        self.assertIn("[bg-local] defer learn_from_turn (turn)", log)

    def test_the_switch_is_read_live_and_fails_safe(self):
        bc = self.bc
        self.assertFalse(bc._background_tag_strict())
        self._p(bc, "BACKGROUND_TAG_STRICT", True)
        self.assertTrue(bc._background_tag_strict())

        class _Bad:
            def __bool__(self):
                raise RuntimeError("unreadable")
        self._p(bc, "BACKGROUND_TAG_STRICT", _Bad())
        self.assertFalse(bc._background_tag_strict())

    def test_the_monolith_installed_the_switch_on_the_shared_gate(self):
        bc = self.bc
        self.assertIs(bc._lt.GATE._strict_src, bc._background_tag_strict)
        self.assertFalse(bc._lt.GATE.strict())


# ════════════════════════════════════════════════════════════════════════════
#  3. The owner never waits
# ════════════════════════════════════════════════════════════════════════════
class OwnerNeverWaitsTests(_Base):
    def test_the_voice_turn_posts_at_once(self):
        bc = self.bc
        bc._turn_in_progress[0] = True
        bc._utterance_in_progress[0] = True
        t0 = time.monotonic()
        self.assertEqual(bc._call_local_llm(
            "sys", [{"role": "user", "content": "hi"}]), "urgent")
        self.assertLess(time.monotonic() - t0, 2.0)
        self.assertEqual(len(self.posted), 1)

    def test_an_untagged_owner_call_on_another_thread_posts_at_once(self):
        bc = self.bc
        bc._turn_in_progress[0] = True
        with contextlib.redirect_stdout(io.StringIO()):
            th, out = self._thread(lambda: bc._llm_quick("sys", "recall"))
            th.join(5)
        self.assertFalse(th.is_alive(), "an owner call waited")
        self.assertEqual(out.get("value"), "urgent")
        self.assertEqual(len(self.posted), 1)


# ════════════════════════════════════════════════════════════════════════════
#  4. Bounded
# ════════════════════════════════════════════════════════════════════════════
class BoundedWaitTests(_Base):
    def test_a_turn_that_never_ends_still_releases_the_job(self):
        bc = self.bc
        self._p(bc, "LOCAL_BACKGROUND_MAX_DEFER_S", 0.2)
        bc._lt.GATE.configure(hard_grace_s=0.3)
        self.addCleanup(bc._lt.GATE.configure,
                        hard_grace_s=bc._lt.DEFAULT_HARD_GRACE_S)
        mod = self._skill("chappie_consciousness")
        bc._turn_in_progress[0] = True          # and it never clears
        t0 = time.monotonic()
        with contextlib.redirect_stdout(io.StringIO()) as log:
            th, out = self._thread(lambda: mod._llm("s", "u"))
            th.join(10)
        self.assertFalse(th.is_alive(), "unbounded wait")
        self.assertLess(time.monotonic() - t0, 5.0)
        self.assertEqual(len(self.posted), 1)
        self.assertRegex(log.getvalue(),
                         r"\[bg-local\] run chappie after \d+ ms \(forced\)")

    def test_the_classifier_cap_ends_a_quiet_window_wait_early(self):
        bc = self.bc
        self._p(bc, "LOCAL_BACKGROUND_MAX_DEFER_S", 60.0)
        self._p(bc, "_conversation_active", return_value=True)   # soft only
        mod = self._skill("notification_triage")
        self._p(mod, "_cloud_allowed", return_value=False)
        self._p(mod, "LLM_LOCAL_MAX_DEFER_S", 0.2)
        t0 = time.monotonic()
        with contextlib.redirect_stdout(io.StringIO()) as log:
            th, out = self._thread(
                lambda: mod._classify_with_llm("App", "T", "B"))
            th.join(10)
        self.assertFalse(th.is_alive())
        self.assertLess(time.monotonic() - t0, 5.0,
                        "the job's own cap was not applied")
        self.assertEqual(out.get("value"), "urgent")
        text = log.getvalue()
        self.assertIn("[bg-local] defer notification-triage (conversation)",
                      text)
        self.assertRegex(text, r"run notification-triage after \d+ ms "
                               r"\(forced\)")


# ════════════════════════════════════════════════════════════════════════════
#  5. An untagged off-thread call mid-turn is named (log only)
# ════════════════════════════════════════════════════════════════════════════
class UntaggedCallAuditTests(_Base):
    def _off_thread(self, fn):
        with contextlib.redirect_stdout(io.StringIO()) as log:
            th, out = self._thread(fn)
            th.join(5)
        self.assertFalse(th.is_alive(), "the audit delayed the call")
        return out, log.getvalue()

    def test_untagged_chat_call_mid_turn_is_named_once(self):
        bc = self.bc
        bc._turn_in_progress[0] = True

        def _stray_summary():
            return bc._call_local_llm("sys", [{"role": "user", "content": "x"}])
        out, log = self._off_thread(lambda: (_stray_summary(),
                                             _stray_summary()))
        self.assertEqual(len(self.posted), 2, "the audit must not hold calls")
        lines = [ln for ln in log.splitlines() if "untagged" in ln]
        self.assertEqual(len(lines), 1, log)
        self.assertIn("[bg-local] untagged chat call during the owner's turn: "
                      "caller=_stray_summary@", lines[0])
        # a second thread running the same caller is still the same caller
        _, log2 = self._off_thread(_stray_summary)
        self.assertNotIn("untagged", log2)

    def test_untagged_complete_call_mid_sentence_is_named(self):
        bc = self.bc
        bc._utterance_in_progress[0] = True

        def _site_writer():
            return bc._local_complete("sys", [{"role": "user", "content": "x"}])
        out, log = self._off_thread(_site_writer)
        self.assertEqual(out.get("value"), "urgent")
        self.assertIn("[bg-local] untagged complete call during the owner's "
                      "utterance: caller=_site_writer@", log)

    def test_untagged_vision_call_mid_turn_is_named(self):
        bc = self.bc
        self._p(bc, "LOCAL_VISION_MODEL", "gemma-test")
        self._p(bc, "_local_vision_usable", return_value=True)
        self._p(bc, "_local_vision_model_already_resident", return_value=True)
        self._p(bc, "_log_gpu_state")
        bc._RESOLVED_LOCAL_LLM_MODEL[0] = "gemma-test"
        bc._turn_in_progress[0] = True

        def _screen_peek():
            return bc._call_local_vision("what is on screen", [b"png"])
        out, log = self._off_thread(_screen_peek)
        self.assertEqual(len(self.posted), 1)
        self.assertIn("[bg-local] untagged vision call during the owner's "
                      "turn: caller=_screen_peek@", log)

    def test_tagged_main_thread_quiet_and_cloud_calls_are_not_named(self):
        bc = self.bc
        msgs = [{"role": "user", "content": "x"}]
        # quiet: nothing to collide with
        _, log = self._off_thread(lambda: bc._call_local_llm("s", msgs))
        self.assertNotIn("untagged", log)
        # tagged, made mid-turn: the gate's business, not the audit's
        def _tagged():
            with bc.background_local_work("ambient-extract"):
                return bc._call_local_llm("s", msgs)
        self.posted.clear()
        _, log = self._held_during(bc._turn_in_progress, _tagged)
        self.assertIn("[bg-local] defer ambient-extract (turn)", log)
        self.assertNotIn("untagged", log)
        bc._turn_in_progress[0] = True
        # the main thread is the voice turn itself
        with contextlib.redirect_stdout(io.StringIO()) as main_log:
            bc._call_local_llm("s", msgs)
        self.assertNotIn("untagged", main_log.getvalue())
        # the cloud chat route: no prefix to protect
        import core.config as cfg
        self._p(cfg, "model_route", return_value="cloud")
        _, log = self._off_thread(lambda: bc._call_local_llm("s", msgs))
        self.assertNotIn("untagged", log)

    def test_the_seen_set_is_bounded(self):
        bc = self.bc
        bc._turn_in_progress[0] = True
        bc._untagged_local_seen.update(
            f"chat:x{i}" for i in range(bc._UNTAGGED_LOCAL_SEEN_MAX))
        _, log = self._off_thread(
            lambda: bc._call_local_llm("s", [{"role": "user", "content": "x"}]))
        self.assertNotIn("untagged", log)
        self.assertEqual(len(bc._untagged_local_seen),
                         bc._UNTAGGED_LOCAL_SEEN_MAX)


# ════════════════════════════════════════════════════════════════════════════
#  6. The bounded path sends the runner options (R5d)
# ════════════════════════════════════════════════════════════════════════════
@requires_monolith
class BoundedPathOptionsTests(MonolithGlobalsTestCase):
    def _chat_kwargs(self, model):
        fake_client = mock.Mock()
        fake_client.chat.return_value = {"message": {"content": "ok"}}
        fake_ollama = types.ModuleType("ollama")
        fake_ollama.Client = mock.Mock(return_value=fake_client)
        with mock.patch.dict(sys.modules, {"ollama": fake_ollama}):
            out = self.bc._ollama_chat_bounded(
                model, [{"role": "user", "content": "hi"}])
        self.assertEqual(out, {"message": {"content": "ok"}})
        fake_client.chat.assert_called_once()
        return fake_client.chat.call_args.kwargs

    def test_payload_carries_num_ctx_and_keep_alive(self):
        kw = self._chat_kwargs("gemma4:26b-a4b-it-qat")
        self.assertEqual(kw["model"], "gemma4:26b-a4b-it-qat")
        self.assertEqual(kw["messages"], [{"role": "user", "content": "hi"}])
        self.assertEqual(kw["options"]["num_ctx"],
                         self.bc._local_num_ctx("gemma4:26b-a4b-it-qat"))
        self.assertEqual(kw["keep_alive"], "20m")

    def test_a_big_model_gets_the_tight_window(self):
        kw = self._chat_kwargs("qwen2.5:32b-instruct")
        self.assertEqual(kw["options"]["num_ctx"], 12288)

    def test_the_patched_resolver_wins_like_on_the_chat_path(self):
        with mock.patch.object(self.bc, "_local_num_ctx", return_value=4321):
            kw = self._chat_kwargs("gemma-test")
        self.assertEqual(kw["options"]["num_ctx"], 4321)


if __name__ == "__main__":
    unittest.main()
