"""A self-voiced action that said nothing must never silence the turn
(2026-10-05).

Live, v2.0.180, 00:49-00:53: five owner turns asked JARVIS to have the desk
device chat about a topic. The brain replied "Right away, sir. [ACTION:
<chat>, <topic>]", the device skill refused before a single line (its result
"... not started: binding." - the transcript did not name the device the way
the skill expects, or the sentence was "you got cut off, try that again"),
and JARVIS then said NOTHING: the reply's own "Right away, sir." was dropped
as self-voiced prose and the refusal was neither news nor a failure to
report. [turn-timing] showed synth_start=- first_play=-. The same request on
the routed (no-brain) path ran the chat and spoke within ~1-2.5 s.

Covers (bobert_companion._run_self_voiced / _self_voiced_did_talk /
_self_voiced_reply / _self_voiced_wait_ready, core/self_voiced.py):
  * a REPLAY of the live sequence through the real parse_and_run_actions and
    the real dialogue session / speak_line / core.dialogue.Runner, with a fake
    device client (it can only say lines - no motion API exists on it): the
    brain-path refusals each get ONE honest line, the hallucinated prose of
    the second turn is not spoken, the routed chat speaks its own lines and
    nothing more;
  * the routed path and the brain path say the SAME line for the same
    refusal;
  * the bounded readiness wait for another device chat still running on
    another thread: it ends inside the wait -> the chat starts; it does not
    -> the action is not run and JARVIS says so, within the bound;
  * a crash is reported (failure follow-up), an owner stop adds nothing,
    another thread's speech never counts as the action's own, _speak and
    _speak_line both count, the crash marker matches the dispatcher's result,
    every one of _dialogue_ready()'s reasons has a clause.

GENERIC fixtures only ("desk device"); the owner's words are paraphrased.
No real audio, no LLM, no network, no device.

    python -m unittest tests.monolith.test_monolith_self_voiced_silent
"""
from __future__ import annotations

import ast
import concurrent.futures
import contextlib
import inspect
import io
import re
import sys
import textwrap
import threading
import time
import types
import unittest
from unittest import mock

from tests._monolith_harness import load_monolith, requires_monolith
from tests.monolith.test_monolith_claim_validation import _Base as _DispatchBase

_OPENER = "Shall we, desk device?"
_DEVICE_LINE = "Hello there."
_CLOSER = "Indeed."
_HONEST = "I'm afraid the chat didn't start, sir."


class _FakeDeskClient:
    """The device end of a chat: it can say a line and report it finished.
    Nothing else - there is no drive / motion call on it at all."""

    def __init__(self):
        self.said: list = []

    def say(self, chunk):
        self.said.append(chunk)
        return "ok"

    def done(self):
        return True


@requires_monolith
class _Base(_DispatchBase):
    def setUp(self):
        super().setUp()
        bc = self.bc
        from core import device_speech_filter as dsf
        dsf._reset_cache_for_tests()
        self.addCleanup(dsf._reset_cache_for_tests)
        # A normal, awake, post-boot, non-staging session (the dialogue
        # tests' base), plus the turn machinery the claim-validation base
        # already stubs (speech recorded in self.spoken, no LLM).
        self._p(bc, "_is_staging", return_value=False)
        self._p(bc, "_mic_input_disabled", return_value=False)
        self._p(bc, "_session_start_time", time.time() - 1000.0)
        self._p(bc, "DIALOGUE_ENABLED", True)
        self._p(bc, "_tts_muted", [False])
        self._p(bc, "_mic_muted", [False])
        self._p(bc, "_sleep_mode", [False])
        self._p(bc, "_standby_mode", [False])
        self._p(bc, "_realtime_session", [None])
        self._p(bc, "_dialogue_active", [False])
        self._p(bc, "_dialogue_current", [None])
        self._p(bc, "_speech_hold_until", [0.0])
        self._p(bc, "_turn_hold_until", [0.0])
        self._p(bc, "_turn_hold_reason", [""])
        self._p(bc, "_pathb_mic_active", [False])
        self._p(bc, "_reprime_after_background", return_value=False)
        self._p(bc, "_UTTERANCE_ROUTES", [])
        self._p(bc, "SKILL_ROUTES_ENABLED", True, create=True)
        self._p(bc, "SELF_VOICED_ACTIONS", set())
        self._p(bc, "_last_user_text", [None])
        self._p(bc, "conversation_history", [])
        self._p(bc, "_SELF_VOICED_READY_WAIT_S", 8.0, create=True)
        # The recorder reports the line heard, like the real _speak, so a
        # dialogue line reads "spoken" and the chat goes on.
        self._p(bc, "_speak",
                side_effect=lambda t, *a, **k: self.spoken.append(t) or True)
        self.client = _FakeDeskClient()
        self.runs: list = []
        self.used: list = []
        self._actions["desk_chat"] = self._desk_chat
        self.assertTrue(bc.register_self_voiced("desk_chat"))
        self.gfr = self._p(bc, "get_followup_response",
                           side_effect=[None] * 8)

    # A device-chat action shaped like the private chat skill: the owner's
    # sentence must name the device (or be a bare "again") and is consumed
    # on every attempt, JARVIS must be ready, then a Runner over the REAL
    # session / speak_line. Its refusals are not spoken by the skill itself.
    def _desk_chat(self, arg=""):
        bc = self.bc
        from core import dialogue as dlg
        raw = bc._last_user_text[0]
        t = str(raw or "").lower()
        self.runs.append(arg)
        if any(u is raw for u in self.used):
            return "Chat not started: binding."
        self.used.append(raw)
        named = "desk device" in t
        encore = bool(re.fullmatch(r"(?:jarvis,? )?(?:do (?:it|that) )?again\.?",
                                   t.strip()))
        if not (named or (encore and arg in ("", "encore"))):
            return "Chat not started: binding."
        why = bc._dialogue_ready()
        if why:
            return f"Chat not started: {why}."
        with bc._dialogue_session("desk device", max_s=30) as ds:
            runner = dlg.Runner(
                speak_self=lambda text, final: bc._speak_line(text),
                device_say=self.client.say, device_done=self.client.done,
                listen=lambda until, *, beat_s, max_s: dlg.ListenCapture(
                    available=False),
                session=ds, beat_s=0.0, sleep=lambda s: None)
            fut = concurrent.futures.Future()
            fut.set_result([dlg.Line("device", _DEVICE_LINE, (_DEVICE_LINE,)),
                            dlg.Line("self", _CLOSER, final=True)])
            out = runner.run(_OPENER, fut, lambda: None,
                             script_deadline=time.monotonic() + 1.0)
        return f"Chat finished: {out.lines_spoken} lines, {out.reason}."

    def _route_desk(self):
        self.assertTrue(self.bc.register_utterance_route(
            lambda t: ("[ACTION: desk_chat]"
                       if "talk to the desk device" in t.lower() else None),
            "desk device"))

    def _brain_turn(self, text, reply):
        """A turn the brain answers (the routed path did not claim it)."""
        bc = self.bc
        bc._last_user_text[0] = text          # _call_llm records it live
        self.llm = self._p(bc, "get_response_with_animation",
                           return_value=reply)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            bc._run_llm_dispatch(text)
        return buf.getvalue()

    def _routed_turn(self, text):
        buf = io.StringIO()
        self.llm = self._p(self.bc, "get_response_with_animation",
                           return_value="the brain must not be asked")
        with contextlib.redirect_stdout(buf):
            self.bc._run_llm_dispatch(text)
        self.llm.assert_not_called()
        return buf.getvalue()


# ── the live sequence, replayed ──────────────────────────────────────────
class LiveReplayTests(_Base):

    def test_misheard_name_on_the_brain_path_says_why(self):
        # 00:49:23 - the brain understood, the skill's binding did not.
        out = self._brain_turn(
            "Jarvis, talk to the desk divice about phones.",
            "[intent:confirmation] As you wish, sir. "
            "[ACTION: desk_chat, phones]")
        self.assertEqual(self.runs, ["phones"])
        self.assertEqual(self.client.said, [])
        self.assertEqual(self.spoken, [_HONEST],
                         "a self-voiced action that said nothing left the "
                         "turn silent")
        self.assertIn("said nothing", out)
        self.gfr.assert_not_called()        # deterministic: no LLM round

    def test_the_brains_invented_prose_is_not_spoken(self):
        # 00:49:41 - "Indeed, sir. [ACTION: ..., encore] It was quite the
        # spirited debate ..." about a chat that never happened.
        self._brain_turn(
            "Jarvis, talked to the desk divice about phones.",
            "[intent:amused] Indeed, sir. [ACTION: desk_chat, encore] It was "
            "quite the lively debate, though the device preferred the "
            "hardware.")
        self.assertEqual(self.spoken, [_HONEST])
        for line in self.spoken:
            self.assertNotIn("debate", line)

    def test_a_routed_chat_speaks_its_own_lines_and_nothing_more(self):
        # 00:49:52 - the routed path: the chat runs and speaks for itself.
        self._route_desk()
        out = self._routed_turn("Jarvis, talk to the desk device.")
        self.assertEqual(self.client.said, [_DEVICE_LINE])
        self.assertEqual(self.spoken, [_OPENER, _CLOSER])
        self.assertIn("[self-voiced] the action did its own talking", out)

    def test_the_same_chat_emitted_twice_adds_nothing_after_it_ran(self):
        # The second token is refused (the sentence was used by the first):
        # "the chat didn't start" after a chat that ran would be false.
        out = self._brain_turn(
            "Jarvis, talk to the desk device about phones.",
            "Right away, sir. [ACTION: desk_chat, phones] "
            "[ACTION: desk_chat, encore]")
        self.assertEqual(self.runs, ["phones", "encore"])
        self.assertEqual(self.client.said, [_DEVICE_LINE])
        self.assertEqual(self.spoken, [_OPENER, _CLOSER])
        self.assertIn("[self-voiced] the action did its own talking", out)

    def test_cut_off_then_try_that_again_says_why(self):
        # 00:53:06 / 00:53:34 - "you got cut off there, try that again":
        # not a bare encore, so the skill refused; JARVIS said nothing.
        self._brain_turn(
            "Jarvis, you were cut off there. Go ahead and try that again.",
            "[intent:confirmation] Understood, sir. I'll keep it brief. "
            "[ACTION: desk_chat, encore]")
        self.assertEqual(self.client.said, [])
        self.assertEqual(self.spoken, [_HONEST])

    def test_the_whole_sequence(self):
        self._route_desk()
        steps = (
            ("brain", "Jarvis, talk to the desk divice about phones.",
             "Right away, sir. [ACTION: desk_chat, phones]", [_HONEST]),
            ("route", "Jarvis, talk to the desk device.", None,
             [_OPENER, _CLOSER]),
            ("brain", "Jarvis, have the desk divise chat about a movie hero.",
             "Of course, sir. [ACTION: desk_chat, a movie hero]", [_HONEST]),
            ("route", "Jarvis, talk to the desk device about a movie hero.",
             None, [_OPENER, _CLOSER]),
            ("brain", "Jarvis, you got cut off. Finish that one.",
             "Right away, sir. [ACTION: desk_chat, encore]", [_HONEST]),
        )
        for kind, text, reply, want in steps:
            with self.subTest(text=text):
                self.spoken.clear()
                if kind == "route":
                    self._routed_turn(text)
                else:
                    self._brain_turn(text, reply)
                self.assertEqual(self.spoken, want)
        self.assertEqual(self.client.said, [_DEVICE_LINE, _DEVICE_LINE])


# ── the routed path and the brain path say the same thing ────────────────
class SameFailureSpeechTests(_Base):

    def test_route_and_brain_say_the_same_line(self):
        bc = self.bc
        self._p(bc, "_mic_muted", [True])
        want = ("I'm afraid the chat didn't start, sir; the microphone is "
                "muted.")
        self._route_desk()
        self._routed_turn("Jarvis, talk to the desk device about pizza.")
        routed = list(self.spoken)
        self.spoken.clear()
        self._brain_turn("Jarvis, have the desk device talk about pizza.",
                         "Right away, sir. [ACTION: desk_chat, pizza]")
        self.assertEqual(routed, [want])
        self.assertEqual(self.spoken, [want])


# ── another chat still running on another thread ────────────────────────
class ReadinessWaitTests(_Base):
    """The previous chat may still be finishing on another thread (an
    after-reply encore, a web-panel chat, the device finishing its line after
    the owner's wake word cut it). The action waits for it, bounded."""

    def _other_chat(self, release_after=None):
        bc = self.bc
        entered = threading.Event()
        release = threading.Event()

        def run():
            with bc._dialogue_session("desk device", max_s=60):
                entered.set()
                release.wait(10.0)

        t = threading.Thread(target=run, name="other-chat", daemon=True)
        t.start()
        self.assertTrue(entered.wait(5.0))

        def _cleanup():
            release.set()
            t.join(5.0)
        self.addCleanup(_cleanup)
        if release_after is not None:
            timer = threading.Timer(release_after, release.set)
            timer.daemon = True
            timer.start()
            self.addCleanup(timer.cancel)
        return t

    def test_it_ends_inside_the_wait_so_the_chat_starts(self):
        self._route_desk()
        self._other_chat(release_after=0.3)
        t0 = time.monotonic()
        out = self._routed_turn("Jarvis, talk to the desk device.")
        waited = time.monotonic() - t0
        self.assertEqual(self.spoken, [_OPENER, _CLOSER])
        self.assertEqual(self.client.said, [_DEVICE_LINE])
        self.assertGreaterEqual(waited, 0.25)
        self.assertIn("another device chat is still running", out)
        self.assertIn("the other chat ended after", out)
        self.bc._processing_filler.cancel.assert_any_call("self-voiced-wait")

    def test_it_never_ends_so_jarvis_says_so_within_the_bound(self):
        self._p(self.bc, "_SELF_VOICED_READY_WAIT_S", 0.4, create=True)
        self._route_desk()
        self._other_chat()
        t0 = time.monotonic()
        out = self._routed_turn("Jarvis, talk to the desk device.")
        waited = time.monotonic() - t0
        from core import self_voiced as sv
        self.assertEqual(self.spoken, [sv.BUSY_LINE])
        self.assertEqual(self.runs, [], "the chat was started anyway")
        self.assertEqual(self.client.said, [])
        self.assertGreaterEqual(waited, 0.35)
        self.assertLess(waited, 3.0, "the wait was not bounded")
        self.assertIn("not started", out)

    def test_the_brain_path_waits_the_same_way(self):
        self._p(self.bc, "_SELF_VOICED_READY_WAIT_S", 0.4, create=True)
        self._other_chat()
        self._brain_turn("Jarvis, talk to the desk device about rain.",
                         "Right away, sir. [ACTION: desk_chat, rain]")
        from core import self_voiced as sv
        self.assertEqual(self.spoken, [sv.BUSY_LINE])
        self.assertEqual(self.runs, [])

    def test_the_wait_is_clamped(self):
        bc = self.bc
        self._p(bc, "_SELF_VOICED_READY_WAIT_S", 1e9)
        calls = []
        seq = iter([True, True, False])
        self._p(bc, "_other_dialogue_running",
                side_effect=lambda: calls.append(1) or next(seq))
        sleeps = []
        self._p(bc.time, "sleep", side_effect=lambda s: sleeps.append(s))
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            self.assertEqual(bc._self_voiced_wait_ready("desk_chat"), "")
        self.assertIn("up to 15s", buf.getvalue())
        self.assertTrue(all(s <= bc._SELF_VOICED_READY_POLL_S for s in sleeps))

    def test_no_wait_for_a_chat_on_this_very_thread(self):
        bc = self.bc
        with contextlib.redirect_stdout(io.StringIO()):
            with bc._dialogue_session("desk device", max_s=10):
                self.assertFalse(bc._other_dialogue_running())
                self.assertEqual(bc._self_voiced_wait_ready("desk_chat"), "")


# ── the rest of the contract ─────────────────────────────────────────────
class ContractTests(_Base):

    def setUp(self):
        # The REAL _speak, before the dispatch base replaces it with a
        # recorder (test_the_real_speak_counts_a_voiced_line).
        self._real_speak = load_monolith()._speak
        super().setUp()

    def test_a_crash_is_reported_not_swallowed(self):
        def boom(arg=""):
            raise RuntimeError("transport gone")
        self._actions["desk_chat"] = boom
        told = "I'm afraid the chat fell over, sir."
        self.gfr.side_effect = [told] + [None] * 8
        self._brain_turn("Jarvis, talk to the desk device.",
                         "Right away, sir. [ACTION: desk_chat]")
        self.gfr.assert_called()
        names = [n for n, _r in self.gfr.call_args_list[0].args[0]]
        self.assertEqual(names, ["desk_chat"])
        self.assertEqual(self.spoken, [told])

    def test_an_owner_stop_before_the_first_line_adds_nothing(self):
        self._actions["desk_chat"] = (
            lambda arg="": "Chat finished: 0 lines, wake.")
        self._brain_turn("Jarvis, talk to the desk device.",
                         "Right away, sir. [ACTION: desk_chat]")
        self.assertEqual(self.spoken, [])
        self.gfr.assert_not_called()

    def test_another_threads_speech_is_not_the_actions_own(self):
        bc = self.bc

        def chat(arg=""):
            t = threading.Thread(target=lambda: bc._speak_line("Tea, sir?"))
            t.start()
            t.join(5.0)
            return "Chat not started: binding."
        self._actions["desk_chat"] = chat
        self._brain_turn("Jarvis, talk to the desk device.",
                         "Right away, sir. [ACTION: desk_chat]")
        self.assertEqual(self.spoken, ["Tea, sir?", _HONEST])

    def test_a_skills_own_terminal_line_is_spoken_as_is(self):
        bc = self.bc
        line = "I didn't catch which device you meant, sir."
        self._actions["desk_chat"] = (
            lambda arg="": bc._TERMINAL_FAILURE_PREFIX + line)
        self._brain_turn("Jarvis, talk to the desk divice.",
                         "Right away, sir. [ACTION: desk_chat]")
        self.assertEqual(self.spoken, [line])

    def test_a_mixed_reply_still_says_the_line(self):
        self._stub("desk_lamp", "ok")
        self._brain_turn(
            "Jarvis, lamp on and talk to the desk divice.",
            "On it, sir. [ACTION: desk_lamp] [ACTION: desk_chat]")
        self.assertEqual(self.calls["desk_lamp"], [""])
        self.assertTrue(self.spoken)
        self.assertEqual(self.spoken[-1], _HONEST)

    def test_did_talk_and_reply_predicates(self):
        bc = self.bc
        fin = "Chat finished: 3 lines, done."
        term = bc._TERMINAL_FAILURE_PREFIX + _HONEST
        crash = "Oops, sir. (action failed; class=unknown; RuntimeError: x)"
        defer = bc._ANSWER_FIRST_DEFERRED_PREFIXES[0] + " desk_chat(x)"
        self.assertTrue(bc._self_voiced_did_talk("desk_chat", fin))
        self.assertTrue(bc._self_voiced_did_talk("DESK_CHAT", fin))
        for res in (term, crash, defer):
            with self.subTest(res=res):
                self.assertFalse(bc._self_voiced_did_talk("desk_chat", res))
        self.assertFalse(bc._self_voiced_did_talk("desk_lamp", fin))
        self.assertTrue(bc._all_self_voiced([("desk_chat", fin, False)]))
        self.assertFalse(bc._all_self_voiced([("desk_chat", term, False)]))
        self.assertTrue(bc._self_voiced_reply([("desk_chat", term, False)]))
        self.assertFalse(bc._self_voiced_reply([("desk_chat", defer, False)]))
        self.assertFalse(bc._self_voiced_reply([]))
        # A silent self-voiced failure counts as a failure for the
        # acknowledgement drop / result hold.
        self.assertEqual(bc._failed_or_refused_actions(
            [("desk_chat", term, False)]), ["desk_chat"])
        self.assertEqual(bc._failed_or_refused_actions(
            [("desk_chat", fin, False)]), [])

    def test_the_crash_marker_matches_the_dispatcher(self):
        bc = self.bc

        def boom(arg=""):
            raise RuntimeError("transport gone")
        self._actions["boom_x"] = boom
        with contextlib.redirect_stdout(io.StringIO()):
            _clean, results = bc.parse_and_run_actions("[ACTION: boom_x]")
        self.assertEqual([r[0] for r in results], ["boom_x"])
        self.assertIn(bc._ACTION_CRASH_MARK, results[0][1])

    def test_speak_line_and_speak_both_count(self):
        bc = self.bc
        n0 = bc._spoken_here()
        bc._speak_line("One.")
        self.assertEqual(bc._spoken_here(), n0 + 1)
        # A line refused before playback is not counted.
        self._p(bc, "_tts_muted", [True])
        self.assertEqual(bc._speak_line("Two."), "muted")
        self.assertEqual(bc._spoken_here(), n0 + 1)

    def test_the_real_speak_counts_a_voiced_line(self):
        bc = self.bc
        self._p(bc, "_speak", self._real_speak)
        self._p(bc, "_is_staging", return_value=True)
        rec = []
        fake = types.ModuleType("staging_instance")
        fake.record_reply = lambda *a, **k: rec.append(a[0])
        with mock.patch.dict(sys.modules, {"staging_instance": fake}):
            n0 = bc._spoken_here()
            bc._speak("[intent:calm] ")             # nothing audible
            self.assertEqual(bc._spoken_here(), n0)
            bc._speak("Good evening, sir.")
            self.assertEqual(bc._spoken_here(), n0 + 1)
        self.assertEqual(rec, ["Good evening, sir."])

    def test_every_dialogue_ready_reason_has_a_clause(self):
        from core import self_voiced as sv
        src = textwrap.dedent(inspect.getsource(self.bc._dialogue_ready))
        reasons = {n.value for n in ast.walk(ast.parse(src))
                   if isinstance(n, ast.Return)
                   for n in [n.value] if isinstance(n, ast.Constant)
                   and isinstance(n.value, str) and n.value}
        self.assertGreaterEqual(len(reasons), 8, reasons)    # blindness floor
        reasons.discard("staging")          # a staging instance never speaks
        self.assertEqual(sorted(reasons - set(sv.REASON_CLAUSES)), [])



if __name__ == "__main__":
    unittest.main()
