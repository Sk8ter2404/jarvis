"""Turn checker wiring (TURN_CHECK_MODE, 2026-10-02).

core/turn_checker.py decides whether a finished turn failed in a way a retry
on Claude would fix; bobert_companion._turn_check_after_chain runs it once per
voice turn, after the follow-up chain and its close-out line.

Drives the REAL _run_llm_dispatch_body (parse_and_run_actions, the follow-up
loop, the claim validator) with stub actions, a recording _speak (the
answer-first harness), a stubbed cloud gate and a stubbed _claude_oneshot.
The jsonl log goes to a temp dir.

Covers:
  * 'off': nothing is checked, printed or written;
  * 'shadow': a failed turn prints one [turn-check] line and writes a row
    with no transcript text; an ok turn prints nothing (row only); the cloud
    is never called; any other mode value reads as 'shadow'; the check runs
    on the "turn-check" worker, never the voice thread, and a full queue
    drops the check, not the turn;
  * 'on': a said_no_action turn speaks ONE_MOMENT_LINE, retries ONCE on
    TURN_CHECK_ESCALATE_MODEL without the failed reply, runs the retry's
    actions once, speaks it and puts it in the history in place of the
    failed reply; one read-back at most, its own actions never run;
  * 'on' never escalates when the cloud is not allowed, a confirmation is
    pending or the turn is ok; a failing / empty / raising cloud call stops
    with the local reply standing;
  * route-reply, glance, barged and self-voiced turns are not checked;
  * a helper exception never escapes the dispatch body (or the worker);
  * synthetic "_" results (_unverified_claim, ...) never count as ran.

No real audio, no LLM, no network.

Run locally (full-deps tier):
    python -m unittest tests.monolith.test_monolith_turn_check
"""
from __future__ import annotations

import ast
import contextlib
import inspect
import io
import json
import os
import queue
import shutil
import tempfile
import threading
import unittest
from unittest import mock

from tests._monolith_harness import requires_monolith
from tests.monolith.test_monolith_answer_first import _Base

# Generic fixtures only.
_ASK = "turn off the desk lamp"
_CLAIM = "Right away, sir."                 # a claim, and no action token
_RETRY = "[ACTION: lamp_off] The desk lamp is off, sir."
_RETRY_SPOKEN = "The desk lamp is off, sir."
_LAMP_RESULT = "Desk lamp off."
_WEATHER_ASK = "what's the weather"
_WEATHER_REPLY = "Currently mild and clear, sir."


@requires_monolith
class _TurnCheckBase(_Base):
    def setUp(self):
        super().setUp()
        bc = self.bc
        self._p(bc, "MISSION_NARRATION_ENABLED", False)
        self._p(bc, "TURN_CHECK_MODE", "shadow")
        self._p(bc, "TURN_CHECK_ESCALATE_MODEL", "claude-sonnet-5-5")
        self._p(bc, "_pending_confirmation", [])
        self._p(bc, "_pending_autocorrect_choice", [])
        self.cloud_allowed = self._p(bc, "_chat_cloud_allowed",
                                     return_value=True)
        self.oneshot = self._p(bc, "_claude_oneshot", return_value=_RETRY)
        self._p(bc, "_publish_turn_brain")
        self.lamp_calls: list = []

        def _lamp(arg=""):
            self.lamp_calls.append(arg)
            return _LAMP_RESULT

        bc.ACTIONS["lamp_off"] = _lamp
        tmp = tempfile.mkdtemp(prefix="jarvis_turn_check_")
        self.addCleanup(shutil.rmtree, tmp, True)
        self.log_path = os.path.join(tmp, "turn_check.jsonl")
        self._p(bc, "_turn_check_log_path", return_value=self.log_path)

    def _run(self, reply, text="what's the weather", during_llm=None):
        """_Base._run (a canned LLM reply honouring _call_llm's history
        contract), plus a bounded wait for the shadow worker INSIDE the
        stdout capture: its line goes to the stdout live at the turn's end."""
        bc = self.bc

        def fake_llm(user_text):
            bc.conversation_history.append({"role": "user",
                                            "content": user_text})
            bc.conversation_history.append({"role": "assistant",
                                            "content": reply})
            if during_llm is not None:
                during_llm()
            return reply

        self._p(bc, "get_response_with_animation", side_effect=fake_llm)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            out = bc._run_llm_dispatch_body(text)
            self.assertTrue(bc._turn_check_flush(10.0), "worker stuck")
        self.assertEqual(out, reply)
        return buf.getvalue()

    def _dispatch_direct(self, text=_ASK):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.bc._run_llm_dispatch_body(text)
            self.assertTrue(self.bc._turn_check_flush(10.0))
        return buf.getvalue()

    def rows(self) -> list:
        if not os.path.exists(self.log_path):
            return []
        with open(self.log_path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def log_text(self) -> str:
        if not os.path.exists(self.log_path):
            return ""
        with open(self.log_path, encoding="utf-8") as f:
            return f.read()

    def turn_lines(self, out: str) -> list:
        return [ln.strip() for ln in out.splitlines()
                if ln.strip().startswith("[turn-check]")]

    def assistant_msgs(self) -> list:
        return [m.get("content") for m in
                self.bc.conversation_history[self._hist_len:]
                if m.get("role") == "assistant"]


class OffModeTests(_TurnCheckBase):
    def test_off_checks_nothing(self):
        bc = self.bc
        self._p(bc, "TURN_CHECK_MODE", "off")
        check = self._p(bc._turn_checker, "check_turn")
        submit = self._p(bc, "_turn_check_submit")
        out = self._run(_CLAIM, text=_ASK)
        check.assert_not_called()
        submit.assert_not_called()
        self.assertEqual(self.turn_lines(out), [])
        self.assertFalse(os.path.exists(self.log_path))
        self.oneshot.assert_not_called()

    def test_mode_normalisation(self):
        bc = self.bc
        for raw, want in (("off", "off"), (" ON ", "on"), ("Shadow", "shadow"),
                          ("bogus", "shadow"), ("", "shadow"),
                          (None, "shadow"), (1, "shadow")):
            with mock.patch.object(bc, "TURN_CHECK_MODE", raw):
                self.assertEqual(bc._turn_check_mode(), want, repr(raw))


class ShadowModeTests(_TurnCheckBase):
    def test_failed_turn_logs_a_line_and_a_row_without_words(self):
        out = self._run(_CLAIM, text=_ASK)
        lines = self.turn_lines(out)
        self.assertEqual(len(lines), 1, out)
        self.assertTrue(lines[0].startswith(
            "[turn-check] said_no_action conf=0.90 ("), lines[0])
        self.assertTrue(lines[0].endswith("- would escalate"), lines[0])
        # The line carries the checker's fixed reason, never the words.
        self.assertNotIn("desk lamp", lines[0].lower())
        self.assertNotIn("right away", lines[0].lower())
        rows = self.rows()
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["mode"], "shadow")
        self.assertEqual(row["kind"], "said_no_action")
        self.assertEqual(row["confidence"], 0.9)
        self.assertIs(row["would_escalate"], True)
        self.assertIs(row["escalated"], False)
        self.assertEqual(row["emitted"], [])
        self.assertEqual(row["ran"], [])
        self.assertEqual(
            set(row), {"ts", "mode", "kind", "confidence", "would_escalate",
                       "cloud_allowed", "needs_confirmation", "escalated",
                       "emitted", "ran"})
        text = self.log_text().lower()
        for words in ("desk", "lamp", "right away", "sir"):
            self.assertNotIn(words, text)
        # Shadow never calls the cloud and never says "one moment".
        self.oneshot.assert_not_called()
        self.assertNotIn(self.bc._turn_checker.ONE_MOMENT_LINE, self.spoken)
        self.assertEqual(self.lamp_calls, [])

    def test_shadow_says_why_it_would_not_escalate(self):
        self.cloud_allowed.return_value = False
        out = self.turn_lines(self._run(_CLAIM, text=_ASK))
        self.assertEqual(len(out), 1)
        self.assertTrue(out[0].endswith(
            "- not escalated: the cloud is not allowed for chat"), out[0])
        row = self.rows()[0]
        self.assertIs(row["would_escalate"], False)
        self.assertIs(row["cloud_allowed"], False)

    def test_ok_turn_prints_nothing_but_writes_a_row(self):
        out = self._run(_WEATHER_REPLY, text=_WEATHER_ASK)
        self.assertEqual(self.turn_lines(out), [])
        rows = self.rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["kind"], "ok")
        self.assertIs(rows[0]["would_escalate"], False)
        self.assertNotIn("mild", self.log_text().lower())
        self.oneshot.assert_not_called()

    def test_action_names_are_logged_as_written(self):
        self._run("[ACTION: lamp_off] Done, sir.", text=_ASK)
        row = self.rows()[0]
        self.assertEqual(row["kind"], "ok")
        self.assertEqual(row["emitted"], ["lamp_off"])
        self.assertEqual(row["ran"], ["lamp_off"])

    def test_unknown_mode_reads_as_shadow(self):
        self._p(self.bc, "TURN_CHECK_MODE", "sometimes")
        out = self._run(_CLAIM, text=_ASK)
        self.assertEqual(len(self.turn_lines(out)), 1)
        self.assertEqual(self.rows()[0]["mode"], "shadow")
        self.oneshot.assert_not_called()

    def test_the_check_runs_on_the_worker_not_the_voice_thread(self):
        bc = self.bc
        real = bc._turn_checker.check_turn
        seen: list = []

        def spy(*a, **k):
            seen.append(threading.current_thread().name)
            return real(*a, **k)

        self._p(bc._turn_checker, "check_turn", side_effect=spy)
        out = self._run(_CLAIM, text=_ASK)
        self.assertEqual(seen, ["turn-check"])
        self.assertEqual(len(self.turn_lines(out)), 1)

    def test_a_full_queue_drops_the_check_not_the_turn(self):
        self._p(self.bc._turn_check_jobs, "put_nowait",
                side_effect=queue.Full)
        out = self._run(_CLAIM, text=_ASK)    # _run asserts the reply
        self.assertEqual(self.turn_lines(out), [])
        self.assertEqual(self.rows(), [])

    def test_a_failing_log_write_never_breaks_the_turn(self):
        self._p(self.bc._stt_parakeet, "append_jsonl",
                side_effect=OSError("disk full"))
        out = self._run(_CLAIM, text=_ASK)     # _run asserts the reply
        self.assertEqual(len(self.turn_lines(out)), 1)
        self.assertNotIn("skipped", out)


class OnModeTests(_TurnCheckBase):
    def setUp(self):
        super().setUp()
        self._p(self.bc, "TURN_CHECK_MODE", "on")

    def test_said_no_action_escalates_once_and_replaces_the_reply(self):
        bc = self.bc
        hist_before = len(bc.conversation_history)
        out = self._run(_CLAIM, text=_ASK)
        self.assertEqual(self.oneshot.call_count, 1, out)
        args, kwargs = self.oneshot.call_args
        self.assertEqual(kwargs.get("model"), "claude-sonnet-5-5")
        system, msgs = args[0], args[1]
        # The history sent ends with THIS turn's request; the failed local
        # reply is not in it.
        self.assertEqual(msgs[-1], {"role": "user", "content": _ASK})
        self.assertNotIn(_CLAIM, [m.get("content") for m in msgs])
        # The system-side note names the failure kind.
        sys_text = (system if isinstance(system, str)
                    else " ".join(b.get("text", "") for b in system))
        self.assertIn("said_no_action", sys_text)
        self.assertTrue(sys_text.startswith(bc._system_prompt[:200]))
        # Spoken: the local reply, then "One moment, sir.", then the retry.
        one = bc._turn_checker.ONE_MOMENT_LINE
        self.assertIn(one, self.spoken)
        self.assertIn(_RETRY_SPOKEN, self.spoken)
        self.assertLess(self.spoken.index(one),
                        self.spoken.index(_RETRY_SPOKEN))
        self.assertEqual(self.spoken.count(one), 1)
        # The retry's action ran exactly once.
        self.assertEqual(self.lamp_calls, [""])
        # The history holds the retry in place of the failed reply.
        assistants = self.assistant_msgs()
        self.assertNotIn(_CLAIM, assistants)
        self.assertEqual(assistants.count(_RETRY), 1)
        users = [m for m in bc.conversation_history[hist_before:]
                 if m.get("role") == "user"]
        self.assertEqual(users, [{"role": "user", "content": _ASK}])
        lines = self.turn_lines(out)
        self.assertTrue(any(ln.endswith("- escalating to claude-sonnet-5-5")
                            for ln in lines), lines)
        row = self.rows()[0]
        self.assertIs(row["escalated"], True)
        self.assertEqual(row["retry"], "spoken")
        self.assertEqual(row["mode"], "on")

    def test_informative_retry_reads_back_once_and_runs_none_of_it(self):
        bc = self.bc
        bc.ACTIONS["weather_briefing"] = lambda a="": _WEATHER_REPLY
        self.oneshot.return_value = "[ACTION: weather_briefing] Checking, sir."
        # Round 1 of the local chain (the claim check's self-correction)
        # gets nothing back; the retry's read-back gets a reply that names
        # an action of its own.
        self.followup.side_effect = ["", "It is mild out, sir. "
                                         "[ACTION: lamp_off]"]
        with mock.patch.object(bc, "INFORMATIVE_ACTIONS",
                               set(bc.INFORMATIVE_ACTIONS)
                               | {"weather_briefing"}):
            self._run(_CLAIM, text=_ASK)
        self.assertEqual(self.oneshot.call_count, 1)
        # One read-back (the second follow-up call of the turn); its own
        # [ACTION: lamp_off] is not run.
        self.assertEqual(self.followup.call_count, 2)
        self.assertEqual(self.followup.call_args[0][0],
                         [("weather_briefing", _WEATHER_REPLY)])
        self.assertEqual(self.lamp_calls, [])
        self.assertIn("It is mild out, sir.", self.spoken)

    def test_cloud_not_allowed_never_escalates(self):
        self.cloud_allowed.return_value = False
        out = self._run(_CLAIM, text=_ASK)
        self.oneshot.assert_not_called()
        self.assertNotIn(self.bc._turn_checker.ONE_MOMENT_LINE, self.spoken)
        self.assertIn(_CLAIM, self.assistant_msgs())
        self.assertTrue(any("not escalated: the cloud is not allowed" in ln
                            for ln in self.turn_lines(out)))
        self.assertIs(self.rows()[0]["escalated"], False)

    def test_pending_confirmation_never_escalates(self):
        self._p(self.bc, "_pending_confirmation", [("delete_file", "x.txt")])
        out = self._run(_CLAIM, text=_ASK)
        self.oneshot.assert_not_called()
        self.assertTrue(any("a confirmation is pending" in ln
                            for ln in self.turn_lines(out)))
        row = self.rows()[0]
        self.assertIs(row["needs_confirmation"], True)
        self.assertIs(row["escalated"], False)

    def test_held_for_confirmation_this_turn_never_escalates(self):
        bc = self.bc
        bc.ACTIONS["wipe_notes"] = lambda a="": "wiped"
        self._p(bc, "_needs_confirmation",
                side_effect=lambda n, a="": n == "wipe_notes")
        self._run("[ACTION: wipe_notes] Clearing them, sir.", text=_ASK)
        self.oneshot.assert_not_called()
        self.assertEqual(self.rows()[0]["ran"], [])

    def test_ok_turn_never_escalates(self):
        out = self._run(_WEATHER_REPLY, text=_WEATHER_ASK)
        self.oneshot.assert_not_called()
        self.assertEqual(self.turn_lines(out), [])
        self.assertEqual(self.rows()[0]["kind"], "ok")

    def test_empty_cloud_reply_stops_quietly(self):
        self.oneshot.return_value = None
        out = self._run(_CLAIM, text=_ASK)
        self.assertEqual(self.oneshot.call_count, 1)
        one = self.bc._turn_checker.ONE_MOMENT_LINE
        # "One moment" was said, then nothing else after it.
        self.assertEqual(self.spoken[-1], one)
        self.assertIn(_CLAIM, self.assistant_msgs())
        self.assertIn("returned nothing", out)
        self.assertEqual(self.rows()[0]["retry"], "empty")

    def test_raising_cloud_call_stops_quietly(self):
        self.oneshot.side_effect = RuntimeError("socket closed")
        out = self._run(_CLAIM, text=_ASK)    # the dispatch still returns
        self.assertIn("the retry failed - RuntimeError", out)
        self.assertIn(_CLAIM, self.assistant_msgs())
        self.assertEqual(self.lamp_calls, [])
        row = self.rows()[0]
        self.assertIs(row["escalated"], True)
        self.assertEqual(row["retry"], "error")

    def test_barge_during_the_retry_runs_nothing(self):
        bc = self.bc

        def _barge(*a, **k):
            bc._tts_interrupt_seq[0] += 1
            return _RETRY

        self.oneshot.side_effect = _barge
        self._run(_CLAIM, text=_ASK)
        self.assertEqual(self.lamp_calls, [])
        self.assertNotIn(_RETRY_SPOKEN, self.spoken)
        self.assertEqual(self.rows()[0]["retry"], "barged")

    def test_never_escalates_inside_a_retry(self):
        bc = self.bc
        bc._turn_check_tls.escalating = True
        self.addCleanup(setattr, bc._turn_check_tls, "escalating", False)
        self._run(_CLAIM, text=_ASK)
        self.oneshot.assert_not_called()


class SkippedTurnTests(_TurnCheckBase):
    def setUp(self):
        super().setUp()
        self._p(self.bc, "TURN_CHECK_MODE", "on")
        self.check = self._p(self.bc._turn_checker, "check_turn",
                             wraps=self.bc._turn_checker.check_turn)

    def _assert_not_checked(self):
        self.check.assert_not_called()
        self.oneshot.assert_not_called()
        self.assertEqual(self.rows(), [])

    def test_route_reply_turn_is_skipped(self):
        self._p(self.bc, "_utterance_route_reply", return_value=_CLAIM)
        self._dispatch_direct()
        self._assert_not_checked()

    def test_glance_turn_is_skipped(self):
        self._p(self.bc, "maybe_glance_response", return_value=_CLAIM)
        self._dispatch_direct()
        self._assert_not_checked()

    def test_shadow_mode_skips_too(self):
        self._p(self.bc, "TURN_CHECK_MODE", "shadow")
        self._p(self.bc, "maybe_glance_response", return_value=_CLAIM)
        self._dispatch_direct()
        self._assert_not_checked()

    def test_barged_turn_is_skipped(self):
        bc = self.bc

        def _barge():
            bc._tts_interrupt_seq[0] += 1

        self._run(_CLAIM, text=_ASK, during_llm=_barge)
        self._assert_not_checked()

    def test_self_voiced_turn_is_skipped(self):
        bc = self.bc
        bc.ACTIONS["talk_to_device"] = lambda a="": "dialogue over"
        self._p(bc, "is_self_voiced",
                side_effect=lambda n: n == "talk_to_device")
        self._run("[ACTION: talk_to_device] Done, sir.", text=_ASK)
        self._assert_not_checked()

    def test_direct_skip_flag(self):
        self.assertIsNone(self.bc._turn_check_after_chain(
            _ASK, [_CLAIM], [], skip=True))
        self.assertTrue(self.bc._turn_check_flush(10.0))
        self._assert_not_checked()


class HelperFaultTests(_TurnCheckBase):
    def test_helper_exception_never_escapes_the_dispatch(self):
        self._p(self.bc, "_turn_check_after_chain",
                side_effect=RuntimeError("boom"))
        out = self._run(_CLAIM, text=_ASK)    # _run asserts the reply
        self.assertIn("[turn-check] skipped - RuntimeError: boom", out)

    def test_checker_exception_never_escapes_the_worker(self):
        self._p(self.bc._turn_checker, "check_turn",
                side_effect=ValueError("bad"))
        out = self._run(_CLAIM, text=_ASK)
        self.assertIn("[turn-check] skipped - ValueError: bad", out)
        self.oneshot.assert_not_called()

    def test_checker_exception_never_escapes_the_dispatch_in_on_mode(self):
        self._p(self.bc, "TURN_CHECK_MODE", "on")
        self._p(self.bc._turn_checker, "check_turn",
                side_effect=ValueError("bad"))
        out = self._run(_CLAIM, text=_ASK)
        self.assertIn("[turn-check] skipped - ValueError: bad", out)
        self.oneshot.assert_not_called()

    def test_the_worker_survives_a_failed_job(self):
        bc = self.bc
        bc._turn_check_submit(lambda: 1 / 0)
        self.assertTrue(bc._turn_check_flush(10.0))
        self._run(_CLAIM, text=_ASK)
        self.assertEqual(self.rows()[0]["kind"], "said_no_action")


class RanNamesTests(_TurnCheckBase):
    def test_synthetic_held_and_unknown_results_never_count_as_ran(self):
        ran = self.bc._turn_check_ran([
            ("_unverified_claim", "reply claims ...", True),
            ("_dropped_step", "reply promised ...", True),
            ("_preemptive_hallucinated_claim", "reply claimed ...", True),
            ("wipe_notes", "⚠  REQUIRES CONFIRMATION: wipe_notes() — say "
                           "'yes' to proceed", False),
            ("close_all", "⚠  PUSHBACK: All of them, sir?", False),
            ("lamp_of", "⚠  AMBIGUOUS: lamp_off vs lamp_on — awaiting "
                        "clarification", False),
            ("bogus", "unknown action: bogus", False),
            ("Lamp_Off", _LAMP_RESULT, False),
        ])
        self.assertEqual(ran, ["lamp_off"])

    def test_unverified_claim_turn_is_still_said_no_action(self):
        # The in-turn claim check adds _unverified_claim (a synthetic result
        # that runs nothing); the checker must still see no action as run.
        out = self._run(_CLAIM, text=_ASK)
        self.assertIn("[validation] reply claims", out)
        self.assertIn("Reading results (depth 1)", out)
        row = self.rows()[0]
        self.assertEqual(row["kind"], "said_no_action")
        self.assertEqual(row["ran"], [])

    def test_an_action_run_in_a_follow_up_round_counts(self):
        # The claim check's self-correction round emits and runs the action:
        # the turn is fixed, and the row shows what ran in that round.
        self.followup.side_effect = ["[ACTION: lamp_off] Done, sir.", ""]
        out = self._run(_CLAIM, text=_ASK)
        self.assertEqual(self.lamp_calls, [""])
        self.assertEqual(self.turn_lines(out), [])
        row = self.rows()[0]
        self.assertEqual(row["kind"], "ok")
        self.assertEqual(row["emitted"], ["lamp_off"])
        self.assertEqual(row["ran"], ["lamp_off"])

    def test_emitted_names_come_from_every_round(self):
        self.assertEqual(
            self.bc._turn_check_emitted(
                ["[ACTION: Lamp_Off] ok", None,
                 "then [ACTION: see_screen, the editor] and [ACTION: x_y]"]),
            ["lamp_off", "see_screen", "x_y"])


class HistoryReplaceTests(_TurnCheckBase):
    def test_failed_replies_are_swapped_in_place_and_others_kept(self):
        bc = self.bc
        hist = bc.conversation_history
        ident = id(hist)
        older = [{"role": "user", "content": _ASK},    # same words, earlier
                 {"role": "assistant", "content": _CLAIM}]
        turn = [{"role": "user", "content": _ASK},
                {"role": "assistant", "content": _CLAIM},
                {"role": "assistant", "content": "Your timer is done, sir."},
                {"role": "assistant", "content": "Still on it, sir."}]
        hist.extend(older + turn)
        gone = bc._turn_check_replace_history(
            _ASK, [_CLAIM, "Still on it, sir."], [_RETRY])
        self.assertEqual(gone, 2)
        self.assertEqual(id(bc.conversation_history), ident)
        self.assertEqual(
            hist[self._hist_len:],
            older + [{"role": "user", "content": _ASK},
                     {"role": "assistant", "content": _RETRY},
                     {"role": "assistant",
                      "content": "Your timer is done, sir."}])

    def test_retry_messages_end_at_this_turns_request(self):
        bc = self.bc
        bc.conversation_history.extend(
            [{"role": "user", "content": _ASK},
             {"role": "assistant", "content": _CLAIM}])
        msgs = bc._turn_check_retry_messages(_ASK)
        self.assertEqual(msgs[-1], {"role": "user", "content": _ASK})
        self.assertIsNone(bc._turn_check_retry_messages("never said"))


class WiringTests(_TurnCheckBase):
    def test_one_guarded_call_after_the_close_out(self):
        # AST, not text: the comments around the call must not count.
        tree = ast.parse(inspect.getsource(self.bc._run_llm_dispatch_body))
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and getattr(n.func, "id", "") == "_turn_check_after_chain"]
        self.assertEqual(len(calls), 1)
        guarded = [t for t in ast.walk(tree) if isinstance(t, ast.Try)
                   and any(c in list(ast.walk(t)) for c in calls)]
        self.assertTrue(guarded, "the call must sit inside a try")
        wanted = ("_chain_close_out_line", "_record_turn_offers",
                  "_turn_check_after_chain", "_trim_conversation_history")
        seq = sorted((n.lineno, n.func.id) for n in ast.walk(tree)
                     if isinstance(n, ast.Call)
                     and getattr(n.func, "id", "") in wanted)
        # The last call of each kind, in source order: close-out, offers,
        # the check, then the trim.
        last = {name: line for line, name in seq}
        self.assertEqual(sorted(wanted, key=last.get), list(wanted), seq)

    def test_config_defaults_and_settings_rows(self):
        from core import config
        from tools import settings_window as sw
        self.assertEqual(config.TURN_CHECK_MODE, "shadow")
        self.assertEqual(config.TURN_CHECK_ESCALATE_MODEL, "claude-sonnet-5-5")
        row = sw.SCHEMA["TURN_CHECK_MODE"]
        self.assertEqual(row["type"], "enum")
        self.assertEqual(list(row["choices"]),
                         list(self.bc._TURN_CHECK_MODES))
        self.assertEqual(sw.SCHEMA["TURN_CHECK_ESCALATE_MODEL"]["choices"],
                         sw.SCHEMA["CLAUDE_MODEL"]["choices"])
        self.assertIn("TURN_CHECK_MODE", self.bc._TURN_FLAG_KEYS)


if __name__ == "__main__":
    unittest.main()
