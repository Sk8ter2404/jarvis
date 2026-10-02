"""Cross-branch interactions on claude/integrate-1002 (2026-10-02).

Each 2026-10-02 branch tested its own feature against origin/main. These
tests drive the REAL _run_llm_dispatch_body with several of them live at once,
where one feature's state is another's input:

  * instant actions x the turn checker: an instant ("on") turn skipped the
    LLM, so it is never turn-checked or retried on Claude - not even when its
    action fails (that gets the normal failure follow-up); in shadow mode both
    features score the SAME brain reply, one row each, and neither changes
    what the owner hears; a turn-check retry is not scored a second time;
  * shipped defaults: with every new switch at its core/config.py literal a
    turn is owner-identical (speech, actions, LLM calls, history) to the same
    turn with both shadow features off - and nothing is sent to Claude, even
    when the cloud is allowed and the brain's reply failed;
  * the turn checker x the retired-model guard (local-features): a retry on a
    model Anthropic already answered not_found for (no successor) is never
    started - it could only say "One moment, sir." and then nothing; with a
    successor configured the retry runs on it; shadow mode is unchanged;
  * R5's background gate x the turn checker's worker: a strict opt-in job
    held in the gate during a turn never blocks the turn or its shadow check,
    the worker never takes the gate, and the job runs once the turn ends;
  * INSTANT_ACTIONS_MODE is on the [turn-flags] line like TURN_CHECK_MODE;
  * the harness forgets the process-wide state these branches added (the
    model guard, R5's once-per-caller set, the turn-check queue).

No real audio, no LLM, no network. Generic fixtures only.

    python -m unittest tests.monolith.test_monolith_integration_1002
"""
from __future__ import annotations

import ast
import contextlib
import io
import json
import os
import shutil
import tempfile
import threading
import time
import unittest
from unittest import mock

from tests._monolith_harness import (MonolithGlobalsTestCase,
                                     requires_monolith)
from tests.monolith.test_monolith_instant_actions import _InstantBase
from tests.monolith.test_monolith_turn_check import (_ASK, _CLAIM,
                                                     _RETRY_SPOKEN,
                                                     _TurnCheckBase)

_PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
_ESCALATE = "claude-sonnet-5-5"


def _config_literals() -> dict:
    """core/config.py's module-level literals, read by AST - the SHIPPED
    values, never this box's user_settings.json."""
    with open(os.path.join(_PROJECT, "core", "config.py"),
              encoding="utf-8") as f:
        tree = ast.parse(f.read())
    lits: dict = {}
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            try:
                lits[node.targets[0].id] = ast.literal_eval(node.value)
            except Exception:
                pass
    return lits


def _jsonl(path) -> list:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ════════════════════════════════════════════════════════════════════════════
#  instant actions x the turn checker
# ════════════════════════════════════════════════════════════════════════════
@requires_monolith
class _BothFeaturesBase(_InstantBase):
    """_InstantBase (instant-actions wiring, stub actions, recording _speak)
    plus the turn checker's seams: its log in a temp dir, the cloud gate
    open, the Claude retry stubbed."""

    TURN_CHECK = "shadow"

    def setUp(self):
        super().setUp()
        bc = self.bc
        self._p(bc, "TURN_CHECK_MODE", self.TURN_CHECK)
        self._p(bc, "TURN_CHECK_ESCALATE_MODEL", _ESCALATE)
        self._p(bc, "_pending_confirmation", [])
        self._p(bc, "_pending_autocorrect_choice", [])
        self.cloud = self._p(bc, "_chat_cloud_allowed", return_value=True)
        self.oneshot = self._p(bc, "_claude_oneshot", return_value=None)
        self._p(bc, "_publish_turn_brain")
        tmp = tempfile.mkdtemp(prefix="jarvis_integ_1002_")
        self.addCleanup(shutil.rmtree, tmp, True)
        self.tc_log = os.path.join(tmp, "turn_check.jsonl")
        self._p(bc, "_turn_check_log_path", return_value=self.tc_log)
        self.check = self._p(bc._turn_checker, "check_turn",
                             wraps=bc._turn_checker.check_turn)
        # No follow-up round answers unless a test says so.
        self.gfr.side_effect = [None] * 10

        # _call_llm's contract: the user line and the FULL reply go to the
        # history (the turn checker's retry anchors on that user line).
        def _honour_history(user_text):
            reply = self.llm.return_value
            self.history.append({"role": "user", "content": user_text})
            self.history.append({"role": "assistant", "content": reply})
            return reply

        self.llm.side_effect = _honour_history
        from core.claude_model_guard import GUARD
        GUARD.reset()
        self.addCleanup(GUARD.reset)

    def _run(self, text):
        """One turn; the shadow worker is drained INSIDE the capture (its
        line goes to the stdout live when the turn ended)."""
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.bc._run_llm_dispatch(text)
            self.assertTrue(self.bc._turn_check_flush(10.0), "worker stuck")
        return buf.getvalue()

    def tc_rows(self):
        return _jsonl(self.tc_log)

    @property
    def one_moment(self):
        return self.bc._turn_checker.ONE_MOMENT_LINE


class InstantOnSkipsTheTurnCheckTests(_BothFeaturesBase):
    MODE = "on"
    TURN_CHECK = "on"

    def test_an_instant_turn_is_never_checked_or_retried(self):
        out = self._run("volume up")
        self.llm.assert_not_called()
        self.assertEqual(self.calls["volume_up"], [""])
        self.assertEqual(self.spoken, ["Volume up, sir."])
        self.check.assert_not_called()
        self.oneshot.assert_not_called()
        self.assertNotIn(self.one_moment, self.spoken)
        self.assertEqual(self.tc_rows(), [])
        self.assertNotIn("[turn-check]", out)

    def test_a_failed_instant_action_gets_the_follow_up_not_a_cloud_retry(self):
        self._stub("pause_music",
                   "I couldn't reach the Windows media controls, sir.")
        self.gfr.side_effect = (["The media controls didn't answer, sir."]
                                + [None] * 8)
        out = self._run("pause the music")
        self.llm.assert_not_called()
        self.gfr.assert_called_once()
        self.assertIn("The media controls didn't answer, sir.", self.spoken)
        self.assertIn("did not run cleanly", out)
        # Still a route-shaped turn: no check, no "One moment", no Claude.
        self.check.assert_not_called()
        self.oneshot.assert_not_called()
        self.assertNotIn(self.one_moment, self.spoken)
        self.assertEqual(self.tc_rows(), [])
        self.assertEqual(self._rows()[0]["ok"], False)


class BothShadowTests(_BothFeaturesBase):
    """The shipped modes: instant 'shadow' and turn check 'shadow'."""

    def test_both_score_the_same_brain_reply_and_change_nothing(self):
        self.gfr.side_effect = ["It is paused, sir."] + [None] * 8
        self._run("pause the music")
        self.llm.assert_called_once()
        self.assertEqual(self.calls["pause_music"], [""])
        self.assertIn("It is paused, sir.", self.spoken)
        self.assertNotIn(self.one_moment, self.spoken)
        self.oneshot.assert_not_called()
        inst = self._rows()
        self.assertEqual(len(inst), 1)
        self.assertEqual((inst[0]["action"], inst[0]["brain"], inst[0]["agree"]),
                         ("pause_music", ["pause_music"], True))
        tc = self.tc_rows()
        self.assertEqual(len(tc), 1)
        # The checker saw what the brain ran: shadow scoring did not touch it.
        self.assertEqual((tc[0]["kind"], tc[0]["ran"]), ("ok", ["pause_music"]))

    def test_a_failed_brain_reply_is_logged_twice_and_spoken_once(self):
        self.llm.return_value = "Right away, sir."     # no token
        out = self._run("volume up")
        self.llm.assert_called_once()
        self.assertEqual(self.calls["volume_up"], [])
        # live-fixes-1202 F1: an unverified execution claim is never voiced;
        # the turn goes straight to the follow-up round (stubbed empty here).
        self.assertEqual(self.spoken, [])
        self.gfr.assert_called()
        self.oneshot.assert_not_called()
        inst = self._rows()
        self.assertEqual((inst[0]["action"], inst[0]["agree"], inst[0]["brain"]),
                         ("volume_up", False, []))
        tc = self.tc_rows()
        self.assertEqual(len(tc), 1)
        self.assertNotEqual(tc[0]["kind"], "ok")
        self.assertIs(tc[0]["would_escalate"], True)
        self.assertIs(tc[0]["escalated"], False)
        self.assertIn("[instant] would run volume_up without the brain", out)
        self.assertIn("would escalate", out)


class ShadowInstantWithTurnCheckOnTests(_BothFeaturesBase):
    TURN_CHECK = "on"

    def test_the_retry_runs_the_action_once_and_is_not_rescored(self):
        self.llm.return_value = "Right away, sir."     # no token
        self.oneshot.return_value = "[ACTION: volume_up] Volume up, sir."
        self._run("volume up")
        self.llm.assert_called_once()
        self.assertEqual(self.oneshot.call_count, 1)
        self.assertEqual(self.oneshot.call_args.kwargs.get("model"), _ESCALATE)
        # The retry ran the action exactly once.
        self.assertEqual(self.calls["volume_up"], [""])
        self.assertEqual(self.spoken.count(self.one_moment), 1)
        self.assertIn("Volume up, sir.", self.spoken)
        self.assertLess(self.spoken.index(self.one_moment),
                        self.spoken.index("Volume up, sir."))
        # Instant shadow scored the brain's FIRST reply, once.
        inst = self._rows()
        self.assertEqual(len(inst), 1)
        self.assertEqual((inst[0]["agree"], inst[0]["brain"]), (False, []))
        tc = self.tc_rows()
        self.assertEqual(len(tc), 1)
        self.assertEqual((tc[0]["escalated"], tc[0]["retry"]), (True, "spoken"))


class ShippedDefaultsAreOwnerIdenticalTests(_BothFeaturesBase):
    """Every 2026-10-02 switch at its core/config.py literal: what the owner
    hears and what runs is the same as with both shadow features off."""

    SCENARIOS = (
        # (utterance, the brain's reply, follow-up replies)
        ("pause the music", "Right away, sir. [ACTION: pause_music]",
         ["It is paused, sir."]),
        ("volume up", "Right away, sir.", []),                 # a failure
        ("next song", "[intent:wry] A fine track, sir.", []),  # wrong reply
        ("what's the weather like", "Mild and clear, sir.", []),
    )

    def _scenario(self, text, reply, followups, *, instant, turn_check):
        bc = self.bc
        self.spoken.clear()
        for v in self.calls.values():
            v.clear()
        self.history.clear()
        self.llm.reset_mock()
        self.llm.return_value = reply
        self.gfr.reset_mock()
        self.gfr.side_effect = list(followups) + [None] * 8
        self.oneshot.reset_mock()
        with mock.patch.object(bc, "INSTANT_ACTIONS_MODE", instant), \
                mock.patch.object(bc, "TURN_CHECK_MODE", turn_check):
            self._run(text)
        return {"spoken": list(self.spoken),
                "calls": {k: list(v) for k, v in self.calls.items()},
                "llm": self.llm.call_count,
                "followups": self.gfr.call_count,
                "history": [dict(m) for m in self.history],
                "cloud": self.oneshot.call_count}

    def test_shipped_defaults_match_both_features_off(self):
        bc = self.bc
        lits = _config_literals()
        # Pin the rest of the batch's switches to their shipped values too
        # (this box's user_settings.json must not decide the comparison).
        for key in ("BACKGROUND_TAG_STRICT", "PROCESSING_FILLER_PRERENDER",
                    "FILLER_DUCK_HOLD", "PROCESSING_FILLER_SKIP_PLEASANTRIES",
                    "PROCESSING_FILLER_LATE_START_S"):
            self._p(bc, key, lits[key])
        self.assertEqual(lits["INSTANT_ACTIONS_MODE"], "shadow")
        self.assertEqual(lits["TURN_CHECK_MODE"], "shadow")
        for text, reply, followups in self.SCENARIOS:
            with self.subTest(text=text):
                shipped = self._scenario(
                    text, reply, followups,
                    instant=lits["INSTANT_ACTIONS_MODE"],
                    turn_check=lits["TURN_CHECK_MODE"])
                off = self._scenario(text, reply, followups,
                                     instant="off", turn_check="off")
                self.assertEqual(shipped, off)
                self.assertEqual(shipped["cloud"], 0)
                self.assertEqual(shipped["llm"], 1)
                self.assertNotIn(self.one_moment, shipped["spoken"])


# ════════════════════════════════════════════════════════════════════════════
#  the turn checker x the retired-model guard
# ════════════════════════════════════════════════════════════════════════════
class RetiredEscalateModelTests(_TurnCheckBase):
    def setUp(self):
        super().setUp()
        from core import config as cfg
        from core.claude_model_guard import GUARD
        self.cfg = cfg
        self.guard = GUARD
        GUARD.reset()
        self.addCleanup(GUARD.reset)
        self._p(cfg, "CLAUDE_MODEL_SUCCESSORS", {})
        self._p(self.bc, "TURN_CHECK_MODE", "on")

    def _retire(self, model=_ESCALATE):
        with contextlib.redirect_stdout(io.StringIO()):
            self.guard.note_not_found(model, where="a test")

    def test_a_known_retired_model_is_never_retried(self):
        self._retire()
        out = self._run(_CLAIM, text=_ASK)
        self.oneshot.assert_not_called()
        self.assertNotIn(self.bc._turn_checker.ONE_MOMENT_LINE, self.spoken)
        lines = self.turn_lines(out)
        self.assertEqual(len(lines), 1, out)
        self.assertTrue(lines[0].endswith(
            f"- not escalated: {_ESCALATE} is retired (Anthropic answered "
            f"not_found this session)"), lines[0])
        row = self.rows()[0]
        self.assertIs(row["would_escalate"], True)
        self.assertIs(row["escalated"], False)
        self.assertEqual(row["retry"], "retired")
        # The local reply stands, unreplaced.
        self.assertIn(_CLAIM, self.assistant_msgs())
        self.assertEqual(self.lamp_calls, [])

    def test_a_configured_successor_takes_the_retry(self):
        self._p(self.cfg, "CLAUDE_MODEL_SUCCESSORS",
                {_ESCALATE: "claude-opus-5-5"})
        self._retire()
        out = self._run(_CLAIM, text=_ASK)
        self.assertEqual(self.oneshot.call_count, 1)
        self.assertEqual(self.oneshot.call_args.kwargs.get("model"),
                         "claude-opus-5-5")
        self.assertTrue(any(ln.endswith("- escalating to claude-opus-5-5")
                            for ln in self.turn_lines(out)))
        self.assertIn(_RETRY_SPOKEN, self.spoken)
        self.assertEqual(self.rows()[0]["retry"], "spoken")

    def test_a_retired_successor_stands_down_too(self):
        self._p(self.cfg, "CLAUDE_MODEL_SUCCESSORS",
                {_ESCALATE: "claude-opus-5-5"})
        self._retire()
        self._retire("claude-opus-5-5")
        self._run(_CLAIM, text=_ASK)
        self.oneshot.assert_not_called()
        self.assertEqual(self.rows()[0]["retry"], "retired")

    def test_a_working_model_still_escalates_unchanged(self):
        out = self._run(_CLAIM, text=_ASK)
        self.assertEqual(self.oneshot.call_count, 1)
        self.assertEqual(self.oneshot.call_args.kwargs.get("model"), _ESCALATE)
        self.assertTrue(any(ln.endswith(f"- escalating to {_ESCALATE}")
                            for ln in self.turn_lines(out)))

    def test_nothing_to_retry_means_no_one_moment_line(self):
        # Same broken-promise shape: the history check comes BEFORE the line.
        self._p(self.bc, "_turn_check_retry_messages", return_value=None)
        out = self._run(_CLAIM, text=_ASK)
        self.oneshot.assert_not_called()
        self.assertNotIn(self.bc._turn_checker.ONE_MOMENT_LINE, self.spoken)
        self.assertIn("no longer in the history", out)
        self.assertEqual(self.rows()[0]["retry"], "no-history")

    def test_shadow_mode_is_unchanged_by_the_guard(self):
        self._p(self.bc, "TURN_CHECK_MODE", "shadow")
        self._retire()
        out = self._run(_CLAIM, text=_ASK)
        lines = self.turn_lines(out)
        self.assertTrue(lines[0].endswith("- would escalate"), lines[0])
        self.assertNotIn("retry", self.rows()[0])
        self.oneshot.assert_not_called()

    def test_the_target_follows_the_guards_one_rule(self):
        bc = self.bc
        target = bc._turn_check_escalate_target
        self.assertEqual(target(), (_ESCALATE, _ESCALATE))
        with mock.patch.object(bc, "TURN_CHECK_ESCALATE_MODEL", ""):
            self.assertEqual(target(), (bc.CLAUDE_MODEL, bc.CLAUDE_MODEL))
        self._retire()
        self.assertEqual(target(), (_ESCALATE, None))
        self._p(self.cfg, "CLAUDE_MODEL_SUCCESSORS", {_ESCALATE: "claude-x-1"})
        self.assertEqual(target(), (_ESCALATE, "claude-x-1"))
        # A guard that cannot be consulted means the configured model, as
        # before the guard existed.
        from core import llm_client
        with mock.patch.object(llm_client, "_guarded_model",
                               side_effect=ValueError("broken")):
            self.assertEqual(target(), (_ESCALATE, _ESCALATE))


class RetiredEscalateModelEndToEndTests(_TurnCheckBase):
    """The same interaction with NOTHING between the turn checker and the
    guard mocked: the real _claude_oneshot -> core.llm_client.complete ->
    create_message, with only the SDK client faked. Before the fix this said
    "One moment, sir." and then printed "cloud fallback also failed
    (RetiredModelError ...)" on every failed turn."""

    def setUp(self):
        self._real_oneshot = self.bc._claude_oneshot
        super().setUp()
        from core import llm_client
        from core.claude_model_guard import GUARD
        GUARD.reset()
        self.addCleanup(GUARD.reset)
        self._p(self.bc, "TURN_CHECK_MODE", "on")
        self._p(self.bc, "_claude_oneshot", side_effect=self._real_oneshot)
        self._p(self.bc, "_claude_reachable", return_value=True)
        self._p(self.bc, "_llm_client", llm_client)
        self.client = mock.MagicMock()
        self._p(llm_client, "_client", return_value=self.client)
        from core import config as cfg
        self._p(cfg, "CLAUDE_MODEL_SUCCESSORS", {})
        with contextlib.redirect_stdout(io.StringIO()):
            GUARD.note_not_found(_ESCALATE, where="a test")

    def test_no_one_moment_line_and_no_request(self):
        out = self._run(_CLAIM, text=_ASK)
        self.client.messages.create.assert_not_called()
        self.assertNotIn(self.bc._turn_checker.ONE_MOMENT_LINE, self.spoken)
        self.assertNotIn("RetiredModelError", out)
        self.assertEqual(self.rows()[0]["retry"], "retired")


# ════════════════════════════════════════════════════════════════════════════
#  R5's background gate x the turn checker's shadow worker
# ════════════════════════════════════════════════════════════════════════════
class BackgroundGateVsTurnCheckWorkerTests(_TurnCheckBase):
    def test_a_held_strict_job_never_blocks_the_turn_or_its_check(self):
        bc = self.bc
        lt = bc._lt
        self._p(bc, "BACKGROUND_TAG_STRICT", True)
        self._p(bc, "_chat_takes_local_branch", return_value=True)
        self._p(bc, "_utterance_in_progress", [False])
        self._p(bc, "_turn_in_progress", [True])
        quiet = threading.Event()
        self._p(bc, "_conversation_active",
                side_effect=lambda now=None: not quiet.is_set())
        acquired_by: list = []
        real_acquire = lt.GATE.acquire

        def spy_acquire(*a, **k):
            acquired_by.append(threading.current_thread().name)
            return real_acquire(*a, **k)

        self._p(lt.GATE, "acquire", side_effect=spy_acquire)
        stop = threading.Event()
        outcome: list = []

        def job():
            with lt.background_work("notification-triage", cancel=stop.is_set,
                                    opt_in=True, max_defer_s=15.0):
                with lt.slot() as p:
                    outcome.append(None if p is None else p.outcome)

        t = threading.Thread(target=job, name="integ-bg-job", daemon=True)
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink):
            t.start()
            try:
                deadline = time.monotonic() + 5.0
                while lt.GATE.waiting() == 0 and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertEqual(lt.GATE.waiting(), 1, "the job never queued")
                out = self._run(_CLAIM, text=_ASK)   # drains the worker
                self.assertEqual(len(self.turn_lines(out)), 1, out)
                self.assertEqual(self.rows()[0]["kind"], "said_no_action")
                self.assertTrue(t.is_alive(), "the strict job ran mid-turn")
                self.assertEqual(outcome, [])
                # The turn ends: the held job goes.
                bc._turn_in_progress[0] = False
                quiet.set()
                t.join(5.0)
                self.assertFalse(t.is_alive(), "the job never ran")
            finally:
                stop.set()
                t.join(5.0)
        self.assertEqual(outcome, ["released"])
        self.assertEqual(acquired_by, ["integ-bg-job"])
        self.assertNotIn("turn-check", acquired_by)
        self.assertIn("[bg-local] defer notification-triage (turn)",
                      sink.getvalue())


# ════════════════════════════════════════════════════════════════════════════
#  the [turn-flags] line and the harness
# ════════════════════════════════════════════════════════════════════════════
class TurnFlagsTests(MonolithGlobalsTestCase):
    def test_both_turn_shaping_modes_are_on_the_flags_line(self):
        bc = self.bc
        for key in ("TURN_CHECK_MODE", "INSTANT_ACTIONS_MODE",
                    "BACKGROUND_TAG_STRICT", "PROCESSING_FILLER_PRERENDER"):
            self.assertIn(key, bc._TURN_FLAG_KEYS)
        out = io.StringIO()
        with mock.patch.object(bc, "INSTANT_ACTIONS_MODE", "on"), \
                contextlib.redirect_stdout(out):
            bc._log_turn_flags()
        self.assertIn("INSTANT_ACTIONS_MODE=on", out.getvalue())


class HarnessForgetsTheBatchStateTests(MonolithGlobalsTestCase):
    def test_restore_resets_the_guard_the_untagged_set_and_the_queue(self):
        from tests._monolith_harness import _restore_monolith_pristine
        from core.claude_model_guard import GUARD
        bc = self.bc
        self.addCleanup(GUARD.reset)
        with contextlib.redirect_stdout(io.StringIO()):
            GUARD.note_not_found("claude-test-gone", where="a test")
        bc._untagged_local_seen.add("chat:some.caller")
        ran = threading.Event()
        self.assertTrue(bc._turn_check_submit(ran.set))
        _restore_monolith_pristine(bc)
        self.assertEqual(GUARD.retired_models(), [])
        self.assertEqual(bc._untagged_local_seen, set())
        self.assertTrue(ran.is_set(), "a queued check outlived its test")


if __name__ == "__main__":   # pragma: no cover
    unittest.main()
