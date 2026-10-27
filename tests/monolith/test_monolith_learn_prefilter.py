"""LEARN_PREFILTER wired into the monolith (v3 W09, core/learn_prefilter.py).

Pinned here:
  * 'off' queues every owner turn exactly as before (no log line);
  * 'shadow' (the shipped default) logs "[learn-prefilter] would skip (why)"
    for a turn that cannot teach and still queues it;
  * 'on' does not queue it, and a batch whose turns were all skipped is never
    sent (no job, no model request);
  * a turn that states a fact is queued in every mode;
  * overheard speech (owner_directed False) is never judged by the filter;
  * the names of the actions the dispatch ran reach the filter;
  * the log line carries a fixed reason, never the turn's text;
  * the owner-voice ambient path drops a too-short line BEFORE the content
    judge's local-LLM call ('on'), and only logs it in 'shadow'.

    python -m unittest tests.monolith.test_monolith_learn_prefilter
"""
from __future__ import annotations

import contextlib
import io
import os
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

_TEXT = "open the garage door for Zelda"


@requires_monolith
class _Base(MonolithGlobalsTestCase):
    mode = "shadow"

    def setUp(self):
        super().setUp()
        bc = self.bc
        self.queued = []
        for name, value in (
                ("LEARN_PREFILTER", self.mode),
                ("LEARN_EVERY_TURN", True),
                ("LEARN_ONLY_FROM_OWNER", False),
                ("_dialogue_gate_active", lambda: False),
                ("_learn_enqueue", self.queued.append)):
            p = mock.patch.object(bc, name, value)
            p.start()
            self.addCleanup(p.stop)

    def run_turn(self, text, reply="Done.", **kw):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.bc.learn_from_turn(text, reply, {}, **kw)
        return buf.getvalue()


class OffTests(_Base):
    mode = "off"

    def test_off_queues_everything_and_prints_nothing(self):
        out = self.run_turn("open Chrome")
        self.run_turn("I like jazz")
        self.assertEqual(len(self.queued), 2)
        self.assertNotIn("learn-prefilter", out)

    def test_off_does_not_put_ran_names_on_the_origin(self):
        self.run_turn("open Chrome")
        self.assertNotIn("ran", self.queued[0][5])

    def test_unknown_value_is_off(self):
        with mock.patch.object(self.bc, "LEARN_PREFILTER", "banana"):
            self.assertEqual(self.bc._learn_prefilter_mode(), "off")


class ShadowTests(_Base):
    mode = "shadow"

    def test_shadow_is_the_shipped_default(self):
        import core.config as cfg
        self.assertEqual(cfg.LEARN_PREFILTER, "shadow")

    def test_shadow_logs_would_skip_and_still_queues(self):
        out = self.run_turn("open Chrome")
        self.assertIn("[learn-prefilter] would skip (", out)
        self.assertEqual(len(self.queued), 1)

    def test_a_fact_is_queued_silently(self):
        out = self.run_turn("I work at an IT company")
        self.assertEqual(len(self.queued), 1)
        self.assertNotIn("learn-prefilter", out)

    def test_the_log_never_carries_the_turn_text(self):
        out = self.run_turn(_TEXT)
        self.assertIn("would skip", out)
        self.assertNotIn("Zelda", out)
        self.assertNotIn("garage", out)


class OnTests(_Base):
    mode = "on"

    def test_a_command_is_not_queued(self):
        out = self.run_turn("pause the music")
        self.assertEqual(self.queued, [])
        self.assertIn("[learn-prefilter] skipped (", out)

    def test_thanks_and_questions_are_not_queued(self):
        for t in ("thanks a lot", "what time is it", "ok"):
            self.run_turn(t)
        self.assertEqual(self.queued, [])

    def test_a_statement_is_queued(self):
        for t in ("I have a paper due this Friday",
                  "remember that the code changed",
                  "my sister lives in Ohio"):
            self.run_turn(t)
        self.assertEqual(len(self.queued), 3)

    def test_overheard_speech_is_never_judged(self):
        out = self.run_turn("open Chrome", "", owner_directed=False)
        self.assertEqual(len(self.queued), 1)
        self.assertNotIn("learn-prefilter", out)

    def test_a_turn_from_before_a_memory_wipe_is_still_dropped_first(self):
        out = self.run_turn("I like jazz", epoch=-5)
        self.assertEqual(self.queued, [])
        self.assertIn("before a memory wipe", out)
        self.assertNotIn("learn-prefilter", out)

    def test_ran_actions_reach_the_filter(self):
        bc = self.bc
        text = "the left monitor then the lamp"       # no verb, no question
        self.run_turn(text)
        self.assertEqual(len(self.queued), 1, "nothing ran: kept")
        bc._turn_grounding.last_ran = frozenset({"window_move"})
        self.addCleanup(setattr, bc._turn_grounding, "last_ran", frozenset())
        self.run_turn(text)
        self.assertEqual(len(self.queued), 1, "an action ran: skipped")

    def test_a_filter_error_queues_the_turn(self):
        with mock.patch.object(self.bc._learn_prefilter_mod, "decide",
                               side_effect=RuntimeError("boom")):
            self.run_turn("open Chrome")
        self.assertEqual(len(self.queued), 1)


@requires_monolith
class GateAndBatchTests(MonolithGlobalsTestCase):
    """The pre-filter runs AFTER the learn gate decided, so a skipped command
    turn still opens the gate's follow-up window; and a batch whose turns
    were all skipped is not sent."""

    def setUp(self):
        super().setUp()
        bc = self.bc
        self.enqueued = []
        for name, value in (
                ("LEARN_PREFILTER", "on"),
                ("LEARN_EVERY_TURN", True),
                ("LEARN_ONLY_FROM_OWNER", True),
                ("_dialogue_gate_active", lambda: False),
                ("_learn_gate_submit", bc._learn_gate_classify),
                ("_learn_enqueue", self.enqueued.append)):
            p = mock.patch.object(bc, name, value)
            p.start()
            self.addCleanup(p.stop)
        bc._learn_gate_state[0] = bc._learn_gate_mod.LearnGate(90)

    def _turn(self, text):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.bc.learn_from_turn(text, "ok", {}, injected=True,
                                    wake=True, audio=None, sample_rate=0)
        return buf.getvalue()

    def test_gate_sees_the_command_then_the_filter_skips_it(self):
        out = self._turn("pause the music")
        self.assertIn("[learn-gate] learning from this turn", out)
        self.assertIn("[learn-prefilter] skipped (", out)
        self.assertEqual(self.enqueued, [])

    def test_gate_path_carries_ran_names_to_the_filter(self):
        bc = self.bc
        bc._turn_grounding.last_ran = frozenset({"lights_set"})
        self.addCleanup(setattr, bc._turn_grounding, "last_ran", frozenset())
        self._turn("the left monitor then the lamp")
        self.assertEqual(self.enqueued, [])
        bc._turn_grounding.last_ran = frozenset()
        self._turn("I use the left monitor for video")
        self.assertEqual(len(self.enqueued), 1)
        self.assertEqual(self.enqueued[0][5].get("ran"), [])


@requires_monolith
class BatchNotSentTests(MonolithGlobalsTestCase):
    def test_all_skipped_turns_make_no_job_and_no_request(self):
        bc = self.bc
        jobs = []
        saved_p = list(bc._learn_pending)
        bc._learn_pending.clear()
        self.addCleanup(lambda: (bc._learn_pending.clear(),
                                 bc._learn_pending.extend(saved_p)))
        for name, value in (
                ("LEARN_PREFILTER", "on"),
                ("LEARN_EVERY_TURN", True),
                ("LEARN_ONLY_FROM_OWNER", False),
                ("LEARN_BATCH_ENABLED", True),
                ("_dialogue_gate_active", lambda: False),
                ("_learn_batch_ensure_jobs",
                 lambda: jobs.append(1) or True)):
            p = mock.patch.object(bc, name, value)
            p.start()
            self.addCleanup(p.stop)
        with contextlib.redirect_stdout(io.StringIO()):
            for t in ("open Chrome", "thanks", "what's the weather",
                      "pause the music"):
                bc.learn_from_turn(t, "ok", {})
        self.assertEqual(bc._learn_pending, [])
        self.assertEqual(jobs, [], "no batch job is even asked for")
        with contextlib.redirect_stdout(io.StringIO()):
            bc.learn_from_turn("I have two cats", "ok", {})
        self.assertEqual(len(bc._learn_pending), 1)
        self.assertEqual(jobs, [1])


@requires_monolith
class AmbientOrderTests(MonolithGlobalsTestCase):
    """The cheap length floor runs before the content judge's LLM call."""

    def _run(self, mode, text):
        bc = self.bc
        judged = []
        learned = []
        patches = [
            ("LEARN_PREFILTER", mode),
            ("AMBIENT_LISTEN_ENABLED", True),
            ("LEARN_BATCH_ENABLED", False),
            ("LEARN_ONLY_FROM_OWNER", False),
            ("_dialogue_gate_active", lambda: False),
            ("_ambient_media_is_playing", lambda: False),
            ("_ambient_owner_voice", lambda a, sr: (True, "owner", 0.8)),
            ("_ambient_validity_ok", lambda *a, **k: True),
            ("_ambient_content_is_media",
             lambda s: judged.append(s) or False),
            ("learn_from_turn", lambda *a, **k: learned.append(a[0])),
        ]
        ps = [mock.patch.object(bc, n, v) for n, v in patches]
        for p in ps:
            p.start()
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                bc._ambient_learn_from_gated(text, {}, audio=object(),
                                             sample_rate=16000)
        finally:
            for p in ps:
                p.stop()
        return judged, learned, buf.getvalue()

    def test_on_drops_a_fragment_before_the_judge(self):
        judged, learned, out = self._run("on", "ok sure")
        self.assertEqual(judged, [])
        self.assertEqual(learned, [])
        self.assertIn("low-information fragment", out)

    def test_shadow_only_logs_the_fragment(self):
        judged, learned, out = self._run("shadow", "ok sure")
        self.assertIn("would skip (ambient fragment)", out)
        self.assertEqual(len(judged), 1)
        self.assertEqual(learned, ["ok sure"])

    def test_off_is_the_old_order(self):
        judged, learned, out = self._run("off", "ok sure")
        self.assertNotIn("learn-prefilter", out)
        self.assertEqual(len(judged), 1)
        self.assertEqual(learned, ["ok sure"])

    def test_a_real_line_still_reaches_the_judge_and_is_learned(self):
        line = "we are flying out to Denver on Tuesday morning"
        judged, learned, _out = self._run("on", line)
        self.assertEqual(judged, [line])
        self.assertEqual(learned, [line])


class DispatchLedgerTests(_Base):
    mode = "on"

    def test_end_of_outermost_dispatch_keeps_the_ran_names(self):
        bc = self.bc
        prev = bc._begin_turn_grounding("turn the lamp off")
        bc._turn_grounding.frame["ran"].add("lights_set")
        bc._end_turn_grounding(prev)
        self.assertEqual(bc._turn_grounding.last_ran,
                         frozenset({"lights_set"}))
        self.assertEqual(bc._learn_prefilter_ran(True), ["lights_set"])
        self.assertIsNone(bc._learn_prefilter_ran(False))
        prev = bc._begin_turn_grounding("next turn")
        self.assertEqual(bc._turn_grounding.last_ran, frozenset())
        bc._end_turn_grounding(prev)


if __name__ == "__main__":
    unittest.main()
