"""TURN_TRACE wired into the monolith (v3 W20, core/turn_trace.py).

Pinned here:
  * the setting ships on, has its own line in _TURN_FLAG_KEYS and is local to
    each machine;
  * _tt("begin") opens the collector and the _tt("emit") that prints the
    [turn-timing] line commits ONE queued record built from it;
  * TURN_TRACE False collects and queues nothing;
  * the log tee feeds shadow "would ..." and validator lines to the open turn;
  * a fault in the trace never reaches the turn;
  * the dispatch body notes its chain (names + ok/fail, the cut) before it
    returns, and the main loop notes the owner's words before the shortcuts.

    python -m unittest tests.monolith.test_monolith_turn_trace
"""
from __future__ import annotations

import io
import os
import re
import shutil
import sys
import tempfile
import unittest
from unittest import mock

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(os.path.dirname(_HERE))
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from core import turn_timing  # noqa: E402
from core import turn_trace as ttr  # noqa: E402
from tests._monolith_harness import (  # noqa: E402
    MonolithGlobalsTestCase, requires_monolith,
)


def _src():
    with open(os.path.join(_PROJECT, "bobert_companion.py"),
              encoding="utf-8") as fh:
        return fh.read()


@requires_monolith
class TurnTraceWiringTests(MonolithGlobalsTestCase):
    def setUp(self):
        super().setUp()
        ttr.reset_for_tests()
        self.dir = tempfile.mkdtemp(prefix="mtt_")
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.addCleanup(ttr.reset_for_tests)
        self.timing = turn_timing.TurnTiming(print_fn=lambda _l: None)
        for p in (mock.patch.object(self.bc, "_turn_timing", self.timing),
                  mock.patch.object(self.bc, "TURN_TRACE", True),
                  mock.patch.object(ttr, "trace_dir", lambda: self.dir),
                  mock.patch.object(ttr, "_start_timer",
                                    lambda d, f: mock.Mock()),
                  mock.patch.object(ttr, "_atexit_armed", [True])):
            p.start()
            self.addCleanup(p.stop)

    def test_setting_ships_on_local_and_listed_on_its_own_line(self):
        import core.config as cfg
        from core import peer_settings
        self.assertIs(cfg.TURN_TRACE, True)
        self.assertFalse(peer_settings.is_shared("TURN_TRACE"))
        self.assertIn("TURN_TRACE", self.bc._TURN_FLAG_KEYS)
        self.assertRegex(_src(), r'\n    "TURN_TRACE",\n')

    def test_begin_then_emit_queues_one_record(self):
        bc = self.bc
        bc._tt("begin", "inject")
        bc._turn_trace_call("note_text", "open chrome", "typed")
        bc._tt("mark", "you")
        bc._tt("emit", "ok")
        self.assertEqual(len(ttr._queue), 1)
        rec = ttr._queue[0]
        self.assertEqual(rec["kind"], "inject")
        self.assertEqual(rec["channel"], "typed")
        self.assertEqual(rec["outcome"], "ok")
        self.assertEqual(rec["you"], "open chrome")
        self.assertIn("end", rec["stages"])

    def test_begin_voice_opens_the_collector_too(self):
        bc = self.bc
        bc._tt("begin_voice")
        bc._turn_trace_call("note_text", "pause", "voice")
        bc._tt("emit", "shortcut")
        self.assertEqual(ttr._queue[0]["outcome"], "shortcut")
        self.assertEqual(ttr._queue[0]["kind"], "voice")

    def test_the_chain_note_carries_names_and_failures(self):
        bc = self.bc
        bc._tt("begin", "inject")
        bc._turn_trace_call("note_chain", "Done, sir.",
                            [("open_app", "Opened", False),
                             ("click", "couldn't find it", True)],
                            "no new progress", 1,
                            failed=lambda r: "couldn't" in r)
        bc._tt("emit", "ok")
        rec = ttr._queue[0]
        self.assertEqual(rec["actions"], [{"n": "open_app", "ok": True},
                                          {"n": "click", "ok": False}])
        self.assertEqual(rec["cut"], "no new progress")

    def test_off_collects_and_queues_nothing(self):
        bc = self.bc
        with mock.patch.object(bc, "TURN_TRACE", False):
            bc._tt("begin", "inject")
            bc._turn_trace_call("note_text", "open chrome", "typed")
            bc._turn_trace_observe("  [head-hold] would drop sentence 1/2")
            bc._tt("emit", "ok")
        self.assertEqual(len(ttr._queue), 0)
        self.assertEqual(ttr._carry, {})
        self.assertEqual(ttr._cur, {})

    def test_the_log_tee_feeds_shadow_and_validator_lines(self):
        bc = self.bc
        tee = bc._TimestampedTee(io.StringIO(), io.StringIO())
        bc._tt("begin", "inject")
        tee.write("  [head-hold] would drop sentence 1/2 (3 word(s))")
        tee.write("\n")
        tee.write("  [validation] dropped 'x': written before")
        tee.write("\n")
        tee.write("  Thinking...\n")
        bc._tt("emit", "ok")
        rec = ttr._queue[0]
        self.assertEqual(rec["shadow"], {"head-hold": 1})
        self.assertEqual(rec["validators"], {"validation": 1})

    def test_a_broken_trace_never_reaches_the_turn(self):
        bc = self.bc
        boom = RuntimeError("x")
        with mock.patch.object(ttr, "begin", side_effect=boom), \
                mock.patch.object(ttr, "commit", side_effect=boom), \
                mock.patch.object(ttr, "note_text", side_effect=boom), \
                mock.patch.object(ttr, "observe", side_effect=boom):
            bc._tt("begin", "inject")
            bc._turn_trace_call("note_text", "hi", "typed")
            bc._turn_trace_observe("  [head-hold] would drop")
            line = bc._tt("emit", "ok")
            tee = bc._TimestampedTee(io.StringIO(), io.StringIO())
            tee.write("  [head-hold] would drop\n")
        self.assertIn("[turn-timing]", line)

    def test_dispatch_body_notes_its_chain_before_it_returns(self):
        src = _src()
        i = src.index("def _run_llm_dispatch_body(")
        j = src.index("def _upgrade_task_signature", i)
        body = src[i:j]
        self.assertIn('_turn_trace_call("note_chain", reply, _chain_results,',
                      body)
        self.assertTrue(re.search(
            r'_turn_trace_call\("note_model", _local_budget_tag\(\)\)\n'
            r'    return reply\n', body))
        self.assertEqual(body.count("\n    return "), 1)

    def test_main_loop_notes_the_owner_words_before_the_shortcuts(self):
        src = _src()
        note = src.index('_turn_trace_call("note_text", text,')
        shortcut = src.index("if _run_voice_shortcuts(text):")
        self.assertLess(note, shortcut)
        self.assertLess(shortcut - note, 400)

    def test_emit_still_returns_the_turn_timing_line(self):
        bc = self.bc
        bc._tt("begin", "inject")
        line = bc._tt("emit", "ok")
        self.assertTrue(line.strip().startswith("[turn-timing]"))


if __name__ == "__main__":
    unittest.main()
