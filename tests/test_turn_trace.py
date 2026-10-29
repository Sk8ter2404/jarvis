"""Tests for core.turn_trace and tools/turn_trace_report.py (v3 W20).

Fakes only: a temp directory for the files and a recording timer factory, so
no real timer thread runs and nothing touches the live data dir.

    python -m unittest tests.test_turn_trace
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core import turn_trace as tt  # noqa: E402
from tools import turn_trace_report as rep  # noqa: E402

LINE = ("  [turn-timing] kind=voice outcome=ok vad_break=0 stt_start=3 "
        "stt_end=1581 you=1584 llm_post=1650 llm_done=4410 actions_done=4418 "
        "synth_start=4420 first_play=5120 end=5890 prompt_eval_count=11873 "
        "prompt_eval_ms=2702 eval_count=84 eval_ms=790 llm_calls=1 "
        "followup_rounds=0 filler=0 filler_ms=- audible_ms=5207 "
        "pe_new=100 pe_state=warm")


class _Timers:
    """Recording replacement for turn_trace._start_timer."""

    def __init__(self):
        self.calls = []

    def __call__(self, delay, fn):
        self.calls.append((delay, fn))
        return mock.Mock()


class _Base(unittest.TestCase):
    def setUp(self):
        tt.reset_for_tests()
        self.dir = tempfile.mkdtemp(prefix="turntrace_")
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.timers = _Timers()
        for p in (mock.patch.object(tt, "trace_dir", lambda: self.dir),
                  mock.patch.object(tt, "_start_timer", self.timers),
                  mock.patch.object(tt, "_atexit_armed", [True])):
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(tt.reset_for_tests)

    def files(self):
        return sorted(os.listdir(self.dir))

    def records(self, day=None):
        out = []
        for name in self.files():
            with open(os.path.join(self.dir, name), encoding="utf-8") as fh:
                out.extend(json.loads(ln) for ln in fh if ln.strip())
        return out


class SchemaTests(_Base):
    def one_turn(self, **kw):
        tt.begin()
        tt.note_text(kw.get("you", "open chrome please"), "voice")
        tt.note_sections(["Browser", "Media", "Browser"])
        tt.note_chain(kw.get("reply", "Opening Chrome, sir."),
                      [("open_app", "ok", False), ("see_screen", "FAILED", True)],
                      cut="no new progress", rounds=2,
                      failed=lambda r: r == "FAILED")
        tt.note_model("gemma4:26b")
        tt.observe("  [head-hold] would drop sentence 1/2 (1 word(s)) written")
        tt.observe("[12:00:01]   [validation] dropped 'x': written before")
        tt.observe("  [head-hold] not speaking sentence 2/2")   # 'on' wording
        return tt.commit(LINE, "ok")

    def test_record_schema(self):
        rec = self.one_turn()
        self.assertEqual(rec["v"], 1)
        self.assertEqual(rec["kind"], "voice")
        self.assertEqual(rec["channel"], "voice")
        self.assertEqual(rec["outcome"], "ok")
        self.assertEqual(rec["sections"], ["Browser", "Media"])
        self.assertEqual(rec["actions"], [{"n": "open_app", "ok": True},
                                          {"n": "see_screen", "ok": False}])
        self.assertEqual(rec["cut"], "no new progress")
        self.assertEqual(rec["rounds"], 2)
        self.assertEqual(rec["shadow"], {"head-hold": 1})
        self.assertEqual(rec["validators"], {"validation": 1})
        self.assertEqual(rec["model"], "gemma4:26b")
        self.assertEqual(rec["pe_state"], "warm")
        self.assertEqual(rec["stages"], {
            "stt": 1578, "llm": 2760, "actions": 8, "first_play": 5120,
            "audible": 5207, "end": 5890})
        self.assertRegex(rec["day"], r"^\d{4}-\d{2}-\d{2}$")
        self.assertEqual(rec["llm_calls"], 1)

    def test_missing_marks_leave_stage_out(self):
        line = LINE.replace("stt_start=3", "stt_start=-").replace(
            "audible_ms=5207", "audible_ms=-")
        tt.begin()
        rec = tt.commit(line, "ok")
        self.assertNotIn("stt", rec["stages"])
        self.assertNotIn("audible", rec["stages"])
        self.assertIn("llm", rec["stages"])

    def test_typed_channel_default_and_first_note_text_wins(self):
        tt.begin()
        tt.note_text("first", "web")
        tt.note_text("second", "voice")
        rec = tt.commit(LINE.replace("kind=voice", "kind=inject"), "ok")
        self.assertEqual(rec["channel"], "web")
        self.assertEqual(rec["you_chars"], len("first"))

    def test_shadow_line_with_no_turn_open_rides_with_next_turn(self):
        tt.observe("  [learn-prefilter] would skip (command)")
        tt.observe("  [learn-prefilter] would skip (thanks)")
        tt.begin()
        rec = tt.commit(LINE, "ok")
        self.assertEqual(rec["shadow"], {"learn-prefilter": 2})
        tt.begin()
        self.assertNotIn("shadow", tt.commit(LINE, "ok"))

    def test_an_abandoned_collector_keeps_its_shadow_lines(self):
        # Live 2026-10-28 15:55: the "would skip (command)" line carried into
        # the 15:55:10 VAD break; that break was dropped as noise (no emit),
        # and the 15:55:17 VAD break wiped it before the 15:55:21 record.
        # bobert_companion._turn_trace_hook calls begin() on EVERY
        # begin_voice (each VAD break) and commit() only on an emit.
        tt.observe("  [learn-prefilter] would skip (command)")
        tt.begin()                                   # 15:55:10, dropped as noise
        tt.observe("  [head-hold] would drop sentence 1/2 (1 word(s)) written")
        tt.begin()                                   # 15:55:17, no commit between
        rec = tt.commit(LINE, "ok")                  # 15:55:21
        self.assertEqual(rec["shadow"], {"learn-prefilter": 1, "head-hold": 1})
        tt.begin()
        self.assertNotIn("shadow", tt.commit(LINE, "ok"))   # counted once only

    def test_ambient_fragment_lines_are_counted_apart(self):
        tt.observe("  [learn-prefilter] would skip (ambient fragment)")
        tt.begin()
        tt.observe("  [learn-prefilter] would skip (ambient fragment)")
        tt.observe("  [learn-prefilter] would skip (thanks)")
        rec = tt.commit(LINE, "ok")
        self.assertEqual(rec["shadow"], {"learn-prefilter": 1,
                                         tt.AMBIENT_SHADOW_KEY: 2})
        s = rep.summarize([rec])
        self.assertEqual(s["shadow"]["learn-prefilter"], 1)
        out = rep.render("2026-10-28", s, {"learn-prefilter": 13,
                                           rep.AMBIENT_KEY: 103})
        self.assertIn("W09 [learn-prefilter]: 1 / 13", out)
        self.assertIn("W09 ambient fragments (no owner turn): 2 / 103", out)
        self.assertEqual(rep.AMBIENT_KEY, tt.AMBIENT_SHADOW_KEY)

    def test_observe_keeps_only_the_tag(self):
        tt.begin()
        tt.observe("  [action-fit] would swap open_app -> open_url for 'secret words'")
        tt.observe("  [loop-guard] would stop (ping-pong web_search/see_screen)")
        tt.observe("  [stt] heard something would be nice")      # other tag
        tt.observe("plain would line")                           # no tag
        rec = tt.commit(LINE, "ok")
        self.assertEqual(rec["shadow"], {"action-fit": 1, "loop-guard": 1})
        self.assertNotIn("secret", json.dumps(rec))

    def test_no_line_or_disabled_returns_none_and_clears(self):
        tt.begin()
        tt.note_text("hello", "voice")
        self.assertIsNone(tt.commit(LINE, "ok", enabled=False))
        self.assertIsNone(tt.commit(LINE, "ok"))     # nothing open any more
        tt.begin()
        self.assertIsNone(tt.commit(None, "ok"))
        self.assertEqual(len(tt._queue), 0)

    def test_notes_with_no_open_turn_are_ignored(self):
        tt.note_text("stray", "voice")
        tt.note_chain("reply", [("a", "x", False)])
        tt.begin()
        rec = tt.commit(LINE, "ok")
        self.assertNotIn("you", rec)
        self.assertNotIn("you_chars", rec)
        self.assertNotIn("actions", rec)

    def test_garbage_inputs_never_raise(self):
        tt.begin()
        tt.note_chain(None, [object(), (1,), None], failed=lambda r: 1 / 0)
        tt.note_sections(None)
        tt.note_text(None)
        tt.observe(None)
        tt.observe(12)
        self.assertIsNotNone(tt.commit("not a timing line", "ok"))


class WriterTests(_Base):
    def test_nothing_is_written_on_the_calling_thread(self):
        tt.begin()
        tt.note_text("open chrome", "voice")
        tt.commit(LINE, "ok")
        # The record is queued and ONE timer is scheduled; no file, no dir.
        self.assertEqual(len(tt._queue), 1)
        self.assertEqual(len(self.timers.calls), 1)
        self.assertEqual(self.timers.calls[0][0], tt.FLUSH_DELAY_S)
        self.assertEqual(self.files(), [])
        # Several more turns share that one timer.
        for _ in range(3):
            tt.begin()
            tt.commit(LINE, "ok")
        self.assertEqual(len(self.timers.calls), 1)
        self.assertEqual(self.files(), [])

    def test_timer_flush_writes_append_only_json_lines(self):
        for n in range(3):
            tt.begin()
            tt.note_text("turn" + "!" * n, "voice")
            tt.commit(LINE, "ok")
        _delay, fn = self.timers.calls[0]
        fn()    # the timer fires
        recs = self.records()
        self.assertEqual([r["you_chars"] for r in recs], [4, 5, 6])
        self.assertEqual(len(self.files()), 1)
        tt.begin()
        tt.note_text("turn 3", "voice")
        tt.commit(LINE, "ok")
        tt.flush()
        self.assertEqual(len(self.records()), 4)   # appended, not rewritten

    def test_free_text_is_scrubbed_and_cut_on_flush(self):
        secret = "sk-ant-api03-" + "A" * 40
        tt.begin()
        tt.note_text("my key is " + secret + " ok", "voice")
        tt.note_chain("x" * 500, [])
        tt.commit(LINE, "ok")
        tt.flush()
        rec = self.records()[0]
        self.assertNotIn(secret, json.dumps(rec))
        self.assertNotIn("reply", rec)
        self.assertGreater(rec["reply_chars"], tt.TEXT_MAX)

    def test_spoken_codes_are_never_stored(self):
        """Security review DAR-1 (2026-10-27): the scrubber misses PINs and
        door / alarm / verification codes, and nothing reads the text, so the
        trace keeps lengths only, never the owner's words or the reply."""
        for said in ("the garage code is 4829", "remember my bank PIN 4829",
                     "my social is 123 45 6789", "my pin is four eight two nine"):
            tt.begin()
            tt.note_text(said, "voice")
            tt.note_chain("Noted, sir: " + said, [])
            rec = tt.commit(LINE, "ok")
            self.assertNotIn("you", rec)
            self.assertNotIn("reply", rec)
            self.assertEqual(rec["you_chars"], len(said))
        tt.flush()
        with open(os.path.join(self.dir, self.files()[0]), encoding="utf-8") as fh:
            body = fh.read()
        for needle in ("4829", "6789", "four eight", "garage", "PIN"):
            self.assertNotIn(needle, body)
        for rec in self.records():
            self.assertNotIn("you", rec)
            self.assertNotIn("reply", rec)

    def test_text_that_still_reads_like_a_credential_is_not_stored(self):
        tt.begin()
        tt.note_text("the password is hunter2 for my account", "voice")
        tt.commit(LINE, "ok")
        tt.flush()
        self.assertNotIn("you", self.records()[0])

    def test_rotation_by_size(self):
        with mock.patch.object(tt, "MAX_FILE_BYTES", 400):
            for _ in range(8):
                tt.begin()
                tt.commit(LINE, "ok")
                tt.flush()
        names = self.files()
        self.assertGreater(len(names), 1)
        self.assertTrue(any(".1.jsonl" in n for n in names), names)
        self.assertEqual(len(self.records()), 8)

    def test_prune_removes_only_files_past_retention(self):
        now = time.time()
        old = tt._today(now - (tt.RETENTION_DAYS + 3) * 86400)
        edge = tt._today(now - (tt.RETENTION_DAYS - 1) * 86400)
        for name in (f"turns-{old}.jsonl", f"turns-{old}.2.jsonl",
                     f"turns-{edge}.jsonl", f"turns-{tt._today(now)}.jsonl",
                     "other.txt", "turns-garbage.jsonl"):
            open(os.path.join(self.dir, name), "w").close()
        self.assertEqual(tt.prune(self.dir, now=now, force=True), 2)
        self.assertEqual(self.files(), sorted([
            f"turns-{edge}.jsonl", f"turns-{tt._today(now)}.jsonl",
            "other.txt", "turns-garbage.jsonl"]))

    def test_prune_runs_once_a_day_from_the_flush(self):
        old = tt._today(time.time() - 40 * 86400)
        p = os.path.join(self.dir, f"turns-{old}.jsonl")
        open(p, "w").close()
        tt.begin()
        tt.commit(LINE, "ok")
        tt.flush()
        self.assertFalse(os.path.exists(p))
        open(p, "w").close()
        tt.begin()
        tt.commit(LINE, "ok")
        tt.flush()
        self.assertTrue(os.path.exists(p))     # already swept today

    def test_a_failing_disk_never_raises_into_the_turn(self):
        tt.begin()
        tt.note_text("hi", "voice")
        with mock.patch.object(tt.os, "makedirs", side_effect=OSError("full")):
            rec = tt.commit(LINE, "ok")
            self.assertIsNotNone(rec)
            self.assertFalse(tt.flush())      # reports failure, no raise
        with mock.patch("builtins.open", side_effect=OSError("denied")):
            tt.begin()
            tt.commit(LINE, "ok")
            self.assertFalse(tt.flush())

    def test_queue_is_bounded(self):
        with mock.patch.object(tt, "QUEUE_MAX", 5):
            for n in range(9):
                tt.submit({"day": "2026-10-27", "n": n})
        self.assertEqual(len(tt._queue), 5)
        self.assertEqual(tt._queue[0]["n"], 4)
        self.assertEqual(tt._dropped[0], 4)

    def test_flush_with_a_second_writer_gives_up_quietly(self):
        tt.begin()
        tt.commit(LINE, "ok")
        tt._flush_lock.acquire()
        try:
            with mock.patch.object(tt, "_flush_lock") as lk:
                lk.acquire.return_value = False
                self.assertFalse(tt.flush())
        finally:
            tt._flush_lock.release()

    def test_real_timer_thread_is_a_daemon(self):
        t = tt.threading.Timer
        self.assertTrue(issubclass(t, threading.Thread))
        src = open(tt.__file__, encoding="utf-8").read()
        self.assertIn("t.daemon = True", src)


class WiringTests(unittest.TestCase):
    def test_setting_default_on_and_local_to_each_machine(self):
        import core.config as cfg
        from core import peer_settings
        self.assertIs(cfg.TURN_TRACE, True)
        self.assertIn("TURN_TRACE", peer_settings.LOCAL_KEYS)
        self.assertFalse(peer_settings.is_shared("TURN_TRACE"))

    def test_the_trace_is_never_part_of_peer_or_chat_paths(self):
        src = open(os.path.join(_ROOT, "core", "turn_trace.py"),
                   encoding="utf-8").read()
        for banned in ("discord_bridge", "parse_and_run_actions", "requests",
                       "urllib", "socket"):
            self.assertNotIn(banned, src, f"turn_trace must not touch {banned}")

    def test_module_never_reads_the_file_back_into_a_decision(self):
        src = open(os.path.join(_ROOT, "core", "turn_trace.py"),
                   encoding="utf-8").read()
        self.assertNotIn("json.load", src)
        self.assertNotIn("json.loads", src)


# ── the report ───────────────────────────────────────────────────────────

def _line(i, pe="warm", first_play=4000, end=5000):
    return (f"  [turn-timing] kind=voice outcome=ok vad_break=0 stt_start=0 "
            f"stt_end=1000 you=1002 llm_post=1100 llm_done={2000 + i} "
            f"actions_done={2010 + i} synth_start=2020 first_play={first_play + i} "
            f"end={end + i} llm_calls=1 followup_rounds=0 filler=0 "
            f"filler_ms=- audible_ms={first_play + i + 90} pe_new=0 pe_state={pe}")


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="turnrep_")
        self.addCleanup(shutil.rmtree, self.dir, True)

    def write(self, day, recs, name=None):
        path = os.path.join(self.dir, name or f"turns-{day}.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            for r in recs:
                fh.write((r if isinstance(r, str) else json.dumps(r)) + "\n")

    def fixture(self, n=149, bad=27):
        """n voice turns, `bad` of them partial / full prefill, built from
        real [turn-timing] lines through the same builder the writer uses."""
        recs = []
        for i in range(n):
            pe = ("partial" if i % 2 else "full") if i < bad else "warm"
            rec = tt.build_record(_line(i, pe), "ok", {"channel": "voice"},
                                  now=time.mktime((2026, 10, 27, 12, 0, 0, 0, 0, -1)))
            recs.append(rec)
        return recs

    def test_reproduces_the_baseline_prefill_rate(self):
        self.write("2026-10-27", self.fixture())
        s = rep.summarize(rep.load_day(self.dir, "2026-10-27"))
        self.assertEqual(s["turns"], 149)
        self.assertEqual(s["prefill"], {"partial_plus_full": 27, "known": 149})
        out = rep.render("2026-10-27", s)
        self.assertIn("partial+full 27/149", out)

    def test_matches_turn_latency_report_on_the_same_turns(self):
        from tools import turn_latency_report as tlr
        recs = self.fixture(60, 5)
        self.write("2026-10-27", recs)
        s = rep.summarize(rep.load_day(self.dir, "2026-10-27"))
        kvs = [tlr.parse_kv(_line(i, "warm")[len("  [turn-timing] "):])
               for i in range(60)]
        self.assertEqual(s["turns"], len(kvs))
        fp = [int(k["first_play"]) for k in kvs]
        self.assertEqual(s["latency"]["first_play"]["p50"], tlr.pct(fp, 50))
        self.assertEqual(s["latency"]["first_play"]["p95"], tlr.pct(fp, 95))

    def test_percentiles_and_counts(self):
        recs = [{"kind": "voice", "channel": "voice", "outcome": "ok",
                 "stages": {"llm": v}, "actions": [{"n": "a", "ok": True},
                                                   {"n": "b", "ok": False}],
                 "cut": "depth cap", "rounds": 1,
                 "validators": {"validation": 1}, "shadow": {"head-hold": 2}}
                for v in (100, 200, 300, 400, 1000)]
        recs.append({"kind": "inject", "channel": "typed", "outcome": "error"})
        s = rep.summarize(recs)
        self.assertEqual(s["turns"], 6)
        self.assertEqual(s["latency"]["llm"], {"n": 5, "p50": 300, "p95": 880})
        self.assertEqual(s["actions"]["ran"], 10)
        self.assertEqual(s["actions"]["failed"], 5)
        self.assertEqual(s["actions"]["top_failed"], [("b", 5)])
        self.assertEqual(s["cuts"], {"depth cap": 5})
        self.assertEqual(s["validators"], {"validation": 5})
        self.assertEqual(s["shadow"]["head-hold"], 10)
        self.assertEqual(s["by_outcome"], {"ok": 5, "error": 1})
        only = rep.summarize(recs, kind="inject")
        self.assertEqual(only["turns"], 1)

    def test_rotated_parts_and_torn_lines_are_read(self):
        a = {"kind": "voice", "stages": {}}
        self.write("2026-10-27", [a, '{"kind": "voice", "sta'], None)
        self.write("2026-10-27", [a, a], "turns-2026-10-27.1.jsonl")
        self.write("2026-10-28", [a], None)
        self.assertEqual(len(rep.load_day(self.dir, "2026-10-27")), 3)
        self.assertEqual(len(rep.load_day(self.dir, "2026-10-28")), 1)
        self.assertEqual(rep.load_day(self.dir, "2026-10-29"), [])

    def test_log_cross_check_counts_the_four_tags(self):
        logs = tempfile.mkdtemp(prefix="turnlogs_")
        self.addCleanup(shutil.rmtree, logs, True)
        with open(os.path.join(logs, "session_2026-10-27_08-00-00.log"), "w",
                  encoding="utf-8") as fh:
            fh.write("[08:00:01]   [head-hold] would drop sentence 1/2\n"
                     "[08:00:02]   [head-hold] not speaking sentence 1/2\n"
                     "[08:00:03]   [learn-prefilter] would skip (command)\n"
                     "[08:00:03]   [learn-prefilter] would skip (ambient fragment)\n"
                     "[08:00:03]   [learn-prefilter] would skip (ambient fragment)\n"
                     "[08:00:04]   [loop-guard] would stop (ping-pong a/b)\n"
                     "[08:00:05]   [stt] would be nice\n")
        with open(os.path.join(logs, "session_2026-10-26_08-00-00.log"), "w",
                  encoding="utf-8") as fh:
            fh.write("[08:00:01]   [head-hold] would drop sentence 1/2\n")
        c = rep.log_shadow_counts(logs, "2026-10-27")
        self.assertEqual(c, {"head-hold": 1, "learn-prefilter": 1,
                             "action-fit": 0, "loop-guard": 1,
                             "learn-prefilter-ambient": 2})
        out = rep.render("2026-10-27", rep.summarize([]), c)
        self.assertIn("W02 [head-hold]: 0 / 1", out)
        self.assertIn("W11 [action-fit]: 0 / 0", out)

    def test_output_has_no_owner_words(self):
        recs = [{"kind": "voice", "channel": "voice", "outcome": "ok",
                 "you": "SECRETWORD open the pod bay doors",
                 "reply": "REPLYWORD I'm afraid I can't", "stages": {}}]
        self.write("2026-10-27", recs)
        import io
        import contextlib
        for extra in ([], ["--json"]):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = rep.main(["--day", "2026-10-27", "--dir", self.dir] + extra)
            self.assertEqual(rc, 0)
            self.assertNotIn("SECRETWORD", buf.getvalue())
            self.assertNotIn("REPLYWORD", buf.getvalue())

    def test_main_errors(self):
        import io
        import contextlib
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertEqual(rep.main(["--dir", os.path.join(self.dir, "nope")]), 2)
        with self.assertRaises(SystemExit):
            with contextlib.redirect_stderr(io.StringIO()):
                rep.main(["--day", "tomorrow", "--dir", self.dir])

    def test_empty_day_prints_a_note(self):
        out = rep.render("2026-10-27", rep.summarize([]))
        self.assertIn("0 turn(s)", out)
        self.assertIn("no records", out)


if __name__ == "__main__":
    unittest.main()
