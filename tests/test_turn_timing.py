"""core/turn_timing.py — the per-turn [turn-timing] line and the Ollama stats
behind the "[local-llm] served via" suffix (speed plan rank 1, 2026-09-29).

Stdlib only, so this runs on the CI-light tier. The monolith wiring is pinned
separately in tests/monolith/test_monolith_turn_timing.py.

Run: python tools/run_tests.py test_turn_timing
"""
from __future__ import annotations

import re
import threading
import time
import unittest

from core import turn_timing as tt

# A real /api/chat body (non-streaming), trimmed to the fields that matter.
_FULL = {
    "model": "gemma-test",
    "message": {"role": "assistant", "content": "It is noon, sir."},
    "done": True,
    "total_duration": 3_512_345_678,
    "load_duration": 12_600_000,
    "prompt_eval_count": 11873,
    "prompt_eval_duration": 2_702_400_000,
    "eval_count": 84,
    "eval_duration": 790_100_000,
}
_BARE = {"model": "gemma-test",
         "message": {"role": "assistant", "content": "Hi."}, "done": True}


class _Clock:
    """Fake perf_counter: every read advances 10 ms, so every mark is
    strictly later than the one before it."""

    def __init__(self, start=100.0, step=0.010):
        self.t = start
        self.step = step
        self.lock = threading.Lock()

    def __call__(self):
        with self.lock:
            self.t += self.step
            return self.t


class _ManualClock:
    """Fake perf_counter that only moves when told to (exact offsets)."""

    def __init__(self, start=500.0):
        self.t = start

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


class _Resp:
    def __init__(self, body=None, exc=None):
        self._body = body
        self._exc = exc

    def json(self):
        if self._exc is not None:
            raise self._exc
        return self._body


def _in_thread(fn):
    th = threading.Thread(target=fn)
    th.start()
    th.join(2)


class OllamaStatsTests(unittest.TestCase):
    def test_full_body(self):
        s = tt.ollama_stats(_FULL)
        self.assertEqual(s["prompt_eval_count"], 11873)
        self.assertEqual(s["prompt_eval_ms"], 2702)
        self.assertEqual(s["eval_count"], 84)
        self.assertEqual(s["eval_ms"], 790)
        self.assertEqual(s["load_ms"], 13)
        self.assertEqual(s["total_ms"], 3512)

    def test_body_without_timing_fields_is_all_none(self):
        s = tt.ollama_stats(_BARE)
        self.assertEqual(set(s), {"prompt_eval_count", "prompt_eval_ms",
                                  "eval_count", "eval_ms", "load_ms",
                                  "total_ms"})
        self.assertTrue(all(v is None for v in s.values()), s)

    def test_partial_body(self):
        # A full KV-cache hit can omit prompt_eval_count but keep eval_*.
        s = tt.ollama_stats({"eval_count": 5, "eval_duration": 50_000_000})
        self.assertIsNone(s["prompt_eval_count"])
        self.assertIsNone(s["prompt_eval_ms"])
        self.assertEqual((s["eval_count"], s["eval_ms"]), (5, 50))

    def test_junk_never_raises(self):
        for junk in (None, "text", 42, [1, 2], {"prompt_eval_count": "x",
                                                "eval_duration": object(),
                                                "eval_count": True}):
            s = tt.ollama_stats(junk)
            self.assertTrue(all(v is None for v in s.values()), (junk, s))

    def test_response_stats(self):
        self.assertEqual(tt.response_stats(_Resp(_FULL))["eval_count"], 84)
        s = tt.response_stats(_Resp(exc=ValueError("not json")))
        self.assertTrue(all(v is None for v in s.values()))
        s = tt.response_stats(object())   # no .json at all
        self.assertTrue(all(v is None for v in s.values()))


class ServedViaSuffixTests(unittest.TestCase):
    def test_full(self):
        self.assertEqual(
            tt.served_via_suffix(tt.ollama_stats(_FULL), "_call_llm@MainThread"),
            "pe=11873/2702 ev=84 caller=_call_llm@MainThread")

    def test_missing_fields_print_question_marks(self):
        self.assertEqual(tt.served_via_suffix(tt.ollama_stats(_BARE), "x@T"),
                         "pe=?/? ev=? caller=x@T")
        self.assertEqual(tt.served_via_suffix(None), "pe=?/? ev=?")
        self.assertEqual(tt.served_via_suffix("junk", ""), "pe=?/? ev=?")


class CallerTagTests(unittest.TestCase):
    def test_plain_caller(self):
        import sys

        def my_background_job():
            return tt.caller_tag(sys._getframe(0))

        self.assertEqual(my_background_job(),
                         f"my_background_job@{threading.current_thread().name}")

    def test_wrapper_names_its_own_caller(self):
        import sys

        def _llm_quick():
            return tt.caller_tag(sys._getframe(0))

        def _extract_facts():
            return _llm_quick()

        self.assertEqual(_extract_facts().split("@")[0],
                         "_llm_quick<_extract_facts")

    def test_thread_name_is_included(self):
        import sys
        out = []

        def worker():
            out.append(tt.caller_tag(sys._getframe(0)))

        th = threading.Thread(target=worker, name="learn-worker")
        th.start()
        th.join(2)
        self.assertEqual(out, ["worker@learn-worker"])

    def test_bad_frame_never_raises(self):
        self.assertEqual(tt.caller_tag(object()), "?")

    def test_unnamed_thread_tag_is_one_token(self):
        # Python 3.10+ names an unnamed Thread "Thread-N (target)"; the space
        # would split the served-via line's caller=<tag> token in two.
        import sys
        out = []

        def _ambient_learn_from_gated():
            out.append(tt.caller_tag(sys._getframe(0)))

        th = threading.Thread(target=_ambient_learn_from_gated, daemon=True)
        th.start()
        th.join(2)
        self.assertEqual(len(out), 1)
        self.assertRegex(out[0], r"^\S+$")
        self.assertTrue(out[0].startswith("_ambient_learn_from_gated@"), out[0])
        suffix = tt.served_via_suffix(tt.ollama_stats(_FULL), out[0])
        self.assertEqual(re.search(r"caller=(\S+)$", suffix).group(1), out[0])


class TurnTimingTests(unittest.TestCase):
    def setUp(self):
        self.lines = []
        self.clock = _Clock()
        self.t = tt.TurnTiming(print_fn=self.lines.append, clock=self.clock)

    def _full_turn(self):
        t = self.t
        since = t.now()
        t.note_vad_break()
        t.begin_voice(since)
        t.mark("stt_start")
        t.mark("stt_end")
        t.mark("you")
        t.mark("llm_post", owner_only=True)
        t.llm_response(tt.ollama_stats(_FULL))
        t.set_first("turn_ctx_chars", 1342)
        t.set_first("sys_chars", 31012)
        t.mark("actions_done", owner_only=True)
        t.mark("synth_start")
        t.mark("first_play")
        return t.emit("ok")

    def test_full_voice_turn_line(self):
        line = self._full_turn()
        self.assertEqual(self.lines, [line])
        self.assertTrue(line.startswith("  [turn-timing] kind=voice outcome=ok "))
        d = tt.parse_line(line)
        self.assertEqual(d["vad_break"], "0")
        offs = [int(d[m]) for m in tt.MARKS]
        self.assertEqual(offs, sorted(offs))
        self.assertEqual(len(set(offs)), len(offs), offs)   # strictly rising
        self.assertGreater(int(d["end"]), offs[-1])
        self.assertEqual(d["prompt_eval_count"], "11873")
        self.assertEqual(d["prompt_eval_ms"], "2702")
        self.assertEqual(d["eval_count"], "84")
        self.assertEqual(d["eval_ms"], "790")
        self.assertEqual(d["llm_calls"], "1")
        self.assertEqual(d["turn_ctx_chars"], "1342")
        self.assertEqual(d["sys_chars"], "31012")
        self.assertEqual(d["followup_rounds"], "0")
        self.assertEqual((d["filler"], d["filler_ms"]), ("0", "-"))

    def test_lead_dropped_defaults_to_zero(self):
        d = tt.parse_line(self._full_turn())
        self.assertEqual(d["lead_dropped"], "0")
        self.assertIn("lead_dropped", tt.STAT_FIELDS)

    def test_budget_trimmed_is_recorded_when_set(self):
        # 2026-10-02: a turn the local prompt budget trimmed is marked, so its
        # prompt_eval_count never enters the chars-per-token calibration as
        # an untrimmed one; '-' when no budget ran (a cloud turn).
        # The brain-prefix fields (2026-10-04) follow it.
        self.assertEqual(tt.STAT_FIELDS[-1 - len(tt.BRAIN_FIELDS)],
                         "budget_trimmed")
        self.assertEqual(tt.parse_line(self._full_turn())["budget_trimmed"],
                         "-")
        t = self.t
        t.begin("inject")
        t.set_first("budget_trimmed", 1)
        t.set_first("budget_trimmed", 0)       # first value wins
        self.assertEqual(tt.parse_line(t.emit())["budget_trimmed"], "1")

    def test_lead_dropped_marked_by_the_turn_thread_only(self):
        t = self.t
        t.begin("inject")
        th = threading.Thread(target=t.note_lead_dropped)
        th.start()
        th.join()
        self.assertEqual(tt.parse_line(t.emit())["lead_dropped"], "0")
        t.begin("inject")
        t.note_lead_dropped()
        t.note_lead_dropped()
        self.assertEqual(tt.parse_line(t.emit())["lead_dropped"], "1")

    def test_lead_dropped_without_a_turn_never_raises(self):
        self.t.note_lead_dropped()
        self.assertIsNone(self.t.emit())

    def test_fixed_schema_order(self):
        line = self._full_turn()
        keys = [tok.split("=")[0] for tok in
                line.split("[turn-timing]")[1].split()]
        self.assertEqual(keys, ["kind", "outcome", *tt.MARKS, "end",
                                *tt.STAT_FIELDS])

    def test_emits_exactly_once(self):
        self._full_turn()
        self.assertIsNone(self.t.emit("error"))
        self.assertEqual(len(self.lines), 1)
        self.assertFalse(self.t.active())

    def test_no_turn_no_line(self):
        self.assertIsNone(self.t.emit())
        self.t.mark("you")
        self.t.llm_response({})
        self.t.note_filler()
        self.assertEqual(self.lines, [])

    def test_partial_line_marks_missing_as_dash(self):
        self.t.begin("inject")
        self.t.mark("you")
        line = self.t.emit("error")
        d = tt.parse_line(line)
        self.assertEqual((d["kind"], d["outcome"]), ("inject", "error"))
        self.assertNotEqual(d["you"], "-")
        for m in ("vad_break", "stt_start", "stt_end", "llm_post", "llm_done",
                  "actions_done", "synth_start", "first_play"):
            self.assertEqual(d[m], "-", m)
        self.assertEqual(d["prompt_eval_count"], "-")
        self.assertEqual(d["llm_calls"], "0")

    def test_stale_vad_is_not_used(self):
        self.t.note_vad_break()          # an older recording's break
        since = self.t.now()
        self.t.begin_voice(since)        # this recording never broke on VAD
        d = tt.parse_line(self.t.emit())
        self.assertEqual(d["vad_break"], "-")

    def test_first_mark_wins(self):
        self.t.begin("inject")
        self.t.mark("you")
        self.t.mark("first_play")
        first = tt.parse_line(self.t.emit())["first_play"]
        self.t.begin("inject")
        self.t.mark("you")
        self.t.mark("first_play")
        self.t.mark("first_play")
        self.t.mark("first_play")
        self.assertEqual(tt.parse_line(self.t.emit())["first_play"], first)

    def test_audio_before_the_transcript_is_not_first_play(self):
        self.t.begin("inject")
        self.t.mark("synth_start")       # a queued reminder, pre-transcript
        self.t.mark("first_play")
        d = tt.parse_line(self.t.emit())
        self.assertEqual((d["synth_start"], d["first_play"]), ("-", "-"))

    def test_other_threads_cannot_touch_turn_owned_fields(self):
        t = self.t
        t.begin("inject")
        t.mark("you")

        def background():
            t.mark("llm_post", owner_only=True)
            t.llm_response(tt.ollama_stats(_FULL))
            t.set_first("turn_ctx_chars", 9)
            t.followup_round()
            self.assertIsNone(t.emit("error"))   # cannot end the turn

        _in_thread(background)
        self.assertTrue(t.active())
        d = tt.parse_line(t.emit())
        self.assertEqual(d["llm_post"], "-")
        self.assertEqual(d["llm_done"], "-")
        self.assertEqual(d["llm_calls"], "0")
        self.assertEqual(d["turn_ctx_chars"], "-")
        self.assertEqual(d["followup_rounds"], "0")

    def test_any_thread_can_note_the_filler(self):
        t = self.t
        t.begin("voice")
        t.mark("you")
        _in_thread(t.note_filler)
        d = tt.parse_line(t.emit())
        self.assertEqual(d["filler"], "1")
        self.assertNotEqual(d["filler_ms"], "-")

    def test_unadopted_thread_audio_is_not_the_answer(self):
        # A reminder / tray command / mid-task status line spoken by another
        # thread during the turn is not the answer's first audio.
        t = self.t
        t.begin("voice")
        t.mark("you")

        def reminder_thread():
            t.mark("synth_start")
            t.mark("first_play")

        _in_thread(reminder_thread)
        d = tt.parse_line(t.emit())
        self.assertEqual((d["synth_start"], d["first_play"]), ("-", "-"))

    def test_adopted_helper_audio_counts(self):
        t = self.t
        t.begin("voice")
        t.mark("you")

        def flush_thread():
            t.mark("synth_start")
            t.mark("first_play")

        th = threading.Thread(target=flush_thread, name="stream-tts-flush")
        t.adopt(th)                      # the owner adopts before start
        th.start()
        th.join(2)
        d = tt.parse_line(t.emit())
        self.assertNotEqual(d["synth_start"], "-")
        self.assertNotEqual(d["first_play"], "-")

    def test_only_the_owner_can_adopt(self):
        t = self.t
        t.begin("voice")
        t.mark("you")

        def flush_thread():
            t.mark("first_play")

        th = threading.Thread(target=flush_thread)
        _in_thread(lambda: t.adopt(th))  # a stranger tries to adopt
        th.start()
        th.join(2)
        self.assertEqual(tt.parse_line(t.emit())["first_play"], "-")

    def test_followups_and_llm_calls_count(self):
        t = self.t
        t.begin("inject")
        t.llm_response(tt.ollama_stats(_FULL))
        t.followup_round()
        t.llm_response(tt.ollama_stats(_BARE))
        d = tt.parse_line(t.emit())
        self.assertEqual((d["llm_calls"], d["followup_rounds"]), ("2", "1"))
        # The main call's counters, not the follow-up's.
        self.assertEqual(d["prompt_eval_count"], "11873")

    def test_unserved_response_counts_but_does_not_answer(self):
        # An empty reply (or HTTP error) before a failover must not supply
        # llm_done or the counters; the response that answered does.
        clock = _ManualClock()
        t = tt.TurnTiming(print_fn=self.lines.append, clock=clock)
        t.begin("inject")
        clock.advance(0.100)
        t.llm_response({"prompt_eval_count": 1, "prompt_eval_ms": 5,
                        "eval_count": 0, "eval_ms": None}, served=False)
        clock.advance(2.000)
        t.llm_response(tt.ollama_stats(_FULL), served=True)
        d = tt.parse_line(t.emit())
        self.assertEqual(d["llm_calls"], "2")
        self.assertEqual(d["llm_done"], "2100")
        self.assertEqual((d["prompt_eval_count"], d["prompt_eval_ms"],
                          d["eval_count"], d["eval_ms"]),
                         ("11873", "2702", "84", "790"))

    def test_only_unserved_responses_leave_stats_blank(self):
        self.t.begin("inject")
        self.t.llm_response(tt.ollama_stats(_FULL), served=False)
        d = tt.parse_line(self.t.emit())
        self.assertEqual((d["llm_calls"], d["llm_done"],
                          d["prompt_eval_count"]), ("1", "-", "-"))

    def test_offsets_are_integer_milliseconds(self):
        clock = _ManualClock()
        t = tt.TurnTiming(print_fn=self.lines.append, clock=clock)
        t.note_vad_break()
        t.begin_voice(None)
        clock.advance(0.0034)
        t.mark("stt_start")
        clock.advance(1.2346)
        t.mark("stt_end")
        clock.advance(0.5)
        d = tt.parse_line(t.emit())
        self.assertEqual((d["vad_break"], d["stt_start"], d["stt_end"],
                          d["end"]), ("0", "3", "1238", "1738"))

    def test_filler_counts_every_clip_and_times_the_first(self):
        clock = _ManualClock()
        t = tt.TurnTiming(print_fn=self.lines.append, clock=clock)
        t.begin("voice")
        clock.advance(0.900)
        t.note_filler()                  # stage 1 clip starts
        clock.advance(4.000)
        t.note_filler()                  # stage 2 clip starts
        d = tt.parse_line(t.emit())
        self.assertEqual((d["filler"], d["filler_ms"]), ("2", "900"))

    def test_defaults_are_the_real_clock_and_print(self):
        t = tt.TurnTiming()
        self.assertIs(t._clock, time.perf_counter)
        self.assertIs(t._print, print)

    def test_begin_replaces_an_unfinished_turn(self):
        self.t.begin("voice")
        self.t.mark("you")
        self.t.begin("inject")
        d = tt.parse_line(self.t.emit())
        self.assertEqual((d["kind"], d["you"]), ("inject", "-"))
        self.assertEqual(len(self.lines), 1)

    def test_discard(self):
        self.t.begin("inject")
        self.t.discard()
        self.assertIsNone(self.t.emit())
        self.assertEqual(self.lines, [])

    def test_never_raises(self):
        def boom(*a, **k):
            raise RuntimeError("boom")

        bad = tt.TurnTiming(print_fn=boom, clock=boom)
        bad.note_vad_break()
        bad.begin_voice(None)
        bad.begin("inject")
        bad.mark("you")
        bad.set_first("turn_ctx_chars", 1)
        bad.llm_response(None)
        bad.followup_round()
        bad.note_filler()
        self.assertIsNone(bad.emit())
        self.assertIsNone(bad.now())
        # A print that raises still reports the line and never propagates.
        loud = tt.TurnTiming(print_fn=boom, clock=_Clock())
        loud.begin("inject")
        self.assertIsNotNone(loud.emit())

    def test_parse_line_on_other_text(self):
        self.assertEqual(tt.parse_line("  [local-llm] served via x"), {})
        self.assertEqual(tt.parse_line(None), {})


# ════════════════════════════════════════════════════════════════════════════
#  Speed plan R1 (2026-10-01): the new fields and TurnTiming.note_stat
# ════════════════════════════════════════════════════════════════════════════
_R1_OWNER = ("tail_ms", "clip_ms", "stt_wait_ms", "stt_engine", "eot",
             "st_p", "st_n", "pre")


class R1SchemaTests(unittest.TestCase):
    def test_new_fields_sit_between_filler_ms_and_lead_dropped(self):
        f = list(tt.STAT_FIELDS)
        i = f.index("filler_ms")
        j = f.index("lead_dropped")
        self.assertEqual(f[i + 1:j], list(tt.NOTE_FIELDS))
        # The prompt budget's budget_trimmed (v2.0.167) follows lead_dropped.
        # ...and the brain-prefix fields (2026-10-04) after that.
        self.assertEqual(f[j:], ["lead_dropped", "budget_trimmed",
                                 "pe_new", "pe_state", "reprime"])
        self.assertEqual(tt.NOTE_FIELDS, (
            "tail_ms", "cap_lag_ms", "clip_ms", "stt_wait_ms", "stt_engine",
            "load_ms", "total_ms", "play_open_ms", "out_lat_ms",
            "filler_clip_ms", "eot", "st_p", "st_n", "pre", "cut",
            "amb_deferred", "cache", "clone", "clone_ms", "keeper"))
        self.assertEqual(len(set(tt.STAT_FIELDS)), len(tt.STAT_FIELDS))

    def test_the_old_fields_keep_their_order(self):
        old = ("prompt_eval_count", "prompt_eval_ms", "eval_count", "eval_ms",
               "llm_calls", "turn_ctx_chars", "sys_chars", "followup_rounds",
               "filler", "filler_ms", "lead_dropped")
        self.assertEqual([k for k in tt.STAT_FIELDS if k in old], list(old))


class NoteStatTests(unittest.TestCase):
    def setUp(self):
        self.lines = []
        self.clock = _ManualClock()
        self.t = tt.TurnTiming(print_fn=self.lines.append, clock=self.clock)

    def _emit(self):
        return tt.parse_line(self.t.emit())

    def test_round_trip_with_every_new_field(self):
        t = self.t
        t.begin("voice")
        t.mark("you")
        values = {"tail_ms": 1410, "clip_ms": 3904, "stt_wait_ms": 0,
                  "stt_engine": "whisper", "eot": "rms", "st_p": 0.8734,
                  "st_n": 2, "pre": 1}
        for k, v in values.items():
            t.note_stat(k, v)
        t.note_stat("play_open_ms", 41)
        t.note_stat("out_lat_ms", 46)
        t.note_stat("cache", "would-hit")
        t.note_stat("clone", 1)
        t.note_stat("clone_ms", 812)
        t.note_stat("keeper", 1)
        _in_thread(lambda: t.note_stat("filler_clip_ms", 2120))
        _in_thread(lambda: t.note_stat("cut", 980))
        _in_thread(lambda: t.note_stat("amb_deferred", 2))
        t.llm_response(tt.ollama_stats(_FULL))
        d = self._emit()
        self.assertEqual(
            {k: d[k] for k in tt.NOTE_FIELDS},
            {"tail_ms": "1410", "cap_lag_ms": "-", "clip_ms": "3904",
             "stt_wait_ms": "0", "stt_engine": "whisper", "load_ms": "13",
             "total_ms": "3512", "play_open_ms": "41", "out_lat_ms": "46",
             "filler_clip_ms": "2120", "eot": "rms",
             "st_p": "0.873", "st_n": "2", "pre": "1", "cut": "980",
             "amb_deferred": "2", "cache": "would-hit", "clone": "1",
             "clone_ms": "812", "keeper": "1"})
        self.assertEqual(d["lead_dropped"], "0")

    def test_an_absent_field_prints_dash(self):
        self.t.begin("inject")
        d = self._emit()
        for k in tt.NOTE_FIELDS:
            self.assertEqual(d[k], "-", k)

    def test_load_and_total_come_only_from_the_answering_response(self):
        t = self.t
        t.begin("inject")
        t.llm_response(dict(tt.ollama_stats(_FULL), load_ms=99, total_ms=1),
                       served=False)
        t.llm_response(tt.ollama_stats(_FULL), served=True)
        t.llm_response(dict(tt.ollama_stats(_FULL), load_ms=7, total_ms=8))
        d = self._emit()
        self.assertEqual((d["load_ms"], d["total_ms"]), ("13", "3512"))

    def test_bare_response_leaves_load_and_total_blank(self):
        self.t.begin("inject")
        self.t.llm_response(tt.ollama_stats(_BARE))
        d = self._emit()
        self.assertEqual((d["load_ms"], d["total_ms"]), ("-", "-"))

    def test_unknown_names_are_refused(self):
        t = self.t
        t.begin("inject")
        t.mark("you")
        for name in ("bogus", "load_ms", "total_ms", "lead_dropped",
                     "filler", "first_play", "llm_calls", "cap_lag_ms", "",
                     None):
            t.note_stat(name, 12345)
        line = t.emit()
        d = tt.parse_line(line)
        self.assertNotIn("bogus", d)
        self.assertNotIn("12345", line)
        self.assertEqual((d["load_ms"], d["lead_dropped"], d["llm_calls"],
                          d["cap_lag_ms"]), ("-", "0", "0", "-"))

    def test_other_threads_cannot_set_owner_fields(self):
        t = self.t
        t.begin("voice")
        t.mark("you")
        _in_thread(lambda: [t.note_stat(k, 7) for k in _R1_OWNER])
        _in_thread(lambda: t.note_stat("play_open_ms", 7))
        _in_thread(lambda: t.note_stat("out_lat_ms", 7))
        _in_thread(lambda: t.note_stat("cache", "hit"))
        _in_thread(lambda: t.note_stat("clone", 1))
        _in_thread(lambda: t.note_stat("clone_ms", 700))
        _in_thread(lambda: t.note_stat("keeper", 1))
        d = self._emit()
        for k in _R1_OWNER + ("play_open_ms", "out_lat_ms", "cache", "clone",
                              "clone_ms", "keeper"):
            self.assertEqual(d[k], "-", k)

    def test_any_thread_fields_are_accepted_from_any_thread(self):
        t = self.t
        t.begin("voice")
        _in_thread(lambda: t.note_stat("filler_clip_ms", 2300))
        _in_thread(lambda: t.note_stat("cut", 450))
        _in_thread(lambda: t.note_stat("amb_deferred", 1))
        d = self._emit()
        self.assertEqual((d["filler_clip_ms"], d["cut"], d["amb_deferred"]),
                         ("2300", "450", "1"))

    def test_play_open_follows_the_first_play_rule(self):
        t = self.t
        t.begin("voice")
        t.note_stat("play_open_ms", 5)        # before "you": a reminder
        t.mark("you")

        def helper():
            t.note_stat("play_open_ms", 33)

        th = threading.Thread(target=helper, name="stream-tts-flush")
        t.adopt(th)
        th.start()
        th.join(2)
        t.note_stat("play_open_ms", 99)       # later playbacks: first wins
        self.assertEqual(self._emit()["play_open_ms"], "33")

    def test_first_value_wins_and_amb_deferred_adds_up(self):
        t = self.t
        t.begin("voice")
        t.note_stat("stt_wait_ms", 12)
        t.note_stat("stt_wait_ms", 900)       # an in-turn capture's wait
        for _ in range(3):
            _in_thread(lambda: t.note_stat("amb_deferred", 1))
        t.note_stat("amb_deferred", "x")      # not a count: ignored
        d = self._emit()
        self.assertEqual((d["stt_wait_ms"], d["amb_deferred"]), ("12", "3"))

    def test_a_lazy_value_is_read_when_the_line_prints(self):
        t = self.t
        box = [None]
        calls = []

        def later():
            calls.append(1)
            return box[0]

        t.begin("voice")
        t.note_stat("tail_ms", later)
        self.assertEqual(calls, [], "read before the line printed")
        box[0] = 1288
        self.assertEqual(self._emit()["tail_ms"], "1288")
        self.assertEqual(calls, [1])

    def test_a_lazy_value_not_ready_or_failing_prints_dash(self):
        t = self.t

        def boom():
            raise RuntimeError("probe died")

        t.begin("voice")
        t.note_stat("tail_ms", lambda: None)
        t.note_stat("clip_ms", boom)
        d = self._emit()
        self.assertEqual((d["tail_ms"], d["clip_ms"]), ("-", "-"))

    def test_values_stay_one_token(self):
        t = self.t
        t.begin("voice")
        t.note_stat("stt_engine", "whisper large v3")
        t.note_stat("eot", "")
        t.note_stat("pre", True)
        line = t.emit()
        d = tt.parse_line(line)
        self.assertEqual(d["stt_engine"], "whisper_large_v3")
        self.assertEqual((d["eot"], d["pre"]), ("-", "1"))
        keys = [tok.split("=")[0] for tok in
                line.split("[turn-timing]")[1].split()]
        self.assertEqual(keys, ["kind", "outcome", *tt.MARKS, "end",
                                *tt.STAT_FIELDS])

    def test_without_a_turn_nothing_prints(self):
        self.t.note_stat("tail_ms", 1)
        self.t.note_stat("filler_clip_ms", 1)
        self.assertIsNone(self.t.emit())
        self.assertEqual(self.lines, [])

    def test_note_stat_never_raises(self):
        def boom(*a, **k):
            raise RuntimeError("boom")

        bad = tt.TurnTiming(print_fn=boom, clock=boom)
        bad.begin("voice")
        bad.note_stat("tail_ms", 1)
        bad.note_stat("amb_deferred", object())
        self.assertIsNone(bad.emit())


class R1ReviewFieldTests(unittest.TestCase):
    """R1 review (2026-10-01): cap_lag_ms travels with the VAD break that is
    the turn's t0, and out_lat_ms follows the play_open_ms rule."""

    def setUp(self):
        self.lines = []
        self.clock = _ManualClock()
        self.t = tt.TurnTiming(print_fn=self.lines.append, clock=self.clock)

    def _emit(self):
        return tt.parse_line(self.t.emit())

    def test_the_vad_break_carries_its_capture_lag(self):
        t = self.t
        since = t.now()
        self.clock.advance(2.0)
        t.note_vad_break(272)
        t.begin_voice(since)
        d = self._emit()
        self.assertEqual((d["vad_break"], d["cap_lag_ms"]), ("0", "272"))

    def test_a_stale_breaks_lag_is_not_adopted(self):
        t = self.t
        t.note_vad_break(500)            # an older capture's break
        self.clock.advance(1.0)
        since = t.now()
        t.begin_voice(since)             # this one hit MAX_RECORDING_SECS
        d = self._emit()
        self.assertEqual((d["vad_break"], d["cap_lag_ms"]), ("-", "-"))

    def test_a_break_without_a_lag_prints_dash(self):
        t = self.t
        since = t.now()
        t.note_vad_break()
        t.begin_voice(since)
        self.assertEqual(self._emit()["cap_lag_ms"], "-")
        for junk in ("abc", float("nan"), float("inf"), True, object()):
            t.note_vad_break(junk)
            t.begin_voice(since)
            self.assertEqual(self._emit()["cap_lag_ms"], "-", junk)

    def test_cap_lag_comes_only_from_the_break(self):
        t = self.t
        since = t.now()
        self.clock.advance(0.1)
        t.note_stat("cap_lag_ms", 9)     # pre-turn: never stashed
        t.note_vad_break()
        t.begin_voice(since)
        t.note_stat("cap_lag_ms", 9)     # in-turn: refused too
        self.assertEqual(self._emit()["cap_lag_ms"], "-")

    def test_reset_forgets_the_lag(self):
        t = self.t
        since = t.now()
        t.note_vad_break(300)
        t.reset()
        t.note_vad_break()
        t.begin_voice(since)
        self.assertEqual(self._emit()["cap_lag_ms"], "-")

    def test_out_lat_follows_the_play_open_rule(self):
        t = self.t
        t.begin("voice")
        t.note_stat("out_lat_ms", 5)      # before "you": a reminder
        t.mark("you")
        _in_thread(lambda: t.note_stat("out_lat_ms", 6))   # a stranger
        t.note_stat("out_lat_ms", 46)
        t.note_stat("out_lat_ms", 99)     # later playbacks: first wins
        self.assertEqual(self._emit()["out_lat_ms"], "46")


class PreTurnStashTests(unittest.TestCase):
    """A standby wake transcribes BEFORE its turn begins (and record_speech
    runs before every voice turn): begin_voice adopts what its own capture
    recorded, and nothing older."""

    def setUp(self):
        self.lines = []
        self.clock = _ManualClock()
        self.t = tt.TurnTiming(print_fn=self.lines.append, clock=self.clock)

    def _emit(self):
        return tt.parse_line(self.t.emit())

    def test_begin_voice_adopts_this_captures_values(self):
        t = self.t
        t.note_stat("stt_wait_ms", 999)     # an EARLIER standby capture
        self.clock.advance(1.0)
        since = t.now()
        self.clock.advance(0.5)
        t.note_vad_break()
        t.note_stat("clip_ms", 2048)
        t.note_stat("stt_wait_ms", 31)
        t.note_stat("tail_ms", lambda: 1344)
        t.begin_voice(since)
        t.mark("you")
        d = self._emit()
        self.assertEqual((d["vad_break"], d["clip_ms"], d["stt_wait_ms"],
                          d["tail_ms"]), ("0", "2048", "31", "1344"))

    def test_another_threads_owner_values_are_not_adopted(self):
        t = self.t
        since = t.now()
        self.clock.advance(0.1)
        # An ambient worker's transcribe while no turn is active.
        _in_thread(lambda: t.note_stat("stt_wait_ms", 1500))
        _in_thread(lambda: t.note_stat("amb_deferred", 1))
        t.begin_voice(since)
        d = self._emit()
        self.assertEqual(d["stt_wait_ms"], "-")
        self.assertEqual(d["amb_deferred"], "1")   # any-thread: adopted

    def test_in_turn_only_values_are_never_stashed(self):
        t = self.t
        since = t.now()
        self.clock.advance(0.1)
        t.note_stat("play_open_ms", 40)
        t.note_stat("cache", "hit")
        t.note_stat("clone", 1)
        t.note_stat("clone_ms", 650)
        t.note_stat("keeper", 1)
        _in_thread(lambda: t.note_stat("filler_clip_ms", 2100))
        _in_thread(lambda: t.note_stat("cut", 500))
        t.begin_voice(since)
        t.mark("you")
        d = self._emit()
        for k in ("play_open_ms", "cache", "clone", "clone_ms", "keeper",
                  "filler_clip_ms", "cut"):
            self.assertEqual(d[k], "-", k)

    def test_no_since_and_other_kinds_adopt_nothing(self):
        t = self.t
        t.note_stat("clip_ms", 100)
        t.begin_voice(None)
        self.assertEqual(self._emit()["clip_ms"], "-")
        t.note_stat("clip_ms", 200)
        t.begin("inject")
        self.assertEqual(self._emit()["clip_ms"], "-")

    def test_values_are_adopted_once(self):
        t = self.t
        since = t.now()
        self.clock.advance(0.1)
        t.note_stat("clip_ms", 300)
        t.begin_voice(since)
        self.assertEqual(self._emit()["clip_ms"], "300")
        t.begin_voice(since)
        self.assertEqual(self._emit()["clip_ms"], "-")

    def test_the_stash_is_bounded(self):
        t = self.t
        for i in range(500):
            t.note_stat("clip_ms", i)
            self.clock.advance(0.001)
        for _ in range(3 * tt._STASH_MAX_KEYS):
            _in_thread(lambda: t.note_stat("stt_wait_ms", 1))
        self.assertLessEqual(len(t._stash), tt._STASH_MAX_KEYS)
        for entries in t._stash.values():
            self.assertLessEqual(len(entries), tt._STASH_MAX_ENTRIES)

    def test_reset_clears_the_stash_and_the_vad_break(self):
        t = self.t
        since = t.now()
        t.note_vad_break()
        t.note_stat("clip_ms", 5)
        t.reset()
        t.begin_voice(since)
        d = self._emit()
        self.assertEqual((d["vad_break"], d["clip_ms"]), ("-", "-"))

    def test_discard_keeps_the_stash(self):
        # The main loop discards at every loop top, BEFORE record_speech;
        # a value recorded after that must survive to begin_voice.
        t = self.t
        since = t.now()
        self.clock.advance(0.1)
        t.note_stat("eot", "rms")
        t.discard()
        t.begin_voice(since)
        self.assertEqual(self._emit()["eot"], "rms")


if __name__ == "__main__":
    unittest.main()
