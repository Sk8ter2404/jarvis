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


if __name__ == "__main__":
    unittest.main()
