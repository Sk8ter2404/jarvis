"""Monolith wiring for the per-turn [turn-timing] line and the served-via
Ollama counters (speed plan rank 1, 2026-09-29).

The line logic itself is covered CI-light by tests/test_turn_timing.py. This
file drives the REAL monolith path for one turn (the inject drain or a fake
mic capture -> _capture_utterance -> _run_llm_dispatch -> _call_local_llm
with a canned Ollama body -> _speak with fake synth/playback) and checks:

  * exactly one [turn-timing] line per turn, offsets rising along the stages;
  * a turn that fails before the LLM still prints a partial line;
  * "[local-llm] served via" carries pe=<count>/<ms> ev=<count> caller=<tag>;
  * _prof_ollama (the JARVIS_PERF_PROBE TSV formatter) on canned bodies;
  * the TSV probe output is unchanged, and timing faults never break a turn.

No real audio, no real Ollama, no real Whisper.

Run locally (full-deps tier):
    python -m unittest tests.monolith.test_monolith_turn_timing
"""
from __future__ import annotations

import contextlib
import inspect
import io
import os
import re
import shutil
import tempfile
import threading
import time
import unittest
from unittest import mock

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith

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
         "message": {"role": "assistant", "content": "It is noon, sir."},
         "done": True}


# A background caller's body: different counters, so a background call
# stamping the turn's stats would be visible.
_BG = dict(_FULL, prompt_eval_count=7, prompt_eval_duration=1_000_000,
           eval_count=3)


class _Resp:
    def __init__(self, body, ok=True, status_code=200, bad_json=False):
        self._body = body
        self.ok = ok
        self.status_code = status_code
        self.text = "body"
        self._bad = bad_json
        self.json_calls = 0

    def json(self):
        self.json_calls += 1
        if self._bad:
            raise ValueError("not json")
        return self._body


class _Clock:
    """Fake perf_counter: each read advances 10 ms (strictly rising marks);
    the faked stages advance it by their own known durations (advance()), so
    the stage deltas on the line are exact."""

    def __init__(self):
        self.t = 1000.0
        self.lock = threading.Lock()

    def __call__(self):
        with self.lock:
            self.t += 0.010
            return self.t

    def peek(self):
        with self.lock:
            return self.t

    def advance(self, seconds):
        with self.lock:
            self.t += seconds


# Faked stage durations (seconds). Each read of the fake clock adds 10 ms,
# and a mark reads it once, so the bracketing marks of a stage sit exactly
# (duration + 10 ms) apart: the deltas asserted below.
_STT_S, _POST_S, _SYNTH_S, _PLAY_S = 1.500, 2.700, 0.300, 0.800


@requires_monolith
class _Base(MonolithGlobalsTestCase):
    def setUp(self):
        super().setUp()
        self._hist_len = len(self.bc.conversation_history)
        self.addCleanup(self._restore_hist)

    def _restore_hist(self):
        del self.bc.conversation_history[self._hist_len:]

    def _p(self, *args, **kwargs):
        patcher = mock.patch.object(*args, **kwargs)
        m = patcher.start()
        self.addCleanup(patcher.stop)
        return m

    def _local_llm(self, body=_FULL, model="gemma-test"):
        """Route _call_local_llm to a canned Ollama /api/chat body."""
        bc = self.bc
        fake_req = mock.Mock()
        fake_req.post.side_effect = lambda url, json=None, timeout=None, **k: \
            _Resp(body)
        self._p(bc, "LOCAL_LLM_FALLBACK", True)
        self._p(bc, "_ollama_alive", return_value=True)
        self._p(bc, "_ollama_has_model", return_value=True)
        self._p(bc, "_get_local_llm_model", return_value=model)
        self._p(bc, "_next_local_llm_fallback", return_value=None)
        self._p(bc, "requests", fake_req)
        return fake_req


# ════════════════════════════════════════════════════════════════════════════
#  _prof_ollama (JARVIS_PERF_PROBE TSV formatter) — unchanged behaviour
# ════════════════════════════════════════════════════════════════════════════
class ProfOllamaTests(_Base):
    def test_full_body(self):
        self.assertEqual(
            self.bc._prof_ollama(_Resp(_FULL)),
            "total=3512.3 load=12.6 peval=2702.4 ptok=11873 eval=790.1 etok=84")

    def test_body_without_timing_fields(self):
        self.assertEqual(
            self.bc._prof_ollama(_Resp(_BARE)),
            "total=0.0 load=0.0 peval=0.0 ptok=0 eval=0.0 etok=0")

    def test_unparseable_never_raises(self):
        self.assertEqual(self.bc._prof_ollama(_Resp(None, bad_json=True)),
                         "unparsed")
        self.assertEqual(self.bc._prof_ollama(object()), "unparsed")


# ════════════════════════════════════════════════════════════════════════════
#  "[local-llm] served via" line format
# ════════════════════════════════════════════════════════════════════════════
class ServedViaLineTests(_Base):
    _RE = re.compile(r"^  \[local-llm\] served via (\S+) "
                     r"pe=(\S+)/(\S+) ev=(\S+) caller=(\S+)$", re.M)

    def _call(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            out = self.bc._call_local_llm(
                "sys", [{"role": "user", "content": "hi"}])
        return out, buf.getvalue()

    def test_stats_and_caller(self):
        self._local_llm(_FULL)
        out, printed = self._call()
        self.assertEqual(out, "It is noon, sir.")
        m = self._RE.search(printed)
        self.assertIsNotNone(m, printed)
        self.assertEqual(m.group(1, 2, 3, 4),
                         ("gemma-test", "11873", "2702", "84"))
        self.assertEqual(m.group(5), f"_call@{threading.current_thread().name}")

    def test_missing_stats_print_question_marks(self):
        self._local_llm(_BARE)
        _, printed = self._call()
        m = self._RE.search(printed)
        self.assertIsNotNone(m, printed)
        self.assertEqual(m.group(2, 3, 4), ("?", "?", "?"))

    def test_wrapper_caller_names_its_caller(self):
        self._local_llm(_FULL)
        buf = io.StringIO()

        def my_background_job():
            return self.bc._local_then_cloud_or_honest(
                "sys", [{"role": "user", "content": "hi"}])

        with contextlib.redirect_stdout(buf):
            my_background_job()
        m = self._RE.search(buf.getvalue())
        self.assertIsNotNone(m, buf.getvalue())
        self.assertTrue(m.group(5).startswith(
            "_local_then_cloud_or_honest<my_background_job@"), m.group(5))

    def test_failover_line_carries_the_alt_stats(self):
        bc = self.bc
        fake_req = self._local_llm(_FULL, model="broken:model")
        self._p(bc, "_next_local_llm_fallback", return_value="good:model")
        # The failover needs a second empty reply in a row (2026-10-04).
        self._p(bc, "_local_empty_streak", [1])
        saved = list(bc._RESOLVED_LOCAL_LLM_MODEL)
        self.addCleanup(lambda: bc._RESOLVED_LOCAL_LLM_MODEL.__setitem__(
            slice(None), saved))

        def post(url, json=None, timeout=None, **k):
            if json["model"] == "broken:model":
                return _Resp({"message": {"content": ""},
                              "prompt_eval_count": 1, "eval_count": 0})
            return _Resp(_FULL)

        fake_req.post.side_effect = post
        _, printed = self._call()
        self.assertIn("  [local-llm] served via good:model (failed over from "
                      "broken:model) pe=11873/2702 ev=84 caller=_call@",
                      printed)


# ════════════════════════════════════════════════════════════════════════════
#  One turn -> one [turn-timing] line
# ════════════════════════════════════════════════════════════════════════════
class TurnLineTests(_Base):
    # True: keep the monolith's REAL global _turn_timing (real print, real
    # perf_counter) instead of a recorder with a fake clock.
    REAL_TIMING = False

    def setUp(self):
        super().setUp()
        bc = self.bc
        import numpy as np
        from core import turn_timing as tt
        self.tt = tt
        self.lines = []
        self.out = io.StringIO()
        self.clock = _Clock()
        if not self.REAL_TIMING:
            self.timing = tt.TurnTiming(print_fn=self.lines.append,
                                        clock=self.clock)
            self._p(bc, "_turn_timing", self.timing)
        # Capture side: no queued speech, no mic hardware.
        self._p(bc, "_speak_pending", return_value=False)
        self._p(bc, "set_state")
        self._p(bc, "_write_hud_state")
        self._p(bc, "_heartbeat")
        # Dispatch side: no glance, a plain local-LLM answer, no actions.
        self._p(bc, "maybe_glance_response", return_value=None)
        self._p(bc, "get_response_with_animation",
                side_effect=lambda text: bc._call_local_llm(
                    "sys", [{"role": "user", "content": text}]))
        self._p(bc, "parse_and_run_actions", side_effect=lambda r: (r, []))
        self._p(bc, "_apply_quip_layer", side_effect=lambda s, r: s)
        # Speech side: real _speak, fake synth + playback.
        self._p(bc, "_processing_filler", mock.Mock())

        def synth(text):
            self.clock.advance(_SYNTH_S)
            return np.zeros(10, dtype=np.float32), 24000

        self._p(bc, "synthesise", side_effect=synth)
        self.played = []
        self.play_starts = []

        def play(audio, sr):
            self.play_starts.append(self.clock.peek())
            self.played.append(sr)
            self.clock.advance(_PLAY_S)

        self._p(bc, "play_with_lipsync", side_effect=play)
        self._p(bc, "_session_start_time", bc.time.time() - 3600)
        self._p(bc, "_tts_layer", None)
        self._p(bc, "_is_staging", lambda: False)
        bc._tts_muted[0] = False
        fake_req = self._local_llm(_FULL)
        # Every POST records (thread name, clock) and takes _POST_S; the
        # turn's own thread gets _FULL, any other thread _BG.
        self.posts = []
        self.owner = threading.current_thread()

        def post(url, json=None, timeout=None, **k):
            self.posts.append((threading.current_thread().name,
                               self.clock.peek()))
            self.clock.advance(_POST_S)
            return _Resp(_FULL if threading.current_thread() is self.owner
                         else _BG)

        fake_req.post.side_effect = post

    def _quiet(self, fn, *a, **k):
        with contextlib.redirect_stdout(self.out):
            return fn(*a, **k)

    def _line(self):
        self.assertEqual(len(self.lines), 1, self.lines)
        return self.tt.parse_line(self.lines[0])

    def _i(self, d, key):
        self.assertNotEqual(d[key], "-", (key, d))
        return int(d[key])

    def _inject_turn(self, text="what time is it"):
        """The main loop's order for an injected turn."""
        bc = self.bc
        bc._tt_loop_top(text)
        cap = self._quiet(bc._capture_utterance, text, {})
        self.assertEqual(cap[0], text)
        bc._tt("mark", "you")
        return self._quiet(bc._run_llm_dispatch, text,
                           voice=False)

    def _offsets(self, d):
        return [(m, int(d[m])) for m in self.tt.MARKS if d[m] != "-"]

    def test_injected_turn_emits_exactly_one_rising_line(self):
        reply = self._inject_turn()
        self.assertEqual(reply, "It is noon, sir.")
        self.assertEqual(self.played, [24000])
        self.assertEqual(len(self.lines), 1, self.lines)
        d = self.tt.parse_line(self.lines[0])
        self.assertEqual((d["kind"], d["outcome"]), ("inject", "ok"))
        offs = self._offsets(d)
        self.assertEqual([m for m, _ in offs],
                         ["you", "llm_post", "llm_done", "actions_done",
                          "synth_start", "first_play"])
        vals = [v for _, v in offs]
        self.assertTrue(all(a < b for a, b in zip(vals, vals[1:])), offs)
        self.assertGreater(int(d["end"]), vals[-1])
        # Each mark brackets what it claims to, in milliseconds: the POST took
        # 2.7 s, synthesis 0.3 s, playback 0.8 s (+10 ms per clock read).
        self.assertEqual(self._i(d, "llm_done") - self._i(d, "llm_post"), 2710)
        self.assertEqual(self._i(d, "first_play") - self._i(d, "synth_start"),
                         310)
        self.assertGreaterEqual(self._i(d, "end") - self._i(d, "first_play"),
                                810)
        self.assertEqual((d["prompt_eval_count"], d["prompt_eval_ms"],
                          d["eval_count"], d["llm_calls"]),
                         ("11873", "2702", "84", "1"))
        self.assertEqual(d["followup_rounds"], "0")
        self.assertEqual(d["filler"], "0")
        # The main loop's error net after a finished turn prints nothing more.
        self.bc._tt("emit", "error")
        self.assertEqual(len(self.lines), 1)

    def test_voice_turn_starts_at_the_vad_break(self):
        bc = self.bc
        import numpy as np
        self._p(bc, "_mic_muted", [False])
        self._p(bc, "_get_realtime_session", return_value=None)
        self._p(bc, "resume_face_tracking")
        self._p(bc, "_audio_music_feed")
        self._p(bc, "apply_capture_auto_gain", side_effect=lambda a, p: (a, 1.0))
        def transcribe(audio):
            self.clock.advance(_STT_S)
            return ("what time is it",
                    {"no_speech_prob": 0.0, "avg_logprob": -0.1})

        self._p(bc, "_transcribe_capture", side_effect=transcribe)

        def fake_record(timeout=None, **_kw):
            bc._tt("note_vad_break")   # what record_speech does at the break
            return np.zeros(bc.SAMPLE_RATE, dtype=np.float32)

        self._p(bc, "record_speech", side_effect=fake_record)
        bc._tt_loop_top(None)
        text, _ = self._quiet(bc._capture_utterance, None, {})
        bc._tt("mark", "you")
        self._quiet(bc._run_llm_dispatch, text, voice=True)
        self.assertEqual(len(self.lines), 1, self.lines)
        d = self.tt.parse_line(self.lines[0])
        self.assertEqual((d["kind"], d["vad_break"]), ("voice", "0"))
        offs = self._offsets(d)
        self.assertEqual([m for m, _ in offs], list(self.tt.MARKS))
        vals = [v for _, v in offs]
        self.assertTrue(all(a < b for a, b in zip(vals, vals[1:])), offs)
        self.assertEqual(self._i(d, "stt_end") - self._i(d, "stt_start"), 1510)
        self.assertEqual(self._i(d, "llm_done") - self._i(d, "llm_post"), 2710)
        self.assertEqual(self._i(d, "first_play") - self._i(d, "synth_start"),
                         310)

    def test_turn_failing_before_the_llm_prints_a_partial_line(self):
        bc = self.bc
        self._p(bc, "maybe_glance_response",
                side_effect=RuntimeError("glance blew up"))
        with self.assertRaises(RuntimeError):
            self._inject_turn()
        self.assertEqual(len(self.lines), 1, self.lines)
        d = self.tt.parse_line(self.lines[0])
        self.assertEqual((d["kind"], d["outcome"]), ("inject", "error"))
        self.assertNotEqual(d["you"], "-")
        for m in ("llm_post", "llm_done", "actions_done", "first_play"):
            self.assertEqual(d[m], "-", m)
        self.assertEqual(d["prompt_eval_count"], "-")
        self.assertEqual(bc.requests.post.call_count, 0)

    def test_background_dispatch_during_the_turn_is_not_counted(self):
        # A whole dispatch on another thread (a tray command / proactive
        # turn) runs while the turn waits on its LLM: its POST, response,
        # actions and speech must not stamp the turn's line.
        bc = self.bc
        orig = bc.get_response_with_animation.side_effect

        def with_background(text):
            if threading.current_thread() is self.owner:
                th = threading.Thread(target=lambda: self._quiet(
                    bc._run_llm_dispatch, "background", voice=False),
                    name="bg-dispatch")
                th.start()
                th.join(5)
            return orig(text)

        bc.get_response_with_animation.side_effect = with_background
        self._inject_turn()
        d = self._line()
        self.assertEqual([n for n, _ in self.posts],
                         ["bg-dispatch", self.owner.name])
        self.assertEqual(len(self.played), 2)   # both spoke
        self.assertEqual(d["llm_calls"], "1")
        self.assertEqual((d["prompt_eval_count"], d["eval_count"]),
                         ("11873", "84"))
        # llm_post is the turn's own POST (2.7 s before its llm_done), not
        # the background one that finished before it began.
        self.assertEqual(self._i(d, "llm_done") - self._i(d, "llm_post"), 2710)
        self.assertGreater(self._i(d, "actions_done"), self._i(d, "llm_done"))
        self.assertGreater(self._i(d, "synth_start"),
                           self._i(d, "actions_done"))
        self.assertEqual(self._i(d, "first_play") - self._i(d, "synth_start"),
                         310)

    def test_failover_turn_line_carries_the_served_models_stats(self):
        bc = self.bc
        self._p(bc, "_get_local_llm_model", return_value="broken:model")
        self._p(bc, "_next_local_llm_fallback", return_value="good:model")
        # The failover needs a second empty reply in a row (2026-10-04).
        self._p(bc, "_local_empty_streak", [1])
        saved = list(bc._RESOLVED_LOCAL_LLM_MODEL)
        self.addCleanup(lambda: bc._RESOLVED_LOCAL_LLM_MODEL.__setitem__(
            slice(None), saved))

        def post(url, json=None, timeout=None, **k):
            broken = json["model"] == "broken:model"
            self.clock.advance(1.000 if broken else _POST_S)
            if broken:
                return _Resp({"message": {"content": ""},
                              "prompt_eval_count": 1,
                              "prompt_eval_duration": 5_000_000,
                              "eval_count": 0})
            return _Resp(_FULL)

        bc.requests.post.side_effect = post
        self.assertEqual(self._inject_turn(), "It is noon, sir.")
        d = self._line()
        self.assertEqual(d["llm_calls"], "2")
        self.assertEqual((d["prompt_eval_count"], d["prompt_eval_ms"],
                          d["eval_count"], d["eval_ms"]),
                         ("11873", "2702", "84", "790"))
        # llm_done is the answer's arrival: after the 1 s empty reply, then
        # the 2.7 s failover call (+10 ms for each of the three clock reads in
        # between: the empty reply's count, the repeat llm_post, the answer).
        self.assertEqual(self._i(d, "llm_done") - self._i(d, "llm_post"), 3730)

    def test_http_error_then_cloud_answer_leaves_llm_done_blank(self):
        bc = self.bc

        def post(url, json=None, timeout=None, **k):
            return _Resp({"error": "boom", "prompt_eval_count": 3},
                         ok=False, status_code=500)

        bc.requests.post.side_effect = post
        bc.get_response_with_animation.side_effect = lambda text: (
            bc._call_local_llm("sys", [{"role": "user", "content": text}])
            or "Cloud answer, sir.")
        self.assertEqual(self._inject_turn(), "Cloud answer, sir.")
        d = self._line()
        self.assertEqual((d["llm_calls"], d["llm_done"],
                          d["prompt_eval_count"]), ("1", "-", "-"))

    def test_unadopted_speech_during_the_turn_is_not_first_play(self):
        # A reminder spoken by another thread while the turn thinks.
        bc = self.bc
        orig = bc.get_response_with_animation.side_effect

        def with_reminder(text):
            th = threading.Thread(target=lambda: self._quiet(
                bc._speak, "Reminder: stretch, sir."), name="reminder")
            th.start()
            th.join(5)
            return orig(text)

        bc.get_response_with_animation.side_effect = with_reminder
        self._inject_turn()
        d = self._line()
        self.assertEqual(len(self.played), 2)
        # first_play is the ANSWER's audio, after the LLM, not the reminder.
        self.assertGreater(self._i(d, "first_play"), self._i(d, "llm_done"))

    def test_streamed_flush_sentence_counts_as_first_play(self):
        # The streaming-TTS flush threads speak part of the answer, so the
        # turn adopts them and their audio is the turn's first_play.
        bc = self.bc
        self.timing.begin("inject")
        self.timing.mark("you")
        buf = bc._SentenceFlushBuffer()
        # A content sentence: a lone pure acknowledgement ("Right away, sir,
        # checking now.") is held back until the next sentence since
        # 2026-09-30 (tests/monolith/test_monolith_ack_before_failure.py).
        self._quiet(buf.feed, "The report is on screen now, sir. ")
        buf.join(5)
        self.assertEqual(self.played, [24000])
        d = self.tt.parse_line(self.timing.emit())
        self.assertEqual(self._i(d, "first_play") - self._i(d, "synth_start"),
                         310)

    def test_followup_rounds_are_counted_in_a_real_dispatch(self):
        bc = self.bc
        calls = []

        def actions(reply):
            calls.append(reply)
            if len(calls) == 1:
                return reply, [("see_screen", "The editor shows main.py.",
                                True)]
            return reply, []

        bc.parse_and_run_actions.side_effect = actions
        fu = self._p(bc, "get_followup_response", return_value=None)
        self._inject_turn()
        fu.assert_called_once()
        self.assertEqual(self._line()["followup_rounds"], "1")

    def test_filler_ms_is_the_start_of_the_first_clip(self):
        bc = self.bc
        import numpy as np
        clips = mock.Mock()
        clips.available.return_value = ["Processing, sir."]
        clips.get.return_value = (np.zeros(10, dtype=np.float32), 24000)
        self._p(bc, "_filler_clips", clips)
        self._p(bc, "_filler_suppressed", return_value=None)
        bc._processing_filler.claim.return_value = "ok"
        self.timing.begin("voice")
        t0 = self.clock.peek()
        self.clock.advance(1.000)
        self.assertEqual(self._quiet(bc._filler_play, object(), 1), "played")
        d = self.tt.parse_line(self.timing.emit())
        self.assertEqual(d["filler"], "1")
        start_ms = int(round((self.play_starts[0] - t0) * 1000))
        self.assertEqual(int(d["filler_ms"]), start_ms)
        self.assertEqual(start_ms, 1010)

    def test_call_llm_local_route_records_prompt_sizes(self):
        # The production route: _call_llm -> _local_then_cloud_or_honest ->
        # _call_local_llm. turn_ctx_chars / sys_chars must be the sizes the
        # JARVIS_PERF_PROBE prompt_end row reports for the same turn.
        bc = self.bc
        import core.config as cfg
        self._p(cfg, "model_route", return_value="local")
        self._p(bc, "_DYNAMIC_LOCAL_PROMPT", True)
        self._p(bc, "_STABLE_LOCAL_PREFIX", True)
        if not (bc.PC_CONTROL_PROMPT
                and bc.PC_CONTROL_PROMPT in bc._system_prompt):
            self._p(bc, "_system_prompt",
                    "BASE IDENTITY\n" + (bc.PC_CONTROL_PROMPT or ""))
        # The section bodies this turn implicates: a known, non-empty block
        # (a plain "what time" may implicate none), so turn_ctx_chars > 0
        # proves the stable-split local route really ran.
        import core.prompt_router as pr
        self._p(pr, "turn_pc_block", return_value="B" * 1234)
        prof = self._p(bc, "_prof")
        bc.get_response_with_animation.side_effect = bc._call_llm
        self.assertEqual(self._inject_turn(), "It is noon, sir.")
        ends = [c.args for c in prof.call_args_list
                if c.args and c.args[0] == "prompt_end"]
        self.assertEqual(len(ends), 1, ends)
        m = re.fullmatch(r"sys=(\d+) ctx=(\d+)", ends[0][1])
        d = self._line()
        self.assertEqual(d["sys_chars"], m.group(1))
        self.assertEqual(d["turn_ctx_chars"], m.group(2))
        self.assertGreaterEqual(int(d["turn_ctx_chars"]), 1234)
        self.assertGreater(int(d["sys_chars"]), 0)
        self.assertIn("caller=_local_then_cloud_or_honest<_call_llm@"
                      f"{self.owner.name}", self.out.getvalue())

    def test_filtered_turn_is_dropped_at_the_next_loop_top(self):
        bc = self.bc
        bc._tt_loop_top("some overheard words")   # gated: never dispatched
        bc._tt("mark", "you")
        bc._tt_loop_top(None)                      # next iteration
        self._quiet(bc._run_llm_dispatch, "hello", voice=False)
        self.assertEqual(self.lines, [])

    def test_no_turn_means_no_line(self):
        self._quiet(self.bc._run_llm_dispatch, "hello", voice=False)
        self.assertEqual(self.lines, [])

    def test_timing_faults_never_break_the_turn(self):
        bc = self.bc

        class Broken:
            def __getattr__(self, name):
                def boom(*a, **k):
                    raise RuntimeError("timing exploded")
                return boom

        def boom(*a, **k):
            raise RuntimeError("timing module fault")

        self._p(bc, "_turn_timing", Broken())
        self._p(bc._tt_mod, "response_stats", side_effect=boom)
        self._p(bc._tt_mod, "caller_tag", side_effect=boom)
        self._p(bc._tt_mod, "served_via_suffix", side_effect=boom)
        self.assertEqual(self._inject_turn(), "It is noon, sir.")
        self.assertEqual(self.played, [24000])
        self.assertNotIn("call failed", self.out.getvalue())
        self.assertIn("  [local-llm] served via gemma-test pe=?/? ev=?",
                      self.out.getvalue())


class RealPrinterTurnLineTests(TurnLineTests):
    """The same turn through the monolith's REAL global _turn_timing: its
    line must reach stdout, timed by perf_counter."""
    REAL_TIMING = True

    def test_real_timing_prints_one_line_to_stdout(self):
        bc = self.bc
        self.assertIs(bc._turn_timing._clock, time.perf_counter)
        self.assertIs(bc._turn_timing._print, print)
        self.assertEqual(self._inject_turn(), "It is noon, sir.")
        lines = [ln for ln in self.out.getvalue().splitlines()
                 if "[turn-timing]" in ln]
        self.assertEqual(len(lines), 1, self.out.getvalue())
        d = self.tt.parse_line(lines[0])
        self.assertEqual((d["kind"], d["outcome"]), ("inject", "ok"))
        for m in ("you", "llm_post", "llm_done", "actions_done",
                  "synth_start", "first_play"):
            self.assertNotEqual(d[m], "-", m)


# Only the real-printer test runs in the subclass; the fake-clock tests it
# inherits need the recorder.
for _name in [n for n in vars(TurnLineTests) if n.startswith("test_")]:
    setattr(RealPrinterTurnLineTests, _name, None)
del _name


# ════════════════════════════════════════════════════════════════════════════
#  JARVIS_PERF_PROBE TSV stays as it was
# ════════════════════════════════════════════════════════════════════════════
class ProbeTsvUnchangedTests(_Base):
    def test_llm_post_and_resp_rows(self):
        bc = self.bc
        d = tempfile.mkdtemp(prefix="jarvis_tt_")
        self.addCleanup(shutil.rmtree, d, True)
        path = os.path.join(d, "probe.tsv")
        self._p(bc, "_PROF_ON", True)
        self._p(bc, "_PROF_PATH", path)
        self._local_llm(_FULL)
        with contextlib.redirect_stdout(io.StringIO()):
            bc._call_local_llm("sys", [{"role": "user", "content": "hi"}])
        with open(path, encoding="utf-8") as f:
            rows = [ln.rstrip("\n").split("\t") for ln in f]
        tags = [r[1] for r in rows]
        self.assertEqual(tags, ["llm_post", "llm_resp"])
        # _prof_ollama's parse is the one extra: stats + reply text + probe.
        self.assertEqual(self.resps[0].json_calls, 3)
        self.assertTrue(rows[0][3].startswith("sys="))
        self.assertEqual(rows[1][3], bc._prof_ollama(_Resp(_FULL)))

    def test_probe_off_writes_nothing(self):
        bc = self.bc
        d = tempfile.mkdtemp(prefix="jarvis_tt_")
        self.addCleanup(shutil.rmtree, d, True)
        path = os.path.join(d, "probe.tsv")
        self._p(bc, "_PROF_ON", False)
        self._p(bc, "_PROF_PATH", path)
        self._local_llm(_FULL)
        with contextlib.redirect_stdout(io.StringIO()):
            bc._call_local_llm("sys", [{"role": "user", "content": "hi"}])
        self.assertFalse(os.path.exists(path))
        # Probe off: _prof_ollama is not even evaluated (the `if _PROF_ON:`
        # guard); only the stats and the reply text parse the body.
        self.assertEqual(self.resps[0].json_calls, 2)

    def _local_llm(self, body=_FULL, model="gemma-test"):
        fake_req = super()._local_llm(body, model)
        self.resps = []

        def post(url, json=None, timeout=None, **k):
            r = _Resp(body)
            self.resps.append(r)
            return r

        fake_req.post.side_effect = post
        return fake_req


# ════════════════════════════════════════════════════════════════════════════
#  Main-loop / _call_llm wiring (source-level: main() cannot run in a test)
# ════════════════════════════════════════════════════════════════════════════
class WiringTests(_Base):
    def test_main_loop_scopes_marks_and_emits(self):
        src = inspect.getsource(self.bc.main)
        drain = src.index("_injected_text = _drain_injected_command()")
        top = src.index("_tt_loop_top(_injected_text)")
        self.assertLess(drain, top)
        self.assertLess(top, src.index("if _sleep_mode[0]:", drain))
        you = src.index('print(f"  You:    {text}")')
        mark_you = src.index('_tt("mark", "you")')
        self.assertLess(you, mark_you)
        # "you" must be stamped before either answer path runs: synth_start /
        # first_play only count after it, and the line prints in the
        # dispatch's finally.
        self.assertLess(mark_you, src.index("if _run_voice_shortcuts(text):"))
        self.assertLess(mark_you, src.index("reply = _run_llm_dispatch(text"))
        self.assertIn('if _run_voice_shortcuts(text):\n'
                      '                    _tt("emit", "shortcut")\n'
                      '                    continue', src)
        net = src.index("except Exception as _loop_exc:")
        self.assertLess(net, src.index('_tt("emit", "error")', net))
        self.assertLess(src.index('_tt("emit", "error")', net),
                        src.index("_recover_from_main_loop_error(_loop_exc)",
                                  net))

    def test_call_llm_records_the_turn_context_size(self):
        src = inspect.getsource(self.bc._call_llm)
        end = src.index('_prof("prompt_end"')
        self.assertLess(end, src.index(
            '_tt("set_first", "turn_ctx_chars", len(_turn_ctx))'))

    def test_record_speech_notes_the_vad_break(self):
        src = inspect.getsource(self.bc.record_speech)
        brk = src.index('_prof("vad_break")')
        note = ('_tt("note_vad_break",\n'
                '                            _capture_lag_ms(audio_q, '
                '_record_stream, CHUNK))')
        self.assertLess(brk, src.index(note))
        self.assertLess(src.index(note), src.index(" break\n", brk))
        self.assertEqual(src.count("note_vad_break"), 1)

    def test_filler_play_is_noted(self):
        src = inspect.getsource(self.bc._filler_play)
        self.assertLess(src.index('_prof("filler_play"'),
                        src.index('_tt("note_filler")'))

    def test_followup_rounds_are_counted(self):
        src = inspect.getsource(self.bc._run_llm_dispatch_body)
        self.assertLess(src.index('_tt("followup_round")'),
                        src.index("followup = get_followup_response("))


# ════════════════════════════════════════════════════════════════════════════
#  Speed plan R1 (2026-10-01): the new fields through the REAL monolith paths
# ════════════════════════════════════════════════════════════════════════════
_R1_RESERVED = ("stt_engine", "eot", "st_p", "st_n", "pre", "cut",
                "amb_deferred", "cache")


class _FakeVad:
    """Stands in for the Silero tail detector (bc._tail_vad)."""

    def __init__(self, tail=1410, gate=None, failed=""):
        self.tail = tail
        self.gate = gate
        self.failed = failed
        self.clips = []

    def speech_tail_ms(self, clip, sr):
        self.clips.append((clip, sr))
        if self.gate is not None:
            self.gate.wait(5)
        return self.tail


class _SlowLock:
    """_stt_lock stand-in: acquiring it 'takes' `wait_s` on the fake clock
    (an ambient decode held Whisper), then behaves like an RLock."""

    def __init__(self, clock, wait_s):
        import threading as _th
        self._clock = clock
        self._wait = wait_s
        self._lock = _th.RLock()

    def __enter__(self):
        self._lock.acquire()
        self._clock.advance(self._wait)
        return self

    def __exit__(self, *exc):
        self._lock.release()
        return False


def _join_probe(bc):
    th = bc._tail_probe_state.get("thread")
    if th is not None:
        th.join(5)


class R1TurnLineTests(TurnLineTests):
    """The TurnLineTests rig (fake clock, real capture -> dispatch -> _speak)
    with the R1 seams faked: the Silero detector, the Whisper body and the
    _stt_lock wait."""

    def setUp(self):
        super().setUp()
        bc = self.bc
        import numpy as np
        self.np = np
        self.vad = _FakeVad()
        self._p(bc, "_tail_vad", self.vad)
        # A probe left running by an earlier test must not make this
        # capture skip its own (one probe in flight at a time).
        self._p(bc, "_tail_probe_state",
                {"thread": None, "off_logged": False})
        self._p(bc, "_mic_muted", [False])
        self._p(bc, "_get_realtime_session", return_value=None)
        self._p(bc, "resume_face_tracking")
        self._p(bc, "_audio_music_feed")
        self._p(bc, "apply_capture_auto_gain",
                side_effect=lambda a, p: (a, 1.0))
        self.stt_threads = []

        def stt_impl(audio):
            self.stt_threads.append(threading.current_thread().name)
            self.clock.advance(_STT_S)
            return ("what time is it",
                    {"no_speech_prob": 0.0, "avg_logprob": -0.1})

        self._p(bc, "_transcribe_impl", side_effect=stt_impl)
        self._p(bc, "_stt_lock", _SlowLock(self.clock, 0.500))
        self.audio = np.full(2 * bc.SAMPLE_RATE, 0.01, dtype=np.float32)

        def fake_record(timeout=None, **_kw):
            bc._tt("note_vad_break")
            return self.audio

        self._p(bc, "record_speech", side_effect=fake_record)

    def _voice_turn(self, join_probe=True):
        bc = self.bc
        bc._tt_loop_top(None)
        text, conf = self._quiet(bc._capture_utterance, None, {})
        self.assertEqual(text, "what time is it")
        if join_probe:
            _join_probe(bc)
        bc._tt("mark", "you")
        self._quiet(bc._run_llm_dispatch, text, voice=True)
        return self._line()

    def test_voice_turn_line_has_clip_tail_and_stt_wait(self):
        d = self._voice_turn()
        self.assertEqual(d["clip_ms"], "2000")
        self.assertEqual(d["tail_ms"], "1410")
        # The lock "took" 500 ms, + the one 10 ms clock read after it.
        self.assertEqual(d["stt_wait_ms"], "510")
        self.assertEqual((d["load_ms"], d["total_ms"]), ("13", "3512"))
        for k in _R1_RESERVED:
            self.assertEqual(d[k], "-", k)
        # The detector got a COPY of the clip Whisper decoded, at 16 kHz.
        clip, sr = self.vad.clips[0]
        self.assertEqual(sr, 16000)
        self.assertFalse(self.np.shares_memory(clip, self.audio),
                         "the probe must work on a copy, never a view")
        self.np.testing.assert_array_equal(clip, self.audio)
        self.assertEqual(self.stt_threads, [self.owner.name])

    def test_line_order_and_one_line_per_turn(self):
        self._voice_turn()
        keys = [tok.split("=")[0] for tok in
                self.lines[0].split("[turn-timing]")[1].split()]
        self.assertEqual(keys, ["kind", "outcome", *self.tt.MARKS, "end",
                                *self.tt.STAT_FIELDS])
        self.bc._tt("emit", "error")        # the loop's error net
        self.assertEqual(len(self.lines), 1)

    def test_a_tail_not_ready_by_the_line_prints_dash(self):
        gate = threading.Event()
        self.vad.gate = gate
        self.addCleanup(gate.set)
        d = self._voice_turn(join_probe=False)
        self.assertEqual(d["tail_ms"], "-")
        self.assertEqual(d["clip_ms"], "2000")
        gate.set()
        _join_probe(self.bc)

    def test_the_transcript_is_unchanged_by_the_probe(self):
        bc = self.bc
        with_probe = self._quiet(bc._transcribe_capture, self.audio)
        _join_probe(bc)
        self._p(bc, "TURN_TAIL_PROBE", False)
        without = self._quiet(bc._transcribe_capture, self.audio)
        self.assertEqual(with_probe, without)

    def test_another_threads_stt_wait_is_not_on_the_line(self):
        bc = self.bc
        bc._tt_loop_top("typed")
        bc._tt("mark", "you")
        th = threading.Thread(target=lambda: bc.transcribe(self.audio),
                              name="ambient-listen")
        th.start()
        th.join(5)
        self.assertEqual(self.stt_threads, ["ambient-listen"])
        self.assertEqual(self.tt.parse_line(bc._tt("emit"))["stt_wait_ms"],
                         "-")
        bc._tt_loop_top("typed")
        bc._tt("mark", "you")
        bc.transcribe(self.audio)            # the owner's own decode
        self.assertEqual(self.tt.parse_line(bc._tt("emit"))["stt_wait_ms"],
                         "510")

    def test_injected_turn_has_llm_load_and_total_only(self):
        self._inject_turn()
        d = self._line()
        self.assertEqual((d["load_ms"], d["total_ms"]), ("13", "3512"))
        for k in ("tail_ms", "clip_ms", "stt_wait_ms", "play_open_ms",
                  "filler_clip_ms") + _R1_RESERVED:
            self.assertEqual(d[k], "-", k)

    def test_filler_clip_ms_is_the_first_clips_length(self):
        bc = self.bc
        np = self.np
        clips = mock.Mock()
        clips.available.return_value = ["Processing, sir."]
        clips.get.side_effect = [(np.zeros(50880, dtype=np.float32), 24000),
                                 (np.zeros(12000, dtype=np.float32), 24000)]
        self._p(bc, "_filler_clips", clips)
        self._p(bc, "_filler_suppressed", return_value=None)
        bc._processing_filler.claim.return_value = "ok"
        self.timing.begin("voice")
        # The filler plays on its own thread: any-thread field.
        for stage in (1, 2):
            th = threading.Thread(target=lambda s=stage: self._quiet(
                bc._filler_play, object(), s), name="processing-filler")
            th.start()
            th.join(5)
        d = self.tt.parse_line(self.timing.emit())
        self.assertEqual((d["filler"], d["filler_clip_ms"]), ("2", "2120"))

    def test_timing_faults_in_the_probe_never_break_the_turn(self):
        bc = self.bc

        class Exploding:
            def __getattr__(self, name):
                raise RuntimeError("vad exploded")

        self._p(bc, "_tail_vad", Exploding())
        self._p(bc._tt_mod.TurnTiming, "note_stat",
                side_effect=RuntimeError("note_stat exploded"))
        d = self._voice_turn()
        self.assertEqual(d["kind"], "voice")
        self.assertEqual(self.played, [24000])


for _name in [n for n in vars(TurnLineTests) if n.startswith("test_")]:
    setattr(R1TurnLineTests, _name, None)
del _name


@requires_monolith
class R1StandbyWakeTests(_Base):
    """A standby wake that carries a command transcribes BEFORE its turn
    begins (_handle_sleep_standby, then begin_voice): the turn must still
    get that capture's clip / tail / lock-wait fields."""

    def setUp(self):
        super().setUp()
        bc = self.bc
        import numpy as np
        from core import turn_timing as tt
        from core.followup_window import FollowupWindow
        self.tt = tt
        self.lines = []
        self.clock = _Clock()
        self._p(bc, "_turn_timing",
                tt.TurnTiming(print_fn=self.lines.append, clock=self.clock))
        for name in ("_mic_muted", "_sleep_mode", "_standby_mode"):
            cell = getattr(bc, name)
            saved = cell[0]
            self.addCleanup(cell.__setitem__, 0, saved)
        bc._mic_muted[0] = False
        bc._sleep_mode[0] = True
        bc._standby_mode[0] = True
        self._p(bc, "_speak")
        self._p(bc, "_speak_pending", return_value=False)
        self._p(bc, "set_state")
        self._p(bc, "_heartbeat")
        self._p(bc, "context_aware_greeting", return_value=("Yes, sir?", 1.0))
        self._p(bc, "OVERNIGHT_FLAG_FILE",
                os.path.join(tempfile.gettempdir(), "jarvis_r1_no_flag"))
        self._p(bc, "_learn_gate_note_wake")
        self._p(bc, "_standby_wake_detected", return_value=None)
        self._p(bc, "_audio_music_should_refuse_wake", return_value=False)
        self._p(bc, "_audio_music_feed")
        self._p(bc, "_device_speech_ignored", return_value=False)
        self._p(bc, "_dialogue_hold_ignored", return_value=False)
        self._p(bc, "_self_echo_ignored", return_value=False)
        self._p(bc, "_followup_window", FollowupWindow(0))
        self._p(bc, "_require_wake_runtime", True)
        self._p(bc, "_ambient_learning_feed")
        self._p(bc, "apply_capture_auto_gain",
                side_effect=lambda a, p: (a, 1.0))
        self.vad = _FakeVad(tail=1290)
        self._p(bc, "_tail_vad", self.vad)
        # A probe left running by an earlier test must not make this
        # capture skip its own (one probe in flight at a time).
        self._p(bc, "_tail_probe_state",
                {"thread": None, "off_logged": False})
        self._p(bc, "_stt_lock", _SlowLock(self.clock, 0.250))
        self._p(bc, "_transcribe_impl", return_value=(
            "Jarvis, what time is it?",
            {"no_speech_prob": 0.0, "avg_logprob": -0.2}))
        self.audio = np.zeros(int(1.5 * bc.SAMPLE_RATE), dtype=np.float32)

        def fake_record(timeout=None, **_kw):
            bc._tt("note_vad_break")
            return self.audio

        self._p(bc, "record_speech", side_effect=fake_record)

    def test_the_carried_command_turn_gets_its_capture_fields(self):
        bc = self.bc
        with contextlib.redirect_stdout(io.StringIO()):
            out = bc._handle_sleep_standby(None)
        self.assertEqual(out[0], "Jarvis, what time is it?")
        _join_probe(bc)
        bc._tt("mark", "you")
        d = self.tt.parse_line(bc._tt("emit"))
        self.assertEqual((d["kind"], d["vad_break"]), ("voice", "0"))
        self.assertEqual((d["clip_ms"], d["tail_ms"], d["stt_wait_ms"]),
                         ("1500", "1290", "260"))

    def test_a_standby_capture_that_wakes_nothing_leaves_no_turn(self):
        bc = self.bc
        bc._transcribe_impl.return_value = ("just the telly", {})
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertIsNone(bc._handle_sleep_standby(None))
        _join_probe(bc)
        self.assertIsNone(bc._tt("emit"))
        # ...and the next turn does not inherit that capture's values.
        bc._tt_loop_top("typed")
        bc._tt("mark", "you")
        d = self.tt.parse_line(bc._tt("emit"))
        self.assertEqual((d["clip_ms"], d["tail_ms"]), ("-", "-"))


@requires_monolith
class R1PlaybackOpenTests(_Base):
    """play_open_ms through the REAL play_with_lipsync body (fake sd, fake
    stream): from just before the duck to sd.play() returning."""

    def setUp(self):
        super().setUp()
        bc = self.bc
        from core import turn_timing as tt
        self.tt = tt
        self.lines = []
        self.clock = _Clock()
        self.timing = tt.TurnTiming(print_fn=self.lines.append,
                                    clock=self.clock)
        self._p(bc, "_turn_timing", self.timing)
        self.stream = mock.Mock()
        self.stream.active = False
        self.sd = mock.Mock()
        self.sd.get_stream.return_value = self.stream
        layer = mock.Mock()
        layer.is_muted.return_value = False
        ducker = mock.Mock()
        ducker.duck.side_effect = lambda: self.clock.advance(0.200)
        self.ducker = ducker
        for name, val in (("sd", self.sd), ("_tts_layer", layer),
                          ("_audio_ducker", ducker),
                          ("BARGE_IN_ENABLED", False),
                          ("ROBOT_ENABLED", False), ("send", mock.Mock())):
            self._p(bc, name, val)
        self._p(bc, "get_output_device", return_value=1)
        self._p(bc, "_write_hud_state")
        self._p(bc, "_feed_playback_reference")

    def _play(self):
        import numpy as np
        with contextlib.redirect_stdout(io.StringIO()):
            self.bc.play_with_lipsync(np.zeros(240, dtype=np.float32), 24000)

    def _turn(self, you=True):
        self.timing.begin("inject")
        if you:
            self.timing.mark("you")

    def _emit(self):
        return self.tt.parse_line(self.timing.emit())

    def test_the_answers_playback_open_is_timed(self):
        self._turn()
        self._play()
        # 200 ms of duck + the one 10 ms clock read after sd.play().
        self.assertEqual(self._emit()["play_open_ms"], "210")
        self.sd.play.assert_called_once()

    def test_the_robot_branch_is_timed_too(self):
        self._p(self.bc, "ROBOT_ENABLED", True)
        self._turn()
        self._play()
        self.assertEqual(self._emit()["play_open_ms"], "210")

    def test_only_the_first_playback_counts(self):
        self._turn()
        self._play()
        self.ducker.duck.side_effect = lambda: self.clock.advance(1.0)
        self._play()
        self.assertEqual(self._emit()["play_open_ms"], "210")

    def test_not_before_you_nor_from_another_thread(self):
        self.stream.latency = 0.0464
        self._turn(you=False)
        self._play()                          # a reminder, pre-transcript
        self.timing.mark("you")
        th = threading.Thread(target=self._play, name="processing-filler")
        th.start()
        th.join(5)
        d = self._emit()
        self.assertEqual((d["play_open_ms"], d["out_lat_ms"]), ("-", "-"))

    def test_flag_off_records_nothing(self):
        self._p(self.bc, "TURN_PLAY_OPEN_PROBE", False)
        self.stream.latency = 0.0464
        self._turn()
        self._play()
        d = self._emit()
        self.assertEqual((d["play_open_ms"], d["out_lat_ms"]), ("-", "-"))
        self.sd.play.assert_called_once()

    # R1 review (2026-10-01): everything between first_play and the open
    # stream is inside play_open_ms - the PortAudio claim, the output-device
    # refresh (a full PortAudio reinit at worst) and the barge-in listener,
    # not only the duck - and the stream's reported output latency is noted.
    def test_the_setup_before_the_duck_is_timed_too(self):
        bc = self.bc
        claim = bc._pa_claim_owner

        def slow_claim(*a, **k):
            self.clock.advance(0.050)
            return claim(*a, **k)

        def slow_device():
            self.clock.advance(0.300)        # a device refresh / reinit
            return 1

        self._p(bc, "_pa_claim_owner", side_effect=slow_claim)
        self._p(bc, "get_output_device", side_effect=slow_device)
        self._turn()
        self._play()
        # 50 claim + 300 refresh + 200 duck + the 10 ms read after sd.play().
        self.assertEqual(self._emit()["play_open_ms"], "560")

    def test_the_output_latency_is_noted_on_both_branches(self):
        for robot in (False, True):
            with self.subTest(robot=robot):
                self._p(self.bc, "ROBOT_ENABLED", robot)
                self.stream.latency = 0.0464
                self._turn()
                self._play()
                self._play()                  # later playbacks: first wins
                d = self._emit()
                self.assertEqual((d["play_open_ms"], d["out_lat_ms"]),
                                 ("210", "46"))

    def test_an_unknown_output_latency_prints_dash(self):
        for junk in (mock.Mock(), None, float("nan"), "0.05", -0.01,
                     (0.01,), True):
            with self.subTest(latency=junk):
                self.stream.latency = junk
                self._turn()
                self._play()
                d = self._emit()
                self.assertEqual(d["out_lat_ms"], "-")
                self.assertEqual(d["play_open_ms"], "210")

    def test_the_reaper_still_takes_exactly_three_args(self):
        seen = []

        def reaper(stream, done_evt, audio_secs):
            seen.append((stream, audio_secs))
            self.bc._pa_close_done()
            done_evt.set()

        self._p(self.bc, "_reap_playback", side_effect=reaper)
        self._turn()
        self._play()
        self.assertEqual(seen, [(self.stream, 0.01)])
        self.assertEqual(self._emit()["play_open_ms"], "210")

    def test_a_timing_fault_never_breaks_playback(self):
        self._p(self.timing, "now", side_effect=RuntimeError("clock"))
        self._turn()
        self._play()
        self.sd.play.assert_called_once()


@requires_monolith
class R1CaptureLagTests(_Base):
    """cap_lag_ms through the REAL record_speech (R1 review, 2026-10-01).
    tail_ms is audio time (last speech -> the clip's last sample); the VAD
    break, t0, is the wall-clock moment the loop handled that sample, later
    by the input stream's latency plus every chunk still queued behind it.
    The break hands that lag to the turn."""

    def setUp(self):
        super().setUp()
        bc = self.bc
        from core import turn_timing as tt
        self.tt = tt
        self.timing = tt.TurnTiming(print_fn=lambda line: None,
                                    clock=_Clock())
        self._p(bc, "_turn_timing", self.timing)
        self.lim = int(bc.SILENCE_SECS * bc.SAMPLE_RATE / 1024)

    def _record(self, n_voiced, n_silent, latency):
        """Queue every frame BEFORE the capture loop reads one (so the
        backlog at the break is exact), run record_speech, then start the
        turn the way the main loop does."""
        bc = self.bc
        np = bc.np

        class FakeStream:
            device = 1

            def __init__(self, *a, callback=None, **k):
                self.cb = callback
                self.latency = latency

            def start(self):
                for amp, n in ((0.2, n_voiced), (0.0, n_silent)):
                    frame = np.full((1024, 1), amp, dtype="float32")
                    for _ in range(n):
                        self.cb(frame, 1024, None, None)

        self._p(bc, "_mic_input_disabled", return_value=False)
        self._p(bc, "_mic_muted", [False])
        self._p(bc, "_capture_holds_mic", return_value=False)
        self._p(bc, "_input_backoff_wait", return_value=False)
        self._p(bc, "get_input_device", return_value=1)
        self._p(bc, "_safe_close_stream", lambda s: None)
        self._p(bc.sd, "InputStream", FakeStream)
        self._p(bc, "_note_live_capture", lambda *a, **k: None)
        self._p(bc, "_filler_capture_mark", lambda *a, **k: None)
        self._p(bc, "_fanout_record_frame", lambda *a, **k: None)
        self._p(bc, "_process_capture_chunk",
                lambda data, sr, skip_ns=False: data)
        self._p(bc, "_spec_stt_should_snapshot", return_value=False)
        self._p(bc, "pause_face_tracking")
        self._p(bc, "set_state")
        self._p(bc, "_write_hud_state")
        self._p(bc, "_heartbeat")
        self._p(bc, "VAD_THRESHOLD", 0.008)
        since = self.timing.now()
        with contextlib.redirect_stdout(io.StringIO()):
            audio = bc.record_speech(timeout=3)
        self.assertIsNotNone(audio, "no utterance was captured")
        # The clip ends at the break: the queued chunks are not in it.
        self.assertEqual(len(audio), (n_voiced + self.lim) * 1024)
        self.timing.begin_voice(since)
        return self.tt.parse_line(self.timing.emit())

    def test_queued_chunks_and_input_latency_make_the_lag(self):
        # 9 chunks (64 ms each) still queued + 100 ms of input latency.
        d = self._record(6, self.lim + 9, 0.100)
        self.assertEqual((d["vad_break"], d["cap_lag_ms"]), ("0", "676"))

    def test_no_backlog_is_the_input_latency_alone(self):
        d = self._record(6, self.lim, 0.100)
        self.assertEqual(d["cap_lag_ms"], "100")

    def test_an_unknown_input_latency_leaves_the_backlog(self):
        d = self._record(6, self.lim + 9, None)
        self.assertEqual(d["cap_lag_ms"], "576")

    def test_the_helper_never_raises(self):
        bc = self.bc
        import queue as _queue
        q = _queue.Queue()
        for _ in range(3):
            q.put(0)

        class S:
            latency = (0.05, 0.2)        # a duplex stream: (input, output)

        self.assertEqual(bc._capture_lag_ms(q, S(), 1024), 242)
        S.latency = float("nan")
        self.assertEqual(bc._capture_lag_ms(q, S(), 1024), 192)
        self.assertEqual(bc._capture_lag_ms(q, object(), 1024), 192)

        class BadQ:
            def qsize(self):
                raise RuntimeError("qsize")

        self.assertIsNone(bc._capture_lag_ms(BadQ(), S(), 1024))
        self.assertIsNone(bc._capture_lag_ms(q, S(), "x"))


@requires_monolith
class R1ReaperMarkTests(_Base):
    def _run(self, stream):
        bc = self.bc
        tags = []
        self._p(bc, "_prof", side_effect=lambda tag, extra="": tags.append(tag))
        done = threading.Event()
        with mock.patch.object(bc, "_pa_close_done"):
            bc._reap_playback(stream, done, 0.1)
        self.assertTrue(done.is_set())
        return tags

    def test_inactive_then_closed(self):
        class _Stream:
            polls = 0

            @property
            def active(self):
                _Stream.polls += 1
                return _Stream.polls < 2

            def stop(self, ignore_errors=True):
                pass

            def close(self, ignore_errors=True):
                pass

        self.assertEqual(self._run(_Stream()),
                         ["reap_inactive", "reap_closed"])

    def test_a_dead_stream_still_marks_closed(self):
        stream = mock.Mock()
        type(stream).active = mock.PropertyMock(
            side_effect=RuntimeError("gone"))
        self.assertEqual(self._run(stream), ["reap_closed"])


@requires_monolith
class R1TailProbeTests(_Base):
    def setUp(self):
        super().setUp()
        bc = self.bc
        import numpy as np
        from core import turn_timing as tt
        self.np = np
        self.tt = tt
        self.timing = tt.TurnTiming(print_fn=lambda line: None)
        self._p(bc, "_turn_timing", self.timing)
        self._p(bc, "transcribe", return_value=("hello", {"x": 1}))
        self.vad = _FakeVad(tail=777)
        self._p(bc, "_tail_vad", self.vad)
        # A probe left running by an earlier test must not make this
        # capture skip its own (one probe in flight at a time).
        self._p(bc, "_tail_probe_state",
                {"thread": None, "off_logged": False})
        self.audio = np.zeros(bc.SAMPLE_RATE, dtype=np.float32)

    def _capture(self, audio=None):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            res = self.bc._transcribe_capture(
                self.audio if audio is None else audio)
        _join_probe(self.bc)
        return res, out.getvalue()

    def _line(self):
        return self.tt.parse_line(self.timing.emit())

    def test_a_latched_detector_is_logged_once_and_skipped(self):
        self.vad.failed = "load failed: OSError: missing"
        self.timing.begin("voice")
        res, log1 = self._capture()
        _, log2 = self._capture()
        self.assertEqual(res, ("hello", {"x": 1}))
        self.assertIn("[turn-timing] tail probe off for this session "
                      "(load failed: OSError: missing)", log1)
        self.assertEqual(log2, "")
        self.assertEqual(self.vad.clips, [])
        d = self._line()
        self.assertEqual((d["clip_ms"], d["tail_ms"]), ("1000", "-"))

    def test_a_warmer_failure_is_its_one_log_line(self):
        bc = self.bc

        class Broken:
            failed = ""

            def warm(self):
                Broken.failed = "load failed: X"
                raise RuntimeError("load failed: X")

        self._p(bc, "_tail_vad", Broken())
        with self.assertRaises(RuntimeError):
            bc._warm_tail_probe()
        _, log = self._capture()
        self.assertEqual(log, "")

    def test_flag_off_never_runs_the_detector(self):
        self._p(self.bc, "TURN_TAIL_PROBE", False)
        self.timing.begin("voice")
        self._capture()
        self.assertEqual(self.vad.clips, [])
        d = self._line()
        self.assertEqual((d["clip_ms"], d["tail_ms"]), ("1000", "-"))

    def test_one_probe_in_flight(self):
        gate = threading.Event()
        self.addCleanup(gate.set)
        self.vad.gate = gate
        self.timing.begin("voice")
        with contextlib.redirect_stdout(io.StringIO()):
            self.bc._transcribe_capture(self.audio)
            self.bc._transcribe_capture(self.audio)
        gate.set()
        _join_probe(self.bc)
        self.assertEqual(len(self.vad.clips), 1)

    def test_not_an_array_is_ignored(self):
        self.timing.begin("voice")
        res, log = self._capture(object())
        self.assertEqual((res, log), (("hello", {"x": 1}), ""))
        self.assertEqual(self.vad.clips, [])
        self.assertEqual(self._line()["clip_ms"], "-")

    def test_a_thread_that_cannot_start_never_breaks_the_capture(self):
        bc = self.bc

        class NoThread:
            def __init__(self, *a, **k):
                raise RuntimeError("can't start new thread")

        self._p(bc.threading, "Thread", NoThread)
        self.timing.begin("voice")
        res, _ = self._capture()
        self.assertEqual(res, ("hello", {"x": 1}))
        self.assertEqual(self._line()["tail_ms"], "-")

    def test_note_stat_helper_for_skills(self):
        self.timing.begin("voice")
        _in = threading.Thread(
            target=lambda: self.bc._tt_note_stat("amb_deferred", 2))
        _in.start()
        _in.join(5)
        self.assertEqual(self._line()["amb_deferred"], "2")


@requires_monolith
class R1BootWarmerTests(_Base):
    def setUp(self):
        super().setUp()
        self.reg = []
        self._p(self.bc, "_boot_warmers", self.reg)
        self._p(self.bc, "_boot_warmers_started", [False])

    def _run(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            th = self.bc._run_boot_warmers()
            if th is not None:
                th.join(5)
        return th, out.getvalue()

    def test_each_warmer_runs_once_in_order_with_one_line(self):
        bc = self.bc
        ran = []

        def boom():
            ran.append("b")
            raise RuntimeError("no model")

        bc._register_boot_warmer("a", lambda: ran.append("a"))
        bc._register_boot_warmer("b", boom)
        bc._register_boot_warmer("c", lambda: ran.append("c"))
        bc._register_boot_warmer("a", lambda: ran.append("A"))   # dup
        bc._register_boot_warmer("x", "not callable")
        th, log = self._run()
        self.assertIsNotNone(th)
        self.assertEqual(ran, ["a", "b", "c"])
        lines = [ln.strip() for ln in log.splitlines()]
        self.assertEqual(len(lines), 3, lines)
        self.assertRegex(lines[0], r"^\[warm\] a ok \(\d+ ms\)$")
        self.assertEqual(lines[1], "[warm] b failed (RuntimeError: no model)")
        self.assertRegex(lines[2], r"^\[warm\] c ok \(\d+ ms\)$")
        self.assertEqual(self._run(), (None, ""))     # once per process

    def test_a_hanging_warmer_never_blocks_the_caller(self):
        bc = self.bc
        release = threading.Event()
        self.addCleanup(release.set)
        bc._register_boot_warmer("slow", lambda: release.wait(10))
        t0 = time.monotonic()
        with contextlib.redirect_stdout(io.StringIO()):
            th = bc._run_boot_warmers()
            self.assertLess(time.monotonic() - t0, 1.0)
            self.assertTrue(th.daemon)
            self.assertTrue(th.is_alive())
            release.set()
            th.join(5)

    def test_nothing_registered_starts_nothing(self):
        self.assertEqual(self._run(), (None, ""))

    def test_a_thread_that_cannot_start_never_raises(self):
        bc = self.bc
        bc._register_boot_warmer("a", lambda: None)

        class NoThread:
            def __init__(self, *a, **k):
                raise RuntimeError("can't start new thread")

        self._p(bc.threading, "Thread", NoThread)
        th, log = self._run()
        self.assertIsNone(th)
        self.assertIn("[warm] could not start", log)



@requires_monolith
class R1TurnFlagsTests(_Base):
    def test_flags_line_has_scalar_tokens_only(self):
        bc = self.bc
        self._p(bc, "PROCESSING_FILLER_DELAY", 0.5)
        self._p(bc, "TURN_TAIL_PROBE", True)
        out = io.StringIO()
        # A reserved key the module does not define is left out. Removed here
        # rather than assumed absent: SMART_TURN_MODE shipped in v2.0.164 and
        # the old "not defined yet" assertion went red.
        with mock.patch.dict(bc.__dict__), contextlib.redirect_stdout(out):
            bc.__dict__.pop("SMART_TURN_MODE", None)
            bc._log_turn_flags()
        line = out.getvalue().strip()
        self.assertTrue(line.startswith("[turn-flags] "), line)
        kv = dict(tok.split("=", 1) for tok in line.split()[1:])
        self.assertEqual(kv["PROCESSING_FILLER_DELAY"], "0.5")
        self.assertEqual(kv["TURN_TAIL_PROBE"], "True")
        self.assertTrue(set(kv) <= set(bc._TURN_FLAG_KEYS))
        self.assertNotIn("SMART_TURN_MODE", kv)

    def test_flag_tokens(self):
        tok = self.bc._turn_flag_token
        self.assertEqual(tok((0.0, 0.2, 0.4)), "0.0,0.2,0.4")
        self.assertEqual(tok("shadow mode"), "shadow_mode")
        self.assertEqual(tok(""), "''")
        self.assertIsNone(tok({"a": 1}))
        self.assertIsNone(tok("x" * 41))


class R1WiringTests(_Base):
    def test_the_tail_probe_registers_its_boot_warmer(self):
        bc = self.bc
        entry = ("silero-tail", bc._warm_tail_probe)
        if bc.TURN_TAIL_PROBE:
            self.assertIn(entry, bc._boot_warmers)
        else:
            self.assertNotIn(entry, bc._boot_warmers)

    def test_boot_warmers_start_right_after_the_boot_whisper_load(self):
        src = inspect.getsource(self.bc.main)
        w = src.index("_ensure_whisper()   # load now")
        run = src.index("_run_boot_warmers()", w)
        flags = src.index("_log_turn_flags()", w)
        self.assertLess(run, flags)
        self.assertLess(flags, src.index("check_dependencies()", w))
        self.assertEqual(src.count("_run_boot_warmers()"), 1)

    def test_transcribe_times_the_lock_and_returns_unchanged(self):
        src = inspect.getsource(self.bc.transcribe)
        lock = src.index("with _stt_lock:")
        note = src.index('_tt_note_elapsed("stt_wait_ms", _stt_w0)')
        self.assertLess(src.index('_stt_w0 = _tt("now")'), lock)
        self.assertLess(lock, note)
        self.assertLess(note, src.index("return _transcribe_impl(audio)"))

    def test_capture_starts_the_probe_before_any_decode(self):
        src = inspect.getsource(self.bc._transcribe_capture)
        self.assertLess(src.index("_tail_probe_start(audio)"),
                        src.index('_spec_stt.get("thread")'))

    def test_playback_open_is_timed_on_both_branches(self):
        src = inspect.getsource(self.bc._play_with_lipsync_body)
        self.assertLess(src.index("_tt_open0 = _tt(\"now\")"),
                        src.index("if not _pa_claim_owner("))
        self.assertLess(src.index("_tt_open0 = _tt(\"now\")"),
                        src.index("_audio_ducker.duck()"))
        self.assertEqual(src.count("_tt_open0 = "), 1)
        self.assertEqual(src.count("_tt_note_out_latency(_stream)"), 2)
        # Both branches note the open through the ONE helper (play_open_ms
        # plus the reply's opens / opens_ms), and only when a stream was
        # opened (PLAYBACK_KEEPER review 2026-10-05: a line cut before its
        # open opens nothing).
        self.assertEqual(src.count("_opened = _play_audio_safe()\n"
                                   "            if _opened:\n"
                                   "                _tt_note_play_open("
                                   "_tt_open0)"), 2)
        self.assertEqual(src.count("_tt_note_play_open("), 2)
        self.assertEqual(src.count("args=(_stream, _done_evt, audio_secs)"),
                         2)

    def test_filler_clip_is_noted_with_the_filler(self):
        src = inspect.getsource(self.bc._filler_play)
        self.assertLess(src.index('_tt("note_filler")'),
                        src.index('_tt_note_clip_ms("filler_clip_ms", '
                                  'audio, sr)'))

    def test_reaper_marks(self):
        src = inspect.getsource(self.bc._reap_playback)
        self.assertLess(src.index('_prof("reap_inactive")'),
                        src.index('_prof("reap_closed")'))


if __name__ == "__main__":
    unittest.main()
