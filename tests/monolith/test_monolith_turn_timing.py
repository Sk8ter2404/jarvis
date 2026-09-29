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

        def fake_record(timeout=None):
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
        self._quiet(buf.feed, "Right away, sir, checking now. ")
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
        self.assertLess(brk, src.index('_tt("note_vad_break")'))
        self.assertLess(src.index('_tt("note_vad_break")'),
                        src.index(" break\n", brk))

    def test_filler_play_is_noted(self):
        src = inspect.getsource(self.bc._filler_play)
        self.assertLess(src.index('_prof("filler_play"'),
                        src.index('_tt("note_filler")'))

    def test_followup_rounds_are_counted(self):
        src = inspect.getsource(self.bc._run_llm_dispatch_body)
        self.assertLess(src.index('_tt("followup_round")'),
                        src.index("followup = get_followup_response("))


if __name__ == "__main__":
    unittest.main()
