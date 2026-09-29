"""Tests for core/processing_filler.py — the "Just a moment, sir." scheduler.

CI-light: stdlib unittest only, no monolith, no audio, no real threads. The
harness is a FakeClock whose wait_fn(evt, t) advances ``now`` by t, runs any
scripted hooks due by then (e.g. ``filler.note_speech()`` at t=5.0) and returns
evt.is_set(). thread_factory records the target without starting it, and each
test calls ``filler._run(turn)`` synchronously.

The monolith glue (bobert_companion._filler_*) is covered separately in
tests/monolith/test_monolith_processing_filler.py (full-deps tier).
"""
from __future__ import annotations

import ast
import contextlib
import io
import math
import os
import re
import sys
import threading
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import processing_filler as pf  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── harness ──────────────────────────────────────────────────────────────────
class FakeClock:
    def __init__(self):
        self.now = 0.0
        self._hooks: list = []
        self.waits: list = []

    def __call__(self):
        return self.now

    def at(self, t, fn):
        self._hooks.append((float(t), fn))
        self._hooks.sort(key=lambda h: h[0])

    def wait(self, evt, t):
        self.waits.append(t)
        if evt.is_set():
            return True
        target = self.now + max(0.0, float(t))
        while self._hooks and self._hooks[0][0] <= target:
            ht, fn = self._hooks.pop(0)
            self.now = max(self.now, ht)
            fn()
            if evt.is_set():
                return True
        self.now = target
        return evt.is_set()


class RecThread:
    def __init__(self, target, args, name, daemon):
        self.target, self.args, self.name, self.daemon = target, args, name, daemon
        self.started = False

    def start(self):
        self.started = True

    def join(self, *a, **k):  # pragma: no cover - must never be reached
        raise AssertionError("the filler must never join a thread")


class RecFactory:
    def __init__(self, raise_on_start=False):
        self.made: list[RecThread] = []
        self.raise_on_start = raise_on_start

    def __call__(self, target=None, args=(), name=None, daemon=None):
        if self.raise_on_start:
            raise RuntimeError("can't start new thread")
        t = RecThread(target, args, name, daemon)
        self.made.append(t)
        return t


def _make(first=2.5, still=12.0, suppressed=None, play_script=None,
          factory=None, max_retry_s=30.0):
    """Build a filler whose play_fn emulates the monolith's: claim() inside
    'the lock', play_done() after an ok claim. play_script, when given, is a
    list of forced return values consumed per stage-2 attempt."""
    clock = FakeClock()
    plays: list = []
    attempts: list = []
    script = list(play_script or [])
    holder: dict = {}
    speech_lock = threading.Lock()   # stands in for _SPEAK_LOCK

    def play_fn(turn, stage):
        attempts.append((stage, clock.now))
        if not speech_lock.acquire(blocking=False):
            return "retry"           # someone is speaking
        speech_lock.release()
        if stage == 2 and script:
            forced = script.pop(0)
            if forced != "claim":
                return forced
        v = holder["f"].claim(turn, stage)
        if v in ("not-yet", "busy"):
            return "retry"
        if v != "ok":
            return "skipped"
        plays.append((stage, clock.now))
        holder["f"].play_done()
        return "played"

    f = pf.ProcessingFiller(
        play_fn=play_fn,
        suppressed_fn=(suppressed or (lambda: None)),
        delays_fn=lambda: (first, still),
        clock=clock, wait_fn=clock.wait,
        thread_factory=factory or RecFactory(),
        max_retry_s=max_retry_s,
    )
    holder["f"] = f
    f.test_speech_lock = speech_lock
    return f, clock, plays, attempts


# ── sanitize_delays ──────────────────────────────────────────────────────────
class SanitizeDelaysTests(unittest.TestCase):
    def test_bad_first_falls_back_to_default(self):
        for bad in (float("nan"), float("inf"), -1, 0, None, "abc"):
            first, _still = pf.sanitize_delays(bad, 12.0)
            self.assertEqual(first, 2.5, bad)

    def test_first_is_clamped(self):
        self.assertEqual(pf.sanitize_delays(0.1, 12.0)[0], 0.5)
        self.assertEqual(pf.sanitize_delays(500, 900)[0], 60.0)

    def test_still_nan_gives_default(self):
        self.assertEqual(pf.sanitize_delays(2.5, float("nan")), (2.5, 12.0))

    def test_still_at_or_below_first_disables_stage_two(self):
        self.assertEqual(pf.sanitize_delays(2.5, 2.5), (2.5, None))
        self.assertEqual(pf.sanitize_delays(4.0, 1.0), (4.0, None))

    def test_still_is_clamped(self):
        self.assertEqual(pf.sanitize_delays(2.5, 10_000)[1], 600.0)

    def test_ints_come_back_as_floats(self):
        first, still = pf.sanitize_delays(3, 15)
        self.assertIsInstance(first, float)
        self.assertIsInstance(still, float)


# ── arm / suppression ────────────────────────────────────────────────────────
class ArmTests(unittest.TestCase):
    def test_suppressed_arm_creates_no_thread(self):
        for reason in ("disabled", "tray-mute", "env-mute", "staging",
                       "standby", "focus", "dnd", "night-owl", "game",
                       "backend", "realtime", "barge-in", "mic-capture"):
            fac = RecFactory()
            f, _c, _p, _a = _make(suppressed=lambda r=reason: r, factory=fac)
            self.assertIsNone(f.arm(), reason)
            self.assertEqual(fac.made, [], reason)
            self.assertFalse(f.armed())
            self.assertEqual(f.last_reason, reason)

    def test_arm_starts_one_named_daemon(self):
        fac = RecFactory()
        f, _c, _p, _a = _make(factory=fac)
        turn = f.arm()
        self.assertIsNotNone(turn)
        self.assertEqual(len(fac.made), 1)
        th = fac.made[0]
        self.assertTrue(th.started)
        self.assertTrue(th.daemon)
        self.assertEqual(th.name, "processing-filler")
        self.assertEqual(th.args, (turn,))

    def test_thread_start_failure_returns_none(self):
        f, _c, _p, _a = _make(factory=RecFactory(raise_on_start=True))
        self.assertIsNone(f.arm())
        self.assertFalse(f.armed())

    def test_new_arm_cancels_previous_turn(self):
        f, _c, _p, _a = _make()
        t1 = f.arm()
        t2 = f.arm()
        self.assertTrue(t1.cancel.is_set())
        self.assertEqual(f.claim(t1, 1), "gone")
        self.assertEqual(f.claim(t2, 1), "ok")

    def test_shutdown_latches_off(self):
        fac = RecFactory()
        f, _c, _p, _a = _make(factory=fac)
        t = f.arm()
        f.shutdown("restart")
        self.assertTrue(t.cancel.is_set())
        self.assertTrue(f.closed())
        self.assertFalse(f.armed())
        self.assertEqual(f.claim(t, 1), "gone")
        self.assertIsNone(f.arm())
        self.assertEqual(len(fac.made), 1)

    def test_note_speech_unarmed_is_noop(self):
        f, _c, _p, _a = _make()
        f.note_speech()      # must not raise
        f.cancel("x")        # must not raise
        f.disarm(None)       # must not raise
        self.assertFalse(f.armed())

    def test_unarmed_hooks_never_read_the_clock(self):
        # The default-off path: _speak / record_speech / get_mic_buffer call
        # these on every utterance, so with nothing armed they must not even
        # read the clock (note_speech also takes no lock).
        # (The hooks swallow exceptions, so count instead of raising.)
        reads = []

        def clock():
            reads.append(1)
            return 0.0

        class CountLock:
            entered = 0

            def __enter__(self_):
                CountLock.entered += 1

            def __exit__(self_, *a):
                return False

        f = pf.ProcessingFiller(play_fn=lambda t, s: "skipped",
                                suppressed_fn=lambda: None,
                                delays_fn=lambda: (2.5, 12.0), clock=clock,
                                thread_factory=RecFactory())
        f._lock = CountLock()
        f.note_speech()
        f.note_speech(owner_only=True)
        self.assertEqual(CountLock.entered, 0)
        self.assertFalse(f.begin_capture())
        f.end_capture()
        self.assertFalse(f.capturing())
        self.assertEqual(reads, [])

    def test_disarm_clears_the_current_turn(self):
        f, _c, _p, _a = _make()
        t = f.arm()
        self.assertTrue(f.armed())
        f.disarm(t)
        self.assertFalse(f.armed())
        self.assertTrue(t.cancel.is_set())

    def test_disarm_of_a_stale_turn_keeps_the_new_one(self):
        f, _c, _p, _a = _make()
        t1 = f.arm()
        t2 = f.arm()
        f.disarm(t1)
        self.assertTrue(f.armed())
        f.disarm(t2)
        self.assertFalse(f.armed())

    def test_turn_records_the_arming_thread(self):
        f, _c, _p, _a = _make()
        t = f.arm()
        self.assertEqual(t.owner, threading.get_ident())


# ── claim ────────────────────────────────────────────────────────────────────
class ClaimTests(unittest.TestCase):
    def test_ok_once_then_gone(self):
        f, _c, _p, _a = _make()
        t = f.arm()
        self.assertEqual(f.claim(t, 1), "ok")
        f.play_done()
        self.assertEqual(f.claim(t, 1), "gone")

    def test_stage_one_gone_after_speech(self):
        f, _c, _p, _a = _make()
        t = f.arm()
        f.note_speech()
        self.assertEqual(f.claim(t, 1), "gone")

    def test_gone_after_disarm_or_cancel(self):
        f, _c, _p, _a = _make()
        t = f.arm()
        f.disarm(t)
        self.assertEqual(f.claim(t, 1), "gone")
        t2 = f.arm()
        f.cancel("action:play_streaming")
        self.assertEqual(f.claim(t2, 1), "gone")

    def test_stage_two_not_yet_until_silence(self):
        f, clock, _p, _a = _make()
        t = f.arm()
        clock.now = 11.0
        self.assertEqual(f.claim(t, 2), "not-yet")
        clock.now = 12.0
        self.assertEqual(f.claim(t, 2), "ok")

    def test_capture_makes_claims_busy(self):
        f, clock, _p, _a = _make()
        t = f.arm()
        t.owner = -1                     # a background capture
        self.assertFalse(f.begin_capture())
        self.assertTrue(f.capturing())
        self.assertEqual(f.claim(t, 1), "busy")
        clock.now = 13.0
        self.assertEqual(f.claim(t, 2), "busy")
        f.end_capture()
        self.assertFalse(f.capturing())
        self.assertEqual(f.claim(t, 1), "ok")
        f.play_done()
        self.assertEqual(f.claim(t, 2), "ok")

    def test_spoken_stage_one_is_gone_even_while_busy(self):
        f, _c, _p, _a = _make()
        t = f.arm()
        f.note_speech()
        f.begin_capture()
        self.assertEqual(f.claim(t, 1), "gone")

    def test_begin_capture_reports_a_clip_claimed_before_it(self):
        f, _c, _p, _a = _make()
        t = f.arm()
        self.assertEqual(f.claim(t, 1), "ok")
        self.assertTrue(f.begin_capture())       # caller must wait_idle
        f.play_done()
        f.end_capture()
        self.assertFalse(f.begin_capture())
        f.end_capture()

    def test_owner_capture_marks_start_and_end(self):
        f, clock, _p, _a = _make()
        t = f.arm()
        clock.now = 4.0
        f.begin_capture()
        self.assertTrue(t.spoke)
        self.assertEqual(t.last_mark, 4.0)
        clock.now = 9.0
        f.end_capture()
        self.assertEqual(t.last_mark, 9.0)
        # The mark and the release are one critical section: once the
        # capture is gone the clock already counts from its end.
        clock.now = 20.0
        self.assertEqual(f.claim(t, 2), "not-yet")
        clock.now = 21.0
        self.assertEqual(f.claim(t, 2), "ok")

    def test_background_thread_capture_is_not_turn_activity(self):
        f, clock, _p, _a = _make()
        t = f.arm()
        clock.now = 4.0
        seen = {}

        def bg():
            seen["busy"] = f.begin_capture()
            seen["claim"] = f.claim(t, 1)
            f.end_capture()
            f.note_speech(owner_only=True)

        th = threading.Thread(target=bg, daemon=True)
        th.start()
        th.join(2.0)
        self.assertFalse(th.is_alive())
        self.assertEqual(seen, {"busy": False, "claim": "busy"})
        self.assertFalse(t.spoke)
        self.assertEqual(t.last_mark, 0.0)
        self.assertFalse(f.capturing())
        # ...while a plain speech mark from any thread still counts.
        f.note_speech()
        self.assertTrue(t.spoke)

    def test_playing_state_and_bounded_wait_idle(self):
        f, clock, _p, _a = _make()
        t = f.arm()
        self.assertTrue(f.wait_idle(3.0))
        self.assertEqual(f.claim(t, 1), "ok")
        self.assertTrue(f.playing())
        before = clock.now
        self.assertFalse(f.wait_idle(10.0))     # capped at 3 s
        self.assertAlmostEqual(clock.now - before, 3.0)
        f.play_done()
        self.assertFalse(f.playing())
        self.assertTrue(f.wait_idle(3.0))


# ── the per-turn thread body ─────────────────────────────────────────────────
class RunTests(unittest.TestCase):
    def test_stage_one_fires_once_at_delay(self):
        f, _c, plays, _a = _make(still=2.0)   # stage 2 off
        t = f.arm()
        f._run(t)
        self.assertEqual(plays, [(1, 2.5)])

    def test_stage_one_skipped_after_early_speech(self):
        f, clock, plays, attempts = _make(still=2.0)
        t = f.arm()
        clock.at(1.0, f.note_speech)
        f._run(t)
        self.assertEqual(plays, [])
        self.assertNotIn(1, [s for s, _ in attempts])

    def test_disarm_before_delay_plays_nothing(self):
        f, clock, plays, attempts = _make()
        t = f.arm()
        clock.at(1.0, lambda: f.disarm(t))
        f._run(t)
        self.assertEqual(plays, [])
        self.assertEqual(attempts, [])

    def test_stage_two_with_no_speech(self):
        f, _c, plays, _a = _make()
        t = f.arm()
        f._run(t)
        self.assertEqual(plays, [(1, 2.5), (2, 12.0)])

    def test_stage_two_counts_from_last_speech(self):
        f, clock, plays, attempts = _make()
        t = f.arm()
        clock.at(5.0, f.note_speech)
        clock.at(7.0, f.note_speech)
        f._run(t)
        # Stage 1 at 2.5 (nothing had spoken yet); the clip itself is NOT a
        # speech mark, so stage 2 counts 12 s from the last mark at 7.0 —
        # and the thread SLEEPS until then (one attempt, no busy-polling).
        self.assertEqual(plays, [(1, 2.5), (2, 19.0)])
        self.assertEqual([ts for s, ts in attempts if s == 2], [19.0])

    def test_stage_two_survives_a_chatty_turn(self):
        # Speech marks spread past t0 + still + max_retry_s: counting from t0
        # would burn the whole retry budget polling and never play stage 2.
        f, clock, plays, attempts = _make()
        t = f.arm()
        for ts in (5.0, 15.0, 25.0, 35.0):
            clock.at(ts, f.note_speech)
        f._run(t)
        self.assertEqual([p for p in plays if p[0] == 2], [(2, 47.0)])
        self.assertEqual([ts for s, ts in attempts if s == 2], [47.0])

    def _background_capture(self, f, clock, start, end):
        clock.at(start, f.begin_capture)
        clock.at(end, f.end_capture)

    def test_stage_one_retries_through_a_background_capture(self):
        # The standby-audio loop holds the mic 3 s of every ~8 s. A stage 1
        # that lands inside such a window waits it out instead of being lost.
        f, clock, plays, attempts = _make(still=2.0)
        t = f.arm()
        t.owner = -1          # captures below run on "another thread"
        self._background_capture(f, clock, 1.0, 4.0)
        f._run(t)
        self.assertEqual(plays, [(1, 4.0)])
        self.assertEqual([ts for s, ts in attempts if s == 1],
                         [2.5, 3.0, 3.5, 4.0])

    def test_stage_one_retry_is_bounded(self):
        f, clock, plays, attempts = _make(still=2.0)
        t = f.arm()
        t.owner = -1
        f.begin_capture()     # never ends
        f._run(t)
        self.assertEqual(plays, [])
        self.assertEqual([ts for s, ts in attempts if s == 1],
                         [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5])

    def test_background_captures_do_not_starve_stage_two(self):
        # Regression: background get_mic_buffer calls used to count as turn
        # activity, so a periodic poll kept resetting the silence clock and
        # stage 2 never fired. Now they only defer a claim while live.
        f, clock, plays, _a = _make()
        t = f.arm()
        t.owner = -1
        for start in (1.0, 9.0, 17.0, 25.0):
            self._background_capture(f, clock, start + 1.0, start + 4.0)
        f._run(t)
        self.assertEqual(plays, [(1, 5.0), (2, 13.0)])

    def test_stage_two_retry_then_plays(self):
        f, _c, plays, attempts = _make(play_script=["retry", "retry", "claim"])
        t = f.arm()
        f._run(t)
        stage2 = [ts for s, ts in attempts if s == 2]
        self.assertEqual(stage2, [12.0, 12.5, 13.0])
        self.assertEqual([p for p in plays if p[0] == 2], [(2, 13.0)])

    def test_stage_two_permanent_retry_gives_up(self):
        f, clock, plays, attempts = _make(play_script=["retry"] * 500,
                                          max_retry_s=30.0)
        t = f.arm()
        f._run(t)
        stage2 = [ts for s, ts in attempts if s == 2]
        self.assertEqual(len(stage2), 61)            # 60 retries + the last
        self.assertLessEqual(clock.now, 12.0 + 30.0 + 0.001)
        self.assertEqual([p for p in plays if p[0] == 2], [])

    def test_stage_two_disabled(self):
        f, _c, plays, attempts = _make(first=2.5, still=2.5)
        t = f.arm()
        f._run(t)
        self.assertEqual(plays, [(1, 2.5)])
        self.assertEqual([s for s, _ in attempts], [1])

    def test_cancel_after_stage_one_blocks_stage_two(self):
        f, clock, plays, _a = _make()
        t = f.arm()
        clock.at(5.0, lambda: f.cancel("action:play_streaming"))
        f._run(t)
        self.assertEqual(plays, [(1, 2.5)])

    def _long_utterance(self, f, clock, start, end, end_mark_inside_lock):
        lk = f.test_speech_lock

        def begin():
            f.note_speech()          # mark (a): before the lock
            lk.acquire()

        def finish():
            if end_mark_inside_lock:
                f.note_speech()      # mark (b) inside the with-block
                lk.release()
            else:
                lk.release()         # the reviewed bug: mark after release

        clock.at(start, begin)
        clock.at(end, finish)
        if not end_mark_inside_lock:
            clock.at(end + 0.25, f.note_speech)

    def test_stage_two_mark_during_long_utterance_defers(self):
        # Regression for the stage-2 race: a 15 s utterance (5.0-20.0) marks at
        # its start and at its END while the speech lock is still held, so the
        # polls that land during it retry and stage 2 moves to 20 + 12 = 32.0,
        # never straight after the answer.
        f, clock, plays, _a = _make()
        t = f.arm()
        self._long_utterance(f, clock, 5.0, 20.0, end_mark_inside_lock=True)
        f._run(t)
        self.assertEqual(plays, [(1, 2.5), (2, 32.0)])

    def test_end_mark_after_release_is_the_bug_shape(self):
        # Documents WHY mark (b) must be inside the lock: marked after the
        # release, a poll landing in the gap claims immediately.
        f, clock, plays, _a = _make()
        t = f.arm()
        self._long_utterance(f, clock, 5.0, 20.0, end_mark_inside_lock=False)
        f._run(t)
        self.assertEqual(plays, [(1, 2.5), (2, 20.0)])

    def test_run_never_sleeps(self):
        f, _c, _p, _a = _make()
        t = f.arm()
        with mock.patch.object(pf.time, "sleep",
                               side_effect=AssertionError("time.sleep")):
            f._run(t)

    def test_play_fn_exception_never_escapes(self):
        clock = FakeClock()

        def boom(turn, stage):
            raise RuntimeError("audio")

        f = pf.ProcessingFiller(play_fn=boom, suppressed_fn=lambda: None,
                                delays_fn=lambda: (2.5, 12.0), clock=clock,
                                wait_fn=clock.wait,
                                thread_factory=RecFactory())
        t = f.arm()
        f._run(t)   # must not raise


# ── ClipCache ────────────────────────────────────────────────────────────────
class _Arr(list):
    """list with .copy() semantics like an ndarray (list.copy is shallow)."""

    def copy(self):
        return _Arr(self)


class ClipCacheTests(unittest.TestCase):
    def _cache(self, render=None, key=("kokoro", "v", False), max_secs=2.0):
        self.key = [key]
        self.rendered: list = []
        self.lock = threading.Lock()

        def _render(text):
            self.rendered.append(text)
            if render is not None:
                return render(text)
            return _Arr([0.1] * 10), 10      # 1.0 s

        return pf.ClipCache(render_fn=_render, lock=self.lock,
                            key_fn=lambda: self.key[0], max_secs=max_secs)

    def _nowait(self):
        clock = FakeClock()
        return clock.wait

    def test_get_never_renders(self):
        c = self._cache()
        self.assertIsNone(c.get("Just a moment, sir."))
        self.assertEqual(self.rendered, [])

    def test_warm_renders_only_missing(self):
        c = self._cache()
        c.put("a", (_Arr([0.1]), 10))
        n = c.warm(["a", "b", "c"], stop_fn=lambda: False,
                   wait_fn=self._nowait())
        self.assertEqual(n, 2)
        self.assertEqual(self.rendered, ["b", "c"])
        self.assertIsNotNone(c.get("b"))

    def test_key_change_invalidates(self):
        c = self._cache()
        c.warm(["a", "b"], stop_fn=lambda: False, wait_fn=self._nowait())
        self.key[0] = ("kokoro", "other", False)
        self.assertIsNone(c.get("a"))
        self.assertEqual(c.missing(["a", "b"]), ["a", "b"])

    def test_too_long_render_is_rejected_once(self):
        c = self._cache(render=lambda t: (_Arr([0.1] * 25), 10))   # 2.5 s
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(c.warm(["long"], stop_fn=lambda: False,
                                    wait_fn=self._nowait()), 0)
            self.assertEqual(c.missing(["long"]), [])
            self.assertEqual(c.rejected(), ["long"])
            # A later warm never re-renders it (it used to, after EVERY
            # enabled voice turn, silently).
            c.warm(["long"], stop_fn=lambda: False, wait_fn=self._nowait())
            self.assertFalse(c.warm_async(["long"], stop_fn=lambda: False,
                                          thread_factory=RecFactory()))
        self.assertEqual(self.rendered, ["long"])
        self.assertEqual(out.getvalue().count("[filler] clip 'long'"), 1)
        # A voice change gives the line another chance.
        self.key[0] = ("kokoro", "other", False)
        self.assertEqual(c.missing(["long"]), ["long"])
        self.assertEqual(c.rejected(), [])

    def test_transient_render_failure_is_retried(self):
        c = self._cache(render=lambda t: None)
        c.warm(["a"], stop_fn=lambda: False, wait_fn=self._nowait())
        self.assertEqual(c.missing(["a"]), ["a"])

    def test_default_max_secs_fits_the_end_of_turn_wait(self):
        c = pf.ClipCache(render_fn=lambda t: None, lock=threading.Lock(),
                         key_fn=lambda: "k")
        # The monolith waits at most 3.0 s for a playing clip; leave room for
        # the ducker / stream-close overhead.
        self.assertLessEqual(c._max_secs, 2.5)

    def test_bad_renders_not_cached(self):
        outs = {"none": None, "empty": (_Arr([]), 10),
                "long": (_Arr([0.1] * 25), 10), "nosr": (_Arr([0.1]), 0)}
        c = self._cache(render=lambda t: outs[t])
        with contextlib.redirect_stdout(io.StringIO()):
            n = c.warm(list(outs), stop_fn=lambda: False,
                       wait_fn=self._nowait())
        self.assertEqual(n, 0)
        self.assertEqual(c.available(list(outs)), [])

    def test_warm_stops_when_stop_fn_flips(self):
        c = self._cache()
        calls = {"n": 0}

        def stop():
            calls["n"] += 1
            return len(self.rendered) >= 1

        c.warm(["a", "b", "c"], stop_fn=stop, wait_fn=self._nowait())
        self.assertEqual(self.rendered, ["a"])

    def test_warm_gives_up_when_lock_held(self):
        c = self._cache()
        self.lock.acquire()
        try:
            n = c.warm(["a"], stop_fn=lambda: False, wait_fn=self._nowait(),
                       max_tries=5)
        finally:
            self.lock.release()
        self.assertEqual(n, 0)
        self.assertEqual(self.rendered, [])

    def test_warm_yields_between_lines_after_release(self):
        c = self._cache()
        seen: list = []

        def wait(evt, t):
            # The lock must be FREE whenever warm yields.
            seen.append((t, self.lock.locked()))
            return False

        c.warm(["a", "b"], stop_fn=lambda: False, wait_fn=wait, yield_s=0.3)
        self.assertEqual(seen, [(0.3, False), (0.3, False)])

    def test_get_returns_a_copy(self):
        c = self._cache()
        c.put("a", (_Arr([0.1, 0.2]), 10))
        got, _sr = c.get("a")
        got[0] = 9.0
        self.assertEqual(c.get("a")[0][0], 0.1)

    def test_warm_async_single_flight(self):
        c = self._cache()
        fac = RecFactory()
        self.assertTrue(c.warm_async(["a"], stop_fn=lambda: False,
                                     wait_fn=self._nowait(),
                                     thread_factory=fac))
        self.assertFalse(c.warm_async(["a"], stop_fn=lambda: False,
                                      wait_fn=self._nowait(),
                                      thread_factory=fac))
        self.assertEqual(len(fac.made), 1)
        self.assertEqual(fac.made[0].name, "filler-warm")
        self.assertTrue(fac.made[0].daemon)
        # Running the recorded target clears the single-flight flag.
        fac.made[0].target(*fac.made[0].args)
        self.assertFalse(c.warming())
        self.assertFalse(c.warm_async(["a"], stop_fn=lambda: False,
                                      thread_factory=fac))   # nothing missing


# ── quiet commands ───────────────────────────────────────────────────────────
class QuietCommandTests(unittest.TestCase):
    def test_quiet_commands_detected(self):
        from core import tone_detector as td
        for t in ("stop", "Stop.", "cancel", "shut up", "Shut up, JARVIS.",
                  "enough", "abort", "stop it", "be quiet please"):
            self.assertTrue(pf.is_quiet_command(t, td._CLIPPED_IMPERATIVES), t)

    def test_real_requests_not_quiet(self):
        from core import tone_detector as td
        for t in ("what's the weather", "go", "do it", "open youtube",
                  "stop the music and play something relaxing", ""):
            self.assertFalse(pf.is_quiet_command(t, td._CLIPPED_IMPERATIVES), t)

    def test_quiet_set_is_drawn_from_the_tone_detector(self):
        from core import tone_detector as td
        self.assertLessEqual(set(pf.QUIET_IMPERATIVES),
                             set(td._CLIPPED_IMPERATIVES))


# ── line banks ───────────────────────────────────────────────────────────────
def _mid_task_bank() -> list[str]:
    with open(os.path.join(_ROOT, "bobert_companion.py"), encoding="utf-8") as fh:
        src = fh.read()
    m = re.search(r"^_MID_TASK_STATUS_LINES\b.*?=\s*(\{.*?^\})", src,
                  re.MULTILINE | re.DOTALL)
    assert m, "could not locate _MID_TASK_STATUS_LINES in the monolith"
    bank = ast.literal_eval(m.group(1))
    return [line for lines in bank.values() for line in lines]


def _minimal_acks() -> list[str]:
    from core import prompts
    text = prompts.BASE_SYSTEM_PROMPT
    i = text.index("Minimal acknowledgements")
    seg = text[i:text.index("These are deliberate", i)]
    return re.findall(r"'([^']+)'", seg)


class LineBankTests(unittest.TestCase):
    ALL = pf.FIRST_LINES + pf.STILL_LINES

    def test_no_line_says_jarvis(self):
        for line in self.ALL:
            self.assertNotIn("jarvis", line.lower())

    def test_lines_are_short(self):
        for line in self.ALL:
            self.assertLessEqual(len(line.split()), 7, line)

    def test_no_collision_with_minimal_acks(self):
        acks = {pf.normalise_line(a) for a in _minimal_acks()}
        self.assertIn("one moment", acks)   # the parse really found them
        for line in self.ALL:
            self.assertNotIn(pf.normalise_line(line), acks, line)

    def test_no_collision_with_mid_task_bank(self):
        bank = _mid_task_bank()
        self.assertIn("Bear with me, sir.", bank)   # the parse really worked
        norm = {pf.normalise_line(b) for b in bank}
        for line in self.ALL:
            self.assertNotIn(pf.normalise_line(line), norm, line)

    def test_banks_do_not_overlap(self):
        a = {pf.normalise_line(x) for x in pf.FIRST_LINES}
        b = {pf.normalise_line(x) for x in pf.STILL_LINES}
        self.assertFalse(a & b)

    def test_normalise_line(self):
        self.assertEqual(pf.normalise_line("One moment, sir."), "one moment")
        self.assertEqual(pf.normalise_line("  Working on it.  "),
                         "working on it")


class KokoroDurationTests(unittest.TestCase):
    """Local tier: needs the Kokoro model files (absent in CI / worktrees)."""

    def test_every_shipped_line_fits_max_secs(self):
        try:
            from core import kokoro_tts
            ok = kokoro_tts.is_available()
        except Exception:
            ok = False
        if not ok:
            self.skipTest("Kokoro model not available on this box")
        limit = pf.ClipCache(render_fn=None, lock=threading.Lock(),
                             key_fn=lambda: "k")._max_secs
        for line in pf.FIRST_LINES + pf.STILL_LINES:
            res = kokoro_tts.synthesize(line, speed=1.0)
            self.assertIsNotNone(res, line)
            audio, sr = res
            self.assertLessEqual(len(audio) / float(sr), limit, line)


class ModuleHygieneTests(unittest.TestCase):
    def test_stdlib_only(self):
        with open(pf.__file__, encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        mods = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods.add(node.module.split(".")[0])
        self.assertLessEqual(mods, {"__future__", "math", "re", "threading",
                                    "time"})

    def test_defaults_match_config(self):
        # Read the SHIPPED literals statically: importing core.config would
        # overlay this box's data/user_settings.json.
        with open(os.path.join(_ROOT, "core", "config.py"),
                  encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        lits = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.value,
                                                           ast.Constant):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name):
                        lits[tgt.id] = node.value.value
        self.assertIs(lits["PROCESSING_FILLER_ENABLED"], False)
        self.assertEqual(pf.DEFAULT_FIRST, lits["PROCESSING_FILLER_DELAY"])
        self.assertEqual(pf.DEFAULT_STILL,
                         lits["PROCESSING_FILLER_STILL_DELAY"])
        self.assertIsInstance(lits["PROCESSING_FILLER_DELAY"], float)
        self.assertIsInstance(lits["PROCESSING_FILLER_STILL_DELAY"], float)
        self.assertTrue(math.isfinite(pf.DEFAULT_FIRST))


if __name__ == "__main__":
    unittest.main()
