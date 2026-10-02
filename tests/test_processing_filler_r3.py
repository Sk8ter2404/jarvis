"""Speed plan R3, Phase 0 — scheduler hygiene in core/processing_filler.py.

  * ``late_fn`` (PROCESSING_FILLER_LATE_START_S): stage 1's retry window, read
    at every arm(). It bounds the retries AND the latest start, on top of the
    fixed first_late_s cap. No late_fn = exactly today's scheduler.
  * ``is_pleasantry``: a pure phrase-list test ("thank you", "hello", ...)
    the monolith uses, with PROCESSING_FILLER_SKIP_PLEASANTRIES on, to arm no
    filler for a bare pleasantry.
  * ``is_owner_thread``: True only on the thread that armed the current turn
    (the R3 pre-render runs only there).
  * ``core.sentence_tts.play_pipelined(first_rendered=...)``: sentence 1 from
    the pre-render is padded and played, never rendered again.

The windowed tests are the non-default twins of three tests in
tests/test_processing_filler.py, which stay unchanged for the default:
test_stage_one_retries_through_a_background_capture,
test_stage_one_retry_is_bounded and
test_ack_state_retry_window_over_is_not_pending.

Light tier: stdlib only, the FakeClock harness of tests/test_processing_filler.

    python -m unittest tests.test_processing_filler_r3
"""
from __future__ import annotations

import math
import threading
import unittest

from core import processing_filler as pf
from core import sentence_tts as st
from tests.test_processing_filler import FakeClock, RecFactory
from tests.test_sentence_tts import _join_workers, _Rec


def _make(first=2.5, still=2.0, window=None, first_late_s=None,
          late_fn=None):
    """Like tests/test_processing_filler._make, plus the late-start knob.
    ``window`` builds a late_fn returning it (``late_fn`` overrides)."""
    clock = FakeClock()
    plays: list = []
    attempts: list = []
    holder: dict = {}
    speech_lock = threading.Lock()

    def play_fn(turn, stage):
        attempts.append((stage, clock.now))
        if not speech_lock.acquire(blocking=False):
            return "retry"
        speech_lock.release()
        v = holder["f"].claim(turn, stage)
        if v in ("not-yet", "busy"):
            return "retry"
        if v != "ok":
            return "skipped"
        plays.append((stage, clock.now))
        holder["f"].play_done()
        return "played"

    if late_fn is None and window is not None:
        late_fn = (lambda: window)
    f = pf.ProcessingFiller(
        play_fn=play_fn, suppressed_fn=lambda: None,
        delays_fn=lambda: (first, still), clock=clock, wait_fn=clock.wait,
        thread_factory=RecFactory(), first_late_s=first_late_s,
        late_fn=late_fn)
    holder["f"] = f
    return f, clock, plays, attempts


def _firsts(attempts):
    return [ts for s, ts in attempts if s == 1]


def _expected_attempts(first, window):
    """Stage-1 attempts with a capture that never ends: one at the delay,
    then every 0.5 s while the attempt is still inside the window."""
    n = int(math.floor(window / 0.5 + 1e-9))
    return [first + 0.5 * k for k in range(n + 1)]


class LateStartWindowTests(unittest.TestCase):
    WINDOWS = (0.6, 1.0, 1.5, 3.0)

    # Twin of test_stage_one_retry_is_bounded (default window 3.0).
    def test_stage_one_retry_is_bounded_by_the_window(self):
        for w in self.WINDOWS:
            with self.subTest(window=w):
                f, clock, plays, attempts = _make(window=w)
                t = f.arm()
                self.assertEqual(t.first_retry, w)
                t.owner = -1
                f.begin_capture()          # never ends
                f._run(t)
                self.assertEqual(plays, [])
                self.assertEqual(_firsts(attempts),
                                 _expected_attempts(2.5, w))
                self.assertLessEqual(max(_firsts(attempts)), 2.5 + w)

    # Twin of test_stage_one_retries_through_a_background_capture.
    def test_stage_one_retries_through_a_capture_inside_the_window(self):
        for w, end, want in ((0.6, 2.8, [(1, 3.0)]), (0.6, 4.0, []),
                             (1.0, 3.4, [(1, 3.5)]), (1.0, 4.0, []),
                             (1.5, 4.0, [(1, 4.0)]), (3.0, 4.0, [(1, 4.0)])):
            with self.subTest(window=w, capture_end=end):
                f, clock, plays, attempts = _make(window=w)
                t = f.arm()
                t.owner = -1
                clock.at(1.0, f.begin_capture)
                clock.at(end, f.end_capture)
                f._run(t)
                self.assertEqual(plays, want)
                for ts in _firsts(attempts):
                    self.assertLessEqual(ts, 2.5 + w + 1e-9)

    def test_a_window_of_0_6_never_starts_after_first_plus_0_6(self):
        # The plan's pick: a stage 1 due at 0.5 s never starts after 1.1 s,
        # whatever holds it off (here the thread itself woke 0.8 s late).
        logs: list = []
        f, clock, plays, _a = _make(first=0.5, window=0.6)
        f._log_fn = logs.append
        t = f.arm()
        t.t0 -= 0.8
        f._run(t)
        self.assertNotIn(1, [s for s, _ in plays])
        self.assertTrue(any("late" in m for m in logs), logs)

    # Twin of test_ack_state_retry_window_over_is_not_pending.
    def test_ack_state_pending_ends_with_the_window(self):
        for w in self.WINDOWS:
            with self.subTest(window=w):
                f, clock, _p, _a = _make(window=w)
                f.arm()
                clock.now = 2.5 + w - 0.1
                self.assertEqual(f.ack_state_here(), "pending")
                clock.now = 2.5 + w + 0.1
                self.assertEqual(f.ack_state_here(), "")

    def test_the_fixed_cap_still_wins_over_a_wider_window(self):
        # The monolith's shape: first_late_s=1.0. A window above 1.0 changes
        # nothing; one below it tightens the start.
        for w, cap in ((3.0, 1.0), (1.5, 1.0), (0.6, 0.6)):
            with self.subTest(window=w):
                f, clock, plays, attempts = _make(window=w, first_late_s=1.0)
                t = f.arm()
                t.owner = -1
                f.begin_capture()
                f._run(t)
                self.assertEqual(_firsts(attempts),
                                 _expected_attempts(2.5, cap))
                f2, c2, _p, _a = _make(window=w, first_late_s=1.0)
                f2.arm()
                c2.now = 2.5 + cap - 0.05
                self.assertEqual(f2.ack_state_here(), "pending")
                c2.now = 2.5 + cap + 0.05
                self.assertEqual(f2.ack_state_here(), "")


class LateStartDefaultTests(unittest.TestCase):
    """The shipped default (3.0) is today's scheduler, attempt for attempt."""

    def _trace(self, **kw):
        f, clock, plays, attempts = _make(**kw)
        t = f.arm()
        t.owner = -1
        clock.at(1.0, f.begin_capture)
        clock.at(4.6, f.end_capture)
        f._run(t)
        return plays, attempts

    def test_default_window_matches_no_late_fn(self):
        for cap in (None, 1.0):
            with self.subTest(first_late_s=cap):
                self.assertEqual(self._trace(first_late_s=cap),
                                 self._trace(first_late_s=cap, window=3.0))

    def test_no_late_fn_leaves_the_turn_unwindowed(self):
        f, _c, _p, _a = _make()
        t = f.arm()
        self.assertIsNone(t.first_retry)
        self.assertEqual(f._first_window(t), f._first_window())

    def test_bad_values_keep_the_default_window(self):
        def boom():
            raise RuntimeError("settings gone")
        for fn in (lambda: None, lambda: "soon", lambda: float("nan"),
                   lambda: True, boom):
            with self.subTest(fn=fn):
                f, _c, _p, _a = _make(late_fn=fn)
                t = f.arm()
                self.assertIsNotNone(t)
                self.assertIsNone(t.first_retry)

    def test_read_at_every_arm(self):
        box = [3.0]
        f, _c, _p, _a = _make(late_fn=lambda: box[0])
        self.assertEqual(f.arm().first_retry, 3.0)
        box[0] = 0.6
        self.assertEqual(f.arm().first_retry, 0.6)

    def test_sanitize_late_start(self):
        cases = ((0.6, 0.6), ("0.6", 0.6), (3, 3.0), (0, 0.1), (-2, 0.1),
                 (1e9, 60.0), (float("inf"), None), (float("nan"), None),
                 (None, None), (True, None), ("x", None))
        for raw, want in cases:
            with self.subTest(raw=raw):
                self.assertEqual(pf.sanitize_late_start(raw), want)


class PleasantryTests(unittest.TestCase):
    def test_pleasantries(self):
        for text in ("Thank you.", "thanks", "Thank you, JARVIS.",
                     "Thanks, sir.", "Hello.", "Hello, Jarvis!", "hi",
                     "Hi, sir.", "Okay.", "OK", "ok jarvis", "Good morning.",
                     "Jarvis, good morning, sir.", "Good night.", "Cool.",
                     "Great!", "  thank   you  "):
            with self.subTest(text=text):
                self.assertTrue(pf.is_pleasantry(text))

    def test_requests_are_not_pleasantries(self):
        for text in ("Thank you, what's the weather?", "thanks for that, "
                     "now play music", "Hello, what time is it?",
                     "Good morning, what's on my calendar?",
                     "great, turn the lights off", "okay play some music",
                     "not great", "thank you very much", "hi there",
                     "", "   ", "jarvis", "sir", "stop", "good",
                     "thank jarvis you", None, 42):
            with self.subTest(text=text):
                self.assertFalse(pf.is_pleasantry(text))

    def test_phrase_list(self):
        self.assertEqual(pf.PLEASANTRIES, frozenset({
            "thanks", "thank you", "hello", "hi", "okay", "ok",
            "good morning", "good night", "cool", "great"}))
        for name in ("is_pleasantry", "PLEASANTRIES", "sanitize_late_start"):
            self.assertIn(name, pf.__all__)

    def test_no_pleasantry_is_a_quiet_command(self):
        # The two gates never disagree about the same phrase.
        for p in pf.PLEASANTRIES:
            self.assertFalse(pf.is_quiet_command(p), p)


class OwnerThreadTests(unittest.TestCase):
    def test_only_the_arming_thread_owns_the_turn(self):
        f, _c, _p, _a = _make()
        self.assertFalse(f.is_owner_thread())
        t = f.arm()
        self.assertTrue(f.is_owner_thread())
        seen = []
        th = threading.Thread(target=lambda: seen.append(f.is_owner_thread()))
        th.start()
        th.join()
        self.assertEqual(seen, [False])
        f.disarm(t)
        self.assertFalse(f.is_owner_thread())

    def test_closed_filler_owns_nothing(self):
        f, _c, _p, _a = _make()
        f.arm()
        f.shutdown("restart")
        self.assertFalse(f.is_owner_thread())


class PlayPipelinedFirstRenderedTests(unittest.TestCase):
    """core.sentence_tts.play_pipelined(first_rendered=...): chunk 1 comes
    from the R3 pre-render. It is padded like any chunk, never rendered
    again, and everything else is unchanged."""
    CHUNKS = ["One.", "Two.", "Three."]

    def tearDown(self):
        _join_workers()

    def test_chunk_one_is_not_rendered_again(self):
        rec = _Rec()
        res = st.play_pipelined(self.CHUNKS, rec.synth, rec.play,
                                lambda: False,
                                first_rendered=(["One*"], 24000))
        rendered = [e[1] for e in rec.log if e[0] == "synth_start"]
        self.assertEqual(rendered, ["Two.", "Three."])
        self.assertEqual([a for a, _ in rec.played],
                         [["One*"], ["Two."], ["Three."]])
        self.assertEqual((res.plays, res.sentences_played), (3, 3))
        self.assertEqual(rec.max_active, 1)

    def test_chunk_one_is_padded_like_any_chunk(self):
        rec = _Rec()
        st.play_pipelined(self.CHUNKS, rec.synth, rec.play, lambda: False,
                          pad=lambda audio, sr: list(audio) + ["<gap>"],
                          first_rendered=(["One*"], 22050))
        self.assertEqual(rec.played[0], (["One*", "<gap>"], 22050))
        self.assertEqual(rec.played[-1], (["Three."], 24000))

    def test_on_first_play_still_runs_before_the_first_play(self):
        rec = _Rec()
        order = []
        st.play_pipelined(self.CHUNKS, rec.synth,
                          lambda a, sr: order.append(("play", tuple(a))),
                          lambda: False,
                          on_first_play=lambda: order.append("first"),
                          first_rendered=(["One*"], 24000))
        self.assertEqual(order[:2], ["first", ("play", ("One*",))])

    def test_a_stop_after_chunk_one_still_ends_the_reply(self):
        rec = _Rec()
        stop = [False]

        def play(audio, sr):
            rec.play(audio, sr)
            stop[0] = True
        res = st.play_pipelined(self.CHUNKS, rec.synth, play,
                                lambda: stop[0],
                                first_rendered=(["One*"], 24000))
        self.assertTrue(res.stopped)
        self.assertEqual([a for a, _ in rec.played], [["One*"]])

    def test_a_later_fallback_still_voices_the_rest_as_one_block(self):
        rec = _Rec()

        def synth(text):
            if text == "Two.":
                raise st.SentenceFallback("missed")
            return rec.synth(text)
        res = st.play_pipelined(self.CHUNKS, synth, rec.play, lambda: False,
                                synth_rest=lambda t: ([t], 24000),
                                first_rendered=(["One*"], 24000))
        self.assertTrue(res.fell_back)
        self.assertEqual([a for a, _ in rec.played],
                         [["One*"], ["Two. Three."]])

    def test_default_is_unchanged(self):
        import inspect
        sig = inspect.signature(st.play_pipelined)
        self.assertIsNone(sig.parameters["first_rendered"].default)
        rec = _Rec()
        st.play_pipelined(self.CHUNKS, rec.synth, rec.play, lambda: False)
        rendered = [e[1] for e in rec.log if e[0] == "synth_start"]
        self.assertEqual(rendered[0], "One.")


if __name__ == "__main__":   # pragma: no cover
    unittest.main()
