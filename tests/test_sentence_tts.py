"""core/sentence_tts.py -- the conservative sentence splitter and the pipelined
player (render sentence 1, play it while a worker renders the rest).

Pure module: no audio, no monolith. The monolith wiring (_speak, the mute
layers, the interrupt counter, ducking and lip-sync per chunk) is covered in
tests/monolith/test_monolith_sentence_tts.py.

    python tools/run_tests.py test_sentence_tts
"""
from __future__ import annotations

import threading
import time
import unittest

from core import sentence_tts as st


class SplitSentencesTableTests(unittest.TestCase):
    CASES = [
        # (text, expected sentences)
        ("It is 3.5 degrees outside. Then it will rain.",
         ["It is 3.5 degrees outside.", "Then it will rain."]),
        ("Pi is 3.14159 and e is 2.718. Both are irrational.",
         ["Pi is 3.14159 and e is 2.718.", "Both are irrational."]),
        ("Use a tool, e.g. A wrench. It works.",
         ["Use a tool, e.g. A wrench.", "It works."]),
        ("That is, i.e. The first one. Done.",
         ["That is, i.e. The first one.", "Done."]),
        ("Mr. Smith lives on Baker St. Near the park. He is out.",
         ["Mr. Smith lives on Baker St. Near the park.", "He is out."]),
        ("Dr. Jones and Mrs. Lee arrived. Welcome them.",
         ["Dr. Jones and Mrs. Lee arrived.", "Welcome them."]),
        ("The meeting is at 2:30 p.m. Tomorrow works too.",
         ["The meeting is at 2:30 p.m. Tomorrow works too."]),
        ("Wake me at 7 a.m. Please. Thanks.",
         ["Wake me at 7 a.m. Please.", "Thanks."]),
        ("Version v2.0.115 shipped. 42 tests passed.",
         ["Version v2.0.115 shipped.", "42 tests passed."]),
        ("He said \"Hello.\" Then he left.",
         ["He said \"Hello.\"", "Then he left."]),
        ("Done. \"Next,\" she said.",
         ["Done.", "\"Next,\" she said."]),
        ("Well... I suppose so. Right.",
         ["Well... I suppose so.", "Right."]),
        ("Wait... Then go ahead.",
         ["Wait... Then go ahead."]),
        ("Hmm\u2026 Maybe later.",
         ["Hmm\u2026 Maybe later."]),
        ("Really?! Yes. Absolutely!",
         ["Really?!", "Yes.", "Absolutely!"]),
        ("J. R. R. Tolkien wrote it. Read it.",
         ["J. R. R. Tolkien wrote it.", "Read it."]),
        ("Item No. 5 is next. No. That one is wrong.",
         ["Item No. 5 is next.", "No.", "That one is wrong."]),
        ("It is made in the U.S. By hand.",
         ["It is made in the U.S. By hand."]),
        ("apples, pears, etc. And more.",
         ["apples, pears, etc. And more."]),
        ("lowercase after a dot. stays joined.",
         ["lowercase after a dot. stays joined."]),
        ("A single sentence with no terminator",
         ["A single sentence with no terminator"]),
        ("One sentence.", ["One sentence."]),
        # A bare list number opening a piece is not a sentence of its own.
        ("1. Open the lid. 2. Press the button.",
         ["1. Open the lid.", "2. Press the button."]),
        ("Steps: 1. Open it. 2. Close it.",
         ["Steps: 1. Open it.", "2. Close it."]),
        ("I counted 12. Then I stopped.",
         ["I counted 12.", "Then I stopped."]),
        ("", []),
        ("   ", []),
    ]

    def test_table(self):
        for text, want in self.CASES:
            with self.subTest(text=text):
                self.assertEqual(st.split_sentences(text), want)

    def test_pieces_rejoin_to_the_input(self):
        for text, _ in self.CASES:
            if not text.strip():
                continue
            with self.subTest(text=text):
                self.assertEqual(" ".join(st.split_sentences(text)),
                                 " ".join(text.split()))


class PlanChunksTests(unittest.TestCase):
    LONG = ("The forecast calls for light rain this afternoon. "
            "Temperatures will stay near 61 degrees. "
            "Bring an umbrella if you head out, sir.")

    def test_long_multi_sentence_splits(self):
        self.assertGreaterEqual(len(self.LONG), st.MIN_CHARS)
        self.assertEqual(st.plan_chunks(self.LONG), [
            "The forecast calls for light rain this afternoon.",
            "Temperatures will stay near 61 degrees.",
            "Bring an umbrella if you head out, sir."])

    def test_short_text_is_one_chunk(self):
        short = "Done. It is set. Anything else?"
        self.assertLess(len(short), st.MIN_CHARS)
        self.assertEqual(st.plan_chunks(short), [short])

    def test_long_single_sentence_is_one_chunk(self):
        one = ("This is one long sentence that keeps going with commas, "
               "clauses, a 3.5 inch measurement and a 2:30 p.m. meeting but "
               "never actually ends")
        self.assertGreaterEqual(len(one), st.MIN_CHARS)
        self.assertEqual(st.plan_chunks(one), [one])

    def test_min_chars_threshold_is_about_120(self):
        self.assertEqual(st.MIN_CHARS, 120)

    def test_threshold_boundary(self):
        base = "Short first sentence here. " + "Y" * 200
        at = base[:st.MIN_CHARS - 1] + "."          # exactly MIN_CHARS long
        below = base[:st.MIN_CHARS - 2] + "."       # one character shorter
        self.assertEqual(len(at), st.MIN_CHARS)
        self.assertEqual(len(below), st.MIN_CHARS - 1)
        self.assertEqual(st.plan_chunks(below), [below])
        self.assertEqual(len(st.plan_chunks(at)), 2)

    def test_blank(self):
        self.assertEqual(st.plan_chunks("  "), [])


class _Rec:
    """Fake synth/play with a shared, ordered log. `synth` returns a list of
    the sentence index (the 'audio'); `play` records starts/ends and tracks
    how many plays run at once."""

    def __init__(self, sr=24000, sr_for=None, synth_delay=0.0):
        self.log = []
        self.lock = threading.Lock()
        self.sr = sr
        self.sr_for = sr_for or {}
        self.synth_delay = synth_delay
        self.active = 0
        self.max_active = 0
        self.played = []
        self.first_play_started = threading.Event()

    def _add(self, *e):
        with self.lock:
            self.log.append(e)

    def synth(self, text):
        self._add("synth_start", text)
        if self.synth_delay:
            time.sleep(self.synth_delay)
        self._add("synth_end", text)
        return [text], self.sr_for.get(text, self.sr)

    def play(self, audio, sr):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        self._add("play_start", tuple(audio))
        self.first_play_started.set()
        time.sleep(0.02)
        self.played.append((list(audio), sr))
        self._add("play_end", tuple(audio))
        with self.lock:
            self.active -= 1


def _concat(parts):
    out = []
    for p in parts:
        out.extend(p)
    return out


def _worker_threads():
    return [t for t in threading.enumerate()
            if t.name == "sentence-tts-synth"]


def _join_workers(timeout=2.0):
    for t in _worker_threads():
        t.join(timeout)


class PlayPipelinedTests(unittest.TestCase):
    CHUNKS = ["One.", "Two.", "Three."]

    def tearDown(self):
        _join_workers()

    def test_first_play_starts_before_the_second_sentence_is_rendered(self):
        rec = _Rec()
        seen = {}

        def synth(text):
            if text != "One.":
                # Renders of the rest wait for sentence 1's play to begin: a
                # player that rendered everything first would time out here.
                seen[text] = rec.first_play_started.wait(timeout=2.0)
            return rec.synth(text)

        res = st.play_pipelined(self.CHUNKS, synth, rec.play, lambda: False)
        self.assertEqual(seen, {"Two.": True, "Three.": True})
        first_play = rec.log.index(("play_start", ("One.",)))
        two_done = rec.log.index(("synth_end", "Two."))
        self.assertLess(first_play, two_done, rec.log)
        self.assertFalse(res.stopped)
        self.assertEqual(res.sentences_played, 3)

    def test_the_rest_renders_while_sentence_one_plays(self):
        # The headline property: sentences 2..n are rendered DURING sentence
        # 1's play, not after it. Sentence 1's play waits (bounded) for both
        # renders; a player that only starts the worker after play 1 returns
        # never lets them finish in time.
        rec = _Rec()

        def play(audio, sr):
            if list(audio) == ["One."]:
                deadline = time.time() + 2.0
                while time.time() < deadline and sum(
                        1 for e in rec.log if e[0] == "synth_end") < 3:
                    time.sleep(0.005)
            rec.play(audio, sr)

        res = st.play_pipelined(self.CHUNKS, rec.synth, play, lambda: False)
        one_end = rec.log.index(("play_end", ("One.",)))
        self.assertLess(rec.log.index(("synth_end", "Two.")), one_end,
                        rec.log)
        self.assertLess(rec.log.index(("synth_end", "Three.")), one_end,
                        rec.log)
        # Rendered ahead or not, every chunk is its own play, in order.
        self.assertEqual([a for a, _ in rec.played],
                         [["One."], ["Two."], ["Three."]])
        self.assertEqual((res.plays, res.sentences_played), (3, 3))

    def test_total_audio_is_the_concatenation_and_plays_never_overlap(self):
        rec = _Rec(synth_delay=0.01)
        st.play_pipelined(self.CHUNKS, rec.synth, rec.play, lambda: False)
        self.assertEqual(_concat(a for a, _ in rec.played),
                         ["One.", "Two.", "Three."])
        self.assertEqual(rec.max_active, 1)
        self.assertEqual(rec.played[0], (["One."], 24000))

    def test_one_play_per_chunk_even_when_everything_is_ready(self):
        rec = _Rec()
        gate = threading.Event()

        def play(audio, sr):
            if list(audio) == ["One."]:
                gate.wait(0.3)          # let the worker finish everything
            rec.play(audio, sr)

        res = st.play_pipelined(self.CHUNKS, rec.synth, play, lambda: False)
        self.assertEqual([a for a, _ in rec.played],
                         [["One."], ["Two."], ["Three."]])
        self.assertEqual(res.plays, 3)

    def test_different_rates_play_at_their_own_rate(self):
        rec = _Rec(sr_for={"Two.": 22050})
        st.play_pipelined(self.CHUNKS, rec.synth, rec.play, lambda: False)
        self.assertEqual([sr for _, sr in rec.played],
                         [24000, 22050, 24000])

    def test_pad_follows_every_chunk_but_the_last(self):
        rec = _Rec()
        st.play_pipelined(self.CHUNKS, rec.synth, rec.play, lambda: False,
                          pad=lambda audio, sr: list(audio) + ["<gap>"])
        self.assertEqual([a for a, _ in rec.played],
                         [["One.", "<gap>"], ["Two.", "<gap>"], ["Three."]])

    def test_stop_during_chunk_one_means_chunk_two_never_plays(self):
        rec = _Rec()
        stopped = [False]

        def play(audio, sr):
            rec.play(audio, sr)
            stopped[0] = True          # an interrupt landed during this play

        res = st.play_pipelined(self.CHUNKS, rec.synth, play,
                                lambda: stopped[0])
        self.assertEqual([a for a, _ in rec.played], [["One."]])
        self.assertTrue(res.stopped)
        self.assertEqual(res.sentences_played, 1)

    def test_stop_tells_the_worker_to_render_nothing_more(self):
        # Deterministic: sentence S2 is held in its render until the player
        # has returned from the stop. Once it is released, the worker must
        # not START another render: S0 (played), S1 and S2 (in flight) are
        # the only renders that may ever begin.
        many = [f"S{i}." for i in range(30)]
        rec = _Rec()
        release = threading.Event()
        stopped = [False]

        def synth(text):
            if text == "S2.":
                release.wait(timeout=2.0)
            return rec.synth(text)

        def play(audio, sr):
            rec.play(audio, sr)
            stopped[0] = True

        res = st.play_pipelined(many, synth, play, lambda: stopped[0])
        self.assertTrue(res.stopped)
        release.set()
        _join_workers()
        self.assertEqual(_worker_threads(), [])
        started = [e[1] for e in rec.log if e[0] == "synth_start"]
        self.assertLessEqual(len(started), 3, started)
        self.assertEqual([a for a, _ in rec.played], [["S0."]])

    def test_a_stop_while_waiting_for_a_wedged_render_returns_at_once(self):
        # The wait for the next render watches the stop too: a STOP that
        # lands while sentence 2 is wedged must not hold the speaker until
        # the render (or wait_timeout) finishes.
        rec = _Rec()
        release = threading.Event()
        stopped = [False]

        def synth(text):
            if text == "Two.":
                release.wait(timeout=5.0)
            return rec.synth(text)

        def play(audio, sr):
            rec.play(audio, sr)
            stopped[0] = True

        t0 = time.monotonic()
        try:
            res = st.play_pipelined(self.CHUNKS, synth, play,
                                    lambda: stopped[0], wait_timeout=5.0)
        finally:
            release.set()
        self.assertLess(time.monotonic() - t0, 0.5)
        self.assertTrue(res.stopped)
        self.assertEqual([a for a, _ in rec.played], [["One."]])

    def test_a_stop_that_lands_during_the_wait_is_seen_mid_wait(self):
        # Stop turns True some time AFTER play 1 returned, while the player
        # is already waiting on sentence 2's (slow) render.
        rec = _Rec()
        release = threading.Event()
        stopped = [False]

        def synth(text):
            if text == "Two.":
                release.wait(timeout=5.0)
            return rec.synth(text)

        def flip():
            time.sleep(0.1)
            stopped[0] = True

        def play(audio, sr):
            rec.play(audio, sr)
            if list(audio) == ["One."]:
                threading.Thread(target=flip, daemon=True).start()

        t0 = time.monotonic()
        try:
            res = st.play_pipelined(self.CHUNKS, synth, play,
                                    lambda: stopped[0], wait_timeout=5.0)
        finally:
            release.set()
        self.assertLess(time.monotonic() - t0, 1.0)
        self.assertTrue(res.stopped)
        self.assertEqual([a for a, _ in rec.played], [["One."]])

    def test_a_stop_raised_during_a_render_prevents_that_play(self):
        rec = _Rec()
        stopped = [False]

        def synth(text):
            out = rec.synth(text)
            if text == "Two.":
                stopped[0] = True      # a mute / STOP lands mid-render
            return out

        res = st.play_pipelined(self.CHUNKS, synth, rec.play,
                                lambda: stopped[0])
        self.assertTrue(res.stopped)
        self.assertEqual([a for a, _ in rec.played], [["One."]])

    def test_on_first_play_runs_after_sentence_one_renders_before_it_plays(self):
        rec = _Rec()
        st.play_pipelined(self.CHUNKS, rec.synth, rec.play, lambda: False,
                          on_first_play=lambda: rec._add("first_play"))
        log = rec.log
        self.assertLess(log.index(("synth_end", "One.")),
                        log.index(("first_play",)))
        self.assertLess(log.index(("first_play",)),
                        log.index(("play_start", ("One.",))))
        self.assertEqual(log.count(("first_play",)), 1)

    def test_first_render_error_raises_and_nothing_plays(self):
        rec = _Rec()

        def synth(text):
            raise RuntimeError("engine gone")

        with self.assertRaises(RuntimeError):
            st.play_pipelined(self.CHUNKS, synth, rec.play, lambda: False)
        self.assertEqual(rec.played, [])

    def test_first_play_error_raises(self):
        rec = _Rec()

        def play(audio, sr):
            raise RuntimeError("device gone")

        with self.assertRaises(RuntimeError):
            st.play_pipelined(self.CHUNKS, rec.synth, play, lambda: False)

    def test_a_later_render_error_ends_the_reply_as_partly_spoken(self):
        # Sentences already heard stay heard: the error is reported on the
        # result, not raised, so the caller does not count the line unspoken
        # (and a streaming flush does not say the opening sentence again).
        rec = _Rec()

        def synth(text):
            if text == "Three.":
                raise RuntimeError("render failed")
            return rec.synth(text)

        res = st.play_pipelined(self.CHUNKS, synth, rec.play, lambda: False)
        self.assertEqual(_concat(a for a, _ in rec.played), ["One.", "Two."])
        self.assertIsInstance(res.error, RuntimeError)
        self.assertEqual(res.sentences_played, 2)

    def test_a_later_play_error_ends_the_reply_as_partly_spoken(self):
        rec = _Rec()

        def play(audio, sr):
            if list(audio) == ["Two."]:
                raise RuntimeError("PortAudio reinit hung")
            rec.play(audio, sr)

        res = st.play_pipelined(self.CHUNKS, rec.synth, play, lambda: False)
        self.assertEqual([a for a, _ in rec.played], [["One."]])
        self.assertIsInstance(res.error, RuntimeError)
        self.assertEqual((res.plays, res.sentences_played), (1, 1))

    def test_a_wedged_render_times_out_instead_of_holding_the_speaker(self):
        rec = _Rec()
        release = threading.Event()

        def synth(text):
            if text != "One.":
                release.wait(timeout=2.0)
            return rec.synth(text)

        try:
            res = st.play_pipelined(self.CHUNKS, synth, rec.play,
                                    lambda: False, wait_timeout=0.1)
        finally:
            release.set()
        self.assertIsInstance(res.error, TimeoutError)
        self.assertEqual([a for a, _ in rec.played], [["One."]])

    def test_a_fallback_voices_the_rest_as_one_block(self):
        # A sentence the per-sentence engine missed (Kokoro timed out or
        # returned nothing) switches the REST of the reply to one block on
        # the fallback path: one voice, one fallback render, no further
        # per-sentence attempts.
        rec = _Rec()
        rest_calls = []

        def synth(text):
            if text == "Two.":
                raise st.SentenceFallback("kokoro missed")
            return rec.synth(text)

        def synth_rest(text):
            rest_calls.append(text)
            return ["REST:" + text], 24000

        res = st.play_pipelined(self.CHUNKS, synth, rec.play, lambda: False,
                                synth_rest=synth_rest,
                                pad=lambda a, sr: list(a) + ["<gap>"])
        self.assertEqual(rest_calls, ["Two. Three."])
        self.assertNotIn(("synth_start", "Three."), rec.log)
        self.assertEqual([a for a, _ in rec.played],
                         [["One.", "<gap>"], ["REST:Two. Three."]])
        self.assertTrue(res.fell_back)
        self.assertEqual((res.plays, res.sentences_played), (2, 3))

    def test_a_fallback_on_sentence_one_voices_the_whole_reply_once(self):
        rec = _Rec()
        rest_calls = []

        def synth(text):
            rec._add("synth_start", text)
            raise st.SentenceFallback("kokoro missed")

        def synth_rest(text):
            rest_calls.append(text)
            return ["WHOLE"], 24000

        res = st.play_pipelined(self.CHUNKS, synth, rec.play, lambda: False,
                                synth_rest=synth_rest)
        self.assertEqual(rest_calls, ["One. Two. Three."])
        self.assertEqual([e for e in rec.log if e[0] == "synth_start"],
                         [("synth_start", "One.")])
        self.assertEqual([a for a, _ in rec.played], [["WHOLE"]])
        self.assertEqual(res.sentences_played, 3)

    def test_single_chunk_is_one_synth_and_one_play(self):
        rec = _Rec()
        res = st.play_pipelined(["Only."], rec.synth, rec.play,
                                lambda: False)
        self.assertEqual([e for e in rec.log if e[0] == "synth_start"],
                         [("synth_start", "Only.")])
        self.assertEqual((res.plays, res.sentences_played), (1, 1))

    def test_no_chunks_does_nothing(self):
        rec = _Rec()
        res = st.play_pipelined([], rec.synth, rec.play, lambda: False)
        self.assertEqual((rec.log, res.plays), ([], 0))


# ════════════════════════════════════════════════════════════════════════════
#  The clone voice server's first-line clause split (2026-10-04)
# ════════════════════════════════════════════════════════════════════════════
class ClauseSplitTests(unittest.TestCase):
    SPLITS = [
        # (line, expected head) -- None: no split
        ("While you were away, sir: I tried to check your account credits "
         "but the console is asking for a login.",
         "While you were away, sir:"),
        ("The printer finished the job; the plate is cooling and the part "
         "should be ready to remove soon.",
         "The printer finished the job;"),
        ("Good evening, sir, the forecast calls for light rain this "
         "afternoon and temperatures near sixty.",
         "Good evening, sir,"),
        ("The forecast is mixed — light rain this afternoon and then "
         "clearing skies by the evening hours, sir.",
         "The forecast is mixed —"),
        ("The forecast is mixed - light rain this afternoon and then "
         "clearing skies by the evening hours, sir.",
         "The forecast is mixed -"),
        # Never inside a number or a time.
        ("The total came to 1,500 dollars and 20 cents after the refund was "
         "applied to the account this morning.", None),
        ("Pick any of 3, 4 or 5 for the shelf height and the bracket will "
         "still fit the frame without trouble.", None),
        ("The meeting moved to 2:30 this afternoon because the conference "
         "room was booked by the facilities team.", None),
        # Never inside a web address (a colon with no space after it).
        ("Open the page at https://example.org/status and check that the "
         "green light is on before you leave.", None),
        # Never right after an abbreviation.
        ("Your alarm is set for 7 a.m., and the coffee maker will start ten "
         "minutes before it rings, sir.", None),
        # Never inside quotes or brackets.
        ('He said "stop, wait, and listen to the whole thing before you '
         'decide" and then he left the room.', None),
        ("The plan (the long one, with the extra stops) still gets us home "
         "well before the late show starts.", None),
        # Never a hyphen inside a word.
        ("A well-known and long-standing tradition continues tonight in the "
         "square near the old town hall.", None),
        # A head that would be too short ("Sir,") is no split.
        ("Sir, the forecast calls for light rain this afternoon and much "
         "cooler temperatures overnight.", None),
        # Short lines are never split.
        ("Short line, sir: all good.", None),
        ("Right away, sir, the lamp is on.", None),
    ]

    def test_table(self):
        for line, head in self.SPLITS:
            with self.subTest(line=line):
                got = st.split_first_clause(line)
                self.assertEqual(None if got is None else got[0], head)

    def test_pieces_rejoin_to_the_line(self):
        for line, head in self.SPLITS:
            got = st.split_first_clause(line)
            if got is not None:
                self.assertEqual(" ".join(got), line)

    def test_never_ends_a_head_on_an_abbreviation(self):
        line = ("Bring the usual gear, e.g., a jacket, gloves and an umbrella "
                "for the long walk home tonight, sir.")
        head, rest = st.split_first_clause(line)
        self.assertFalse(head.endswith("e.g.,"), head)
        self.assertEqual(head, "Bring the usual gear, e.g., a jacket,")

    def test_the_head_is_short_and_the_rest_is_not(self):
        for line, head in self.SPLITS:
            got = st.split_first_clause(line)
            if got is None:
                continue
            h, r = got
            self.assertGreaterEqual(len(h), st.HEAD_MIN_CHARS)
            self.assertLessEqual(len(h), st.HEAD_MAX_FRAC * len(line))
            self.assertGreaterEqual(len(r), st.REST_MIN_CHARS)

    def test_threshold(self):
        self.assertEqual(st.CLAUSE_SPLIT_MIN_CHARS, 70)
        base = "The garage door is closed, and the porch light is on"
        line = base + " " + "x" * (70 - len(base) - 2) + "."
        self.assertEqual(len(line), 70)
        self.assertIsNone(st.split_first_clause(line))          # 70: whole
        self.assertEqual(st.split_first_clause(line[:-1] + "x.")[0],
                         "The garage door is closed,")         # 71: split


class PlanCloneChunksTests(unittest.TestCase):
    LONG_FIRST = ("Good evening, sir, the forecast calls for light rain this "
                  "afternoon and temperatures near sixty.")

    def test_a_long_first_sentence_becomes_a_head_and_a_tail(self):
        text = self.LONG_FIRST + " Bring an umbrella. The roads are clear."
        chunks = st.plan_clone_chunks(text)
        self.assertEqual(chunks, [
            "Good evening, sir,",
            "the forecast calls for light rain this afternoon and "
            "temperatures near sixty.",
            "Bring an umbrella.", "The roads are clear."])
        head, tail = chunks[0], chunks[1]
        self.assertIsInstance(head, st.Chunk)
        self.assertEqual((head.clause, head.gap_s, head.budget_chars),
                         ("head", st.CLAUSE_GAP_S, None))
        self.assertEqual((tail.clause, tail.gap_s, tail.budget_chars),
                         ("tail", None, len(self.LONG_FIRST)))
        self.assertLess(st.CLAUSE_GAP_S, st.SENTENCE_GAP_S)
        # Only the first line is ever split; the rest are plain sentences.
        self.assertNotIsInstance(chunks[2], st.Chunk)

    def test_a_long_single_sentence_reply_is_split(self):
        chunks = st.plan_clone_chunks(self.LONG_FIRST)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(" ".join(chunks), self.LONG_FIRST)
        # Kokoro's plan is unchanged: one chunk.
        self.assertEqual(st.plan_chunks(self.LONG_FIRST), [self.LONG_FIRST])

    def test_a_short_reply_of_several_sentences_goes_sentence_by_sentence(self):
        text = ("Good morning, sir. It is currently half past eight, and the "
                "sky is overcast.")
        self.assertGreater(len(text), st.CLAUSE_SPLIT_MIN_CHARS)
        self.assertLess(len(text), st.MIN_CHARS)
        self.assertEqual(st.plan_chunks(text), [text])        # Kokoro
        self.assertEqual(st.plan_clone_chunks(text), [
            "Good morning, sir.",
            "It is currently half past eight, and the sky is overcast."])

    def test_short_or_unsplittable_lines_are_unchanged(self):
        for text in ("Right away, sir.",
                     "Right away, sir, the lamp is on and the door is shut.",
                     "A well-known and long-standing tradition continues "
                     "tonight in the square near the old town hall."):
            self.assertEqual(st.plan_clone_chunks(text), st.plan_chunks(text))
        self.assertEqual(st.plan_clone_chunks("  "), [])

    def test_a_chunk_is_a_plain_string_to_everything_else(self):
        c = st.Chunk("Good evening, sir,", gap_s=0.05, clause="head")
        self.assertEqual(c, "Good evening, sir,")
        self.assertEqual(hash(c), hash("Good evening, sir,"))
        self.assertIs(type(" ".join([c, "x"])), str)
        self.assertIs(type(c.strip()), str)


# ════════════════════════════════════════════════════════════════════════════
#  needed_by(): when a chunk rendered ahead will be played (2026-10-04)
# ════════════════════════════════════════════════════════════════════════════
class NeededByTests(unittest.TestCase):
    """Audio is a list of N samples at SR samples/s, so its length is
    N / SR seconds, and play blocks for that long as the speaker does."""

    SR = 100

    def tearDown(self):
        _join_workers()

    def _run(self, chunks, samples, render_s=0.0, pad=None, **kw):
        seen = {}
        plays = []

        def synth(text):
            seen[str(text)] = (st.needed_by(), time.monotonic())
            if render_s:
                time.sleep(render_s)
            return ["x"] * samples[str(text)], self.SR

        def play(audio, sr):
            plays.append((time.monotonic(), len(audio) / sr))
            time.sleep(len(audio) / sr)

        st.play_pipelined(chunks, synth, play, lambda: False, pad=pad, **kw)
        return seen, plays

    def test_outside_play_pipelined_it_is_none(self):
        self.assertIsNone(st.needed_by())

    def test_the_first_chunk_has_none_and_the_rest_count_the_queue(self):
        seen, plays = self._run(["A.", "B.", "C."],
                                {"A.": 40, "B.": 30, "C.": 10})
        self.assertIsNone(seen["A."][0])
        t_play1 = plays[0][0]
        # B is needed when A (0.4 s) has played; C when A and B have.
        self.assertAlmostEqual(seen["B."][0], t_play1 + 0.4, delta=0.08)
        self.assertAlmostEqual(seen["C."][0], t_play1 + 0.7, delta=0.08)
        # ... and they really were played about then.
        self.assertAlmostEqual(plays[1][0], seen["B."][0], delta=0.1)
        self.assertAlmostEqual(plays[2][0], seen["C."][0], delta=0.1)
        self.assertIsNone(st.needed_by())      # cleared after every render

    def test_padding_counts(self):
        seen, plays = self._run(["A.", "B."], {"A.": 20, "B.": 10},
                                pad=lambda a, sr: list(a) + ["-"] * 30)
        self.assertAlmostEqual(seen["B."][0], plays[0][0] + 0.5, delta=0.08)

    def test_a_late_render_is_needed_now(self):
        # Rendering slower than playback: the queue runs dry, so the next
        # chunk is needed at once (no slack to wait for).
        seen, plays = self._run(["A.", "B.", "C."],
                                {"A.": 5, "B.": 5, "C.": 5}, render_s=0.2)
        self.assertLess(seen["C."][0] - seen["C."][1], 0.06)

    def test_a_prerendered_first_chunk_counts(self):
        seen = {}

        def synth(text):
            seen[text] = st.needed_by()
            return ["x"] * 10, self.SR
        t0 = time.monotonic()
        st.play_pipelined(["A.", "B."], synth, lambda a, sr: time.sleep(0.6),
                          lambda: False, first_rendered=(["x"] * 60, self.SR))
        self.assertNotIn("A.", seen)
        self.assertAlmostEqual(seen["B."], t0 + 0.6, delta=0.08)

    def test_a_clause_head_is_padded_with_its_own_gap(self):
        calls = []

        def pad(audio, sr, gap_s=None):
            calls.append(gap_s)
            return list(audio) + ["-"]
        head = st.Chunk("Good evening, sir,", gap_s=0.05, clause="head")
        tail = st.Chunk("the rest.", clause="tail")
        rec = _Rec()
        st.play_pipelined([head, tail, "Next."], rec.synth, rec.play,
                          lambda: False, pad=pad)
        self.assertEqual(calls, [0.05, None])
        # A pre-rendered head gets its gap too.
        calls.clear()
        st.play_pipelined([head, tail], rec.synth, rec.play, lambda: False,
                          pad=pad, first_rendered=(["H"], 24000))
        self.assertEqual(calls, [0.05])


if __name__ == "__main__":
    unittest.main()
