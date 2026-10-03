"""Monolith wiring for per-sentence speech (SENTENCE_TTS_ENABLED, Kokoro).

`_speak` used to synthesise the whole reply and only then play it. With the
Kokoro backend it now renders sentence 1, starts playing it, and renders the
rest while it plays (core/sentence_tts.py). These tests drive the REAL
`_speak` (and, where it matters, the real `play_with_lipsync` with a fake
`sd`) and check:

  * the first sentence plays before the second is rendered, and the rest
    renders WHILE sentence 1 plays; one play per sentence, each followed by
    the fixed sentence gap except the last; plays never overlap;
  * an interrupt during sentence 1 (request_tts_interrupt, whose Event the
    play's finally clears) means sentence 2 never plays -- and so does one
    that lands BETWEEN sentences, while the next one is still rendering;
  * a failure after sentence 1 was heard counts the line as spoken, so a
    streaming flush never says the opening sentence again;
  * a Kokoro miss voices the rest of the reply as one fallback block;
  * tray mute and env mute (MUTE_TTS) -> nothing reaches sd.play, as before;
  * SENTENCE_TTS_ENABLED=False, a non-Kokoro backend, the voice clone, an
    unavailable Kokoro, a short reply and a 'wry' reply all keep today's single
    synthesise + play of the whole text;
  * lip-sync values and the music duck run per chunk, in order, with the duck
    held across the reply (one fade down, one restore);
  * the prosody preset is resolved ONCE for the reply and every sentence
    renders at that preset's speed;
  * [turn-timing] first_play marks sentence 1's play.

No real audio, no real Kokoro, no LLM.

    python -m unittest tests.monolith.test_monolith_sentence_tts
"""
from __future__ import annotations

import threading
import unittest
from unittest import mock

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith

S1 = "The forecast calls for light rain this afternoon."
S2 = "Temperatures will stay near 61 degrees."
S3 = "Bring an umbrella if you head out, sir."
REPLY = f"{S1} {S2} {S3}"
SENTENCES = [S1, S2, S3]
NEUTRAL = ("neutral", {"rate": "+0%", "pitch": "+0Hz", "gain": 1.0})


@requires_monolith
class _Base(MonolithGlobalsTestCase):
    def _p(self, *args, **kwargs):
        patcher = mock.patch.object(*args, **kwargs)
        m = patcher.start()
        self.addCleanup(patcher.stop)
        return m

    def setUp(self):
        super().setUp()
        bc = self.bc
        import numpy as np
        self.np = np
        from core import kokoro_tts
        self._p(bc, "set_state")
        self._p(bc, "_write_hud_state")
        self._p(bc, "_processing_filler", mock.Mock())
        self._p(bc, "_session_start_time", bc.time.time() - 3600)
        self._p(bc, "_tts_layer", None)
        self._p(bc, "_is_staging", lambda: False)
        self._p(bc, "TTS_BACKEND", "kokoro", create=True)
        self._p(bc, "VOICE_CLONE_ENABLED", False, create=True)
        # The in-process clone engine whatever this box's settings select.
        self._p(bc, "VOICE_CLONE_MODEL", "chatterbox", create=True)
        self._p(bc, "SENTENCE_TTS_ENABLED", True, create=True)
        self._p(bc, "BARGE_IN_ENABLED", False)
        self._p(kokoro_tts, "is_available", return_value=True)
        self.resolve = self._p(bc, "_resolve_tts_preset",
                               return_value=NEUTRAL)
        bc._tts_muted[0] = False
        seq = bc._tts_interrupt_seq[0]
        self.addCleanup(bc._tts_interrupt_seq.__setitem__, 0, seq)
        self.addCleanup(bc._tts_interrupt.clear)
        self.addCleanup(bc._tts_playback_active.__setitem__, 0, False)
        if hasattr(bc, "_tts_reply_active"):
            self.addCleanup(bc._tts_reply_active.__setitem__, 0, False)
        self._p(bc, "_barge_in_interrupted", False)
        self.lock = threading.Lock()
        self.log = []
        self.played = []
        self.active = 0
        self.max_active = 0
        self.first_play_started = threading.Event()
        self.synth_waits = {}

    # -- fakes --------------------------------------------------------------
    def _audio_for(self, text):
        """Sentence i renders to 800 samples of amplitude 0.01*(i+1); any
        other text (the whole reply) to 0.5."""
        amp = 0.01 * (SENTENCES.index(text) + 1) if text in SENTENCES else 0.5
        return self.np.full(800, amp, dtype=self.np.float32)

    def _rec(self, *e):
        with self.lock:
            self.log.append(e)

    def fake_synth(self, text):
        self._rec("pin", getattr(self.bc._TTS_PRESET_PIN, "value", None))
        if text in SENTENCES[1:]:
            # The rest may only finish rendering once sentence 1 plays: a
            # player that renders everything first times out here (False).
            self.synth_waits[text] = self.first_play_started.wait(timeout=2.0)
        self._rec("synth", text)
        return self._audio_for(text), 24000

    def fake_play(self, audio, sr):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        self._rec("play", round(float(audio[0]), 4), len(audio))
        self.first_play_started.set()
        self.played.append((audio.copy(), sr))
        with self.lock:
            self.active -= 1

    def use_fakes(self):
        bc = self.bc
        self.synth = self._p(bc, "synthesise", side_effect=self.fake_synth)
        self.play = self._p(bc, "play_with_lipsync",
                            side_effect=self.fake_play)
        self.sentences_spy = self._p(bc, "_speak_sentences",
                                     wraps=bc._speak_sentences)

    def speak(self, text=REPLY, **kw):
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()) as out:
            ok = self.bc._speak(text, **kw)
        self.out = out.getvalue()
        return ok

    def synth_texts(self):
        return [e[1] for e in self.log if e[0] == "synth"]

    def gap_samples(self, sr=24000):
        from core import sentence_tts
        return int(sr * sentence_tts.SENTENCE_GAP_S)

    def expected_plays(self, texts):
        """One array per sentence: its audio, then the sentence gap's zeros
        (every sentence but the last)."""
        np = self.np
        out = []
        for i, t in enumerate(texts):
            a = self._audio_for(t)
            if i < len(texts) - 1:
                a = np.concatenate([a, np.zeros(self.gap_samples(),
                                                dtype=np.float32)])
            out.append(a)
        return out

    def assert_plays_equal(self, want, scale=1.0):
        self.assertEqual(len(self.played), len(want))
        for (got, sr), w in zip(self.played, want):
            self.assertEqual(sr, 24000)
            self.assertEqual(len(got), len(w))
            self.assertTrue(self.np.allclose(got, w * scale))

    def assert_whole_text_path(self, text=REPLY):
        self.assertEqual(self.synth_texts(), [text])
        self.assertEqual(self.play.call_count, 1)
        self.assertEqual(float(self.played[0][0][0]), 0.5)
        # The whole-text path proper: never the per-sentence player, and
        # synthesise resolves its own preset (no pin).
        if hasattr(self, "sentences_spy"):
            self.sentences_spy.assert_not_called()
        self.assertEqual([e[1] for e in self.log if e[0] == "pin"], [None])


class PipelinedSpeakTests(_Base):
    def setUp(self):
        super().setUp()
        self.use_fakes()

    def test_first_sentence_plays_before_the_second_is_rendered(self):
        self.assertTrue(self.speak())
        self.assertEqual(self.synth_texts()[0], S1)
        self.assertEqual(sorted(self.synth_texts()), sorted(SENTENCES))
        self.assertEqual(self.synth_waits, {S2: True, S3: True})
        first_play = next(i for i, e in enumerate(self.log) if e[0] == "play")
        s2_done = self.log.index(("synth", S2))
        self.assertLess(first_play, s2_done, self.log)
        self.assertEqual(self.log[first_play][1:],
                         (0.01, 800 + self.gap_samples()))

    def test_rest_renders_while_sentence_one_plays(self):
        # Sentence 1's play is held (bounded) until S2 and S3 have rendered:
        # a player that renders only after play 1 would never get there.
        orig = self.fake_play

        def play(audio, sr):
            if self.play.call_count == 1:
                self.first_play_started.set()   # sentence 1 is "playing"
                deadline = self.bc.time.time() + 2.0
                while self.bc.time.time() < deadline and not (
                        ("synth", S2) in self.log
                        and ("synth", S3) in self.log):
                    self.bc.time.sleep(0.005)
                self._rec("play1_end_check",
                          ("synth", S2) in self.log,
                          ("synth", S3) in self.log)
            orig(audio, sr)

        self.play.side_effect = play
        self.assertTrue(self.speak())
        self.assertIn(("play1_end_check", True, True), self.log)
        self.assertEqual(self.play.call_count, 3)

    def test_one_play_per_sentence_with_a_gap_and_never_overlapping(self):
        self.speak()
        self.assertEqual(self.play.call_count, 3)
        self.assert_plays_equal(self.expected_plays(SENTENCES))
        self.assertEqual(self.max_active, 1)
        # The gap is real silence and sits at each sentence boundary only.
        first, last = self.played[0][0], self.played[-1][0]
        self.assertTrue(self.np.all(first[800:] == 0.0))
        self.assertEqual(len(first), 800 + self.gap_samples())
        self.assertEqual(len(last), 800)

    def test_volume_scale_applies_to_every_sentence(self):
        self.speak(volume_scale=0.5)
        self.assert_plays_equal(self.expected_plays(SENTENCES), scale=0.5)

    def test_interrupt_during_sentence_one_stops_the_rest(self):
        bc = self.bc

        def play(audio, sr):
            self.fake_play(audio, sr)
            if self.play.call_count == 1:
                # A web STOP lands while sentence 1 plays ...
                bc._tts_playback_active[0] = True
                self.assertTrue(bc.request_tts_interrupt(source="web",
                                                         acoustic=False))
                # ... and play_with_lipsync's finally clears the Event.
                bc._tts_interrupt.clear()
                bc._tts_playback_active[0] = False

        self.play.side_effect = play
        self.assertTrue(self.speak())
        self.assertEqual(self.play.call_count, 1)
        self.assertAlmostEqual(float(self.played[0][0][0]), 0.01)
        self.assertIn("reply stopped after 1/3 sentences", self.out)

    def _interrupt_between_sentences(self, text=REPLY, **kw):
        """S2's render waits until sentence 1's play has ENDED (playback flag
        off, as play_with_lipsync's finally leaves it), then a STOP arrives
        from another thread while the player is still waiting on S2."""
        bc = self.bc
        play1_done = threading.Event()
        accepted = []
        orig_play = self.fake_play

        def play(audio, sr):
            orig_play(audio, sr)
            bc._tts_playback_active[0] = False
            play1_done.set()

        def synth(text):
            if text == S2:
                play1_done.wait(timeout=2.0)
                accepted.append(bc.request_tts_interrupt(**kw))
            return self.fake_synth(text)

        self.play.side_effect = play
        self.synth.side_effect = synth
        ok = self.speak(text)
        return ok, accepted

    def test_a_stop_between_sentences_is_accepted_and_ends_the_reply(self):
        ok, accepted = self._interrupt_between_sentences(source="web",
                                                         acoustic=False)
        self.assertEqual(accepted, [True])
        self.assertTrue(ok)
        self.assertEqual(self.play.call_count, 1)
        self.assertIn("reply stopped after 1/3 sentences", self.out)

    def test_a_wake_barge_in_between_sentences_is_accepted(self):
        self._p(self.bc, "_barge_in_wake_enabled", return_value=True)
        ok, accepted = self._interrupt_between_sentences(source="wake-word")
        self.assertEqual(accepted, [True])
        self.assertEqual(self.play.call_count, 1)

    def test_between_sentences_the_echo_gate_still_holds(self):
        # The acoustic "jarvis" echo gate still applies to the reply's text.
        self._p(self.bc, "_barge_in_wake_enabled", return_value=True)
        text = f"{S1} {S2} Bring an umbrella, Jarvis says, if you head out."
        ok, accepted = self._interrupt_between_sentences(
            text=text, source="wake-word")
        self.assertEqual(accepted, [False])
        self.assertEqual(self.play.call_count, 3)

    def test_no_interrupt_is_accepted_once_the_reply_is_over(self):
        bc = self.bc
        self.speak()
        self.assertFalse(bc.request_tts_interrupt(source="web",
                                                  acoustic=False))
        self.assertIs(bc._tts_reply_active[0], False)

    def test_legacy_barge_in_stops_the_rest_without_counting_an_interrupt(self):
        # The legacy RMS barge-in (play_with_lipsync's _barge_watch) only
        # raises _barge_in_interrupted during a play; the sentence reply
        # reads that flag after each chunk. The accepted-interrupt counter
        # is left alone (streaming-flush / filler gates are unchanged).
        bc = self.bc
        seq0 = bc._tts_interrupt_seq[0]
        orig_play = self.fake_play

        def play(audio, sr):
            orig_play(audio, sr)
            if self.play.call_count == 1:
                bc._barge_in_interrupted = True

        self.play.side_effect = play
        self.speak()
        self.assertEqual(self.play.call_count, 1)
        self.assertEqual(bc._tts_interrupt_seq[0], seq0)
        self.assertIn("reply stopped after 1/3 sentences", self.out)

    def test_a_stale_legacy_barge_flag_does_not_cut_a_new_reply(self):
        self.bc._barge_in_interrupted = True      # left over from long ago
        self.speak()
        self.assertEqual(self.play.call_count, 3)

    def test_tray_mute_mid_reply_stops_the_rest(self):
        bc = self.bc

        def play(audio, sr):
            self.fake_play(audio, sr)
            bc._tts_muted[0] = True

        self.play.side_effect = play
        self.speak()
        self.assertEqual(self.play.call_count, 1)

    def test_tray_mute_plays_nothing(self):
        self.bc._tts_muted[0] = True
        self.speak()
        self.assertEqual(self.synth.call_count, 0)
        self.assertEqual(self.play.call_count, 0)

    def test_speech_state_is_reset_after_the_reply(self):
        bc = self.bc
        self.speak()
        self.assertEqual(bc._tts_current_text[0], "")
        self.assertIs(bc._tts_playback_active[0], False)
        self.assertIsNone(getattr(bc._TTS_PRESET_PIN, "value", None))

    def test_turn_timing_first_play_marks_sentence_one(self):
        bc = self.bc
        self._p(bc, "_tt", side_effect=lambda op, *a, **k:
                self._rec("tt", op, *a))
        self.speak()
        log = self.log
        synth_start = log.index(("tt", "mark", "synth_start"))
        first_play = log.index(("tt", "mark", "first_play"))
        s1 = log.index(("synth", S1))
        play1 = next(i for i, e in enumerate(log) if e[0] == "play")
        self.assertLess(synth_start, s1)
        self.assertLess(s1, first_play)
        self.assertLess(first_play, play1)
        self.assertEqual(log[play1][1], 0.01)
        self.assertEqual(log.count(("tt", "mark", "first_play")), 1)

    def test_pipeline_wiring(self):
        from core import sentence_tts
        np = self.np
        seen = {}

        def capture(chunks, synth, play, should_stop, **kw):
            seen.update(kw)
            return sentence_tts.PipelineResult()

        self._p(sentence_tts, "play_pipelined", side_effect=capture)
        self.speak()
        # The wait cap sits above the whole fallback ladder's worst case
        # (Kokoro 30 s + edge-tts 3 x 30 s + backoff + pyttsx3 15 s + SAPI5).
        self.assertEqual(seen["wait_timeout"], self.bc._SENTENCE_TTS_WAIT_S)
        self.assertGreaterEqual(seen["wait_timeout"], 240.0)
        self.assertIsNotNone(seen["synth_rest"])
        pad = seen["pad"]
        padded = pad(np.ones(3, dtype=np.float32), 24000)
        self.assertEqual(padded.dtype, np.float32)
        self.assertEqual(len(padded), 3 + self.gap_samples())
        self.assertTrue(np.all(padded[3:] == 0.0))

    def _fresh_ducker(self):
        bc = self.bc
        log = []
        ducker = bc._AudioDucker()
        ducker._check_available = lambda: True
        ducker._enumerate_targets = lambda: [("session", 0.8)]
        ducker._ensure_worker = lambda: None
        ducker._work_queue = _FakeDuckQueue(log)
        self._p(bc, "_audio_ducker", ducker)
        return ducker

    def test_a_render_error_after_sentence_one_counts_as_spoken(self):
        ducker = self._fresh_ducker()

        def synth(text):
            if text == S3:
                raise RuntimeError("engine gone")
            return self.fake_synth(text)

        self.synth.side_effect = synth
        self.assertTrue(self.speak())
        self.assertIn("[speak] playback failed after 2/3 sentences", self.out)
        self.assertIsNone(getattr(self.bc._TTS_PRESET_PIN, "value", None))
        self.assertEqual(ducker._holds, 0)

    def test_a_first_sentence_render_error_is_a_whole_text_failure(self):
        ducker = self._fresh_ducker()

        def synth(text):
            raise RuntimeError("engine gone")

        self.synth.side_effect = synth
        self.assertFalse(self.speak())
        self.assertIn("[speak] playback failed: RuntimeError", self.out)
        self.assertEqual(ducker._holds, 0)
        self.assertIs(self.bc._tts_reply_active[0], False)

    def test_a_later_play_failure_counts_the_line_as_spoken(self):
        ducker = self._fresh_ducker()
        orig_play = self.fake_play

        def play(audio, sr):
            if self.play.call_count == 2:
                raise RuntimeError("PortAudio reinit hung >1s")
            orig_play(audio, sr)

        self.play.side_effect = play
        self.assertTrue(self.speak())
        self.assertIn("[speak] playback failed after 1/3 sentences", self.out)
        self.assertEqual(ducker._holds, 0)

    def test_a_partly_played_flush_piece_is_never_said_again(self):
        # Streaming flush (#18 ledger) with the REAL _speak: the piece's
        # second sentence fails to play after its first was heard. The piece
        # must be recorded as spoken, so the tail strip removes it.
        bc = self.bc
        orig_play = self.fake_play

        def play(audio, sr):
            if self.play.call_count == 2:
                raise RuntimeError("PortAudio reinit hung >1s")
            orig_play(audio, sr)

        self.play.side_effect = play
        piece = f"{S1} {S2} {S3} "
        buf = bc._SentenceFlushBuffer()
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            buf._dispatch(piece)
            buf.join(timeout=10)
        self.assertEqual(buf.spoken_prefix, piece)
        with mock.patch.object(bc, "_stream_spoken_prefix",
                               [buf.spoken_prefix]):
            self.assertEqual(
                bc._strip_stream_spoken_prefix(piece + "Anything else?"),
                "Anything else?")

    def test_tags_and_markdown_are_stripped_before_splitting(self):
        from core import sentence_tts
        text = ("[mood:calm_efficient] **Right**, sir. The *forecast* calls "
                "for light rain this afternoon. Temperatures will stay near "
                "61 degrees, so bring a `jacket`.")
        self.speak(text)
        chunks = self.synth_texts()
        self.assertGreaterEqual(len(chunks), 2)
        for c in chunks:
            self.assertNotIn("[", c)
            self.assertNotIn("*", c)
            self.assertNotIn("`", c)
        stripped = self.bc._strip_markdown_for_speech(
            self.bc._parse_mood_tag(text)[1])
        self.assertEqual(chunks, sentence_tts.split_sentences(stripped))


class WholeTextPathTests(_Base):
    """Everything that must keep today's single synthesise + play."""

    def setUp(self):
        super().setUp()
        self.use_fakes()

    def test_disabled_setting(self):
        self._p(self.bc, "SENTENCE_TTS_ENABLED", False, create=True)
        self.speak()
        self.assert_whole_text_path()

    def test_edge_backend(self):
        self._p(self.bc, "TTS_BACKEND", "edge", create=True)
        self.speak()
        self.assert_whole_text_path()

    def test_other_backends(self):
        for backend in ("pyttsx3", "xtts"):
            with self.subTest(backend=backend):
                self.log.clear()
                self.played.clear()
                self.play.reset_mock()
                with mock.patch.object(self.bc, "TTS_BACKEND", backend,
                                       create=True):
                    self.speak()
                self.assert_whole_text_path()

    def test_voice_clone_armed(self):
        self._p(self.bc, "VOICE_CLONE_ENABLED", True, create=True)
        self.speak()
        self.assert_whole_text_path()

    def test_kokoro_unavailable(self):
        from core import kokoro_tts
        self._p(kokoro_tts, "is_available", return_value=False)
        self.speak()
        self.assert_whole_text_path()

    def test_short_reply(self):
        short = "Done. It is set. Anything else?"
        self.speak(short)
        self.assert_whole_text_path(short)

    def test_single_long_sentence(self):
        one = ("This reply is a single long sentence that runs on past one "
               "hundred and twenty characters, 3.5 inches at 2:30 p.m. and "
               "never ends")
        self.speak(one)
        self.assert_whole_text_path(one)

    def test_wry_reply_keeps_its_whole_text_beat(self):
        self.resolve.return_value = ("wry", {"rate": "+0%", "pitch": "+0Hz",
                                             "gain": 1.0})
        self.speak()
        self.assert_whole_text_path()


@requires_monolith
class PresetPinTests(_Base):
    """Real synthesise(): one preset resolution per reply, every sentence at
    that preset's Kokoro speed."""

    def test_every_sentence_uses_the_replys_preset(self):
        bc = self.bc
        from core import kokoro_tts
        np = self.np
        speeds = []

        def k_synth(text, speed=1.0):
            speeds.append((text, speed))
            return np.zeros(10, dtype=np.float32), 24000

        self._p(kokoro_tts, "synthesize", side_effect=k_synth)

        def resolve(text, tone):
            if text == REPLY:
                return "calm", {"rate": "-10%", "pitch": "+0Hz", "gain": 1.0}
            return "urgent", {"rate": "+15%", "pitch": "+0Hz", "gain": 1.0}

        self.resolve.side_effect = resolve
        self.play = self._p(bc, "play_with_lipsync",
                            side_effect=self.fake_play)
        self.speak()
        self.assertEqual(sorted(t for t, _ in speeds), sorted(SENTENCES))
        self.assertEqual({s for _, s in speeds}, {0.9})
        self.assertEqual(self.resolve.call_count, 1)
        self.assertEqual(self.resolve.call_args[0][0], REPLY)
        self.assertIn("3 sentences preset=calm", self.out)

    def _real_synth_setup(self, kokoro_miss):
        bc = self.bc
        from core import kokoro_tts
        np = self.np
        self.k_calls, self.edge_calls = [], []

        def k_synth(text, speed=1.0):
            self.k_calls.append(text)
            if text in kokoro_miss:
                return None
            return np.full(10, 0.1, dtype=np.float32), 24000

        def edge(text, rate, pitch):
            self.edge_calls.append((text, rate))
            return np.full(20, 0.2, dtype=np.float32), 24000

        self._p(kokoro_tts, "synthesize", side_effect=k_synth)
        self._p(bc, "_render_edge_tts", side_effect=edge)
        self.resolve.side_effect = None
        self.resolve.return_value = ("calm", {"rate": "-10%",
                                              "pitch": "+0Hz", "gain": 1.0})
        self.play = self._p(bc, "play_with_lipsync",
                            side_effect=self.fake_play)

    def test_a_kokoro_miss_voices_the_rest_as_one_fallback_block(self):
        self._real_synth_setup(kokoro_miss={S2})
        self.assertTrue(self.speak())
        # Kokoro is tried for S1 and S2 only; the rest (S2 + S3) is ONE edge
        # render at the reply's preset -- no per-sentence retry of a dead
        # engine, and no voice switch back and forth.
        self.assertEqual(self.k_calls, [S1, S2])
        self.assertEqual(self.edge_calls, [(f"{S2} {S3}", "-10%")])
        self.assertEqual(self.play.call_count, 2)
        self.assertEqual(len(self.played[1][0]), 20)
        self.assertIn("rest of the reply in one block", self.out)

    def test_a_kokoro_miss_on_sentence_one_voices_the_reply_whole(self):
        self._real_synth_setup(kokoro_miss={S1})
        self.assertTrue(self.speak())
        self.assertEqual(self.k_calls, [S1])
        self.assertEqual(self.edge_calls, [(REPLY, "-10%")])
        self.assertEqual(self.play.call_count, 1)
        self.assertIsNone(getattr(self.bc._TTS_PRESET_PIN, "value", None))
        self.assertIsNone(getattr(self.bc._TTS_PRESET_PIN, "mode", None))


class _FakeDuckQueue:
    """Stands in for _AudioDucker's fade worker queue: records each fade job
    (target level; None = restore) and completes it at once."""

    def __init__(self, log):
        self.log = log

    def put(self, job):
        plans, target, _cancellable, done = job
        self.log.append(("fade", target))
        if done is not None:
            done.set()


@requires_monolith
class RealPlaybackTests(_Base):
    """The REAL play_with_lipsync with a fake sounddevice: lip-sync values and
    the duck per chunk, in order; env mute reaches no sd.play."""

    def setUp(self):
        super().setUp()
        bc = self.bc
        self.synth = self._p(bc, "synthesise", side_effect=self.fake_synth)
        self.sd = mock.Mock()
        self.sd.PortAudioError = type("PortAudioError", (Exception,), {})
        self.sd.get_stream.return_value = None
        self.sd.play.side_effect = self._sd_play
        self._p(bc, "sd", self.sd)
        self._p(bc, "get_output_device", return_value=None)
        self._p(bc, "_feed_playback_reference")
        self._p(bc, "ROBOT_ENABLED", True)
        self._p(bc, "send", side_effect=lambda **k: self._rec(
            "mouth", round(k.get("mouth", 0.0), 2)))
        self._p(bc, "AUDIO_DUCKING_ENABLED", True)
        ducker = bc._AudioDucker()
        ducker._check_available = lambda: True
        ducker._enumerate_targets = lambda: [("session", 0.8)]
        ducker._ensure_worker = lambda: None
        ducker._work_queue = _FakeDuckQueue(self.log)
        self._p(bc, "_audio_ducker", ducker)

    def _sd_play(self, audio, sr, device=None):
        self._rec("sd.play", round(float(audio[0]), 4), len(audio))
        self.first_play_started.set()
        self.played.append((audio.copy(), sr))

    def test_lip_sync_and_ducking_run_per_chunk_in_order(self):
        self.speak()
        sd_plays = [e for e in self.log if e[0] == "sd.play"]
        self.assertEqual(len(sd_plays), 3)          # one play per sentence
        self.assert_plays_equal(self.expected_plays(SENTENCES))
        # Mouth values follow the sentences in order (0.01/0.02/0.03 RMS x
        # MOUTH_SCALE), each chunk closing the mouth after its audio.
        scale = float(self.bc.MOUTH_SCALE)
        want = [round(min(1.0, a * scale), 2) for a in (0.01, 0.02, 0.03)]
        # (the 33 ms lip-sync window that straddles a sentence's end and its
        # gap reads a lower value; only full-window values are compared)
        opened = [e[1] for e in self.log
                  if e[0] == "mouth" and e[1] >= want[0] * 0.9]
        dedup = [v for i, v in enumerate(opened) if i == 0 or opened[i - 1] != v]
        self.assertEqual(dedup, want)
        closes = [e for e in self.log if e == ("mouth", 0.0)]
        self.assertGreaterEqual(len(closes), len(sd_plays))
        mouths = [e for e in self.log if e[0] == "mouth"]
        self.assertEqual(mouths[-1], ("mouth", 0.0))
        # One fade down before the first sd.play, one restore after the last:
        # the duck is held across the reply, not released between sentences.
        fades = [(i, e[1]) for i, e in enumerate(self.log) if e[0] == "fade"]
        self.assertEqual([t for _, t in fades],
                         [self.bc.AUDIO_DUCKING_LEVEL, None])
        first_sd = self.log.index(sd_plays[0])
        last_sd = self.log.index(sd_plays[-1])
        self.assertLess(fades[0][0], first_sd)
        self.assertGreater(fades[1][0], last_sd)
        self.assertEqual(self.bc._audio_ducker._holds, 0)

    def test_env_mute_reaches_no_sd_play(self):
        bc = self.bc
        layer = mock.Mock()
        layer.parse_wry_tag.side_effect = lambda t: (False, t)
        layer.is_muted.return_value = True
        self._p(bc, "_tts_layer", layer)
        self.speak()
        self.sd.play.assert_not_called()
        # Today's muted path exactly: one whole-text render, no split.
        self.assertEqual(self.synth_texts(), [REPLY])


@requires_monolith
class AudioDuckerHoldTests(MonolithGlobalsTestCase):
    def _ducker(self, log):
        d = self.bc._AudioDucker()
        d._ensure_worker = lambda: None
        d._work_queue = _FakeDuckQueue(log)
        d._saved = [("session", 0.8)]
        return d

    def test_restore_is_a_no_op_while_held(self):
        log = []
        d = self._ducker(log)
        d.hold()
        d.restore()
        self.assertEqual(log, [])
        self.assertEqual(d._saved, [("session", 0.8)])
        d.release()
        self.assertEqual(log, [("fade", None)])
        self.assertEqual(d._saved, [])

    def test_nested_holds_restore_on_the_last_release(self):
        log = []
        d = self._ducker(log)
        d.hold()
        d.hold()
        d.release()
        self.assertEqual(log, [])
        d.release()
        self.assertEqual(log, [("fade", None)])
        d.release()                        # extra release: harmless
        self.assertEqual(d._holds, 0)

    def test_unheld_restore_is_unchanged(self):
        log = []
        d = self._ducker(log)
        d.restore()
        self.assertEqual(log, [("fade", None)])


if __name__ == "__main__":
    unittest.main()
