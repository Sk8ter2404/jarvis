"""The clone voice server wired into the monolith (2026-10-03).

VOICE_CLONE_MODEL = "chatterbox_turbo_server": every sentence is one bounded
POST /tts to a separate process (core/clone_voice_client.py). These tests run
the REAL client against a FAKE loopback server (tests/_clone_voice_fake.py)
and drive the real synthesise / _speak / _speak_sentences / pre-render /
filler code; only Kokoro (a constant 0.25 sentinel), the speaker
(play_with_lipsync) and the HUD are faked. Nothing is played, no GPU.

  * default off: the shipped config never touches a server, and the
    in-process engine keeps today's gates;
  * healthy: the clone voices the line (gain and the wry pause honoured),
    [turn-timing] clone / clone_ms are noted, and the Kokoro fast paths
    (per-sentence speech, the filler, the R3 pre-render) run with it;
  * slow: the line times out inside its deadline and THAT line is Kokoro;
  * down at boot: one log line, Kokoro speaks, nothing waits on the server;
  * cool-down (2026-10-04, was latch-off for the session): MAX_FAILURES
    latency-critical misses in a row, then no requests until it ends, then
    the clone voices again;
  * speak contracts: self-echo remember before the play / refresh after it,
    a barge-in between sentences stops the reply, volume_scale, mute;
  * live budgeting (2026-10-04): the 10:36 briefing and the two one-line
    replies that latched the clone off live, replayed through the real
    _speak as the server logged them, keep the clone voice and never cool
    down; a filler warm never counts toward the cool-down; a short opener
    before long lines keeps one voice too (the next line is held: a pause,
    not Kokoro), bounded by its budget; a long first sentence is split at a
    clause (clone only), the rest keeps the head's voice and the whole
    line's budget, also through the R3 pre-render; a barge-in drops the
    line still rendering ahead (no wait, no miss, no Kokoro render); one
    "[tts] clone voice" line per line.

    python -m unittest tests.monolith.test_monolith_clone_server
"""
from __future__ import annotations

import contextlib
import io
import threading
import time
import types
import unittest
from unittest import mock

from tests._clone_voice_fake import (LIVE_1036_AUDIO_S, LIVE_1036_LINES,
                                     LIVE_1036_RENDER_S, LIVE_1037_AWAY,
                                     LIVE_1037_AWAY_PIECES, LIVE_1037_MORNING,
                                     LIVE_1037_MORNING_PIECES,
                                     FakeCloneServer, ProfileDir, free_port,
                                     live_1036_server, make_wav)
from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith
from tests.monolith.test_monolith_processing_filler import _Base as _FillerBase

S1 = "The forecast calls for light rain this afternoon."
S2 = "Temperatures will stay near sixty degrees."
S3 = "Bring an umbrella if you head out, sir."
REPLY = f"{S1} {S2} {S3}"
SHORT = "Right away, sir."
NEUTRAL = ("neutral", {"rate": "+0%", "pitch": "+0Hz", "gain": 1.0})
KOKORO = 0.25


class _CloneMixin:
    """A box with the clone voice server selected and a fake server up.
    Mixed into a MonolithGlobalsTestCase subclass that provides _p()."""

    def clone_setup(self, *, ready=True, server_factory=None, **server_kw):
        bc = self.bc
        import numpy as np
        self.np = np
        from core import clone_voice_client as cvc
        from core import kokoro_tts
        from core import voice_clone as vc
        self.cvc = cvc
        self.prof = ProfileDir("butler")
        self.addCleanup(self.prof.cleanup)
        self._p(vc, "PROFILES_DIR", self.prof.root)
        self.srv = (server_factory(self.prof.sha) if server_factory else
                    FakeCloneServer(ref_sha=self.prof.sha, **server_kw)).start()
        self.addCleanup(self.srv.stop)
        self.logs = []
        self.client = cvc.CloneVoiceClient(
            log=self.logs.append, boot_wait_s=2.0,
            sleep=lambda s: time.sleep(min(s, 0.02)))
        self._p(cvc, "CLIENT", self.client)
        self._p(bc, "TTS_BACKEND", "kokoro", create=True)
        self._p(bc, "VOICE_CLONE_ENABLED", True, create=True)
        self._p(bc, "VOICE_CLONE_MODEL", cvc.MODEL_ID, create=True)
        self._p(bc, "VOICE_CLONE_PROFILE", "butler", create=True)
        self._p(bc, "VOICE_CLONE_SERVER_URL", self.srv.url, create=True)
        self._p(bc, "VOICE_CLONE_SERVER_CMD", "", create=True)
        self._p(bc, "VOICE_CLONE_TIMEOUT_S", 2.5, create=True)
        self.kokoro_texts = []

        def _kokoro(text, speed=1.0):
            self.kokoro_texts.append(text)
            return np.full(2400, KOKORO, dtype=np.float32), 24000
        self._p(kokoro_tts, "is_available", return_value=True)
        self._p(kokoro_tts, "synthesize", side_effect=_kokoro)
        self._p(bc, "_render_edge_tts",
                side_effect=AssertionError("edge-tts must not run"))
        self.stats = []
        self._p(bc, "_tt_note_stat",
                side_effect=lambda n, v: self.stats.append((n, v)))
        if ready:
            self.assertEqual(
                self.client.start(url=self.srv.url, cmd="", profile="butler"),
                "ready", self.logs)
            self.logs.clear()

    # helpers
    def rest_clone(self):
        """Put the client in a cool-down that lasts the whole test."""
        self.client._status = "cooldown"
        self.client._cool_until = self.client._clock() + 3600.0

    def is_kokoro(self, audio) -> bool:
        return bool(self.np.allclose(audio, KOKORO))

    def voice_of(self, audio) -> str:
        """'kokoro' / 'clone' / '?' for a played clip, its sentence pause
        (exact zeros) ignored."""
        a = self.np.asarray(audio)
        voiced = a[a != 0.0]
        if voiced.size and bool(self.np.allclose(voiced, KOKORO)):
            return "kokoro"
        return "clone" if self.is_clone(a) else "?"

    def is_clone(self, audio) -> bool:
        a = self.np.asarray(audio)
        return a.size > 0 and not self.is_kokoro(a) and float(
            self.np.max(self.np.abs(a))) > 0.05

    def quiet(self, fn, *a, **kw):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            r = fn(*a, **kw)
        self.out = out.getvalue()
        return r


@requires_monolith
class _Base(_CloneMixin, MonolithGlobalsTestCase):
    def _p(self, *args, **kwargs):
        patcher = mock.patch.object(*args, **kwargs)
        m = patcher.start()
        self.addCleanup(patcher.stop)
        return m

    def setUp(self):
        super().setUp()
        bc = self.bc
        self._p(bc, "_resolve_tts_preset", return_value=NEUTRAL)
        self._p(bc, "_tts_layer", None)
        self._p(bc, "_last_voice_route", [{"addendum": "", "mood": "casual"}])
        self._p(bc, "_last_user_tone", [None])
        self._p(bc, "_last_mood", [None])


# ════════════════════════════════════════════════════════════════════════════
#  Default off
# ════════════════════════════════════════════════════════════════════════════
class DefaultOffTests(_Base):
    def test_shipped_defaults(self):
        from core import config
        self.assertIs(config.VOICE_CLONE_ENABLED, False)
        self.assertEqual(config.VOICE_CLONE_MODEL, "chatterbox")
        self.assertEqual(config.VOICE_CLONE_SERVER_URL, "http://127.0.0.1:8767")
        self.assertEqual(config.VOICE_CLONE_SERVER_CMD, "")
        self.assertEqual(config.VOICE_CLONE_TIMEOUT_S, 2.5)

    def test_clone_off_never_touches_the_server(self):
        self.clone_setup()
        bc = self.bc
        self._p(bc, "VOICE_CLONE_ENABLED", False)
        audio, sr = self.quiet(bc.synthesise, SHORT)
        self.assertTrue(self.is_kokoro(audio))
        self.assertEqual(self.srv.tts_texts(), [])
        self.assertEqual(bc._tts_engine_kind(), "kokoro")
        self.assertFalse(bc._clone_server_selected())
        self.assertNotIn("clone", [n for n, _v in self.stats])

    def test_default_off_never_starts_anything(self):
        self.clone_setup(ready=False)
        bc = self.bc
        self._p(bc, "VOICE_CLONE_ENABLED", False)
        self._p(bc, "_clone_server_may_start", return_value=True)
        start = self._p(self.client, "start_async")
        self.quiet(bc.synthesise, SHORT)
        self.assertFalse(bc._clone_server_kick())
        start.assert_not_called()
        self.assertEqual(self.client.status()[0], "idle")

    def test_in_process_engine_keeps_todays_gates(self):
        self.clone_setup()
        bc = self.bc
        self._p(bc, "VOICE_CLONE_MODEL", "chatterbox")
        self.assertFalse(bc._clone_server_selected())
        self.assertEqual(bc._tts_engine_kind(), "other")
        self.assertIsNone(bc._sentence_tts_plan(REPLY))
        from core import voice_clone as vc
        with mock.patch.object(vc, "is_available", return_value=False):
            audio, _sr = self.quiet(bc.synthesise, SHORT)
        self.assertTrue(self.is_kokoro(audio))
        self.assertEqual(self.srv.tts_texts(), [])


# ════════════════════════════════════════════════════════════════════════════
#  Healthy
# ════════════════════════════════════════════════════════════════════════════
class HealthyTests(_Base):
    def setUp(self):
        super().setUp()
        self.clone_setup()

    def test_the_clone_voices_the_line(self):
        bc = self.bc
        audio, sr = self.quiet(bc.synthesise, SHORT)
        self.assertTrue(self.is_clone(audio))
        self.assertEqual(sr, 24000)
        self.assertEqual(self.srv.tts_texts(), [SHORT])
        self.assertEqual(self.kokoro_texts, [])
        self.assertEqual(bc._tts_engine_kind(), "clone")
        names = dict(self.stats)
        self.assertEqual(names.get("clone"), 1)
        self.assertIsInstance(names.get("clone_ms"), int)

    def test_preset_gain_is_applied(self):
        bc = self.bc
        full, _ = self.quiet(bc.synthesise, SHORT)
        self._p(bc, "_resolve_tts_preset",
                return_value=("calm", {"rate": "-5%", "pitch": "+0Hz",
                                       "gain": 0.5}))
        half, _ = self.quiet(bc.synthesise, SHORT)    # same take: the cache
        self.np.testing.assert_allclose(half, full * 0.5, atol=1e-6)

    def test_wry_line_is_two_renders_with_the_pause(self):
        bc = self.bc
        layer = types.SimpleNamespace(
            split_for_wry_pause=lambda t: ("Splendid.", "Another meeting, sir."),
            WRY_PAUSE_MS=200, is_muted=lambda: False)
        self._p(bc, "_tts_layer", layer)
        self._p(bc, "_resolve_tts_preset",
                return_value=("wry", {"rate": "+0%", "pitch": "+0Hz",
                                      "gain": 1.0}))
        audio, sr = self.quiet(bc.synthesise, "Splendid. Another meeting, sir.")
        self.assertEqual(self.srv.tts_texts(),
                         ["Splendid.", "Another meeting, sir."])
        # A run of exact zeros at least as long as the pause sits inside.
        z = (audio == 0.0).astype(int)
        longest, run = 0, 0
        for v in z:
            run = run + 1 if v else 0
            longest = max(longest, run)
        self.assertGreaterEqual(longest, int(sr * 0.2))
        self.assertEqual(self.kokoro_texts, [])

    def test_a_wry_line_whose_tail_fails_is_all_kokoro(self):
        # Both clauses or neither: a clone head must never ship alone (or
        # twice), and Kokoro then voices the WHOLE line with its own pause.
        bc = self.bc
        layer = types.SimpleNamespace(
            split_for_wry_pause=lambda t: ("Splendid.", "Another meeting, sir."),
            WRY_PAUSE_MS=200, is_muted=lambda: False)
        self._p(bc, "_tts_layer", layer)
        self._p(bc, "_resolve_tts_preset",
                return_value=("wry", {"rate": "+0%", "pitch": "+0Hz",
                                      "gain": 1.0}))
        self.srv.fail_texts = {"Another meeting, sir."}
        audio, sr = self.quiet(bc.synthesise, "Splendid. Another meeting, sir.")
        self.assertEqual(self.srv.tts_texts(),
                         ["Splendid.", "Another meeting, sir."])
        self.assertEqual(self.kokoro_texts,
                         ["Splendid.", "Another meeting, sir."])
        # Only Kokoro's constant and the spliced pause: no clone samples.
        a = self.np.asarray(audio)
        self.assertTrue(bool(self.np.all((a == 0.0) | self.np.isclose(a, KOKORO))))
        self.assertEqual(int(self.np.sum(self.np.isclose(a, KOKORO))), 2 * 2400)
        self.assertIn("Kokoro voices this line", self.out)

    def test_the_fast_paths_run_with_the_clone(self):
        bc = self.bc
        plan = self.quiet(bc._sentence_tts_plan, REPLY)
        self.assertIsNotNone(plan)
        self.assertEqual(plan[0], [S1, S2, S3])
        key_clone = bc._filler_voice_key()
        self.rest_clone()
        self.assertEqual(bc._tts_engine_kind(), "kokoro")
        self.assertNotEqual(bc._filler_voice_key(), key_clone)

    def test_skip_kokoro_block_skips_the_clone_too(self):
        bc = self.bc
        bc._TTS_PRESET_PIN.value = NEUTRAL
        bc._TTS_PRESET_PIN.mode = "skip_kokoro"
        self.addCleanup(setattr, bc._TTS_PRESET_PIN, "value", None)
        self.addCleanup(setattr, bc._TTS_PRESET_PIN, "mode", None)
        self._p(bc, "_render_edge_tts",
                return_value=(self.np.full(10, 0.75, dtype=self.np.float32),
                              24000))
        audio, _ = self.quiet(bc.synthesise, S2)
        self.assertTrue(self.np.allclose(audio, 0.75))
        self.assertEqual(self.srv.tts_texts(), [])

    def test_the_voiced_flag_describes_the_last_line_only(self):
        # _CLONE_LINE.voiced: did the clone voice the LAST synthesise() on
        # this thread. A stale True would let the rest of a sentence whose
        # head Kokoro voiced switch to the clone halfway.
        bc = self.bc
        self.quiet(bc.synthesise, SHORT)
        self.assertTrue(bc._CLONE_LINE.voiced)
        self.srv.fail_texts = {"A line that fails."}
        audio, _ = self.quiet(bc.synthesise, "A line that fails.")
        self.assertTrue(self.is_kokoro(audio))
        self.assertFalse(bc._CLONE_LINE.voiced)

    def test_boot_label_names_the_clone(self):
        self.assertIn("clone voice server (ready)", self.bc._boot_tts_label())


# ════════════════════════════════════════════════════════════════════════════
#  Slow, failing, cooling down
# ════════════════════════════════════════════════════════════════════════════
class SlowAndFailingTests(_Base):
    def test_slow_line_falls_back_to_kokoro_inside_the_timeout(self):
        self.clone_setup(tts_delay=5.0)
        bc = self.bc
        self._p(bc, "VOICE_CLONE_TIMEOUT_S", 0.5)
        t0 = time.monotonic()
        audio, _ = self.quiet(bc.synthesise, SHORT)
        self.assertLess(time.monotonic() - t0, 2.0)
        self.assertTrue(self.is_kokoro(audio))
        self.assertEqual(self.kokoro_texts, [SHORT])
        self.assertIn("clone voice timed out", self.out)
        self.assertIn("Kokoro voices this line", self.out)
        # One compact line: render ms, first line, the deadline it had.
        self.assertIn("(first line, deadline 0.5 s)", self.out)
        self.assertEqual(dict(self.stats).get("clone"), 0)
        self.assertEqual(bc._tts_engine_kind(), "clone")   # one miss: still on

    def test_cooldown_after_max_failures_then_the_clone_returns(self):
        self.clone_setup(tts_status=500)
        bc = self.bc
        now = [1000.0]
        self.client._clock = lambda: now[0]
        n = self.cvc.MAX_FAILURES
        for i in range(n):
            audio, _ = self.quiet(bc.synthesise, f"Line {i}.")
            self.assertTrue(self.is_kokoro(audio))
        self.assertEqual(self.client.status()[0], "cooldown")
        rests = [m for m in self.logs if "rests for 5 min" in m]
        self.assertEqual(len(rests), 1, self.logs)
        self.assertNotIn("for this session", " ".join(self.logs))
        sent = len(self.srv.tts_texts())
        audio, _ = self.quiet(bc.synthesise, "During the rest.")
        self.assertTrue(self.is_kokoro(audio))
        self.assertEqual(len(self.srv.tts_texts()), sent)   # no more requests
        self.assertNotIn("clone voice", self.out)           # and no noise
        self.assertEqual(bc._tts_engine_kind(), "kokoro")
        # Five minutes later the clone is tried again -- not off for the
        # session any more.
        now[0] += self.cvc.COOLDOWN_BASE_S
        self.srv.tts_status = 200
        self.assertEqual(bc._tts_engine_kind(), "clone")
        audio, _ = self.quiet(bc.synthesise, "After the rest.")
        self.assertTrue(self.is_clone(audio))
        self.assertEqual(self.srv.tts_texts()[-1], "After the rest.")
        self.assertEqual(len([m for m in self.logs if "cool-down over" in m]),
                         1, self.logs)

    def test_a_repeated_line_logs_cached(self):
        self.clone_setup()
        self.quiet(self.bc.synthesise, SHORT)
        audio, _ = self.quiet(self.bc.synthesise, SHORT)
        self.assertTrue(self.is_clone(audio))
        self.assertIn("[tts] clone voice 0 ms (first line, cached)", self.out)
        self.assertEqual(self.srv.tts_texts(), [SHORT])

    def test_too_long_line_goes_to_kokoro_without_a_failure(self):
        self.clone_setup()
        long_line = "word " * 300
        audio, _ = self.quiet(self.bc.synthesise, long_line)
        self.assertTrue(self.is_kokoro(audio))
        self.assertEqual(self.srv.tts_texts(), [])
        self.assertEqual(self.client.failures(), 0)
        self.assertIn("too long", self.out)


# ════════════════════════════════════════════════════════════════════════════
#  Down at boot
# ════════════════════════════════════════════════════════════════════════════
class DownAtBootTests(_Base):
    def _wait_started(self, timeout=5.0):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self.client.status()[0] not in ("idle", "starting"):
                return
            time.sleep(0.02)
        self.fail(f"start never finished: {self.client.status()}")

    def test_down_at_boot_logs_one_line_and_kokoro_speaks(self):
        self.clone_setup(ready=False)
        bc = self.bc
        self._p(bc, "VOICE_CLONE_SERVER_URL", f"http://127.0.0.1:{free_port()}")
        self._p(bc, "_clone_server_may_start", return_value=True)
        t0 = time.monotonic()
        self.assertTrue(bc._clone_server_kick())     # boot: non-blocking
        self.assertLess(time.monotonic() - t0, 0.5)
        audio, _ = self.quiet(bc.synthesise, SHORT)  # while it starts
        self.assertTrue(self.is_kokoro(audio))
        self._wait_started()
        self.assertEqual(self.client.status()[0], "down")
        self.assertEqual(len(self.logs), 1, self.logs)
        self.assertIn("Kokoro keeps speaking", self.logs[0])
        audio, _ = self.quiet(bc.synthesise, SHORT)
        self.assertTrue(self.is_kokoro(audio))
        self.assertEqual(self.srv.tts_texts(), [])
        self.assertEqual(bc._tts_engine_kind(), "kokoro")
        self.assertIsNotNone(self.quiet(bc._sentence_tts_plan, REPLY))
        self.assertFalse(bc._clone_server_kick())    # once per session

    def test_boot_kick_reuses_a_running_server_and_rewarms_the_filler(self):
        self.clone_setup(ready=False)
        bc = self.bc
        self._p(bc, "_clone_server_may_start", return_value=True)
        warm = self._p(bc, "_filler_warm_if_needed")
        self.assertTrue(bc._clone_server_kick())
        self._wait_started()
        self.assertEqual(self.client.status()[0], "ready", self.logs)
        warm.assert_called_once_with()
        self.assertEqual(bc._tts_engine_kind(), "clone")

    def test_selected_at_runtime_starts_it_lazily_once(self):
        self.clone_setup(ready=False)
        bc = self.bc
        self._p(bc, "_clone_server_may_start", return_value=True)
        calls = []
        self._p(self.client, "start_async",
                side_effect=lambda **kw: (calls.append(kw), False)[1])
        self.assertFalse(bc._clone_server_active())
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["profile"], "butler")
        self.assertEqual(calls[0]["url"], self.srv.url)

    def test_never_starts_from_the_test_harness(self):
        self.clone_setup(ready=False)
        bc = self.bc
        start = self._p(self.client, "start_async")
        self.assertFalse(bc._clone_server_active())
        self.assertFalse(bc._clone_server_kick())
        start.assert_not_called()


# ════════════════════════════════════════════════════════════════════════════
#  Speak contracts through the real _speak
# ════════════════════════════════════════════════════════════════════════════
class _SpeakBase(_Base):
    """The real _speak / _speak_sentences / synthesise with the speaker
    (play_with_lipsync), HUD, ducker and self-echo faked. PLAY_SLEEPS: the
    fake speaker blocks for the audio's length, as the real one does."""

    PLAY_SLEEPS = False

    def clone_kw(self) -> dict:
        return {}

    def setUp(self):
        super().setUp()
        self.clone_setup(**self.clone_kw())
        bc = self.bc
        self._p(bc, "set_state")
        self._p(bc, "_write_hud_state")
        self._p(bc, "_processing_filler", mock.Mock())
        self._p(bc, "_audio_ducker", mock.Mock())
        self._p(bc, "_session_start_time", time.time() - 3600)
        self._p(bc, "_is_staging", lambda: False)
        self._p(bc, "BARGE_IN_ENABLED", False)
        self._p(bc, "SENTENCE_TTS_ENABLED", True, create=True)
        self._p(bc, "PROCESSING_FILLER_PRERENDER", False, create=True)
        bc._tts_muted[0] = False
        seq = bc._tts_interrupt_seq[0]
        self.addCleanup(bc._tts_interrupt_seq.__setitem__, 0, seq)
        self.addCleanup(bc._tts_interrupt.clear)
        self.addCleanup(bc._tts_playback_active.__setitem__, 0, False)
        self.addCleanup(bc._tts_reply_active.__setitem__, 0, False)
        self._p(bc, "_barge_in_interrupted", False)
        self.events = []
        self.played = []
        self.on_play = None

        def play(audio, sr):
            self.events.append(("play", len(self.played)))
            self.played.append(self.np.array(audio, copy=True))
            if self.on_play is not None:
                self.on_play(len(self.played))
            if self.PLAY_SLEEPS:
                time.sleep(len(audio) / float(sr))
        self._p(bc, "play_with_lipsync", side_effect=play)
        self._p(bc._self_echo, "remember",
                side_effect=lambda t: (self.events.append(("remember", t)), 7)[1])
        self._p(bc._self_echo, "refresh",
                side_effect=lambda tok: self.events.append(("refresh", tok)))

    def speak(self, text=REPLY, **kw):
        return self.quiet(self.bc._speak, text, **kw)

    def join_workers(self):
        """Let a render left in flight by a stopped reply finish (by design)
        before the fake server goes away, its log line kept out of the
        test output."""
        with contextlib.redirect_stdout(io.StringIO()):
            for th in threading.enumerate():
                if th.name == "sentence-tts-synth":
                    th.join(timeout=5.0)


class SpeakContractTests(_SpeakBase):

    def test_each_sentence_is_one_clone_render_in_order(self):
        self.assertTrue(self.speak())
        self.assertEqual(self.srv.tts_texts(), [S1, S2, S3])
        self.assertEqual(len(self.played), 3)
        for a in self.played:
            self.assertTrue(self.is_clone(a))
        self.assertEqual(self.kokoro_texts, [])

    def test_a_failed_sentence_keeps_the_rest_of_the_reply_in_kokoro(self):
        # One voice per reply (owner, 2026-10-04): once a line missed the
        # clone, the reply never switches back to the clone voice.
        self.srv.fail_texts = {S2}
        self.assertTrue(self.speak())
        self.assertEqual(len(self.played), 3)
        self.assertTrue(self.is_clone(self.played[0]))
        self.assertTrue(self.is_kokoro(
            self.played[1][: -int(24000 * 0.15)]))   # before the sentence gap
        self.assertTrue(self.is_kokoro(self.played[2]))
        self.assertEqual(self.kokoro_texts, [S2, S3])
        self.assertEqual(self.srv.tts_texts(), [S1, S2])  # S3 never asked
        self.assertIn("one voice per reply", self.out)

    def test_a_kokoro_first_line_keeps_the_whole_reply_in_kokoro(self):
        self.srv.fail_texts = {S1}
        self.assertTrue(self.speak())
        self.assertEqual(len(self.played), 3)
        self.assertTrue(self.is_kokoro(
            self.played[0][: -int(24000 * 0.15)]))   # before the sentence gap
        self.assertTrue(self.is_kokoro(self.played[2]))
        self.assertEqual(self.kokoro_texts, [S1, S2, S3])
        self.assertEqual(self.srv.tts_texts(), [S1])

    def test_self_echo_is_remembered_before_the_first_play(self):
        self.speak(SHORT)
        kinds = [e[0] for e in self.events]
        self.assertEqual(kinds, ["remember", "play", "refresh"])
        self.assertEqual(self.events[0][1], SHORT)
        self.assertEqual(self.events[2][1], 7)

    def test_barge_in_between_sentences_stops_the_reply(self):
        bc = self.bc

        def barge(n):
            if n == 1:
                bc._tts_interrupt_seq[0] += 1     # an accepted interrupt
        self.on_play = barge
        self.speak()
        self.assertEqual(len(self.played), 1)
        self.assertIn("reply stopped after 1/3", self.out)
        # The render already in flight on the worker finishes on its own
        # (by design) -- let it, before the fake server goes away.
        self.join_workers()
        self.assertEqual(len(self.played), 1)

    def test_volume_scale_applies_to_the_clone(self):
        self.speak(SHORT)
        self.speak(SHORT, volume_scale=0.5)      # same take: the cache
        self.assertEqual(len(self.played), 2)
        self.np.testing.assert_allclose(self.played[1], self.played[0] * 0.5,
                                        atol=1e-6)

    def test_tray_mute_never_reaches_the_server(self):
        self.bc._tts_muted[0] = True
        self.addCleanup(self.bc._tts_muted.__setitem__, 0, False)
        self.speak()
        self.assertEqual(self.srv.tts_texts(), [])
        self.assertEqual(self.played, [])


# ════════════════════════════════════════════════════════════════════════════
#  Live budgeting (2026-10-04): the 10:36 briefing, replayed through _speak
# ════════════════════════════════════════════════════════════════════════════
class LiveBudgetReplayTests(_SpeakBase):
    """2026-10-04 10:36-10:37 as the clone server logged it
    (tests/_clone_voice_fake): a 7-sentence briefing whose line 7 missed its
    fixed per-line budget (Kokoro mid-reply, miss 1), then two one-line
    replies that each missed their first-line budget (misses 2 and 3: the
    clone latched off for the session). Replayed here in that order with
    every time scaled by SCALE (budget, per-char allowance, margin, sentence
    gap, render and audio lengths), through the REAL _speak ->
    _speak_sentences -> play_pipelined -> synthesise -> client, a serial fake
    server and a fake speaker that blocks for each clip's length. On
    origin/main d5931da the same replay gives [clone x6, Kokoro], [Kokoro],
    [Kokoro] and the latch -- the live incident."""

    SCALE = 0.25
    PLAY_SLEEPS = True

    def clone_kw(self) -> dict:
        return {"server_factory":
                lambda sha: live_1036_server(sha, self.SCALE, later=True)}

    def setUp(self):
        super().setUp()
        s = self.SCALE
        from core import sentence_tts
        self._p(self.bc, "_clone_timeout_s", return_value=2.5 * s)
        self._p(self.cvc, "PER_CHAR_S", self.cvc.PER_CHAR_S * s)
        # create=True: the same replay runs against a client without the
        # needed-by deadline (origin/main before 2026-10-04) and fails there
        # on behaviour, not on a missing name.
        self._p(self.cvc, "NEEDED_BY_MARGIN_S",
                getattr(self.cvc, "NEEDED_BY_MARGIN_S", 0.5) * s, create=True)
        self._p(sentence_tts, "SENTENCE_GAP_S",
                sentence_tts.SENTENCE_GAP_S * s)

    def test_the_live_sequence_keeps_the_clone_voice_and_never_cools_down(self):
        replies = (" ".join(LIVE_1036_LINES), LIVE_1037_AWAY,
                   LIVE_1037_MORNING)
        voices, outs = [], []
        for text in replies:
            n = len(self.played)
            self.assertTrue(self.speak(text))
            self.join_workers()
            voices.append([self.voice_of(a) for a in self.played[n:]])
            outs.append(self.out)
        self.assertEqual(self.srv.tts_texts(),
                         list(LIVE_1036_LINES) + list(LIVE_1037_AWAY_PIECES)
                         + list(LIVE_1037_MORNING_PIECES))
        # One voice per reply, the clone's, every time.
        self.assertEqual(voices, [["clone"] * 7, ["clone"] * 2,
                                  ["clone"] * 2])
        self.assertEqual(self.kokoro_texts, [])
        self.assertEqual(self.client.failures(), 0)
        self.assertEqual(self.client.status(), ("ready", ""))   # no latch
        self.assertEqual(self.logs, [])               # no cool-down either
        out = "".join(outs)
        self.assertNotIn("Kokoro voices this line", out)
        # One compact line per clone line: ms, first / look-ahead, deadline,
        # and how late a held line came back.
        lines = [ln for ln in out.splitlines() if "[tts] clone voice" in ln]
        self.assertEqual(len(lines), 11, out)
        first = {0, 7, 9}                  # each reply's first line
        for i, ln in enumerate(lines):
            if i in first:
                self.assertRegex(ln, r"clone voice \d+ ms \(first line, "
                                     r"deadline 0\.6 s\)$")
            else:
                self.assertRegex(ln, r"clone voice \d+ ms \(look-ahead, "
                                     r"deadline \d+\.\d s"
                                     r"(, \d+\.\d s late)?\)$")


# ════════════════════════════════════════════════════════════════════════════
#  The first-line clause split (clone only)
# ════════════════════════════════════════════════════════════════════════════
LONG = ("Good evening, sir, the forecast calls for light rain this afternoon "
        "and temperatures near sixty.")
HEAD = "Good evening, sir,"
TAIL = ("the forecast calls for light rain this afternoon and temperatures "
        "near sixty.")


class ClauseSplitSpeakTests(_SpeakBase):
    def test_a_long_first_sentence_is_two_renders_head_first(self):
        from core import sentence_tts
        self.speak(f"{LONG} {S3}")
        self.assertEqual(self.srv.tts_texts(), [HEAD, TAIL, S3])
        self.assertEqual(len(self.played), 3)
        for a in self.played:
            self.assertTrue(self.is_clone(a))
        self.assertEqual(self.kokoro_texts, [])
        # Every render is the same take, so the lengths differ only by the
        # pause after it: a short clause pause after the head, the full
        # sentence pause after the end of the sentence, none at the end.
        clip = len(self.played[2])
        self.assertEqual(len(self.played[0]) - clip,
                         int(24000 * sentence_tts.CLAUSE_GAP_S))
        self.assertEqual(len(self.played[1]) - clip,
                         int(24000 * sentence_tts.SENTENCE_GAP_S))
        self.assertIn("split=clause", self.out)
        self.assertIn("(first line, deadline", self.out)
        self.assertIn("(look-ahead, deadline", self.out)

    def test_kokoro_is_unchanged_when_the_clone_is_off(self):
        self._p(self.bc, "VOICE_CLONE_ENABLED", False)
        self.speak(LONG)
        self.assertEqual(self.kokoro_texts, [LONG])
        self.assertEqual(len(self.played), 1)
        self.assertEqual(self.srv.tts_texts(), [])
        self.assertNotIn("split=clause", self.out)

    def test_no_split_while_the_clone_rests(self):
        self.rest_clone()
        self.speak(LONG)
        self.assertEqual(self.kokoro_texts, [LONG])
        self.assertEqual(len(self.played), 1)
        self.assertEqual(self.srv.tts_texts(), [])

    def test_the_rest_of_a_sentence_keeps_the_heads_voice(self):
        self.srv.fail_texts = {HEAD}
        self.speak(LONG)
        self.assertEqual(self.srv.tts_texts(), [HEAD])   # the rest: no clone
        self.assertEqual(self.kokoro_texts, [HEAD, TAIL])
        self.assertTrue(self.is_kokoro(self.played[0][:2400]))
        self.assertTrue(self.is_kokoro(self.played[1]))
        self.assertIn("Kokoro voiced the start of this sentence", self.out)

    def test_the_rest_keeps_the_whole_lines_budget(self):
        # The rest alone (77 chars) gets the 0.5 s budget; the whole line
        # (96 chars) 0.5 + 0.03 x 16 = 0.98 s. Rendering it in 0.75 s keeps
        # the clone: splitting never gives the rest less time than the
        # unsplit line had.
        self._p(self.bc, "VOICE_CLONE_TIMEOUT_S", 0.5)
        self.srv.latency_for = {TAIL: 0.75}
        self.speak(LONG)
        self.join_workers()
        self.assertEqual(self.srv.tts_texts(), [HEAD, TAIL])
        self.assertEqual(self.kokoro_texts, [])
        for a in self.played:
            self.assertTrue(self.is_clone(a))

    def test_a_prerendered_head_tells_the_rest_which_voice_to_keep(self):
        bc = self.bc
        plan = self.quiet(bc._sentence_tts_plan, LONG)
        self.assertEqual(plan[0], [HEAD, TAIL])
        kokoro_head = (self.np.full(2400, KOKORO, dtype=self.np.float32),
                       24000)
        self.quiet(bc._speak_sentences, plan[0], plan[1],
                   first_rendered=kokoro_head, first_clone=False)
        self.assertEqual(self.srv.tts_texts(), [])
        self.assertEqual(self.kokoro_texts, [TAIL])
        self.quiet(bc._speak_sentences, plan[0], plan[1],
                   first_rendered=kokoro_head, first_clone=True)
        self.assertEqual(self.srv.tts_texts(), [TAIL])


class _ScaledSpeakBase(_SpeakBase):
    """The real _speak against a serial fake server whose renders and clips
    are the live-measured ones scaled by SCALE (so is every budget), with a
    Kokoro whose clip lasts as long as the line (55 ms per char, scaled)."""

    SCALE = 0.2
    PLAY_SLEEPS = True
    TIMEOUT_S = 2.5
    LINES: tuple = ()
    RENDER_S: tuple = ()
    AUDIO_S: tuple = ()

    def clone_kw(self) -> dict:
        s = self.SCALE
        lat = {t: r * s for t, r in zip(self.LINES, self.RENDER_S)}
        wavs = {t: make_wav(lead_s=0.0, speech_s=a * s, tail_s=0.0, amp=0.3)
                for t, a in zip(self.LINES, self.AUDIO_S)}
        return {"server_factory": lambda sha: FakeCloneServer(
            ref_sha=sha, latency_for=lat, wav_for=wavs, serial=True)}

    def setUp(self):
        super().setUp()
        s = self.SCALE
        from core import kokoro_tts, sentence_tts
        self._p(self.bc, "_clone_timeout_s", return_value=self.TIMEOUT_S * s)
        self._p(self.cvc, "PER_CHAR_S", self.cvc.PER_CHAR_S * s)
        self._p(self.cvc, "NEEDED_BY_MARGIN_S",
                self.cvc.NEEDED_BY_MARGIN_S * s)
        self._p(sentence_tts, "SENTENCE_GAP_S",
                sentence_tts.SENTENCE_GAP_S * s)
        np = self.np

        def _kokoro(text, speed=1.0):
            self.kokoro_texts.append(str(text))
            n = int(24000 * 0.055 * len(str(text)) * s)
            return np.full(max(n, 240), KOKORO, dtype=np.float32), 24000
        self._p(kokoro_tts, "synthesize", side_effect=_kokoro)


class ShortOpenerReplayTests(_ScaledSpeakBase):
    """10:36 lines 4, 7 and 6 as one reply at the SHIPPED
    VOICE_CLONE_TIMEOUT_S 2.5: a 23-char opener (1.6 s of audio), then the
    114-char line that took 3.76 s against a 3.52 s budget. Only ~1.75 s of
    audio is queued ahead of it, so waiting until it is needed is not
    enough: on 995fad8 it went to Kokoro mid-reply (clone, Kokoro, clone).
    The reply already speaks in the clone, so the line is held -- about 2 s
    of pause, one voice."""

    SCALE = 0.3                   # the hold's 1.5 s live margin -> 0.45 s
    LINES = tuple(LIVE_1036_LINES[i] for i in (3, 6, 5))
    RENDER_S = tuple(LIVE_1036_RENDER_S[i] for i in (3, 6, 5))
    AUDIO_S = tuple(LIVE_1036_AUDIO_S[i] for i in (3, 6, 5))

    def test_a_short_opener_then_long_lines_keeps_one_voice(self):
        self.assertTrue(self.speak(" ".join(self.LINES)))
        self.join_workers()
        voices = [self.voice_of(a) for a in self.played]
        self.assertEqual(voices, ["clone"] * 3,
                         f"voices={voices} kokoro={self.kokoro_texts}\n"
                         f"{self.out}")
        self.assertEqual(self.kokoro_texts, [])
        self.assertEqual(self.client.failures(), 0)
        # The held line came back after it was due: the log says how late
        # (scaled: about 0.6 s here, about 2 s live).
        lines = [ln for ln in self.out.splitlines()
                 if "[tts] clone voice" in ln]
        self.assertEqual(len(lines), 3, self.out)
        self.assertRegex(lines[1], r"\(look-ahead, deadline \d+\.\d s, "
                                   r"\d+\.\d s late\)$")


class HoldTests(_SpeakBase):
    """Once a reply speaks in the clone, the next line may run up to its
    budget past the time it is due -- and no further."""

    def test_a_clone_head_is_never_followed_by_a_kokoro_tail(self):
        # The whole sentence's budget (0.5 + 0.03 x 16 = 0.98 s) is shorter
        # than the rest's 1.3 s render; unsplit (before 2026-10-04) this
        # sentence was ONE render in ONE voice. The head was the clone's,
        # so the rest is held for it.
        self._p(self.bc, "VOICE_CLONE_TIMEOUT_S", 0.5)
        self.srv.latency_for = {TAIL: 1.3}
        self.speak(LONG)
        self.join_workers()
        self.assertEqual(self.srv.tts_texts(), [HEAD, TAIL])
        voices = [self.voice_of(a) for a in self.played]
        self.assertEqual(voices, ["clone", "clone"],
                         f"one sentence, two voices: {voices}\n{self.out}")
        self.assertEqual(self.client.failures(), 0)

    def test_the_hold_is_bounded_and_a_miss_past_it_counts(self):
        # A wedged line: held for its budget past the time it was due, then
        # Kokoro voices it (counted), and -- one voice per reply (owner,
        # 2026-10-04) -- the rest of the reply stays in Kokoro, not held.
        self._p(self.bc, "VOICE_CLONE_TIMEOUT_S", 0.5)
        self.srv.latency_for = {S2: 5.0}
        t0 = time.monotonic()
        self.assertTrue(self.speak())
        self.assertLess(time.monotonic() - t0, 3.0)
        self.assertEqual([self.voice_of(a) for a in self.played],
                         ["clone", "kokoro", "kokoro"], self.out)
        self.assertEqual(self.kokoro_texts, [S2, S3])
        self.assertEqual(self.client.failures(), 1)   # S2's miss counted
        self.assertRegex(self.out, r"clone voice timed out after \d+ ms "
                                   r"\(look-ahead, deadline \d+\.\d s\); "
                                   r"Kokoro voices this line")
        self.join_workers()

    def test_a_lookahead_given_up_before_it_was_due_logs_not_counted(self):
        # Seconds of audio queued ahead, the wait capped below that: given
        # up on before anyone waited for it -- Kokoro, and not counted.
        self._p(self.bc, "VOICE_CLONE_TIMEOUT_S", 0.5)
        self._p(self.cvc, "LOOKAHEAD_MAX_S", 0.8)
        self.srv.wav_for = {S1: make_wav(lead_s=0.0, speech_s=4.0,
                                         tail_s=0.0, amp=0.3)}
        self.srv.latency_for = {S2: 3.0}
        self.speak()
        self.assertEqual(self.kokoro_texts, [S2, S3])   # one voice per reply
        self.assertEqual(self.client.failures(), 0)
        self.assertRegex(self.out, r"clone voice timed out after \d+ ms "
                                   r"\(look-ahead, deadline 0\.8 s, not "
                                   r"counted\); Kokoro voices this line")
        self.join_workers()


class StoppedReplyTests(_SpeakBase):
    def test_a_barge_in_drops_the_line_still_rendering_ahead(self):
        # S2 renders for 5 s on the worker when the listener barges in
        # during S1. Nothing will play S2: the wait ends at once, it is not
        # a miss, and Kokoro does not render it (that would only hold
        # Kokoro's lock against the next reply).
        bc = self.bc
        self.srv.latency_for = {S2: 5.0}

        def barge(n):
            if n == 1:
                time.sleep(0.2)                    # S2 is in flight by now
                bc._tts_interrupt_seq[0] += 1
        self.on_play = barge
        self.speak()
        buf = io.StringIO()
        t0 = time.monotonic()
        with contextlib.redirect_stdout(buf):
            for th in threading.enumerate():
                if th.name == "sentence-tts-synth":
                    th.join(timeout=5.0)
        self.assertLess(time.monotonic() - t0, 1.5)
        self.assertEqual(len(self.played), 1)
        self.assertEqual(self.kokoro_texts, [])
        self.assertEqual(self.client.failures(), 0)
        self.assertEqual(self.client.status()[0], "ready")
        self.assertIn("clone voice line dropped", self.out + buf.getvalue())
        self.assertEqual(self.srv.tts_texts(), [S1, S2])

    def test_a_dropped_line_is_a_short_silence_never_kokoro(self):
        # synthesise() for a line of a reply that is already stopped (as
        # the worker sees it after a barge-in): the clone gives up and the
        # line comes back as a short silence -- an array, never the
        # sentinel, and never a Kokoro render.
        from core import sentence_tts
        stop = threading.Event()
        stop.set()
        ctx = sentence_tts._render_ctx
        ctx.stop, ctx.needed_by = stop, time.monotonic() + 5.0
        self.addCleanup(setattr, ctx, "stop", None)
        self.addCleanup(setattr, ctx, "needed_by", None)
        audio, sr = self.quiet(self.bc.synthesise, S2)
        self.assertIsInstance(audio, self.np.ndarray)
        self.assertFalse(bool(self.np.any(audio)))
        self.assertLess(len(audio) / float(sr), 0.2)
        self.assertEqual(self.kokoro_texts, [])
        self.assertEqual(self.srv.tts_texts(), [])
        self.assertEqual(self.client.failures(), 0)
        self.assertIn("clone voice line dropped", self.out)

    def test_a_barge_in_during_a_split_first_line_stops_after_the_head(self):
        bc = self.bc

        def barge(n):
            if n == 1:
                bc._tts_interrupt_seq[0] += 1
        self.on_play = barge
        self.speak(f"{LONG} {S3}")
        self.join_workers()
        self.assertEqual(len(self.played), 1)
        self.assertTrue(self.is_clone(self.played[0]))
        self.assertIn("reply stopped after 1/3 parts", self.out)


class PrerenderThroughSpeakTests(_SpeakBase):
    """The R3 filler pre-render through the REAL _speak: the pre-rendered
    clause head tells the rest of the sentence which voice to keep."""

    def setUp(self):
        super().setUp()
        bc = self.bc
        self._p(bc, "_resolve_tts_preset", bc._RESOLVE_TTS_PRESET_ORIG)
        self._p(bc, "PROCESSING_FILLER_PRERENDER", True, create=True)
        self._p(bc, "_processing_filler",
                mock.Mock(is_owner_thread=lambda: True))
        bc._filler_on_device[0] = True
        self.addCleanup(bc._filler_on_device.__setitem__, 0, False)

    def test_a_kokoro_prerendered_head_keeps_the_rest_in_kokoro(self):
        self.srv.fail_texts = {HEAD}
        self.assertTrue(self.speak(LONG))
        self.join_workers()
        self.assertIn(("pre", 1), self.stats)          # the pre-render used
        self.assertEqual(self.srv.tts_texts(), [HEAD])  # the rest: no clone
        self.assertEqual(self.kokoro_texts, [HEAD, TAIL])
        self.assertEqual([self.voice_of(a) for a in self.played],
                         ["kokoro", "kokoro"])
        self.assertIn("Kokoro voiced the start of this sentence", self.out)

    def test_a_clone_prerendered_head_holds_the_rest_for_the_clone(self):
        self._p(self.bc, "VOICE_CLONE_TIMEOUT_S", 0.5)
        self.srv.latency_for = {TAIL: 1.3}
        self.assertTrue(self.speak(LONG))
        self.join_workers()
        self.assertIn(("pre", 1), self.stats)
        self.assertEqual(self.srv.tts_texts(), [HEAD, TAIL])
        self.assertEqual(self.kokoro_texts, [])
        self.assertEqual([self.voice_of(a) for a in self.played],
                         ["clone", "clone"], self.out)


# ════════════════════════════════════════════════════════════════════════════
#  The processing filler and the R3 pre-render with the clone
# ════════════════════════════════════════════════════════════════════════════
@requires_monolith
class FillerAndPrerenderTests(_CloneMixin, _FillerBase):
    def setUp(self):
        super().setUp()
        self._enable()                      # filler allowed, Kokoro, quiet off
        self.clone_setup()                  # ...then the clone selected + up
        self._p(self.bc, "_tts_layer", None)

    def test_the_filler_is_not_suppressed_by_the_clone(self):
        self.assertIsNone(self.bc._filler_suppressed())
        self.client._status = "down"        # Kokoro clips then
        self.assertIsNone(self.bc._filler_suppressed())
        self._p(self.bc, "VOICE_CLONE_MODEL", "chatterbox")
        self.assertEqual(self.bc._filler_suppressed(), "backend")

    def test_filler_clips_render_in_the_clone_voice(self):
        bc = self.bc
        audio, sr = bc._filler_render("Just a moment, sir.")
        self.assertTrue(self.is_clone(audio))
        self.assertEqual(self.srv.tts_texts(), ["Just a moment, sir."])
        self.assertEqual(self.kokoro_texts, [])
        self.rest_clone()
        audio, _ = bc._filler_render("One moment.")
        self.assertTrue(self.is_kokoro(audio))

    def test_filler_warm_misses_never_rest_the_clone(self):
        # The clips are warmed in the background (nobody waits for them):
        # however many of them fail, the clone is not sent to rest -- and a
        # warm that succeeds says nothing about the answers' misses either.
        bc = self.bc
        self.srv.tts_status = 500
        for line in ("Just a moment, sir.", "One moment.", "Working on it.",
                     "Bear with me, sir."):
            self.assertIsNone(bc._filler_render(line))
        self.assertEqual(len(self.srv.tts_texts()), 4)
        self.assertEqual(self.client.failures(), 0)
        self.assertEqual(self.client.status(), ("ready", ""))
        self.assertEqual(self.logs, [])
        self.srv.tts_status = 200
        self.client._failed("timed out", self.client._clock())
        self.assertIsNotNone(bc._filler_render("Right away, sir."))
        self.assertEqual(self.client.failures(), 1)

    def test_a_clip_rendered_across_an_engine_switch_is_not_filed(self):
        # ClipCache.warm files a clip under the key it reads AFTER the
        # render. The server coming up mid-render must not file a Kokoro
        # clip under the clone key (it would play in the wrong voice).
        bc = self.bc
        from core import processing_filler as pf
        clips = pf.ClipCache(render_fn=bc._filler_render, lock=bc._SPEAK_LOCK,
                             key_fn=bc._filler_voice_key)
        self._p(bc, "_filler_clips", clips)
        self.client._status = "starting"           # Kokoro voices clips now
        real_k = bc._filler_voice_key()

        def _kokoro_then_ready(text, speed=1.0):
            self.kokoro_texts.append(text)
            self.client._status = "ready"          # the server came up
            return self.np.full(2400, KOKORO, dtype=self.np.float32), 24000
        from core import kokoro_tts
        self._p(kokoro_tts, "synthesize", side_effect=_kokoro_then_ready)
        line = "Processing, sir."
        stored = bc._filler_clips.warm([line], stop_fn=lambda: False,
                                       wait_fn=lambda ev, s: None)
        self.assertEqual(stored, 0)
        self.assertEqual(self.kokoro_texts, [line])
        self.assertNotEqual(bc._filler_voice_key(), real_k)
        self.assertIsNone(bc._filler_clips.get(line))
        # The next warm renders it in the voice now speaking.
        self.assertEqual(bc._filler_clips.warm([line], stop_fn=lambda: False,
                                               wait_fn=lambda ev, s: None), 1)
        audio, _sr = bc._filler_clips.get(line)
        self.assertTrue(self.is_clone(audio))

    def test_prerender_goes_through_the_clone_and_is_taken_in_the_lock(self):
        bc = self.bc
        self._p(bc, "_processing_filler",
                mock.Mock(is_owner_thread=lambda: True))
        bc._filler_on_device[0] = True
        self.addCleanup(bc._filler_on_device.__setitem__, 0, False)
        self.assertTrue(bc._prerender_allowed())
        pre = self.quiet(bc._speak_prerender, SHORT, None, False, None)
        self.assertIsNotNone(pre)
        self.assertTrue(self.is_clone(pre["audio"]))
        self.assertEqual(self.srv.tts_texts(), [SHORT])
        got = self.quiet(bc._prerender_take, pre, SHORT, None)
        self.assertIsNotNone(got)
        self.assertIs(got[0], pre["audio"])

    def test_a_latch_between_prerender_and_take_drops_it(self):
        bc = self.bc
        self._p(bc, "_processing_filler",
                mock.Mock(is_owner_thread=lambda: True))
        bc._filler_on_device[0] = True
        self.addCleanup(bc._filler_on_device.__setitem__, 0, False)
        pre = self.quiet(bc._speak_prerender, SHORT, None, False, None)
        self.assertIsNotNone(pre)
        self.rest_clone()
        got = self.quiet(bc._prerender_take, pre, SHORT, None)
        self.assertIsNone(got)
        self.assertIn("pre-render dropped (voice changed)", self.out)

    def test_prerender_renders_the_clause_head_the_plan_starts_with(self):
        # One planner (_speech_chunks) for the pre-render and the plan: the
        # pre-rendered chunk is the plan's clause head, so the take keeps it.
        bc = self.bc
        self._p(bc, "_processing_filler",
                mock.Mock(is_owner_thread=lambda: True))
        self._p(bc, "SENTENCE_TTS_ENABLED", True, create=True)
        bc._filler_on_device[0] = True
        self.addCleanup(bc._filler_on_device.__setitem__, 0, False)
        pre = self.quiet(bc._speak_prerender, LONG, None, False, None)
        self.assertIsNotNone(pre)
        self.assertEqual((pre["chunk"], pre["multi"], pre["clone"]),
                         (HEAD, True, True))
        self.assertEqual(self.srv.tts_texts(), [HEAD])
        plan = self.quiet(bc._sentence_tts_plan, LONG)
        self.assertEqual(plan[0], [HEAD, TAIL])
        got = self.quiet(bc._prerender_take, pre, LONG, plan)
        self.assertIsNotNone(got, self.out)
        self.assertIs(got[0], pre["audio"])

    def test_no_prerender_with_the_in_process_clone(self):
        bc = self.bc
        self._p(bc, "_processing_filler",
                mock.Mock(is_owner_thread=lambda: True))
        self._p(bc, "VOICE_CLONE_MODEL", "chatterbox")
        self.assertFalse(bc._prerender_allowed())


if __name__ == "__main__":
    unittest.main()
