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
  * live budgeting (2026-10-04): the 10:36 briefing replayed through the
    real _speak keeps the clone voice and never cools down; a long first
    sentence is split at a clause (clone only), the rest keeps the head's
    voice and the whole line's budget; one "[tts] clone voice" line per line.

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

from tests._clone_voice_fake import (LIVE_1036_LINES, FakeCloneServer,
                                     ProfileDir, free_port, live_1036_server)
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

    def test_a_failed_sentence_is_kokoro_and_the_next_is_clone_again(self):
        self.srv.fail_texts = {S2}
        self.assertTrue(self.speak())
        self.assertEqual(len(self.played), 3)
        self.assertTrue(self.is_clone(self.played[0]))
        self.assertTrue(self.is_kokoro(
            self.played[1][: -int(24000 * 0.15)]))   # before the sentence gap
        self.assertTrue(self.is_clone(self.played[2]))
        self.assertEqual(self.kokoro_texts, [S2])
        self.assertEqual(self.client.failures(), 0)

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
    """The 7-sentence briefing of 2026-10-04 10:36: renders of 1.0-3.8 s
    (lines 5-7 slower than the fixed per-line budget) while seconds of
    earlier audio were still queued. Live, Kokoro voiced lines mid-reply and
    three such misses latched the clone off for the session. Replayed here
    with every time scaled by SCALE (budget, per-char allowance, margin,
    sentence gap, render and audio lengths), through the REAL _speak ->
    _speak_sentences -> play_pipelined -> synthesise -> client, a serial fake
    server and a fake speaker that blocks for each clip's length."""

    SCALE = 0.2
    PLAY_SLEEPS = True

    def clone_kw(self) -> dict:
        return {"server_factory":
                lambda sha: live_1036_server(sha, self.SCALE)}

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

    def test_the_1036_briefing_keeps_the_clone_voice_and_never_cools_down(self):
        self.assertTrue(self.speak(" ".join(LIVE_1036_LINES)))
        self.join_workers()
        self.assertEqual(self.srv.tts_texts(), list(LIVE_1036_LINES))
        self.assertEqual(len(self.played), 7)
        voices = [self.voice_of(a) for a in self.played]
        self.assertEqual(voices, ["clone"] * 7)       # never changed voice
        self.assertEqual(self.kokoro_texts, [])
        self.assertEqual(self.client.failures(), 0)
        self.assertEqual(self.client.status(), ("ready", ""))   # no latch
        self.assertEqual(self.logs, [])               # no cool-down either
        self.assertNotIn("Kokoro voices this line", self.out)
        # One compact line per clone line: ms, first / look-ahead, deadline.
        lines = [ln for ln in self.out.splitlines()
                 if "[tts] clone voice" in ln]
        self.assertEqual(len(lines), 7, self.out)
        self.assertRegex(lines[0], r"clone voice \d+ ms \(first line, "
                                   r"deadline 0\.5 s\)$")
        for ln in lines[1:]:
            self.assertRegex(ln, r"clone voice \d+ ms \(look-ahead, "
                                 r"deadline \d+\.\d s\)$")


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
