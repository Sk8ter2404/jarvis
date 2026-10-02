"""Speed plan R3 — the filler handoff, monolith glue (bobert_companion).

Phase 0 (scheduler hygiene; every flag ships at today's behaviour):
  * PROCESSING_FILLER_LATE_START_S reaches the shared ProcessingFiller through
    its late_fn; the default 3.0 is today's window.
  * PROCESSING_FILLER_SKIP_PLEASANTRIES: _filler_should_arm refuses a bare
    "thank you" / "hello" only when the flag is on.
  * FILLER_DUCK_HOLD: _filler_play takes ONE _audio_ducker hold after an "ok"
    claim; _filler_end_turn / _filler_teardown give it back and the next voice
    turn drops a stale one, so the count stays balanced on success, on a
    playback exception, on a raising turn and on teardown. While a hold is up
    with the flag on, a second duck() is a no-op even when the first scan
    matched nothing.

Phase 1 (PROCESSING_FILLER_PRERENDER, off by default):
  * the answer's first audio is rendered before _SPEAK_LOCK, on the armed
    turn's thread, only while a clip is on the device, Kokoro only, never
    muted / cloned / wry / night-owl-wrapped;
  * it never touches the speaking state before the lock (ordered spy);
  * inside the lock it is played only when provably identical (text, voice,
    re-resolved preset, interrupt seq, mute); else today's path runs — the
    preset-mismatch tests are the mutation targets for that check;
  * sentence 1 of a split reply is never rendered twice;
  * _filler_play's handoff wait is a bounded pure Event wait, and (real
    threads) the answer is next on the lock after the filler.
RealThreadPrerenderTests is the flag-on twin of test_monolith_processing_
filler.RealThreadOrderingTests.test_a_filler_first_answer_waits, which stays
unchanged as the proof of the default path.

The scheduler itself: tests/test_processing_filler_r3.py (light tier).
No real audio; fakes only.

    python -m unittest tests.monolith.test_monolith_filler_handoff
"""
from __future__ import annotations

import contextlib
import io
import threading
import time
import unittest
from unittest import mock

from tests.monolith.test_monolith_processing_filler import _Base


class _FakeDucker:
    """Records hold / release / duck / restore and keeps the real count."""

    def __init__(self):
        self.calls: list = []
        self.holds = 0

    def hold(self):
        self.calls.append("hold")
        self.holds += 1

    def release(self):
        self.calls.append("release")
        self.holds = max(0, self.holds - 1)

    def duck(self):
        self.calls.append("duck")

    def restore(self):
        self.calls.append("restore")


class _DuckBase(_Base):
    def setUp(self):
        super().setUp()
        bc = self.bc
        self.ducker = _FakeDucker()
        self._p(bc, "_audio_ducker", self.ducker)
        bc._filler_duck_held[0] = False
        self.addCleanup(bc._filler_duck_held.__setitem__, 0, False)


# ════════════════════════════════════════════════════════════════════════════
#  PROCESSING_FILLER_LATE_START_S
# ════════════════════════════════════════════════════════════════════════════
class LateStartWiringTests(_Base):
    def test_shared_filler_reads_the_knob_at_every_arm(self):
        bc = self.bc
        f = bc._processing_filler
        self.assertIsNotNone(f._late_fn)
        self._p(bc, "PROCESSING_FILLER_LATE_START_S", 0.6)
        self.assertEqual(f._read_late(), 0.6)
        self._p(bc, "PROCESSING_FILLER_LATE_START_S", 2.0)
        self.assertEqual(f._read_late(), 2.0)

    def test_shipped_default_is_todays_window(self):
        bc = self.bc
        from core import config
        from core import processing_filler as pf
        self.assertEqual(config.PROCESSING_FILLER_LATE_START_S, 3.0)
        self.assertIsInstance(config.PROCESSING_FILLER_LATE_START_S, float)
        f = bc._processing_filler
        self.assertEqual(f._first_retry_s, 3.0)
        self._p(bc, "PROCESSING_FILLER_LATE_START_S", 3.0)
        turn = pf.FillerTurn(0.0, 0.5, None, first_retry=f._read_late())
        # min(3.0, the fixed 1.0 cap) — exactly the window without the knob.
        self.assertEqual(f._first_window(turn), f._first_window())
        self.assertEqual(f._first_window(turn), bc._FILLER_FIRST_LATE_S)


# ════════════════════════════════════════════════════════════════════════════
#  PROCESSING_FILLER_SKIP_PLEASANTRIES
# ════════════════════════════════════════════════════════════════════════════
class PleasantryGateTests(_Base):
    PLEASANT = ("Thank you.", "Thanks, JARVIS.", "Hello.", "Good night, sir.",
                "Okay.")

    def test_flag_off_keeps_todays_gate(self):
        self._enable()
        self._p(self.bc, "PROCESSING_FILLER_SKIP_PLEASANTRIES", False)
        for text in self.PLEASANT:
            self.assertTrue(self.bc._filler_should_arm(text), text)

    def test_flag_on_skips_bare_pleasantries_only(self):
        bc = self.bc
        self._enable()
        self._p(bc, "PROCESSING_FILLER_SKIP_PLEASANTRIES", True)
        for text in self.PLEASANT:
            self.assertFalse(bc._filler_should_arm(text), text)
        for text in ("Thank you, what's the weather?", "hello, play music",
                     "what time is it"):
            self.assertTrue(bc._filler_should_arm(text), text)
        self.assertFalse(bc._filler_should_arm("stop"))   # quiet still wins

    def test_shipped_default_is_off(self):
        from core import config
        self.assertIs(config.PROCESSING_FILLER_SKIP_PLEASANTRIES, False)

    def test_dispatch_never_arms_for_a_pleasantry_with_the_flag_on(self):
        bc = self.bc
        self._enable()
        self._p(bc, "PROCESSING_FILLER_SKIP_PLEASANTRIES", True)
        self._p(bc, "_run_llm_dispatch_body", return_value="You're welcome.")
        self._p(bc, "_filler_warm_if_needed")
        fake = mock.Mock()
        fake.playing.return_value = False
        self._p(bc, "_processing_filler", fake)
        bc._run_llm_dispatch("Thank you, JARVIS.", voice=True)
        fake.arm.assert_not_called()


# ════════════════════════════════════════════════════════════════════════════
#  FILLER_DUCK_HOLD — the turn's hold
# ════════════════════════════════════════════════════════════════════════════
class DuckHoldTests(_DuckBase):
    def setUp(self):
        super().setUp()
        bc = self.bc
        self._enable()
        self.lock = threading.Lock()
        self._p(bc, "_SPEAK_LOCK", self.lock)
        self.f = self._fresh_filler()
        self._fresh_clips()
        self.play = self._p(bc, "play_with_lipsync")
        self._p(bc, "set_state")
        self._p(bc, "_filler_warm_if_needed")

    def _play(self, turn, stage=1):
        with contextlib.redirect_stdout(io.StringIO()):
            return self.bc._filler_play(turn, stage)

    def test_flag_off_takes_no_hold(self):
        bc = self.bc
        self._p(bc, "FILLER_DUCK_HOLD", False)
        turn = self.f.arm()
        self.assertEqual(self._play(turn), "played")
        bc._filler_end_turn(turn)
        self.assertEqual(self.ducker.calls, [])
        self.assertFalse(bc._filler_duck_held[0])

    def test_hold_taken_after_the_claim_and_given_back_at_turn_end(self):
        bc = self.bc
        self._p(bc, "FILLER_DUCK_HOLD", True)
        order = []
        self.ducker.hold = lambda: (order.append("hold"),
                                    setattr(self.ducker, "holds", 1))
        real_claim = self.f.claim

        def claim(turn, stage):
            order.append("claim")
            return real_claim(turn, stage)
        self._p(self.f, "claim", side_effect=claim)
        self.play.side_effect = lambda a, sr: order.append("play")
        turn = self.f.arm()
        self.assertEqual(self._play(turn), "played")
        self.assertEqual(order, ["claim", "hold", "play"])
        self.assertTrue(bc._filler_duck_held[0])
        bc._filler_end_turn(turn)
        self.assertFalse(bc._filler_duck_held[0])
        self.assertEqual(self.ducker.calls, ["release"])
        self.assertEqual(self.ducker.holds, 0)

    def test_stage_two_reuses_the_turns_hold(self):
        bc = self.bc
        self._p(bc, "FILLER_DUCK_HOLD", True)
        turn = self.f.arm()
        self.assertEqual(self._play(turn, 1), "played")
        turn.still = 0.0                       # stage 2 due at once
        self.assertEqual(self._play(turn, 2), "played")
        self.assertEqual(self.ducker.calls.count("hold"), 1)
        bc._filler_end_turn(turn)
        self.assertEqual(self.ducker.holds, 0)

    def test_no_hold_without_an_ok_claim(self):
        bc = self.bc
        self._p(bc, "FILLER_DUCK_HOLD", True)
        turn = self.f.arm()
        self.f.note_speech()                   # stage 1 is gone
        self.assertEqual(self._play(turn), "skipped")
        self.assertEqual(self.ducker.calls, [])

    def test_balanced_when_the_clip_fails_to_play(self):
        bc = self.bc
        self._p(bc, "FILLER_DUCK_HOLD", True)
        self.play.side_effect = RuntimeError("PortAudio reinit hung")
        turn = self.f.arm()
        self.assertEqual(self._play(turn), "played")
        bc._filler_end_turn(turn)
        self.assertEqual(self.ducker.holds, 0)
        self.assertFalse(bc._filler_duck_held[0])

    def test_balanced_when_the_turn_raises(self):
        # The clip plays (hold taken) inside the turn, then the turn raises:
        # the dispatch wrapper's finally (_filler_end_turn) gives it back.
        bc = self.bc
        self._p(bc, "FILLER_DUCK_HOLD", True)
        seen = []

        def body(text):
            turn = self.f._current            # armed by the wrapper
            self.assertEqual(self._play(turn), "played")
            seen.append(self.ducker.holds)
            raise RuntimeError("x")
        self._p(bc, "_run_llm_dispatch_body", side_effect=body)
        with self.assertRaises(RuntimeError), \
                contextlib.redirect_stdout(io.StringIO()):
            bc._run_llm_dispatch("what's the weather", voice=True)
        self.assertEqual(seen, [1])
        self.assertEqual(self.ducker.holds, 0)
        self.assertEqual(self.ducker.calls, ["hold", "release"])

    def test_balanced_on_teardown(self):
        bc = self.bc
        self._p(bc, "FILLER_DUCK_HOLD", True)
        turn = self.f.arm()
        self._play(turn)
        self.assertEqual(self.ducker.holds, 1)
        bc._filler_teardown("tray:restart")
        self.assertEqual(self.ducker.holds, 0)
        self.assertFalse(bc._filler_duck_held[0])
        bc._filler_teardown("again")           # nothing left to give back
        self.assertEqual(self.ducker.calls.count("release"), 1)

    def test_next_voice_turn_drops_a_stale_hold(self):
        bc = self.bc
        self._p(bc, "FILLER_DUCK_HOLD", True)
        turn = self.f.arm()
        self._play(turn)
        self.f.disarm(turn)                    # its end_turn never ran
        self.assertEqual(self.ducker.holds, 1)
        bc._filler_should_arm("what's the weather")
        self.assertEqual(self.ducker.holds, 0)
        self.assertFalse(bc._filler_duck_held[0])

    def test_release_without_a_hold_is_a_no_op(self):
        bc = self.bc
        bc._filler_duck_release()
        bc._filler_end_turn(None)
        bc._filler_should_arm("x")
        self.assertEqual(self.ducker.calls, [])

    def test_hold_helpers_never_raise(self):
        bc = self.bc
        broken = mock.Mock()
        broken.hold.side_effect = RuntimeError("x")
        broken.release.side_effect = RuntimeError("y")
        self._p(bc, "_audio_ducker", broken)
        bc._filler_duck_hold()
        self.assertFalse(bc._filler_duck_held[0])
        bc._filler_duck_held[0] = True
        bc._filler_duck_release()
        self.assertFalse(bc._filler_duck_held[0])

    def test_shipped_default_is_off(self):
        from core import config
        self.assertIs(config.FILLER_DUCK_HOLD, False)


# ════════════════════════════════════════════════════════════════════════════
#  FILLER_DUCK_HOLD — _AudioDucker.duck: one scan per hold
# ════════════════════════════════════════════════════════════════════════════
class _DuckQueue:
    def __init__(self, log):
        self.log = log

    def put(self, job):
        _plans, target, _cancellable, done = job
        self.log.append(("fade", target))
        if done is not None:
            done.set()


class DuckerOneScanTests(_Base):
    def _ducker(self, matched):
        bc = self.bc
        d = bc._AudioDucker()
        self.scans = []
        self.fades = []

        def enum():
            self.scans.append(1)
            return list(matched)
        d._check_available = lambda: True
        d._enumerate_targets = enum
        d._ensure_worker = lambda: None
        d._work_queue = _DuckQueue(self.fades)
        self._p(bc, "AUDIO_DUCKING_ENABLED", True)
        return d

    def test_flag_off_a_held_unmatched_duck_scans_every_time(self):
        self._p(self.bc, "FILLER_DUCK_HOLD", False)
        d = self._ducker([])
        d.hold()
        d.duck()
        d.duck()
        self.assertEqual(len(self.scans), 2)       # today's behaviour
        d.release()

    def test_flag_on_a_held_duck_scans_once_even_when_nothing_matched(self):
        self._p(self.bc, "FILLER_DUCK_HOLD", True)
        d = self._ducker([])
        d.hold()
        d.duck()
        d.duck()
        d.duck()
        self.assertEqual(len(self.scans), 1)
        d.release()
        self.assertFalse(d._held_ducked)
        d.duck()                                   # not held: scans again
        self.assertEqual(len(self.scans), 2)

    def test_flag_on_without_a_hold_scans_every_time(self):
        self._p(self.bc, "FILLER_DUCK_HOLD", True)
        d = self._ducker([])
        d.duck()
        d.duck()
        self.assertEqual(len(self.scans), 2)
        self.assertFalse(d._held_ducked)

    def test_flag_on_matched_duck_restores_once_at_the_last_release(self):
        self._p(self.bc, "FILLER_DUCK_HOLD", True)
        d = self._ducker([("session", 0.8)])
        self._p(self.bc, "AUDIO_DUCKING_FADE_MS", 1)
        d.hold()
        d.duck()                  # the filler clip
        d.restore()               # its end: held, no swell
        d.duck()                  # the answer: no second scan
        d.restore()
        self.assertEqual(len(self.scans), 1)
        self.assertEqual(self.fades, [("fade", self.bc.AUDIO_DUCKING_LEVEL)])
        d.release()               # the end of the turn
        self.assertEqual(self.fades[-1], ("fade", None))
        self.assertEqual(d._holds, 0)

    def test_a_failed_scan_is_retried_by_the_next_duck(self):
        self._p(self.bc, "FILLER_DUCK_HOLD", True)
        d = self._ducker([])
        boom = [True]

        def enum():
            self.scans.append(1)
            if boom[0]:
                boom[0] = False
                raise OSError("COM")
            return []
        d._enumerate_targets = enum
        d.hold()
        with contextlib.redirect_stdout(io.StringIO()):
            d.duck()
        d.duck()
        d.duck()
        self.assertEqual(len(self.scans), 2)
        d.release()


# ════════════════════════════════════════════════════════════════════════════
#  Phase 1 — PROCESSING_FILLER_PRERENDER
# ════════════════════════════════════════════════════════════════════════════
_SHORT = "The answer, sir."
_LONG = ("The first sentence of this answer is long enough to split, sir. "
         "The second sentence follows it closely. A third one ends the reply.")
_NEUTRAL = ("neutral", {"rate": "+0%", "pitch": "+0Hz", "gain": 1.0})
_CALM = ("calm_low", {"rate": "-5%", "pitch": "+0Hz", "gain": 0.9})
_PRE, _LOCKED = 0.3, 0.6     # the fake render's sample value, by where it ran


class _FakeLayer:
    """Just the core.tts surface _speak's path uses; anything else raises."""

    def __init__(self, presets=(_NEUTRAL,), wry=False, muted=False):
        self.presets = list(presets)
        self.wry = wry
        self.muted = muted
        self.resolves = 0

    def parse_wry_tag(self, text):
        return self.wry, text

    def is_muted(self):
        return self.muted

    def resolve_tts_preset(self, text, user_tone, **kw):
        self.resolves += 1
        if kw.get("wry"):
            return "wry", dict(_NEUTRAL[1])
        chosen, preset = (self.presets.pop(0) if len(self.presets) > 1
                          else self.presets[0])
        return chosen, dict(preset)


class _SpyCell(list):
    def __init__(self, name, log, init):
        super().__init__(init)
        self._name, self._log = name, log

    def __setitem__(self, i, v):
        self._log.append(("cell", self._name, v))
        super().__setitem__(i, v)


class _PreBase(_Base):
    """A box where the pre-render may run: filler on, Kokoro, flag on, the
    turn armed on THIS thread and a clip 'on the device'. _SPEAK_LOCK is a
    spy that logs enter / exit; the fake synthesise returns _PRE outside the
    lock and _LOCKED inside it, so every play says which render it was."""

    def setUp(self):
        super().setUp()
        bc = self.bc
        import numpy as np
        self.np = np
        self._enable()
        self._p(bc, "PROCESSING_FILLER_PRERENDER", True)
        self._p(bc, "SENTENCE_TTS_ENABLED", True)
        self.f = self._fresh_filler()
        self.turn = self.f.arm()
        bc._filler_on_device[0] = True
        self.addCleanup(bc._filler_on_device.__setitem__, 0, False)
        self.log = log = []
        self.depth = depth = [0]

        class SpyLock:
            def __enter__(s):
                log.append("enter")
                depth[0] += 1
                return s

            def __exit__(s, *a):
                depth[0] -= 1
                log.append("exit")
                return False
        self._p(bc, "_SPEAK_LOCK", SpyLock())
        self.on_pre_synth = None

        def synth(text):
            pin = getattr(bc._TTS_PRESET_PIN, "value", None)
            mode = getattr(bc._TTS_PRESET_PIN, "mode", None)
            inside = depth[0] > 0
            log.append(("synth", text, pin, mode, inside))
            if not inside and self.on_pre_synth is not None:
                r = self.on_pre_synth()
                if r is not None:
                    return r
            return (np.full(100, _LOCKED if inside else _PRE,
                            dtype=np.float32), 24000)
        self.synth = self._p(bc, "synthesise", side_effect=synth)

        def play(audio, sr):
            log.append(("play", round(float(audio[0]), 3), len(audio)))
        self.play = self._p(bc, "play_with_lipsync", side_effect=play)
        self._p(bc, "set_state", side_effect=lambda s: log.append(("state", s)))
        self._p(bc, "_write_hud_state")
        self._p(bc, "_session_start_time", time.time() - 3600)
        self._p(bc, "_audio_ducker", mock.Mock())
        self.stats = []
        self._p(bc, "_tt_note_stat",
                side_effect=lambda n, v: self.stats.append((n, v)))
        from core import kokoro_tts
        self._p(kokoro_tts, "is_available", return_value=True)

    # helpers
    def speak(self, text=_SHORT, **kw):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            ok = self.bc._speak(text, **kw)
        self.out = out.getvalue()
        return ok

    def synths(self):
        return [e for e in self.log if isinstance(e, tuple) and e[0] == "synth"]

    def plays(self):
        return [e for e in self.log if isinstance(e, tuple) and e[0] == "play"]

    def outside(self):
        return [e for e in self.synths() if not e[4]]

    def layer(self, **kw):
        lay = _FakeLayer(**kw)
        self._p(self.bc, "_tts_layer", lay)
        return lay

    def assert_today(self):
        """No render outside the lock; the line rendered and played inside."""
        self.assertEqual(self.outside(), [], self.log)
        self.assertTrue(self.plays(), self.log)
        self.assertEqual(self.plays()[0][1], _LOCKED, self.log)
        self.assertNotIn(("pre", 1), self.stats)


class PrerenderSingleTests(_PreBase):
    def test_renders_before_the_lock_and_plays_that_render(self):
        self.assertTrue(self.speak())
        synths = self.synths()
        self.assertEqual(len(synths), 1, self.log)
        _s, text, pin, mode, inside = synths[0]
        self.assertEqual((text, inside, mode), (_SHORT, False, "kokoro_only"))
        self.assertEqual(pin, _NEUTRAL)
        self.assertLess(self.log.index(synths[0]), self.log.index("enter"))
        self.assertEqual(self.plays(), [("play", _PRE, 100)])
        self.assertIn(("pre", 1), self.stats)
        self.assertTrue(self.bc._prerender_idle.is_set())
        self.assertIsNone(getattr(self.bc._TTS_PRESET_PIN, "value", None))

    def test_no_speaking_state_is_touched_before_the_lock(self):
        bc = self.bc
        log = self.log
        self._p(bc, "_tts_current_text", _SpyCell("tct", log, [""]))
        for name in ("_last_intent_override", "_last_wry", "_last_mood"):
            self._p(bc, name, _SpyCell(name, log, [getattr(bc, name)[0]]))
        self._p(bc._self_echo, "remember",
                side_effect=lambda t: (log.append(("remember", t)), 0)[1])
        self.speak()
        enter = log.index("enter")
        before = log[:enter]
        self.assertTrue(any(isinstance(e, tuple) and e[0] == "synth"
                            for e in before), log)
        for e in before:
            self.assertFalse(isinstance(e, tuple)
                             and e[0] in ("state", "remember", "cell"), e)
        after = log[enter:]
        for want in (("state", "speaking"), ("remember", _SHORT),
                     ("cell", "tct", _SHORT.lower())):
            self.assertIn(want, after)
        self.assertLess(after.index(("cell", "tct", _SHORT.lower())),
                        after.index(("play", _PRE, 100)))

    def test_synth_start_is_marked_at_the_prerender(self):
        bc = self.bc
        log = self.log
        self._p(bc, "_tt", side_effect=lambda op, *a, **k: log.append(
            ("tt", op) + a))
        self.speak()
        first = log.index(("tt", "mark", "synth_start"))
        self.assertLess(first, log.index("enter"))
        self.assertLess(log.index(("tt", "mark", "first_play")),
                        log.index(("play", _PRE, 100)))

    def test_volume_scale_applies_to_the_prerender(self):
        self.speak(volume_scale=0.5)
        self.assertEqual(self.plays(), [("play", round(_PRE * 0.5, 3), 100)])

    def test_the_preset_line_prints_once_as_synthesise_would(self):
        self.layer(presets=(_CALM,))
        self.speak()
        self.assertEqual(self.plays()[0][1], _PRE)
        self.assertEqual(self.out.count("[tts] preset=calm_low"), 1, self.out)
        self.assertIn("rate=-5% pitch=+0Hz gain=0.90", self.out)
        self.assertNotIn("pre-render dropped", self.out)

    def test_mood_and_intent_reach_the_prerender_resolve(self):
        bc = self.bc
        seen = []
        real = bc._resolve_tts_preset_from

        def spy(text, tone, **kw):
            seen.append(kw)
            return real(text, tone, **kw)
        self._p(bc, "_resolve_tts_preset_from", side_effect=spy)
        self.speak("[mood:concerned_soft] " + _SHORT)
        self.assertEqual(seen[0]["mood"], "concerned_soft")
        self.assertIsNone(seen[0]["intent_override"])
        self.assertFalse(seen[0]["wry"])


class PrerenderGateTests(_PreBase):
    def test_flag_off_is_todays_path(self):
        self._p(self.bc, "PROCESSING_FILLER_PRERENDER", False)
        self.speak()
        self.assert_today()
        self.assertEqual(self.stats, [])

    def test_no_clip_on_the_device(self):
        self.bc._filler_on_device[0] = False
        self.speak()
        self.assert_today()

    def test_not_from_another_thread(self):
        self.turn.owner = -1
        self.speak()
        self.assert_today()

    def test_not_for_a_closed_or_mocked_filler(self):
        self.f.shutdown("restart")
        self.speak()
        self.assert_today()
        self.log.clear()
        self._p(self.bc, "_processing_filler", mock.Mock())
        self.speak()
        self.assert_today()

    def test_not_when_env_muted(self):
        self.layer(muted=True)
        self.speak()
        self.assertEqual(self.outside(), [])

    def test_not_when_tray_muted(self):
        self.bc._tts_muted[0] = True
        self.addCleanup(self.bc._tts_muted.__setitem__, 0, False)
        self.speak()
        self.assertEqual(self.synths(), [])

    def test_not_with_the_voice_clone(self):
        self._p(self.bc, "VOICE_CLONE_ENABLED", True)
        self.speak()
        self.assert_today()

    def test_not_on_edge(self):
        self._p(self.bc, "TTS_BACKEND", "edge")
        self.speak()
        self.assert_today()

    def test_not_for_a_wry_line(self):
        self.layer(wry=True)
        self.speak("[wry] Splendid. Another meeting, sir.")
        self.assert_today()

    def test_not_while_night_owl_wraps_the_resolver(self):
        bc = self.bc
        orig = bc._resolve_tts_preset

        def night_owl(text, user_tone):
            return orig(text, user_tone)
        self._p(bc, "_resolve_tts_preset", night_owl)
        self.speak()
        self.assert_today()

    def test_a_failed_prerender_falls_back_in_the_same_call(self):
        bc = self.bc
        from core.sentence_tts import SentenceFallback

        def fail():
            raise SentenceFallback("kokoro did not voice it")
        self.on_pre_synth = fail
        self.assertTrue(self.speak())
        self.assertEqual(len(self.outside()), 1)
        self.assertEqual(self.plays(), [("play", _LOCKED, 100)])
        self.assertNotIn("pre", [n for n, _v in self.stats])
        self.assertTrue(bc._prerender_idle.is_set())

    def test_a_silent_prerender_is_not_used(self):
        np = self.np
        self.on_pre_synth = lambda: (np.zeros(100, dtype=np.float32), 24000)
        self.speak()
        self.assertEqual(self.plays(), [("play", _LOCKED, 100)])

    def test_the_prerender_never_takes_the_speech_lock(self):
        # A real lock held by "the filler": the pre-render still runs (it
        # never asks for the lock); only the in-lock part waits for it.
        bc = self.bc
        lock = threading.Lock()
        self._p(bc, "_SPEAK_LOCK", lock)
        lock.acquire()
        seen = []

        def at_render():
            seen.append(lock.locked())
            lock.release()        # the clip ends after the render

        def answer():
            self.turn.owner = threading.get_ident()   # this thread's turn
            self.speak()
        self.on_pre_synth = at_render
        th = threading.Thread(target=answer, daemon=True)
        th.start()
        th.join(2.0)
        if th.is_alive():         # never strand the thread on the lock
            lock.release()
            th.join(2.0)
        self.assertFalse(th.is_alive())
        self.assertEqual(seen, [True])
        self.assertEqual(self.plays(), [("play", _PRE, 100)])


class PrerenderDiscardTests(_PreBase):
    def test_an_interrupt_during_the_prerender_discards_it(self):
        bc = self.bc
        self.on_pre_synth = lambda: bc._tts_interrupt_seq.__setitem__(
            0, bc._tts_interrupt_seq[0] + 1)
        self.speak()
        self.assertNotIn(_PRE, [p[1] for p in self.plays()])
        self.assertEqual(self.plays(), [("play", _LOCKED, 100)])
        self.assertIn(("pre", 0), self.stats)
        self.assertIn("pre-render dropped (interrupted)", self.out)

    def test_a_preset_mismatch_inside_the_lock_falls_back(self):
        # The pre-render resolved calm_low; by the time the lock is taken the
        # same resolve says neutral: today's render must run, unpinned.
        self.layer(presets=(_CALM, _NEUTRAL))
        self.speak()
        self.assertNotIn(_PRE, [p[1] for p in self.plays()])
        self.assertEqual(self.plays(), [("play", _LOCKED, 100)])
        inside = [e for e in self.synths() if e[4]]
        self.assertEqual(len(inside), 1)
        self.assertIsNone(inside[0][2])            # today's unpinned render
        self.assertIn(("pre", 0), self.stats)
        self.assertIn("pre-render dropped (preset changed)", self.out)

    def test_mute_switched_on_during_the_prerender_discards_it(self):
        bc = self.bc
        self.addCleanup(bc._tts_muted.__setitem__, 0, False)
        self.on_pre_synth = lambda: bc._tts_muted.__setitem__(0, True)
        self.speak()
        self.assertNotIn(_PRE, [p[1] for p in self.plays()])
        self.assertIn("pre-render dropped (muted)", self.out)

    def test_a_voice_change_during_the_prerender_discards_it(self):
        bc = self.bc
        self.on_pre_synth = lambda: setattr(bc, "VOICE_CLONE_ENABLED", True)
        self.speak()
        self.assertNotIn(_PRE, [p[1] for p in self.plays()])
        self.assertIn("pre-render dropped (voice changed)", self.out)

    def test_take_never_raises(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertIsNone(self.bc._prerender_take({}, _SHORT, None))
            self.assertIsNone(self.bc._prerender_take(None, _SHORT, None))


class PrerenderSentenceTests(_PreBase):
    def setUp(self):
        super().setUp()
        from core import sentence_tts
        self.chunks = sentence_tts.plan_chunks(_LONG)
        self.assertGreaterEqual(len(self.chunks), 3)
        self.gap = int(24000 * sentence_tts.SENTENCE_GAP_S)

    def test_sentence_one_is_prerendered_and_never_rendered_again(self):
        self.speak(_LONG)
        texts = [e[1] for e in self.synths()]
        self.assertEqual(texts.count(self.chunks[0]), 1, self.log)
        first = self.synths()[0]
        self.assertEqual((first[1], first[4], first[3]),
                         (self.chunks[0], False, "kokoro_only"))
        self.assertEqual(self.plays(), [
            ("play", _PRE, 100 + self.gap), ("play", _LOCKED, 100 + self.gap),
            ("play", _LOCKED, 100)])
        self.assertIn(("pre", 1), self.stats)
        self.assertIn(f"[tts] {len(self.chunks)} sentences preset=", self.out)

    def test_the_sentence_pin_is_the_prerenders_preset(self):
        self.layer(presets=(_CALM,))
        self.speak(_LONG)
        pins = [e[2] for e in self.synths()]
        self.assertEqual(len(pins), len(self.chunks))
        for pin in pins:
            self.assertEqual(pin, _CALM)

    def test_volume_scale_applies_to_the_prerendered_sentence(self):
        self.speak(_LONG, volume_scale=0.5)
        self.assertEqual(self.plays()[0][1], round(_PRE * 0.5, 3))
        self.assertEqual(self.plays()[1][1], round(_LOCKED * 0.5, 3))

    def test_a_preset_mismatch_renders_sentence_one_again(self):
        self.layer(presets=(_CALM, _NEUTRAL))
        self.speak(_LONG)
        texts = [e[1] for e in self.synths() if e[4]]
        self.assertEqual(texts[0], self.chunks[0])
        self.assertEqual(self.plays()[0][1], _LOCKED)
        self.assertIn(("pre", 0), self.stats)
        self.assertIn("pre-render dropped (preset changed)", self.out)

    def test_split_disabled_inside_the_lock_drops_a_sentence_prerender(self):
        bc = self.bc
        self.on_pre_synth = lambda: setattr(bc, "SENTENCE_TTS_ENABLED", False)
        self.speak(_LONG)
        self.assertEqual(self.plays(), [("play", _LOCKED, 100)])
        self.assertIn("pre-render dropped (split changed)", self.out)


# ════════════════════════════════════════════════════════════════════════════
#  Phase 1 — the filler side: device flag + bounded handoff wait
# ════════════════════════════════════════════════════════════════════════════
class _FakeIdle:
    def __init__(self, is_set):
        self._set = is_set
        self.waits = []

    def is_set(self):
        return self._set

    def wait(self, timeout=None):
        self.waits.append(timeout)
        return self._set

    def set(self):
        self._set = True


class HandoffTests(_Base):
    def setUp(self):
        super().setUp()
        self._enable()
        self.lock = threading.Lock()
        self._p(self.bc, "_SPEAK_LOCK", self.lock)
        self.f = self._fresh_filler()
        self._fresh_clips()
        self.seen = []
        self.play = self._p(
            self.bc, "play_with_lipsync",
            side_effect=lambda a, sr: self.seen.append(
                self.bc._filler_on_device[0]))
        self._p(self.bc, "set_state")

    def _play(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            r = self.bc._filler_play(self.f.arm(), 1)
        self.out = out.getvalue()
        return r

    def test_device_flag_only_while_the_clip_plays(self):
        self.assertFalse(self.bc._filler_on_device[0])
        self.assertEqual(self._play(), "played")
        self.assertEqual(self.seen, [True])
        self.assertFalse(self.bc._filler_on_device[0])

    def test_device_flag_cleared_when_the_clip_fails(self):
        self.play.side_effect = RuntimeError("PortAudio")
        self.assertEqual(self._play(), "played")
        self.assertFalse(self.bc._filler_on_device[0])

    def test_no_wait_when_nothing_is_in_flight(self):
        idle = _FakeIdle(True)
        self._p(self.bc, "_prerender_idle", idle)
        self._play()
        self.assertEqual(idle.waits, [])

    def test_the_handoff_wait_is_bounded(self):
        idle = _FakeIdle(False)          # a pre-render that never finishes
        self._p(self.bc, "_prerender_idle", idle)
        self.assertEqual(self._play(), "played")
        self.assertEqual(idle.waits, [self.bc._PRERENDER_HANDOFF_WAIT_S])
        self.assertLessEqual(self.bc._PRERENDER_HANDOFF_WAIT_S, 1.5)
        self.assertIn("answer still rendering", self.out)
        self.assertTrue(self.lock.acquire(blocking=False))   # released
        self.lock.release()
        self.assertFalse(self.f.playing())

    def test_the_wait_happens_after_the_clip_while_holding_the_lock(self):
        bc = self.bc
        order = []
        lock = self.lock
        self.play.side_effect = lambda a, sr: order.append("clip")

        class Idle(_FakeIdle):
            def wait(s, timeout=None):
                order.append(("wait", lock.locked(),
                              bc._filler_on_device[0]))
                return True
        self._p(bc, "_prerender_idle", Idle(False))
        self._play()
        self.assertEqual(order, ["clip", ("wait", True, False)])

    def test_a_real_event_wait_ends_when_the_prerender_does(self):
        bc = self.bc
        ev = threading.Event()
        self._p(bc, "_prerender_idle", ev)
        timer = threading.Timer(0.1, ev.set)
        timer.start()
        self.addCleanup(timer.cancel)
        t0 = time.monotonic()
        self._play()
        waited = time.monotonic() - t0
        self.assertGreaterEqual(waited, 0.05)
        self.assertLess(waited, 1.4)

    def test_a_real_event_wait_gives_up_at_the_bound(self):
        bc = self.bc
        self._p(bc, "_prerender_idle", threading.Event())   # never set
        self._p(bc, "_PRERENDER_HANDOFF_WAIT_S", 0.1)
        t0 = time.monotonic()
        self.assertEqual(self._play(), "played")
        self.assertLess(time.monotonic() - t0, 1.0)

    def test_shipped_default_is_off(self):
        from core import config
        self.assertIs(config.PROCESSING_FILLER_PRERENDER, False)


class _OrderIdle:
    """A real Event that logs clear / set into `order`."""

    def __init__(self, order):
        self._e = threading.Event()
        self._e.set()
        self._order = order

    def clear(self):
        self._order.append("idle.clear")
        self._e.clear()

    def set(self):
        self._order.append("idle.set")
        self._e.set()

    def is_set(self):
        return self._e.is_set()

    def wait(self, timeout=None):
        return self._e.wait(timeout)


class _OrderCell(list):
    """The device-flag cell, logging every read of it into `order`."""

    def __init__(self, order, init):
        super().__init__(init)
        self._order = order

    def __getitem__(self, i):
        self._order.append("device.read")
        return super().__getitem__(i)


class HandoffOrderTests(_PreBase):
    """The pre-render half of the race-free handoff (see the comments in
    _speak_prerender and _filler_play). The pre-render marks itself in flight
    BEFORE it reads the device flag; _filler_play clears the flag BEFORE it
    reads the Event (pinned by HandoffTests.test_the_wait_happens_after_the_
    clip_while_holding_the_lock). With both orders, either the pre-render
    sees the clip gone and stops, or the filler sees it in flight and waits.
    Read the flag first and a clip ending in between lets the filler release
    the lock with no wait while the answer is still rendered outside it.
    Nothing else catches that swap: every other test is single-threaded or
    too coarse to land in the gap."""

    def test_in_flight_is_marked_before_the_device_flag_is_read(self):
        bc = self.bc
        order = []
        self._p(bc, "_prerender_idle", _OrderIdle(order))
        self._p(bc, "_filler_on_device", _OrderCell(order, [True]))
        self.speak()
        self.assertEqual(self.plays(), [("play", _PRE, 100)], self.log)
        self.assertIn("device.read", order)
        self.assertEqual(order[0], "idle.clear", order)
        self.assertLess(order.index("idle.clear"),
                        order.index("device.read"))
        self.assertEqual(order[-1], "idle.set", order)

    def test_a_clip_gone_before_the_read_still_clears_and_sets(self):
        # No clip on the device: nothing is rendered, and the in-flight mark
        # is still cleared then set again (never left clear).
        bc = self.bc
        order = []
        self._p(bc, "_prerender_idle", _OrderIdle(order))
        self._p(bc, "_filler_on_device", _OrderCell(order, [False]))
        self.speak()
        self.assert_today()
        self.assertEqual(order, ["idle.clear", "device.read", "idle.set"])
        self.assertTrue(bc._prerender_idle.is_set())


# ════════════════════════════════════════════════════════════════════════════
#  Phase 1 — real threads (the flag-on twin of RealThreadOrderingTests
#  .test_a_filler_first_answer_waits, which stays as the default-off proof)
# ════════════════════════════════════════════════════════════════════════════
class _RecLock:
    """A real lock that records which thread got it, in order."""

    def __init__(self):
        self._l = threading.Lock()
        self.got = []

    def acquire(self, blocking=True, timeout=-1):
        ok = self._l.acquire(blocking, timeout)
        if ok:
            self.got.append(threading.current_thread().name)
        return ok

    def release(self):
        self._l.release()

    def locked(self):
        return self._l.locked()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *a):
        self.release()
        return False


class RealThreadPrerenderTests(_Base):
    def setUp(self):
        super().setUp()
        import numpy as np
        bc = self.bc
        self._enable()
        self._p(bc, "PROCESSING_FILLER_PRERENDER", True)
        self.lock = _RecLock()
        self._p(bc, "_SPEAK_LOCK", self.lock)
        self.f = self._fresh_filler()
        self._fresh_clips()
        self.order = []
        self.active = [0]
        self.overlap = [False]
        self.filler_started = threading.Event()
        self.filler_clip_done = threading.Event()
        self.release_filler = threading.Event()
        self.answer_playing = threading.Event()
        self.renders = []
        self.at_render = None

        def _play(audio, sr):
            self.active[0] += 1
            if self.active[0] > 1:
                self.overlap[0] = True
            try:
                if len(audio) == 2400:        # the cached filler clip
                    self.order.append("filler")
                    self.filler_started.set()
                    self.release_filler.wait(2.0)
                    self.filler_clip_done.set()
                else:
                    self.order.append("answer")
                    self.answer_playing.set()
            finally:
                self.active[0] -= 1
        self._p(bc, "play_with_lipsync", side_effect=_play)

        def _synth(text):
            pinned = getattr(bc._TTS_PRESET_PIN, "value", None) is not None
            self.renders.append((threading.current_thread().name, pinned,
                                 self.lock.locked(), list(self.lock.got)))
            if self.at_render is not None:
                self.at_render()
            return np.full(10, 0.1, dtype=np.float32), 24000
        self._p(bc, "synthesise", side_effect=_synth)
        self._p(bc, "set_state")
        self._p(bc, "_write_hud_state")
        self._p(bc, "_session_start_time", time.time() - 3600)
        self._p(bc, "_tt_note_stat")

    def _thread(self, fn, *a, name=None):
        th = threading.Thread(target=fn, args=a, daemon=True, name=name)
        th.start()
        self.addCleanup(th.join, 2.0)
        return th

    def test_a_filler_first_answer_is_rendered_during_the_clip(self):
        bc = self.bc
        turn = self.f.arm()                     # this thread owns the turn
        me = threading.current_thread().name
        # The clip ends as soon as the answer has been rendered.
        self.at_render = self.release_filler.set
        with contextlib.redirect_stdout(io.StringIO()):
            tf = self._thread(bc._filler_play, turn, 1, name="filler-t")
            self.assertTrue(self.filler_started.wait(2.0))
            bc._speak("The answer, sir.")
            tf.join(2.0)
        # Filler then answer, never overlapping on the device.
        self.assertEqual(self.order, ["filler", "answer"])
        self.assertFalse(self.overlap[0])
        # The render happened before the answer acquired the lock: on the
        # turn's thread, pinned, while the filler still held the lock.
        self.assertEqual(self.renders, [(me, True, True, ["filler-t"])])
        # ...and the lock went to the answer next.
        self.assertEqual(self.lock.got, ["filler-t", me])

    def test_the_filler_keeps_the_lock_until_the_render_is_done(self):
        bc = self.bc
        armed = threading.Event()
        go = threading.Event()
        render_started = threading.Event()
        render_go = threading.Event()
        box = {}

        def answer():
            box["turn"] = self.f.arm()          # the answer thread owns it
            armed.set()
            go.wait(2.0)
            bc._speak("The answer, sir.")

        def at_render():
            render_started.set()
            render_go.wait(2.0)
        self.at_render = at_render
        with contextlib.redirect_stdout(io.StringIO()):
            ta = self._thread(answer, name="answer-t")
            self.assertTrue(armed.wait(2.0))
            tf = self._thread(bc._filler_play, box["turn"], 1,
                              name="filler-t")
            self.assertTrue(self.filler_started.wait(2.0))
            go.set()
            self.assertTrue(render_started.wait(2.0))
            self.release_filler.set()           # the clip ends mid-render
            self.assertTrue(self.filler_clip_done.wait(2.0))
            time.sleep(0.15)
            # Clip over, render still running: the filler holds the lock
            # (nobody else can take it) and the answer is not playing yet.
            self.assertTrue(self.lock.locked())
            self.assertTrue(tf.is_alive())
            self.assertFalse(self.answer_playing.is_set())
            render_go.set()
            self.assertTrue(self.answer_playing.wait(2.0))
            tf.join(2.0)
            ta.join(2.0)
        self.assertEqual(self.order, ["filler", "answer"])
        self.assertEqual(self.lock.got, ["filler-t", "answer-t"])
        self.assertEqual([r[:2] for r in self.renders], [("answer-t", True)])
        self.assertFalse(self.overlap[0])


if __name__ == "__main__":   # pragma: no cover
    unittest.main()
