"""Monolith glue for the processing filler (bobert_companion._filler_*).

The scheduler itself (core/processing_filler.py) is covered CI-light by
tests/test_processing_filler.py. This file pins the monolith wiring, including
the nine review fixes of 2026-09-29:

  1. never speak while an in-turn mic capture is live (record_speech, Path B,
     enrolment, the self-diagnostic probe, any get_mic_buffer call), and
     captures count as activity for the silence clock;
  2. _speak's end-of-speech mark is INSIDE _SPEAK_LOCK;
  3. the dispatch wrapper waits (bounded) for a playing clip before returning;
  4. suppressed while the wake-word barge-in listener runs;
  5. clips render on CPU Kokoro only, and the warm yields between lines;
  6. never on the realtime voice path;
  7. latched off on every restart / shutdown / blue-green entry;
  8. never armed for a stop / cancel / quiet command;
  9. (line banks — CI file).

Every test patches in a FRESH ProcessingFiller / ClipCache (or a fake) instead
of mutating the shared instances. No real audio, no real LLM; the only real
threads are the bounded ordering tests (Events with <= 2 s timeouts).

Run locally (full-deps tier):
    python -m unittest tests.monolith.test_monolith_processing_filler
"""
from __future__ import annotations

import ast
import contextlib
import io
import os
import sys
import threading
import time
import types
import unittest
from unittest import mock

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


_SKILL_KEYS = ("skill_dnd_focus_mode", "skill_night_owl_mode",
               "skill_game_mode", "skill_wake_listener",
               "skill_kinect_gestures")


def _restore_module(key, had, saved):
    if had:
        sys.modules[key] = saved
    else:
        sys.modules.pop(key, None)


class _Stop(BaseException):
    """Raised by a fake _filler_teardown to halt the function under test right
    after the latch (BaseException: the production guards catch Exception)."""


class _Other(BaseException):
    """Raised on any OTHER host attribute access, so a mutant that skips the
    latch is stopped before it can restart / exit anything."""


class _RecHost:
    """A fake monolith host that records attribute reads in order."""

    def __init__(self, **allowed):
        object.__setattr__(self, "seen", [])
        object.__setattr__(self, "_allowed", allowed)

    def __getattr__(self, name):
        self.seen.append(name)
        if name == "_filler_teardown":
            def _latch(reason):
                raise _Stop(reason)
            return _latch
        if name in self._allowed:
            return self._allowed[name]
        raise _Other(name)


class _RecFactory:
    """thread_factory that records instead of starting."""

    def __init__(self):
        self.made = []

    def __call__(self, target=None, args=(), name=None, daemon=None):
        th = types.SimpleNamespace(target=target, args=args, name=name,
                                   daemon=daemon, started=False)

        def _start():
            th.started = True
        th.start = _start
        self.made.append(th)
        return th


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

    def _enable(self):
        """A box where the filler is allowed: enabled, Kokoro, nothing quiet."""
        bc = self.bc
        self._p(bc, "PROCESSING_FILLER_ENABLED", True)
        self._p(bc, "TTS_BACKEND", "kokoro")
        self._p(bc, "VOICE_CLONE_ENABLED", False)
        self._p(bc, "_tts_layer", None)
        self._p(bc, "_is_staging", lambda: False)
        self._p(bc, "_GESTURE_BARGE_IN_ENABLED", False)
        bc._tts_muted[0] = False
        bc._sleep_mode[0] = False
        bc._standby_mode[0] = False
        bc._focus_mode[0] = False
        bc._realtime_session[0] = None
        self.addCleanup(lambda: bc._realtime_session.__setitem__(0, None))
        for cell in ("_record_speech_active", "_pathb_mic_active",
                     "_enroll_capture_active"):
            getattr(bc, cell)[0] = False
        bc._diag_capture_active[0] = 0
        bc._ambient_stream_active[0] = 0
        # No skill modules loaded (tests add fakes back per case); restored
        # exactly afterwards.
        for key in _SKILL_KEYS:
            had = key in sys.modules
            saved = sys.modules.pop(key, None)
            self.addCleanup(_restore_module, key, had, saved)

    def _fresh_filler(self, **kw):
        bc = self.bc
        from core import processing_filler as pf
        kw.setdefault("thread_factory", _RecFactory())
        f = pf.ProcessingFiller(
            play_fn=bc._filler_play, suppressed_fn=bc._filler_suppressed,
            delays_fn=lambda: (2.5, 12.0), **kw)
        self._p(bc, "_processing_filler", f)
        return f

    def _fresh_clips(self, lines=None):
        bc = self.bc
        import numpy as np
        from core import processing_filler as pf
        c = pf.ClipCache(render_fn=lambda t: None, lock=bc._SPEAK_LOCK,
                         key_fn=lambda: ("k",))
        for line in (lines if lines is not None
                     else pf.FIRST_LINES + pf.STILL_LINES):
            c.put(line, (np.full(2400, 0.1, dtype=np.float32), 24000))
        self._p(bc, "_filler_clips", c)
        return c


# ════════════════════════════════════════════════════════════════════════════
#  _run_llm_dispatch wrapper
# ════════════════════════════════════════════════════════════════════════════
class DispatchWrapperTests(_Base):
    def setUp(self):
        super().setUp()
        self.body = self._p(self.bc, "_run_llm_dispatch_body",
                            return_value="reply")
        self.fake = mock.Mock()
        self.fake.arm.return_value = "TURN"
        self.fake.playing.return_value = False
        self._p(self.bc, "_processing_filler", self.fake)
        self.warm = self._p(self.bc, "_filler_warm_if_needed")

    def test_no_kwarg_never_arms(self):
        self._enable()
        self.assertEqual(self.bc._run_llm_dispatch("what time is it"), "reply")
        self.fake.arm.assert_not_called()
        self.fake.disarm.assert_not_called()
        self.body.assert_called_once_with("what time is it")

    def test_voice_turn_arms_and_disarms_once(self):
        self._enable()
        self.bc._run_llm_dispatch("what's the weather", voice=True)
        self.fake.arm.assert_called_once_with()
        self.fake.disarm.assert_called_once_with("TURN")
        self.warm.assert_called_once_with()

    def test_disarm_runs_when_body_raises(self):
        self._enable()
        self.body.side_effect = RuntimeError("boom")
        with self.assertRaises(RuntimeError):
            self.bc._run_llm_dispatch("what's the weather", voice=True)
        self.fake.disarm.assert_called_once_with("TURN")

    def test_arm_failure_never_breaks_the_turn(self):
        self._enable()
        self.fake.arm.side_effect = RuntimeError("no threads")
        self.assertEqual(
            self.bc._run_llm_dispatch("what's the weather", voice=True), "reply")
        self.fake.disarm.assert_called_once_with(None)

    def test_disabled_never_arms(self):
        self._enable()
        self._p(self.bc, "PROCESSING_FILLER_ENABLED", False)
        self.bc._run_llm_dispatch("what's the weather", voice=True)
        self.fake.arm.assert_not_called()

    def test_quiet_commands_never_arm(self):   # fix 8
        self._enable()
        for text in ("stop", "Shut up, JARVIS.", "cancel", "enough", "abort"):
            self.bc._run_llm_dispatch(text, voice=True)
        self.fake.arm.assert_not_called()
        self.bc._run_llm_dispatch("play something relaxing", voice=True)
        self.fake.arm.assert_called_once_with()

    def test_realtime_path_never_arms(self):   # fix 6
        self._enable()
        self.bc._realtime_session[0] = object()
        self.bc._run_llm_dispatch("what's the weather", voice=True)
        self.fake.arm.assert_not_called()

    def test_waits_bounded_for_a_playing_clip(self):   # fix 3
        self._enable()
        self.fake.playing.return_value = True
        self.fake.wait_idle.return_value = True
        self.bc._run_llm_dispatch("what's the weather", voice=True)
        self.fake.wait_idle.assert_called_once_with(3.0)
        self.assertLessEqual(self.bc._FILLER_END_WAIT_S, 3.0)

    def test_main_passes_voice_only_for_mic_turns(self):
        with io.open(os.path.join(_ROOT, "bobert_companion.py"),
                     encoding="utf-8") as fh:
            src = fh.read()
        self.assertIn("reply = _run_llm_dispatch(text, voice=_injected_text "
                      "is None)", src)


class DefaultOffTests(_Base):
    def test_default_off_creates_no_thread(self):
        bc = self.bc
        self._p(bc, "PROCESSING_FILLER_ENABLED", False)  # the shipped default
        fac = _RecFactory()
        f = self._fresh_filler(thread_factory=fac)
        self._p(bc, "_run_llm_dispatch_body", return_value="r")
        warm = self._p(bc._filler_clips, "warm_async")
        bc._run_llm_dispatch("what's the weather", voice=True)
        self.assertEqual(fac.made, [])
        self.assertFalse(f.armed())
        warm.assert_not_called()

    def test_hooks_never_raise(self):
        bc = self.bc
        broken = mock.Mock()
        for name in ("note_speech", "shutdown", "begin_capture",
                     "end_capture", "playing", "wait_idle"):
            getattr(broken, name).side_effect = RuntimeError("x")
        self._p(bc, "_processing_filler", broken)
        bc._filler_note_speech()
        bc._filler_teardown("restart")
        bc._filler_capture_begin()
        bc._filler_capture_end()
        bc._filler_capture_mark(wait=True)
        self._p(bc, "_get_mic_buffer_impl", return_value="AUDIO")
        self.assertEqual(bc.get_mic_buffer(1.0), "AUDIO")

    def test_default_off_capture_never_waits(self):
        bc = self.bc
        self._p(bc, "PROCESSING_FILLER_ENABLED", False)
        f = self._fresh_filler()
        waits = self._p(f, "wait_idle")
        self._p(bc, "_get_mic_buffer_impl", return_value="AUDIO")
        self.assertEqual(bc.get_mic_buffer(1.0), "AUDIO")
        waits.assert_not_called()
        self.assertFalse(f.capturing())


# ════════════════════════════════════════════════════════════════════════════
#  _speak marks (fix 2)
# ════════════════════════════════════════════════════════════════════════════
class SpeakMarkTests(_Base):
    def setUp(self):
        super().setUp()
        bc = self.bc
        import numpy as np
        self.events = []
        fake = mock.Mock()
        fake.note_speech.side_effect = lambda: self.events.append("note")
        self._p(bc, "_processing_filler", fake)
        events = self.events

        class SpyLock:
            def __enter__(self_):
                events.append("enter")
                return self_

            def __exit__(self_, *a):
                events.append("exit")
                return False

        self._p(bc, "_SPEAK_LOCK", SpyLock())
        self._p(bc, "synthesise",
                side_effect=lambda t: (np.zeros(10, dtype=np.float32), 24000))
        self._p(bc, "play_with_lipsync",
                side_effect=lambda a, sr: events.append("play"))
        self._p(bc, "set_state")
        self._p(bc, "_write_hud_state")
        self._p(bc, "_session_start_time", time.time() - 3600)
        self._p(bc, "_tts_layer", None)
        self._p(bc, "_is_staging", lambda: False)
        bc._tts_muted[0] = False

    def test_marks_before_lock_and_inside_it(self):
        self.bc._speak("The weather is fine, sir.")
        self.assertEqual(self.events, ["note", "enter", "play", "note", "exit"])

    def test_no_mark_on_early_returns(self):
        bc = self.bc
        bc._tts_muted[0] = True
        bc._speak("muted line")
        bc._tts_muted[0] = False
        bc._speak("   ")
        with mock.patch.object(bc, "_is_staging", lambda: True), \
                mock.patch.dict(sys.modules,
                                {"staging_instance": mock.Mock()}):
            bc._speak("staged line")
        self.assertEqual(self.events, [])


# ════════════════════════════════════════════════════════════════════════════
#  _filler_suppressed (fixes 1, 4, 5, 6)
# ════════════════════════════════════════════════════════════════════════════
class SuppressedTests(_Base):
    def setUp(self):
        super().setUp()
        self._enable()

    def test_clear_box_is_not_suppressed(self):
        self.assertIsNone(self.bc._filler_suppressed())

    def test_each_reason(self):
        bc = self.bc
        cases = [
            ("disabled", lambda: self._p(bc, "PROCESSING_FILLER_ENABLED", False)),
            ("tray-mute", lambda: bc._tts_muted.__setitem__(0, True)),
            ("env-mute", lambda: self._p(
                bc, "_tts_layer", mock.Mock(is_muted=lambda: True))),
            ("staging", lambda: self._p(bc, "_is_staging", lambda: True)),
            ("standby", lambda: bc._standby_mode.__setitem__(0, True)),
            ("standby", lambda: bc._sleep_mode.__setitem__(0, True)),
            ("focus", lambda: bc._focus_mode.__setitem__(0, True)),
            ("backend", lambda: self._p(bc, "TTS_BACKEND", "edge")),
            ("backend", lambda: self._p(bc, "VOICE_CLONE_ENABLED", True)),
            ("realtime", lambda: bc._realtime_session.__setitem__(0, object())),
            ("dnd", lambda: sys.modules.__setitem__(
                "skill_dnd_focus_mode",
                types.SimpleNamespace(is_focus_mode_active=lambda: True))),
            ("night-owl", lambda: sys.modules.__setitem__(
                "skill_night_owl_mode",
                types.SimpleNamespace(is_night_owl_active=lambda: True))),
            ("game", lambda: sys.modules.__setitem__(
                "skill_game_mode",
                types.SimpleNamespace(_st=types.SimpleNamespace(active=True)))),
            ("barge-in", lambda: sys.modules.__setitem__(
                "skill_wake_listener",
                types.SimpleNamespace(_detector=mock.Mock(
                    is_running=lambda: True)))),
            ("barge-in", lambda: (
                self._p(bc, "_GESTURE_BARGE_IN_ENABLED", True),
                sys.modules.__setitem__("skill_kinect_gestures", object()))),
            ("mic-capture", lambda: bc._record_speech_active.__setitem__(0, True)),
            ("mic-capture", lambda: bc._pathb_mic_active.__setitem__(0, True)),
            ("mic-capture", lambda: bc._enroll_capture_active.__setitem__(0, True)),
            ("mic-capture", lambda: bc._diag_capture_active.__setitem__(0, 1)),
            ("mic-capture", lambda: self._fresh_filler().begin_capture()),
        ]
        for want, apply in cases:
            with self.subTest(want=want):
                apply()
                self.assertEqual(bc._filler_suppressed(), want)
                # Undo by re-running the clean-box setup on top.
                self._enable()
                self._fresh_filler()
                self.assertIsNone(bc._filler_suppressed(),
                                  f"setup did not reset after {want}")

    def test_a_live_capture_does_not_stop_arming(self):
        # At arm time the only capture that can be live is a BACKGROUND one
        # (the dispatch thread is the one arming); refusing the whole turn for
        # it lost the filler on ~1/3 of turns on a default box.
        bc = self.bc
        from core import processing_filler as pf
        f = pf.ProcessingFiller(play_fn=bc._filler_play,
                                suppressed_fn=bc._filler_arm_suppressed,
                                delays_fn=lambda: (2.5, 12.0),
                                thread_factory=_RecFactory())
        self._p(bc, "_processing_filler", f)
        bc._pathb_mic_active[0] = True
        self.assertEqual(bc._filler_suppressed(), "mic-capture")
        self.assertIsNone(bc._filler_arm_suppressed())
        self.assertIsNotNone(f.arm())
        # ...but every other reason still refuses the arm.
        bc._standby_mode[0] = True
        self.assertEqual(bc._filler_arm_suppressed(), "standby")

    def test_module_filler_arms_through_the_transient_gate(self):
        self.assertIs(self.bc._processing_filler._suppressed_fn,
                      self.bc._filler_arm_suppressed)

    def test_inactive_modes_and_idle_listener_do_not_suppress(self):
        bc = self.bc
        sys.modules["skill_dnd_focus_mode"] = types.SimpleNamespace(
            is_focus_mode_active=lambda: False)
        sys.modules["skill_night_owl_mode"] = types.SimpleNamespace(
            is_night_owl_active=lambda: False)
        sys.modules["skill_game_mode"] = types.SimpleNamespace(
            _st=types.SimpleNamespace(active=False))
        sys.modules["skill_wake_listener"] = types.SimpleNamespace(
            _detector=mock.Mock(is_running=lambda: False))
        self.assertIsNone(bc._filler_suppressed())

    def test_ambient_stream_is_excluded(self):
        self.bc._ambient_stream_active[0] = 2
        self.assertIsNone(self.bc._filler_suppressed())

    def test_never_imports_skills_package_modules(self):
        before = {k for k in sys.modules if k.startswith("skills.")}
        self.bc._filler_suppressed()
        after = {k for k in sys.modules if k.startswith("skills.")}
        self.assertEqual(after, before)

    def test_error_is_a_reason(self):
        self._p(self.bc, "_is_staging", mock.Mock(side_effect=RuntimeError))
        self.assertEqual(self.bc._filler_suppressed(), "error")


# ════════════════════════════════════════════════════════════════════════════
#  capture hooks (fix 1)
# ════════════════════════════════════════════════════════════════════════════
class CaptureHookTests(_Base):
    def setUp(self):
        super().setUp()
        self._enable()
        self.f = self._fresh_filler()

    def _release_after(self, delay):
        """Finish the 'playing' clip from another thread after `delay` s."""
        t = threading.Timer(delay, self.f.play_done)
        t.daemon = True
        t.start()
        self.addCleanup(t.cancel)

    def test_get_mic_buffer_holds_the_filler_off(self):
        bc = self.bc
        turn = self.f.arm()
        turn.owner = -1          # behave like a background capture
        seen = {}

        def impl(seconds, sr):
            seen["reason"] = bc._filler_suppressed()
            seen["claim"] = self.f.claim(turn, 1)
            return "AUDIO"

        self._p(bc, "_get_mic_buffer_impl", side_effect=impl)
        self.assertEqual(bc.get_mic_buffer(15.0, 16000), "AUDIO")
        self.assertEqual(seen, {"reason": "mic-capture", "claim": "busy"})
        self.assertFalse(self.f.capturing())
        self.assertEqual(self.f.claim(turn, 1), "ok")   # free again
        self.f.play_done()

    def test_get_mic_buffer_releases_on_error(self):
        bc = self.bc
        self._p(bc, "_get_mic_buffer_impl", side_effect=RuntimeError("x"))
        with self.assertRaises(RuntimeError):
            bc.get_mic_buffer(1.0)
        self.assertFalse(self.f.capturing())

    def test_get_mic_buffer_waits_for_a_clip_claimed_before_it(self):
        # enroll_voice / whos_talking capture straight from the action with no
        # _speak first, so a stage-1 clip claimed just before must finish
        # before the mic opens (else it lands in the owner's voiceprint).
        bc = self.bc
        turn = self.f.arm()
        self.assertEqual(self.f.claim(turn, 1), "ok")
        seen = {}

        def impl(seconds, sr):
            seen["playing"] = self.f.playing()
            return "AUDIO"

        self._p(bc, "_get_mic_buffer_impl", side_effect=impl)
        self._release_after(0.2)
        t0 = time.monotonic()
        self.assertEqual(bc.get_mic_buffer(2.0, 16000), "AUDIO")
        waited = time.monotonic() - t0
        self.assertEqual(seen, {"playing": False})
        self.assertGreaterEqual(waited, 0.15)
        self.assertLess(waited, 3.0)

    def test_record_speech_waits_for_a_clip_claimed_before_it(self):
        bc = self.bc
        turn = self.f.arm()
        self.assertEqual(self.f.claim(turn, 1), "ok")
        seen = {}

        def claim_owner(cell, *a, **k):
            seen["playing"] = self.f.playing()
            return False            # refuse: record_speech returns None

        self._p(bc, "_mic_input_disabled", lambda: False)
        self._p(bc, "get_input_device", lambda: None)
        self._p(bc, "_pa_claim_owner", side_effect=claim_owner)
        self._release_after(0.2)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertIsNone(bc.record_speech(timeout=1.0))
        self.assertEqual(seen, {"playing": False})

    def test_turn_thread_capture_is_turn_activity(self):
        bc = self.bc
        turn = self.f.arm()                # armed on this (the turn) thread
        mark0 = turn.last_mark
        self._p(bc, "_get_mic_buffer_impl", return_value="AUDIO")
        bc.get_mic_buffer(1.0)
        self.assertTrue(turn.spoke)
        self.assertGreaterEqual(turn.last_mark, mark0)

    def test_background_capture_is_not_turn_activity(self):
        # skills/standby_audio_detect polls get_mic_buffer from its own daemon
        # every ~5 s all day; counting those as turn activity killed stage 1
        # on ~1/3 of turns and stage 2 on all of them.
        bc = self.bc
        turn = self.f.arm()
        mark0 = turn.last_mark
        seen = {}

        def impl(seconds, sr):
            seen["capturing"] = self.f.capturing()
            return "AUDIO"

        self._p(bc, "_get_mic_buffer_impl", side_effect=impl)
        th = threading.Thread(target=bc.get_mic_buffer, args=(3.0,),
                              name="standby-audio-loop", daemon=True)
        th.start()
        th.join(2.0)
        self.assertFalse(th.is_alive())
        self.assertEqual(seen, {"capturing": True})
        self.assertFalse(turn.spoke)
        self.assertEqual(turn.last_mark, mark0)
        self.assertFalse(self.f.capturing())

    def test_record_speech_marks_on_entry(self):
        bc = self.bc
        turn = self.f.arm()
        self._p(bc, "_mic_input_disabled", lambda: False)
        self._p(bc, "get_input_device", lambda: None)
        self._p(bc, "_pa_claim_owner", lambda *a, **k: False)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertIsNone(bc.record_speech(timeout=1.0))
        self.assertTrue(turn.spoke)

    def _record_speech_fn(self):
        with io.open(os.path.join(_ROOT, "bobert_companion.py"),
                     encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == "record_speech":
                return node
        self.fail("record_speech not found")

    @staticmethod
    def _call_name(stmt):
        if (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)
                and isinstance(stmt.value.func, ast.Name)):
            return stmt.value.func.id
        return None

    @staticmethod
    def _is_flag_clear(stmt):
        return (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Subscript)
                and isinstance(stmt.targets[0].value, ast.Name)
                and stmt.targets[0].value.id == "_record_speech_active"
                and isinstance(stmt.value, ast.Constant)
                and stmt.value.value is False)

    def test_record_speech_marks_on_exit_before_the_flag_drops(self):
        # AST, not substrings: a commented-out call or a mark moved after the
        # flag clear (the release-then-mark race fix 2 removed from _speak)
        # both fail here.
        fn = self._record_speech_fn()
        finals = [n.finalbody for n in ast.walk(fn)
                  if isinstance(n, ast.Try) and n.finalbody
                  and any(self._is_flag_clear(x) for x in n.finalbody)
                  and any(self._call_name(x) == "_safe_close_stream"
                          for x in n.body)]
        self.assertEqual(len(finals), 1, "the live-stream close finally")
        body = finals[0]
        names = [self._call_name(x) for x in body]
        clear_at = next(i for i, x in enumerate(body) if self._is_flag_clear(x))
        self.assertIn("_filler_capture_mark", names[:clear_at])
        # ...and the entry mark (with its wait) precedes the ownership claim.
        entry = [x for x in fn.body
                 if self._call_name(x) == "_filler_capture_mark"]
        self.assertEqual(len(entry), 1)
        self.assertEqual([k.arg for k in entry[0].value.keywords], ["wait"])
        claim_line = next(
            n.lineno for n in ast.walk(fn)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            and n.func.id == "_pa_claim_owner")
        self.assertLess(entry[0].lineno, claim_line)

    def test_enroll_voice_capture_is_covered(self):
        # skills/enroll_voice captures through bc.get_mic_buffer (held above)
        # and, on fallback, claims _enroll_capture_active (a suppress cell).
        with io.open(os.path.join(_ROOT, "skills", "enroll_voice.py"),
                     encoding="utf-8") as fh:
            src = fh.read()
        self.assertIn('getattr(bc, "get_mic_buffer"', src)
        self.assertIn('"_enroll_capture_active"', src)


# ════════════════════════════════════════════════════════════════════════════
#  _filler_play
# ════════════════════════════════════════════════════════════════════════════
class FillerPlayTests(_Base):
    def setUp(self):
        super().setUp()
        self._enable()
        self.lock = threading.Lock()
        self._p(self.bc, "_SPEAK_LOCK", self.lock)
        self.f = self._fresh_filler()
        self.clips = self._fresh_clips()
        self.played = []
        bc = self.bc

        def _play(audio, sr):
            self.played.append((bc._tts_current_text[0], len(audio), sr))
            bc._tts_playback_active[0] = True
        self.play = self._p(bc, "play_with_lipsync", side_effect=_play)
        self.set_state = self._p(bc, "set_state")

    def test_retry_when_someone_is_speaking(self):
        turn = self.f.arm()
        self.lock.acquire()
        try:
            self.assertEqual(self.bc._filler_play(turn, 1), "retry")
        finally:
            self.lock.release()
        self.play.assert_not_called()

    def test_gone_claim_skips_and_releases(self):
        turn = self.f.arm()
        self.f.note_speech()
        self.assertEqual(self.bc._filler_play(turn, 1), "skipped")
        self.assertTrue(self.lock.acquire(blocking=False))
        self.lock.release()
        self.play.assert_not_called()

    def test_mic_capture_is_a_retry(self):
        turn = self.f.arm()
        self.bc._record_speech_active[0] = True
        self.assertEqual(self.bc._filler_play(turn, 2), "retry")
        self.play.assert_not_called()

    def test_capture_starting_while_waiting_for_the_lock_is_a_retry(self):
        # The IN-LOCK re-check (fix 1): the pre-lock check passes, then a
        # capture goes live before the lock is taken.
        bc = self.bc
        turn = self.f.arm()

        class RaceLock:
            def acquire(self_, blocking=True):
                bc._record_speech_active[0] = True
                return True

            def release(self_):
                pass

        self._p(bc, "_SPEAK_LOCK", RaceLock())
        self.assertEqual(bc._filler_play(turn, 1), "retry")
        self.play.assert_not_called()
        bc._record_speech_active[0] = False
        self.assertEqual(self.f.claim(turn, 1), "ok")   # stage not consumed
        self.f.play_done()

    def test_not_yet_is_a_retry(self):
        # A speech mark between the poll's sleep and the claim moves the
        # silence clock: stage 2 must retry, not be dropped.
        turn = self.f.arm()
        self.assertEqual(self.bc._filler_play(turn, 2), "retry")
        self.play.assert_not_called()
        self.assertNotIn(2, turn.fired)

    def test_busy_claim_is_a_retry(self):
        # claim() refusing for a capture registered after the gate check.
        turn = self.f.arm()
        turn.owner = -1
        self.f.begin_capture()
        self._p(self.bc, "_filler_mic_capture_live", lambda: False)
        self.assertEqual(self.bc._filler_play(turn, 1), "retry")
        self.play.assert_not_called()
        self.f.end_capture()
        self.assertEqual(self.f.claim(turn, 1), "ok")
        self.f.play_done()

    def test_no_clip_means_skipped(self):
        self._fresh_clips(lines=[])
        turn = self.f.arm()
        self.assertEqual(self.bc._filler_play(turn, 1), "skipped")

    def test_success(self):
        bc = self.bc
        from core import processing_filler as pf
        turn = self.f.arm()
        hist = list(bc.conversation_history)
        prefix = bc._stream_spoken_prefix[0]
        out = io.StringIO()
        bc._tts_interrupt.set()
        self.addCleanup(bc._tts_interrupt.clear)
        with contextlib.redirect_stdout(out):
            self.assertEqual(bc._filler_play(turn, 1), "played")
        self.assertFalse(bc._tts_interrupt.is_set())
        self.assertEqual(len(self.played), 1)
        text, n, sr = self.played[0]
        self.assertIn(text, [x.lower() for x in pf.FIRST_LINES])
        self.assertEqual((n, sr), (2400, 24000))
        self.assertEqual(bc._tts_current_text[0], "")
        self.assertFalse(bc._tts_playback_active[0])
        self.assertTrue(self.lock.acquire(blocking=False))
        self.lock.release()
        self.set_state.assert_not_called()
        self.assertEqual(bc.conversation_history, hist)
        self.assertEqual(bc._stream_spoken_prefix[0], prefix)
        self.assertIn("[filler] stage 1:", out.getvalue())
        self.assertNotIn("JARVIS:", out.getvalue())
        self.assertFalse(self.f.playing())
        self.assertEqual(self.f.claim(turn, 1), "gone")   # consumed

    def test_playback_error_consumes_the_stage(self):
        bc = self.bc
        self.play.side_effect = RuntimeError("PortAudio reinit hung")
        turn = self.f.arm()
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(bc._filler_play(turn, 1), "played")
        self.assertTrue(self.lock.acquire(blocking=False))
        self.lock.release()
        self.assertFalse(bc._tts_playback_active[0])
        self.assertEqual(bc._tts_current_text[0], "")
        self.assertFalse(self.f.playing())

    def test_never_bumps_the_barge_seq(self):
        bc = self.bc
        seq0 = bc._tts_interrupt_seq[0]
        turn = self.f.arm()
        with contextlib.redirect_stdout(io.StringIO()):
            bc._filler_play(turn, 1)
        self.assertEqual(bc._tts_interrupt_seq[0], seq0)


# ════════════════════════════════════════════════════════════════════════════
#  render + warm (fix 5)
# ════════════════════════════════════════════════════════════════════════════
class RenderWarmTests(_Base):
    def setUp(self):
        super().setUp()
        self._enable()

    def _kokoro(self, result):
        fake = types.SimpleNamespace(is_available=lambda: True,
                                     synthesize=mock.Mock(return_value=result),
                                     _VOICE="bm_george")
        import core
        self._p(core, "kokoro_tts", fake, create=True)
        p = mock.patch.dict(sys.modules, {"core.kokoro_tts": fake})
        p.start()
        self.addCleanup(p.stop)
        return fake

    def test_render_uses_kokoro_at_neutral_speed(self):
        import numpy as np
        fake = self._kokoro((np.full(100, 0.2, dtype=np.float32), 24000))
        audio, sr = self.bc._filler_render("Just a moment, sir.")
        fake.synthesize.assert_called_once_with("Just a moment, sir.",
                                                speed=1.0)
        self.assertEqual((len(audio), sr), (100, 24000))

    def test_render_rejects_silence_and_other_backends(self):
        import numpy as np
        fake = self._kokoro((np.zeros(100, dtype=np.float32), 24000))
        self.assertIsNone(self.bc._filler_render("x"))
        synth = self._p(self.bc, "synthesise")
        self._p(self.bc, "TTS_BACKEND", "edge")
        self.assertIsNone(self.bc._filler_render("x"))
        synth.assert_not_called()          # never the network edge-tts path
        self.assertEqual(fake.synthesize.call_count, 1)

    def test_warm_only_when_available(self):
        bc = self.bc
        self._fresh_filler()
        clips = self._fresh_clips(lines=[])
        warm = self._p(clips, "warm_async", return_value=True)
        self._p(bc, "TTS_BACKEND", "edge")
        self.assertFalse(bc._filler_warm_if_needed())
        warm.assert_not_called()
        self._p(bc, "TTS_BACKEND", "kokoro")
        self.assertTrue(bc._filler_warm_if_needed())
        kwargs = warm.call_args.kwargs
        self.assertIs(kwargs["stop_fn"], bc._filler_warm_stop)

    def test_warm_stops_when_armed_or_closed(self):
        f = self._fresh_filler()
        self.assertFalse(self.bc._filler_warm_stop())
        turn = f.arm()
        self.assertTrue(self.bc._filler_warm_stop())
        f.disarm(turn)
        self.assertFalse(self.bc._filler_warm_stop())
        f.shutdown("restart")
        self.assertTrue(self.bc._filler_warm_stop())

    def test_voice_change_invalidates_clips(self):
        import numpy as np
        from core import processing_filler as pf
        fake = self._kokoro((np.full(10, 0.2, dtype=np.float32), 24000))
        fake._VOICE = "voice_a"
        c = pf.ClipCache(render_fn=lambda t: None, lock=threading.Lock(),
                         key_fn=self.bc._filler_voice_key)
        self.assertTrue(c.put("Just a moment, sir.",
                              (np.full(2400, 0.1, dtype=np.float32), 24000)))
        self.assertIsNotNone(c.get("Just a moment, sir."))
        fake._VOICE = "voice_b"
        self.assertIsNone(c.get("Just a moment, sir."))

    def test_first_voice_turn_really_renders_the_cache(self):
        # End to end: a real voice dispatch arms, disarms and then warms. If
        # disarm left the turn current, the warm would stop before its first
        # line and the filler would stay silent forever.
        import numpy as np
        from core import processing_filler as pf
        bc = self.bc
        self._kokoro((np.full(2400, 0.2, dtype=np.float32), 24000))
        f = self._fresh_filler()
        clips = pf.ClipCache(render_fn=bc._filler_render, lock=bc._SPEAK_LOCK,
                             key_fn=bc._filler_voice_key)
        self._p(bc, "_filler_clips", clips)
        fac = _RecFactory()
        orig = clips.warm_async

        def warm_async(lines, stop_fn):
            return orig(lines, stop_fn, wait_fn=lambda e, t: False,
                        thread_factory=fac)

        self._p(clips, "warm_async", side_effect=warm_async)
        self._p(bc, "_run_llm_dispatch_body", return_value="r")
        bc._run_llm_dispatch("what's the weather", voice=True)
        self.assertFalse(f.armed())
        self.assertEqual(len(fac.made), 1)
        fac.made[0].target(*fac.made[0].args)
        lines = pf.FIRST_LINES + pf.STILL_LINES
        self.assertEqual(sorted(clips.available(lines)), sorted(lines))

    def test_clip_limit_fits_the_end_of_turn_wait(self):
        self.assertLessEqual(self.bc._filler_clips._max_secs + 0.5,
                             self.bc._FILLER_END_WAIT_S)


# ════════════════════════════════════════════════════════════════════════════
#  parse_and_run_actions cancel + teardown latch (fix 7)
# ════════════════════════════════════════════════════════════════════════════
class ActionCancelTests(_Base):
    def setUp(self):
        super().setUp()
        bc = self.bc
        self.fake = mock.Mock()
        self._p(bc, "_processing_filler", self.fake)
        self._p(bc, "_needs_confirmation", lambda n, a: False)
        self._p(bc, "_jarvis_pushback", lambda n, a: None)
        self._p(bc, "_speak", lambda *a, **k: None)
        self._p(bc, "MID_TASK_STATUS_ENABLED", True)

        class FakeTimer:
            def __init__(self, delay, fn, args=()):
                self.daemon = False

            def start(self):
                pass

            def cancel(self):
                pass

            def join(self, timeout=None):
                pass

        self._p(bc.threading, "Timer", FakeTimer)

    def _run(self, name):
        acts = dict(self.bc.ACTIONS)
        acts[name] = lambda a: "done"
        with mock.patch.object(self.bc, "ACTIONS", acts), \
                contextlib.redirect_stdout(io.StringIO()):
            self.bc.parse_and_run_actions(f"[ACTION: {name}, x]")

    def test_long_running_and_fire_and_exit_cancel(self):
        for name in ("play_streaming", "restart"):
            self.fake.reset_mock()
            self._run(name)
            self.fake.cancel.assert_called_once_with("action:" + name)

    def test_silent_work_actions_keep_the_filler(self):
        for name in ("web_search", "see_screen"):
            self.fake.reset_mock()
            self._run(name)
            self.fake.cancel.assert_not_called()


class TeardownTests(_Base):
    def test_tray_restart_and_shutdown_latch_first(self):
        bc = self.bc
        order = []
        fake = mock.Mock()
        fake.shutdown.side_effect = lambda r: order.append(("latch", r))
        self._p(bc, "_processing_filler", fake)
        self._p(bc, "_act_restart", lambda a: order.append(("restart", a)))
        self._p(bc, "_act_shutdown_jarvis",
                lambda a: order.append(("shutdown", a)))
        with contextlib.redirect_stdout(io.StringIO()):
            bc._dispatch_tray_command("restart", {})
            bc._dispatch_tray_command("shutdown", {})
        self.assertEqual(order, [("latch", "tray:restart"), ("restart", ""),
                                 ("latch", "tray:shutdown"), ("shutdown", "")])

    def test_latched_filler_never_arms_again(self):
        self._enable()
        f = self._fresh_filler()
        turn = f.arm()
        self.bc._filler_teardown("restart")
        self.assertTrue(turn.cancel.is_set())
        self.assertIsNone(f.arm())

    def test_core_actions_latch_through_the_host(self):
        import core.actions as ca
        host = types.SimpleNamespace(_filler_teardown=mock.Mock())
        ca._filler_teardown_via_bc(host, "restart")
        host._filler_teardown.assert_called_once_with("restart")
        ca._filler_teardown_via_bc(types.SimpleNamespace(), "x")   # no hook
        ca._filler_teardown_via_bc(
            types.SimpleNamespace(_filler_teardown=mock.Mock(
                side_effect=RuntimeError)), "x")                  # never raises

    def _core_actions_with(self, host):
        import core.actions as ca
        self._p(ca, "_bc", lambda: host)
        return ca

    def test_act_restart_latches_before_anything_else(self):
        host = _RecHost()
        ca = self._core_actions_with(host)
        with self.assertRaises(_Stop) as cm:
            ca._act_restart("")
        self.assertEqual(cm.exception.args, ("restart",))
        self.assertEqual(host.seen, ["_filler_teardown"])

    def test_act_shutdown_latches_before_the_goodbye(self):
        host = _RecHost(_sleep_mode=[False])
        ca = self._core_actions_with(host)
        with self.assertRaises(_Stop) as cm:
            ca._act_shutdown_jarvis("")
        self.assertEqual(cm.exception.args, ("shutdown",))
        self.assertEqual(host.seen, ["_sleep_mode", "_filler_teardown"])

    def test_release_native_resources_latches_first(self):
        import core.actions as ca
        host = _RecHost()
        with self.assertRaises(_Stop) as cm:
            ca._release_native_resources(host)
        self.assertEqual(cm.exception.args, ("teardown",))
        self.assertEqual(host.seen, ["_filler_teardown"])

    def test_blue_green_teardown_latches_first(self):
        bc = self.bc

        def latch(reason):
            raise _Stop(reason)

        self._p(bc, "_filler_teardown", side_effect=latch)
        intent = self._p(bc, "mark_intentional_exit", side_effect=_Other)
        with contextlib.redirect_stdout(io.StringIO()), \
                self.assertRaises(_Stop) as cm:
            bc._blue_green_teardown_and_exit()
        self.assertEqual(cm.exception.args, ("blue-green",))
        intent.assert_not_called()

    def test_harness_restore_unlatches_the_shared_filler(self):
        # A real teardown entry run by a test latches the SHARED instance;
        # the harness restore must reset it, or every later test in the
        # process sees arm() -> None for the wrong reason.
        from tests import _monolith_harness as h
        bc = self.bc
        shared = bc._processing_filler
        bc._filler_teardown("tray:restart")
        self.assertTrue(shared.closed())
        h._restore_monolith_pristine(bc)
        self.assertIs(bc._processing_filler, shared)
        self.assertFalse(bc._processing_filler.closed())
        self.assertFalse(bc._processing_filler.armed())
        self.assertFalse(bc._processing_filler.capturing())


# ════════════════════════════════════════════════════════════════════════════
#  real-thread ordering (bounded)
# ════════════════════════════════════════════════════════════════════════════
class RealThreadOrderingTests(_Base):
    def setUp(self):
        super().setUp()
        import numpy as np
        bc = self.bc
        self._enable()
        self.lock = threading.Lock()
        self._p(bc, "_SPEAK_LOCK", self.lock)
        self.f = self._fresh_filler()
        self._fresh_clips()
        self.order = []
        self.active = [0]
        self.overlap = [False]
        self.filler_started = threading.Event()
        self.release_filler = threading.Event()
        self.answer_playing = threading.Event()

        def _play(audio, sr):
            self.active[0] += 1
            if self.active[0] > 1:
                self.overlap[0] = True
            try:
                if len(audio) == 2400:        # the cached filler clip
                    self.order.append("filler")
                    self.filler_started.set()
                    self.release_filler.wait(2.0)
                else:
                    self.order.append("answer")
                    self.answer_playing.set()
            finally:
                self.active[0] -= 1
        self._p(bc, "play_with_lipsync", side_effect=_play)
        self._p(bc, "synthesise",
                side_effect=lambda t: (np.full(10, 0.1, dtype=np.float32),
                                       24000))
        self._p(bc, "set_state")
        self._p(bc, "_write_hud_state")
        self._p(bc, "_session_start_time", time.time() - 3600)

    def _thread(self, fn, *a):
        th = threading.Thread(target=fn, args=a, daemon=True)
        th.start()
        self.addCleanup(th.join, 2.0)
        return th

    def test_a_filler_first_answer_waits(self):
        bc = self.bc
        turn = self.f.arm()
        with contextlib.redirect_stdout(io.StringIO()):
            tf = self._thread(bc._filler_play, turn, 1)
            self.assertTrue(self.filler_started.wait(2.0))
            ta = self._thread(bc._speak, "The answer, sir.")
            # The answer must be parked on _SPEAK_LOCK while the clip plays.
            self.assertFalse(self.answer_playing.wait(0.3))
            self.release_filler.set()
            self.assertTrue(self.answer_playing.wait(2.0))
            tf.join(2.0)
            ta.join(2.0)
        self.assertEqual(self.order, ["filler", "answer"])
        self.assertFalse(self.overlap[0])

    def test_b_answer_marks_first_filler_is_gone(self):
        bc = self.bc
        turn = self.f.arm()
        with contextlib.redirect_stdout(io.StringIO()):
            bc._speak("The answer, sir.")
            self.assertEqual(bc._filler_play(turn, 1), "skipped")
        self.assertEqual(self.order, ["answer"])

    def test_end_turn_waits_for_a_playing_clip(self):   # fix 3, real Event
        bc = self.bc
        self._p(bc, "_filler_warm_if_needed")
        turn = self.f.arm()
        with contextlib.redirect_stdout(io.StringIO()):
            tf = self._thread(bc._filler_play, turn, 1)
            self.assertTrue(self.filler_started.wait(2.0))
            threading.Timer(0.2, self.release_filler.set).start()
            t0 = time.monotonic()
            bc._filler_end_turn(turn)
            waited = time.monotonic() - t0
            tf.join(2.0)
        self.assertFalse(self.f.playing())
        self.assertGreaterEqual(waited, 0.15)
        self.assertLess(waited, 3.0)


if __name__ == "__main__":
    unittest.main()
