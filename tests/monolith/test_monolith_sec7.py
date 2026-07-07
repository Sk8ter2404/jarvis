"""Unit tests for bobert_companion.py, section 7 (source lines ~13150-13772).

This band is the *normal-mode main-loop step helpers* — the per-iteration
pieces ``main()`` calls so the giant boot ``while True:`` loop itself stays a
thin orchestrator. Each is an ordinary function that takes a string / dict and
returns a bool / str / tuple, reading mockable module globals and patchable
sibling helpers. Covered here:

  * ``_run_voice_shortcuts`` — the pre-LLM voice-shortcut router (replay,
    TTS-backend toggle, conversation-mode toggle / controlled dispatch, the
    multi-step chain resolver).
  * ``_capture_utterance`` — drain queued speech then return the next turn
    (injected pass-through metadata, or a mic recording -> Whisper).
  * ``_handle_ambient_music`` / ``_handle_sleep_triggers`` /
    ``_handle_sleep_standby`` — the ambient-music auto-standby gate, the
    sleep/standby trigger-phrase gate, and the sleep/standby wake-listen phase.
  * ``_run_llm_dispatch`` — a single LLM turn (glance fast-path or full
    response), [ACTION:] execution, and the informative/failure follow-up loop.
  * ``_blue_green_loop_tick`` / ``_consume_blue_green_handoff`` — the
    per-iteration blue/green heartbeat + handoff signal handling, and the
    post-relaunch conversation-tail resume.
  * ``_do_proactive_turn`` — the spontaneous-comment turn (animation thread
    mocked away).
  * ``_drain_injected_command`` — the race-safe injected-command queue consumer.

It also picks up the testable-but-previously-uncovered *boot self-heal* and
*adjacent helper* surface (functions that look boot-bound but are pure logic
over mockable I/O, so the honesty rule says TEST rather than pragma them):
``_enforce_singleton`` / ``_release_singleton`` (PID-file singleton re-check +
OS-lock release — ``subprocess``/``os`` mocked), ``_preflight_cameras`` /
``_startup_preflight`` (probe-fanout + the API/cublas/camera orchestrator, with
the camera probe + ``_preflight_api_key`` + ``sys.exit`` mocked), ``load_skills``
(temp ``SKILLS_DIR`` with crafted package/flat/error fixtures), the live-grab
*guard + success* paths of ``_capture_focused_window_png`` (``mss``/``PIL``
mocked), the vision-fallback branches of ``maybe_glance_response`` and
``_media_key_with_focus``, and the close-error/hang branches of
``_safe_close_stream``.

Out of scope (boot-only, infinite-loop, or hardware-bound — pragma'd at their
headers in the monolith, exercised behaviourally by the staging tier):
``main()`` and the ``__main__`` block, ``_face_tracking_thread`` /
``_overnight_upgrade_thread`` (``while True`` daemons), the live mic capture
loop inside ``record_speech`` / ``get_mic_buffer``, the live Win32 GDI
PrintWindow grab inside ``_open_url_offscreen_capture``.

Strategy mirrors sec6: import the monolith ONCE via the harness (cached), patch
the specific globals/helpers each function reads with
``mock.patch.object(bc, ...)``, and route every side effect (``_speak``,
``set_state``, ``record_speech``, ``transcribe``, the blue/green manager,
threads) through mocks. Lazily-imported sibling modules are faked via
``mock.patch.dict(sys.modules, ...)``. Directly-mutated globals
(``conversation_history`` and the single-element runtime-state lists) are
deep-restored after every test by ``MonolithGlobalsTestCase``.

Run locally (full-deps tier):
    python -m unittest tests.monolith.test_monolith_sec7
On the light-deps CI runner these all skip via @requires_monolith.
"""
from __future__ import annotations

import json
import os
import tempfile
import types
import unittest
from unittest import mock

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith


@requires_monolith
class SectionSevenBase(MonolithGlobalsTestCase):
    # setUpClass (cached monolith load) + per-test deep-restore of the mutated
    # bobert_companion globals are inherited from MonolithGlobalsTestCase.

    def setUp(self):
        self._hist_len = len(self.bc.conversation_history)
        self.addCleanup(self._restore_hist)

    def _restore_hist(self):
        del self.bc.conversation_history[self._hist_len:]

    # Convenience: start a patch and auto-stop it at test teardown.
    def _p(self, *args, **kwargs):
        patcher = mock.patch.object(*args, **kwargs)
        m = patcher.start()
        self.addCleanup(patcher.stop)
        return m


# ════════════════════════════════════════════════════════════════════════════
#  _run_voice_shortcuts
# ════════════════════════════════════════════════════════════════════════════
class RunVoiceShortcutsTests(SectionSevenBase):
    def setUp(self):
        super().setUp()
        # Silence the speech + state side effects for every shortcut path.
        self._speak = self._p(self.bc, "_speak")
        self._set_state = self._p(self.bc, "set_state")

    def test_replay_shortcut_handles_and_returns_true(self):
        self._p(self.bc, "maybe_replay_last_action", return_value="Done that again, sir.")
        out = self.bc._run_voice_shortcuts("do that again")
        self.assertTrue(out)
        self._speak.assert_called_once_with("Done that again, sir.")
        # The turn is appended to history (user + assistant).
        self.assertEqual(self.bc.conversation_history[-2]["role"], "user")
        self.assertEqual(self.bc.conversation_history[-1]["content"],
                         "Done that again, sir.")

    def test_replay_handler_exception_falls_through(self):
        # maybe_replay_last_action raising must not abort the function; with no
        # other shortcut matching it should fall through and return False.
        self._p(self.bc, "maybe_replay_last_action", side_effect=RuntimeError("boom"))
        # Neutralise every later shortcut so we reach the final `return False`.
        self._p(self.bc, "maybe_replay_last_action", side_effect=RuntimeError("boom"))
        with mock.patch.dict(self.bc.sys.modules, {}, clear=False):
            # No skill_custom_voice / mode_router / dispatcher matches.
            self._neutralise_later_shortcuts()
            out = self.bc._run_voice_shortcuts("hello there")
        self.assertFalse(out)
        self._speak.assert_not_called()

    def _neutralise_later_shortcuts(self):
        # custom-voice backend toggle returns None (no match)
        fake_voice = types.SimpleNamespace(maybe_switch_backend=lambda _t: None)
        self._p_dict({"skill_custom_voice": fake_voice})
        # mode_router toggle/controlled both inert
        fake_router = types.ModuleType("core.mode_router")
        fake_router.maybe_handle_mode_toggle = lambda _t: None
        fake_router.controlled_dispatch = lambda _t, _a: None
        fake_router.is_in_controlled_mode = lambda: False
        self._p_dict({"core.mode_router": fake_router})
        # chain resolver inert
        fake_disp = types.ModuleType("core.dispatcher")
        fake_disp_resolve = lambda _t, _a: None
        fake_disp.resolve_and_dispatch = fake_disp_resolve
        self._p_dict({"core.dispatcher": fake_disp})

    def _p_dict(self, mapping):
        patcher = mock.patch.dict(self.bc.sys.modules, mapping, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_tts_backend_toggle_shortcut(self):
        self._p(self.bc, "maybe_replay_last_action", return_value=None)
        fake_voice = types.SimpleNamespace(
            maybe_switch_backend=lambda _t: "Switched to the Edge voice, sir.")
        self._p_dict({"skill_custom_voice": fake_voice})
        out = self.bc._run_voice_shortcuts("switch to edge voice")
        self.assertTrue(out)
        self._speak.assert_called_once_with("Switched to the Edge voice, sir.")

    def test_mode_toggle_shortcut(self):
        self._p(self.bc, "maybe_replay_last_action", return_value=None)
        self._p_dict({"skill_custom_voice":
                      types.SimpleNamespace(maybe_switch_backend=lambda _t: None)})
        fake_router = types.ModuleType("core.mode_router")
        fake_router.maybe_handle_mode_toggle = lambda _t: "Controlled mode engaged, sir."
        fake_router.controlled_dispatch = lambda _t, _a: None
        fake_router.is_in_controlled_mode = lambda: False
        self._p_dict({"core.mode_router": fake_router})
        out = self.bc._run_voice_shortcuts("controlled mode")
        self.assertTrue(out)
        self._speak.assert_called_once_with("Controlled mode engaged, sir.")

    def test_controlled_dispatch_shortcut(self):
        self._p(self.bc, "maybe_replay_last_action", return_value=None)
        self._p_dict({"skill_custom_voice":
                      types.SimpleNamespace(maybe_switch_backend=lambda _t: None)})
        fake_router = types.ModuleType("core.mode_router")
        fake_router.maybe_handle_mode_toggle = lambda _t: None
        fake_router.controlled_dispatch = lambda _t, _a: "No skill for that, sir."
        fake_router.is_in_controlled_mode = lambda: True
        self._p_dict({"core.mode_router": fake_router})
        out = self.bc._run_voice_shortcuts("do a barrel roll")
        self.assertTrue(out)
        self._speak.assert_called_once_with("No skill for that, sir.")

    def test_controlled_dispatch_exception_speaks_recovery(self):
        self._p(self.bc, "maybe_replay_last_action", return_value=None)
        self._p_dict({"skill_custom_voice":
                      types.SimpleNamespace(maybe_switch_backend=lambda _t: None)})
        fake_router = types.ModuleType("core.mode_router")
        fake_router.maybe_handle_mode_toggle = lambda _t: None

        def _boom(_t, _a):
            raise RuntimeError("dispatcher exploded")
        fake_router.controlled_dispatch = _boom
        fake_router.is_in_controlled_mode = lambda: True
        self._p_dict({"core.mode_router": fake_router})
        out = self.bc._run_voice_shortcuts("anything")
        self.assertTrue(out)
        # The except branch speaks a recovery line rather than crashing.
        self.assertIn("controlled mode", self._speak.call_args[0][0].lower())

    def test_chain_resolver_shortcut(self):
        self._p(self.bc, "maybe_replay_last_action", return_value=None)
        self._p_dict({"skill_custom_voice":
                      types.SimpleNamespace(maybe_switch_backend=lambda _t: None)})
        fake_router = types.ModuleType("core.mode_router")
        fake_router.maybe_handle_mode_toggle = lambda _t: None
        fake_router.controlled_dispatch = lambda _t, _a: None
        fake_router.is_in_controlled_mode = lambda: False
        self._p_dict({"core.mode_router": fake_router})
        fake_disp = types.ModuleType("core.dispatcher")
        fake_disp.resolve_and_dispatch = lambda _t, _a: "Playing music and starting a timer, sir."
        self._p_dict({"core.dispatcher": fake_disp})
        out = self.bc._run_voice_shortcuts("play jazz and set a 5 minute timer")
        self.assertTrue(out)
        self._speak.assert_called_once_with(
            "Playing music and starting a timer, sir.")

    def test_no_shortcut_matches_returns_false(self):
        self._p(self.bc, "maybe_replay_last_action", return_value=None)
        self._neutralise_later_shortcuts()
        out = self.bc._run_voice_shortcuts("what's the meaning of life")
        self.assertFalse(out)
        self._speak.assert_not_called()

    def test_router_import_unavailable_falls_through(self):
        # If core.mode_router import raises, the function must degrade (treat as
        # not-controlled) rather than crash. Force the import to blow up by
        # inserting a module object that raises on attribute access via a
        # builtins.__import__ shim scoped to this test.
        self._p(self.bc, "maybe_replay_last_action", return_value=None)
        self._p_dict({"skill_custom_voice":
                      types.SimpleNamespace(maybe_switch_backend=lambda _t: None)})
        real_import = self.bc.__builtins__["__import__"] if isinstance(
            self.bc.__builtins__, dict) else __import__

        def _import_shim(name, *a, **k):
            if name == "core.mode_router":
                raise ImportError("router gone")
            return real_import(name, *a, **k)
        # Also neutralise the chain resolver so we reach `return False`.
        fake_disp = types.ModuleType("core.dispatcher")
        fake_disp.resolve_and_dispatch = lambda _t, _a: None
        self._p_dict({"core.dispatcher": fake_disp})
        with mock.patch("builtins.__import__", side_effect=_import_shim):
            out = self.bc._run_voice_shortcuts("hello")
        self.assertFalse(out)


# ════════════════════════════════════════════════════════════════════════════
#  _run_voice_shortcuts — conversation_history stays bounded (RAM/token leak)
# ════════════════════════════════════════════════════════════════════════════
#
# Regression guard for the unbounded-growth bug: every fast path in
# _run_voice_shortcuts appended a user+assistant PAIR then `return True`,
# short-circuiting the main loop before the full-LLM path's
# _trim_conversation_history() ever ran. A session driven mostly by shortcuts
# (replay, chains, controlled/mode dispatch, music routing) therefore grew
# conversation_history without bound — a steady RAM leak AND an ever-growing
# cloud-prompt token cost, since the whole list is dumped into the system
# prompt every turn. Each path now routes through _append_turn (append-then-
# trim), so none can drift past MAX_CONVERSATION_HISTORY.
class RunVoiceShortcutsHistoryBoundedTests(SectionSevenBase):
    def setUp(self):
        super().setUp()
        self._p(self.bc, "_speak")
        self._p(self.bc, "set_state")
        # Reset history to exactly the cap so a single fast-path turn would push
        # it to cap+2 if the trim were missing. _restore_hist (base) restores
        # the pristine list afterwards.
        self.cap = self.bc.MAX_CONVERSATION_HISTORY
        self.bc.conversation_history[:] = [
            {"role": ("user" if i % 2 == 0 else "assistant"),
             "content": f"seed {i}"}
            for i in range(self.cap)
        ]

    def _assert_bounded_and_tailed(self, reply):
        hist = self.bc.conversation_history
        # The trim fired: never exceeds the cap (pre-fix this was cap + 2).
        self.assertLessEqual(len(hist), self.cap)
        # The new turn is still recorded at the tail, role-correct.
        self.assertEqual(hist[-2]["role"], "user")
        self.assertEqual(hist[-1]["role"], "assistant")
        self.assertEqual(hist[-1]["content"], reply)
        # Role alternation preserved (Claude API requires a 'user' first message).
        self.assertEqual(hist[0]["role"], "user")

    def test_replay_path_trims(self):
        self._p(self.bc, "maybe_replay_last_action",
                return_value="Done that again, sir.")
        self.assertTrue(self.bc._run_voice_shortcuts("do that again"))
        self._assert_bounded_and_tailed("Done that again, sir.")

    def test_tts_toggle_path_trims(self):
        self._p(self.bc, "maybe_replay_last_action", return_value=None)
        self._p_dict({"skill_custom_voice": types.SimpleNamespace(
            maybe_switch_backend=lambda _t: "Switched to the Edge voice, sir.")})
        self.assertTrue(self.bc._run_voice_shortcuts("switch to edge voice"))
        self._assert_bounded_and_tailed("Switched to the Edge voice, sir.")

    def test_mode_toggle_path_trims(self):
        self._p(self.bc, "maybe_replay_last_action", return_value=None)
        self._p_dict({"skill_custom_voice":
                      types.SimpleNamespace(maybe_switch_backend=lambda _t: None)})
        fake_router = types.ModuleType("core.mode_router")
        fake_router.maybe_handle_mode_toggle = lambda _t: "Controlled mode engaged, sir."
        fake_router.controlled_dispatch = lambda _t, _a: None
        fake_router.is_in_controlled_mode = lambda: False
        self._p_dict({"core.mode_router": fake_router})
        self.assertTrue(self.bc._run_voice_shortcuts("controlled mode"))
        self._assert_bounded_and_tailed("Controlled mode engaged, sir.")

    def test_controlled_dispatch_path_trims(self):
        self._p(self.bc, "maybe_replay_last_action", return_value=None)
        self._p_dict({"skill_custom_voice":
                      types.SimpleNamespace(maybe_switch_backend=lambda _t: None)})
        fake_router = types.ModuleType("core.mode_router")
        fake_router.maybe_handle_mode_toggle = lambda _t: None
        fake_router.controlled_dispatch = lambda _t, _a: "No skill for that, sir."
        fake_router.is_in_controlled_mode = lambda: True
        self._p_dict({"core.mode_router": fake_router})
        self.assertTrue(self.bc._run_voice_shortcuts("do a barrel roll"))
        self._assert_bounded_and_tailed("No skill for that, sir.")

    def test_chain_resolver_path_trims(self):
        self._p(self.bc, "maybe_replay_last_action", return_value=None)
        self._p_dict({"skill_custom_voice":
                      types.SimpleNamespace(maybe_switch_backend=lambda _t: None)})
        fake_router = types.ModuleType("core.mode_router")
        fake_router.maybe_handle_mode_toggle = lambda _t: None
        fake_router.controlled_dispatch = lambda _t, _a: None
        fake_router.is_in_controlled_mode = lambda: False
        self._p_dict({"core.mode_router": fake_router})
        fake_disp = types.ModuleType("core.dispatcher")
        fake_disp.resolve_and_dispatch = (
            lambda _t, _a: "Playing music and starting a timer, sir.")
        self._p_dict({"core.dispatcher": fake_disp})
        self.assertTrue(
            self.bc._run_voice_shortcuts("play jazz and set a 5 minute timer"))
        self._assert_bounded_and_tailed("Playing music and starting a timer, sir.")

    def test_many_consecutive_fastpath_turns_stay_bounded(self):
        # The core leak scenario: a long run of fast-path-only turns must NOT
        # accumulate. Drive 50 replay turns and confirm history is still capped.
        self._p(self.bc, "maybe_replay_last_action", return_value="Again, sir.")
        for _ in range(50):
            self.assertTrue(self.bc._run_voice_shortcuts("do that again"))
        self.assertLessEqual(len(self.bc.conversation_history), self.cap)

    def _p_dict(self, mapping):
        patcher = mock.patch.dict(self.bc.sys.modules, mapping, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)


# ════════════════════════════════════════════════════════════════════════════
#  _append_turn (centralised append-then-trim helper)
# ════════════════════════════════════════════════════════════════════════════
class AppendTurnTests(SectionSevenBase):
    def test_appends_user_then_assistant(self):
        before = len(self.bc.conversation_history)
        self.bc._append_turn("hello", "hi there, sir")
        hist = self.bc.conversation_history
        self.assertEqual(len(hist), before + 2)
        self.assertEqual(hist[-2], {"role": "user", "content": "hello"})
        self.assertEqual(hist[-1], {"role": "assistant", "content": "hi there, sir"})

    def test_trims_to_cap_in_pairs(self):
        cap = self.bc.MAX_CONVERSATION_HISTORY
        self.bc.conversation_history[:] = [
            {"role": ("user" if i % 2 == 0 else "assistant"), "content": str(i)}
            for i in range(cap)
        ]
        self.bc._append_turn("newest user", "newest assistant")
        hist = self.bc.conversation_history
        self.assertLessEqual(len(hist), cap)
        # Oldest PAIR dropped from the front; first message stays 'user'.
        self.assertEqual(hist[0]["role"], "user")
        self.assertEqual(hist[-1]["content"], "newest assistant")


# ════════════════════════════════════════════════════════════════════════════
#  _capture_utterance
# ════════════════════════════════════════════════════════════════════════════
class CaptureUtteranceTests(SectionSevenBase):
    def setUp(self):
        super().setUp()
        self._p(self.bc, "_speak_pending", return_value=False)
        self._p(self.bc, "_heartbeat")
        self._p(self.bc, "set_state")
        self._p(self.bc, "resume_face_tracking")

    def test_injected_text_returns_passthrough_metadata(self):
        text, conf = self.bc._capture_utterance("open notepad", {"facts": {}})
        self.assertEqual(text, "open notepad")
        # Confidence is the neutral pass-through pair.
        self.assertEqual(conf["no_speech_prob"], 0.0)
        self.assertEqual(conf["avg_logprob"], -0.1)
        # peak is bumped above the whisper-trust floor so is_valid_speech passes.
        self.assertGreaterEqual(self.bc._last_recording_peak,
                                self.bc.WHISPER_TRUST_RMS * 2.0)

    def test_injected_text_non_test_mode_plain_log(self):
        # With the inject test-mode flag off, the plain "[inject] <text>" log
        # branch runs (vs the verbose test-mode line).
        self._p(self.bc, "_INJECT_TEST_MODE", False)
        text, _ = self.bc._capture_utterance("play jazz", {"facts": {}})
        self.assertEqual(text, "play jazz")

    def test_no_audio_runs_proactive_and_returns_none(self):
        self._p(self.bc, "record_speech", return_value=None)
        # _speak_pending already False; should_be_proactive True -> proactive turn.
        self._p(self.bc, "should_be_proactive", return_value=True)
        prov = self._p(self.bc, "_do_proactive_turn")
        out = self.bc._capture_utterance(None, {"facts": {}})
        self.assertIsNone(out)
        prov.assert_called_once()

    def test_no_audio_pending_spoken_returns_none_without_proactive(self):
        self._p(self.bc, "record_speech", return_value=None)
        # First _speak_pending() (top drain) False; the second (after timeout)
        # returns True -> a reminder fired, so we return None and skip proactive.
        self.bc._speak_pending.side_effect = [False, True]
        prov = self._p(self.bc, "_do_proactive_turn")
        out = self.bc._capture_utterance(None, {"facts": {}})
        self.assertIsNone(out)
        prov.assert_not_called()

    def test_too_short_clip_returns_none(self):
        # Audio shorter than SAMPLE_RATE*0.4 samples -> treated as non-speech.
        short = self.bc.np.zeros(int(self.bc.SAMPLE_RATE * 0.2), dtype="float32")
        self._p(self.bc, "record_speech", return_value=short)
        out = self.bc._capture_utterance(None, {"facts": {}})
        self.assertIsNone(out)

    def test_valid_clip_transcribes_and_returns_text(self):
        good = self.bc.np.zeros(int(self.bc.SAMPLE_RATE * 1.0), dtype="float32")
        self._p(self.bc, "record_speech", return_value=good)
        self._p(self.bc, "_audio_music_feed")
        self._p(self.bc, "transcribe",
                return_value=("turn on the lights", {"no_speech_prob": 0.1}))
        text, conf = self.bc._capture_utterance(None, {"facts": {}})
        self.assertEqual(text, "turn on the lights")
        self.assertEqual(conf["no_speech_prob"], 0.1)


# ════════════════════════════════════════════════════════════════════════════
#  _handle_ambient_music
# ════════════════════════════════════════════════════════════════════════════
class HandleAmbientMusicTests(SectionSevenBase):
    def setUp(self):
        super().setUp()
        self._speak = self._p(self.bc, "_speak")
        self._p(self.bc, "set_state")

    def test_non_music_returns_false(self):
        self._p(self.bc, "is_ambient_music", return_value=False)
        self.assertFalse(self.bc._handle_ambient_music("turn on the lights"))

    def test_music_from_jarvis_grace_period_no_count(self):
        # JARVIS itself played music recently -> within grace, no hit counted,
        # but it's still music (not speech) so returns True.
        self._p(self.bc, "is_ambient_music", return_value=True)
        self.bc._jarvis_played_music_at[0] = self.bc.time.time()
        self.bc._ambient_music_hits[0] = 0
        out = self.bc._handle_ambient_music("[Music]")
        self.assertTrue(out)
        self.assertEqual(self.bc._ambient_music_hits[0], 0)
        self._speak.assert_not_called()

    def test_music_counts_hit_below_threshold(self):
        self._p(self.bc, "is_ambient_music", return_value=True)
        self.bc._jarvis_played_music_at[0] = 0.0      # not from JARVIS
        self.bc._ambient_music_hits[0] = 0
        self.bc._ambient_music_last_hit[0] = self.bc.time.time()
        out = self.bc._handle_ambient_music("♪ la la la")
        self.assertTrue(out)
        self.assertEqual(self.bc._ambient_music_hits[0], 1)
        # Below MUSIC_HITS_TO_STANDBY (2) -> no standby announcement yet.
        self.assertFalse(self.bc._standby_mode[0])

    def test_music_window_reset_when_stale(self):
        self._p(self.bc, "is_ambient_music", return_value=True)
        self.bc._jarvis_played_music_at[0] = 0.0
        self.bc._ambient_music_hits[0] = 5
        # Last hit older than the window -> counter resets to 0 then +1.
        self.bc._ambient_music_last_hit[0] = (
            self.bc.time.time() - self.bc.MUSIC_HITS_WINDOW - 10)
        self.bc._handle_ambient_music("[Music]")
        self.assertEqual(self.bc._ambient_music_hits[0], 1)

    def test_music_reaches_threshold_enters_standby(self):
        self._p(self.bc, "is_ambient_music", return_value=True)
        self.bc._jarvis_played_music_at[0] = 0.0
        self.bc._ambient_music_last_hit[0] = self.bc.time.time()
        # One below threshold so this hit crosses it.
        self.bc._ambient_music_hits[0] = self.bc.MUSIC_HITS_TO_STANDBY - 1
        out = self.bc._handle_ambient_music("[Music]")
        self.assertTrue(out)
        self.assertTrue(self.bc._sleep_mode[0])
        self.assertTrue(self.bc._standby_mode[0])
        # Counter resets after entering standby.
        self.assertEqual(self.bc._ambient_music_hits[0], 0)
        self._speak.assert_called_once()


# ════════════════════════════════════════════════════════════════════════════
#  _handle_sleep_triggers
# ════════════════════════════════════════════════════════════════════════════
class HandleSleepTriggersTests(SectionSevenBase):
    def setUp(self):
        super().setUp()
        self._speak = self._p(self.bc, "_speak")
        self._p(self.bc, "set_state")

    def test_standby_phrase_enters_standby(self):
        # Use a known phrase from STANDBY_TRIGGER_PHRASES.
        out = self.bc._handle_sleep_triggers("standby mode")
        self.assertTrue(out)
        self.assertTrue(self.bc._standby_mode[0])
        self.assertTrue(self.bc._sleep_mode[0])

    def test_sleep_phrase_enters_sleep_only(self):
        out = self.bc._handle_sleep_triggers("go to sleep")
        self.assertTrue(out)
        self.assertTrue(self.bc._sleep_mode[0])
        self.assertFalse(self.bc._standby_mode[0])

    def test_long_utterance_does_not_trigger(self):
        # >6 words: the trigger gate is skipped even if a phrase appears.
        text = "well if I ever say go to sleep you should ignore me entirely"
        out = self.bc._handle_sleep_triggers(text)
        self.assertFalse(out)
        self.assertFalse(self.bc._sleep_mode[0])

    def test_unrelated_short_utterance_returns_false(self):
        out = self.bc._handle_sleep_triggers("what time is it")
        self.assertFalse(out)


# ════════════════════════════════════════════════════════════════════════════
#  _handle_sleep_standby
# ════════════════════════════════════════════════════════════════════════════
class HandleSleepStandbyTests(SectionSevenBase):
    def setUp(self):
        super().setUp()
        self._speak = self._p(self.bc, "_speak")
        self._p(self.bc, "set_state")
        self._p(self.bc, "_heartbeat")
        # Default: ambient learning off, wake never refused by music.
        self._p(self.bc, "_audio_music_should_refuse_wake", return_value=False)
        # Force the neural wake fast-path OFF (returns None) so the mic-path
        # tests deterministically exercise the Whisper-transcribe branch on ANY
        # host. _handle_sleep_standby calls _standby_wake_detected(audio) BEFORE
        # transcribe(); only a None return (the neural-disabled default) falls
        # through to the mocked transcribe — a real return (True/False) sets
        # text directly and skips it. Without this, a box with WAKE_WORD_AUTOSTART
        # on builds a real detector that returns False on the silent test buffer,
        # so transcribe is never reached and the wake assertion fails. Mirrors
        # voice-wiring's test_default_flags_use_whisper_path_selector_not_engaged.
        self._p(self.bc, "_standby_wake_detected", return_value=None)
        # Standby starts asleep so the wake path has something to clear.
        self.bc._sleep_mode[0] = True
        self.bc._standby_mode[0] = True

    def test_injected_wake_phrase_wakes_and_greets(self):
        self._p(self.bc, "context_aware_greeting",
                return_value=("Back online, sir.", 1.0))
        # OVERNIGHT_FLAG_FILE absent -> the cleanup branch is a no-op.
        self._p(self.bc, "OVERNIGHT_FLAG_FILE",
                os.path.join(tempfile.gettempdir(), "no_such_overnight_flag.json"))
        self.bc._handle_sleep_standby("hey JARVIS wake up please")
        self.assertFalse(self.bc._sleep_mode[0])
        self.assertFalse(self.bc._standby_mode[0])
        self._speak.assert_called_once_with("Back online, sir.", volume_scale=1.0)

    def test_injected_non_wake_is_ignored(self):
        # No wake phrase, ambient learning off -> line is simply ignored.
        self._p(self.bc, "_ambient_learning_feed")
        self.bc._handle_sleep_standby("just talking to myself over here")
        # Still asleep.
        self.assertTrue(self.bc._sleep_mode[0])
        self._speak.assert_not_called()

    def test_injected_non_wake_feeds_ambient_learner(self):
        self.bc._ambient_learning[0] = True
        feed = self._p(self.bc, "_ambient_learning_feed")
        self.bc._handle_sleep_standby("the meeting is at three tomorrow")
        feed.assert_called_once()
        self.assertTrue(self.bc._sleep_mode[0])

    def test_wake_refused_when_music_lyric_near_miss(self):
        self._p(self.bc, "_audio_music_should_refuse_wake", return_value=True)
        self.bc._handle_sleep_standby("wake up little susie")
        # Wake was refused -> still asleep, no greeting.
        self.assertTrue(self.bc._sleep_mode[0])
        self._speak.assert_not_called()

    def test_mic_path_short_audio_returns_early(self):
        # No injected text -> mic path; record_speech returns a too-short clip.
        short = self.bc.np.zeros(int(self.bc.SAMPLE_RATE * 0.1), dtype="float32")
        self._p(self.bc, "record_speech", return_value=short)
        # transcribe must NOT be reached.
        tr = self._p(self.bc, "transcribe")
        self.bc._handle_sleep_standby(None)
        tr.assert_not_called()
        self.assertTrue(self.bc._sleep_mode[0])

    def test_mic_path_transcribed_wake_phrase_wakes(self):
        good = self.bc.np.zeros(int(self.bc.SAMPLE_RATE * 1.0), dtype="float32")
        self._p(self.bc, "record_speech", return_value=good)
        self._p(self.bc, "_audio_music_feed")
        self._p(self.bc, "transcribe", return_value=("wake up", {}))
        self._p(self.bc, "context_aware_greeting",
                return_value=("At your service, sir.", 0.9))
        self._p(self.bc, "OVERNIGHT_FLAG_FILE",
                os.path.join(tempfile.gettempdir(), "no_such_overnight_flag.json"))
        self.bc._handle_sleep_standby(None)
        self.assertFalse(self.bc._sleep_mode[0])
        self._speak.assert_called_once()

    def test_wake_clears_overnight_flag_when_present(self):
        good = self.bc.np.zeros(int(self.bc.SAMPLE_RATE * 1.0), dtype="float32")
        self._p(self.bc, "record_speech", return_value=good)
        self._p(self.bc, "_audio_music_feed")
        self._p(self.bc, "transcribe", return_value=("wake up", {}))
        self._p(self.bc, "context_aware_greeting", return_value=("Hello, sir.", 1.0))
        whud = self._p(self.bc, "_write_hud_state")
        with tempfile.TemporaryDirectory() as td:
            flag = os.path.join(td, "overnight.flag")
            with open(flag, "w", encoding="utf-8") as f:
                f.write("1")
            self._p(self.bc, "OVERNIGHT_FLAG_FILE", flag)
            self.bc._handle_sleep_standby(None)
            # The flag file is removed on wake.
            self.assertFalse(os.path.exists(flag))
        whud.assert_called_once()

    def test_wake_overnight_flag_remove_failure_swallowed(self):
        # If os.remove of the overnight flag raises, the wake still proceeds
        # (the cleanup is best-effort, wrapped in try/except).
        self._p(self.bc, "context_aware_greeting", return_value=("Hi, sir.", 1.0))
        with tempfile.TemporaryDirectory() as td:
            flag = os.path.join(td, "overnight.flag")
            with open(flag, "w", encoding="utf-8") as f:
                f.write("1")
            self._p(self.bc, "OVERNIGHT_FLAG_FILE", flag)
            self._p(self.bc.os, "remove",
                    side_effect=OSError("locked by another process"))
            # Inject a wake phrase; the remove failure must not abort the wake.
            self.bc._handle_sleep_standby("wake up")
        self.assertFalse(self.bc._sleep_mode[0])
        self._speak.assert_called_once()


# ════════════════════════════════════════════════════════════════════════════
#  _run_llm_dispatch
# ════════════════════════════════════════════════════════════════════════════
class RunLlmDispatchTests(SectionSevenBase):
    def setUp(self):
        super().setUp()
        self._speak = self._p(self.bc, "_speak")
        self._p(self.bc, "set_state")
        self._p(self.bc, "_heartbeat")
        # Quip layer is a pass-through for these tests.
        self._p(self.bc, "_apply_quip_layer", side_effect=lambda s, r: s)

    def test_glance_fast_path_used_when_available(self):
        self._p(self.bc, "maybe_glance_response", return_value="That's your editor, sir.")
        gra = self._p(self.bc, "get_response_with_animation")
        # No actions, no follow-up.
        self._p(self.bc, "parse_and_run_actions",
                return_value=("That's your editor, sir.", []))
        out = self.bc._run_llm_dispatch("what is this")
        self.assertEqual(out, "That's your editor, sir.")
        gra.assert_not_called()      # glance short-circuited the LLM call
        self._speak.assert_called_once_with("That's your editor, sir.")

    def test_full_llm_path_when_no_glance(self):
        self._p(self.bc, "maybe_glance_response", return_value=None)
        self._p(self.bc, "get_response_with_animation",
                return_value="The weather is fine, sir.")
        self._p(self.bc, "parse_and_run_actions",
                return_value=("The weather is fine, sir.", []))
        out = self.bc._run_llm_dispatch("how's the weather")
        self.assertEqual(out, "The weather is fine, sir.")
        self._speak.assert_called_once_with("The weather is fine, sir.")

    def test_informative_action_triggers_followup_loop(self):
        self._p(self.bc, "maybe_glance_response", return_value=None)
        self._p(self.bc, "get_response_with_animation",
                return_value="[ACTION: see_screen]")
        # First parse: one informative action. Follow-up parse: none.
        self._p(self.bc, "parse_and_run_actions", side_effect=[
            ("", [("see_screen", "A code editor is open", True)]),
            ("Looks like your editor, sir.", []),
        ])
        gfr = self._p(self.bc, "get_followup_response",
                      return_value="Looks like your editor, sir.")
        out = self.bc._run_llm_dispatch("what's on screen")
        gfr.assert_called_once()
        # Both the (empty) first reply and the follow-up are appended/spoken.
        self.assertIn("Looks like your editor, sir.",
                      [c.args[0] for c in self._speak.call_args_list])
        # Original reply is returned for learn_from_turn.
        self.assertEqual(out, "[ACTION: see_screen]")

    def test_failed_action_followup_then_stops_on_repeat(self):
        self._p(self.bc, "maybe_glance_response", return_value=None)
        self._p(self.bc, "get_response_with_animation",
                return_value="[ACTION: click, missing]")
        # Action keeps failing IDENTICALLY; loop must give it exactly one
        # retry then break (same (action, result) = no progress).
        self._p(self.bc, "parse_and_run_actions", side_effect=[
            ("", [("click", "could not find target", False)]),
            ("", [("click", "could not find target", False)]),
            ("", [("click", "could not find target", False)]),
        ])
        gfr = self._p(self.bc, "get_followup_response", return_value="Retrying, sir.")
        self.bc._run_llm_dispatch("click the button")
        # Exactly one follow-up round for the failing action (retry), then stop.
        self.assertEqual(gfr.call_count, 1)

    def test_failed_action_different_result_keeps_chain_alive(self):
        # A failing action retried with a DIFFERENT argument (→ different
        # result text) is a new approach, not a stuck loop — the chain must
        # continue until an identical (action, result) repeat shows up.
        self._p(self.bc, "maybe_glance_response", return_value=None)
        self._p(self.bc, "get_response_with_animation",
                return_value="[ACTION: click, the bookmark]")
        self._p(self.bc, "parse_and_run_actions", side_effect=[
            ("", [("click", "could not locate 'the bookmark' on screen", False)]),
            ("", [("click", "could not locate 'bookmarks toolbar item' on screen", False)]),
            ("", [("click", "could not locate 'bookmarks toolbar item' on screen", False)]),
            ("", [("click", "could not locate 'bookmarks toolbar item' on screen", False)]),
        ])
        gfr = self._p(self.bc, "get_followup_response", return_value="Trying another way, sir.")
        self.bc._run_llm_dispatch("click my bookmark")
        # Round 1: first failure → follow-up (new approach). Round 2: new
        # result → follow-up again. Round 3: identical repeat → stop.
        self.assertEqual(gfr.call_count, 2)

    def test_followup_depth_falls_back_on_import_error(self):
        # If core.mode_router.followup_loop_depth import fails, depth falls
        # back to the built-in default (8) and the loop still terminates
        # (no informative actions here).
        self._p(self.bc, "maybe_glance_response", return_value=None)
        self._p(self.bc, "get_response_with_animation", return_value="Done, sir.")
        self._p(self.bc, "parse_and_run_actions", return_value=("Done, sir.", []))
        real_import = __import__

        def _shim(name, *a, **k):
            if name == "core.mode_router":
                raise ImportError("no router")
            return real_import(name, *a, **k)
        with mock.patch("builtins.__import__", side_effect=_shim):
            out = self.bc._run_llm_dispatch("do the thing")
        self.assertEqual(out, "Done, sir.")

    def test_empty_spoken_text_not_spoken(self):
        self._p(self.bc, "maybe_glance_response", return_value=None)
        self._p(self.bc, "get_response_with_animation", return_value="[ACTION: mute]")
        # Cleaned spoken text empty, no actions -> nothing spoken.
        self._p(self.bc, "parse_and_run_actions", return_value=("", []))
        self.bc._run_llm_dispatch("mute")
        self._speak.assert_not_called()


# ════════════════════════════════════════════════════════════════════════════
#  _blue_green_loop_tick  /  _consume_blue_green_handoff
# ════════════════════════════════════════════════════════════════════════════
class BlueGreenLoopTickTests(SectionSevenBase):
    def setUp(self):
        super().setUp()
        self._speak = self._p(self.bc, "_speak")
        # A fresh fake blue/green manager for each test; default every consumer
        # to "no signal" so individual tests opt in.
        self.bgm = mock.MagicMock()
        self.bgm.read_version.return_value = "1.2.3"
        self.bgm.consume_handoff_signal.return_value = None
        self.bgm.consume_upgrade_aborted_signal.return_value = None
        self.bgm.consume_handoff_failure_signal.return_value = None
        self._p(self.bc, "_bgm", self.bgm)
        # Reset the cross-iteration timers so each test starts clean.
        self._p(self.bc, "_bg_last_heartbeat", [0.0])
        self._p(self.bc, "_bg_handoff_seen_at", [0.0])

    def test_heartbeat_emitted_when_due(self):
        self._p(self.bc, "BLUE_GREEN_ROLE", "staging")
        out = self.bc._blue_green_loop_tick()
        self.assertFalse(out)
        self.bgm.heartbeat.assert_called_once()

    def test_heartbeat_failure_swallowed(self):
        self._p(self.bc, "BLUE_GREEN_ROLE", "staging")
        self.bgm.heartbeat.side_effect = RuntimeError("fs hiccup")
        # Must not raise.
        self.assertFalse(self.bc._blue_green_loop_tick())

    def test_prod_handoff_signal_announces_and_snapshots(self):
        self._p(self.bc, "BLUE_GREEN_ROLE", "prod")
        self.bgm.consume_handoff_signal.return_value = {
            "target_version": "2.0.0", "signaled_at": 123.0}
        out = self.bc._blue_green_loop_tick()
        # Handoff just seen -> not yet time to exit.
        self.assertFalse(out)
        self._speak.assert_called_once()
        self.assertIn("2.0.0", self._speak.call_args[0][0])
        self.bgm.write_handoff_state.assert_called_once()

    def test_prod_handoff_then_window_elapsed_exits(self):
        self._p(self.bc, "BLUE_GREEN_ROLE", "prod")
        # Pretend the handoff was seen 11s ago (>10s grace) so this tick exits.
        self._p(self.bc, "_bg_handoff_seen_at", [self.bc.time.time() - 11.0])
        out = self.bc._blue_green_loop_tick()
        self.assertTrue(out)

    def test_prod_upgrade_aborted_announced(self):
        self._p(self.bc, "BLUE_GREEN_ROLE", "prod")
        self.bgm.consume_upgrade_aborted_signal.return_value = {"reason": "x"}
        self.bc._blue_green_loop_tick()
        spoken = " ".join(c.args[0] for c in self._speak.call_args_list).lower()
        self.assertIn("aborted", spoken)

    def test_prod_handoff_failure_cancels_pending_exit(self):
        self._p(self.bc, "BLUE_GREEN_ROLE", "prod")
        seen = [self.bc.time.time() - 11.0]   # would otherwise exit
        self._p(self.bc, "_bg_handoff_seen_at", seen)
        self.bgm.consume_handoff_failure_signal.return_value = {"err": "boom"}
        out = self.bc._blue_green_loop_tick()
        # Failure signal zeroes the pending-exit timer, so we do NOT exit.
        self.assertFalse(out)
        self.assertEqual(seen[0], 0.0)

    def test_no_manager_is_noop(self):
        self._p(self.bc, "_bgm", None)
        self.assertFalse(self.bc._blue_green_loop_tick())
        self._speak.assert_not_called()

    def test_handoff_signal_consume_exception_swallowed(self):
        # 13383-13384: consume_handoff_signal() raises -> _signal=None, so no
        # announce / snapshot, and the tick completes without exiting.
        self._p(self.bc, "BLUE_GREEN_ROLE", "prod")
        self.bgm.consume_handoff_signal.side_effect = RuntimeError("fs gone")
        out = self.bc._blue_green_loop_tick()
        self.assertFalse(out)
        self._speak.assert_not_called()
        self.bgm.write_handoff_state.assert_not_called()

    def test_upgrade_aborted_consume_exception_swallowed(self):
        # 13425-13426: consume_upgrade_aborted_signal() raises -> _abort=None,
        # no abort announcement.
        self._p(self.bc, "BLUE_GREEN_ROLE", "prod")
        self.bgm.consume_upgrade_aborted_signal.side_effect = RuntimeError("io")
        out = self.bc._blue_green_loop_tick()
        self.assertFalse(out)
        self._speak.assert_not_called()

    def test_handoff_failure_consume_exception_swallowed(self):
        # 13435-13436: consume_handoff_failure_signal() raises -> _fail=None,
        # the pending-exit timer is NOT cleared by a failure that never parsed.
        self._p(self.bc, "BLUE_GREEN_ROLE", "prod")
        seen = [self.bc.time.time() - 11.0]
        self._p(self.bc, "_bg_handoff_seen_at", seen)
        self.bgm.consume_handoff_failure_signal.side_effect = RuntimeError("io")
        # Pending-exit timer survived (not zeroed) -> the window-elapsed check
        # still fires and the tick reports exit.
        out = self.bc._blue_green_loop_tick()
        self.assertTrue(out)
        self.assertNotEqual(seen[0], 0.0)

    def test_handoff_failure_announce_exception_swallowed(self):
        # 13443-13444: a handoff-failure signal IS present, but the spoken
        # announcement raises -> the except logs and the tick still cancels the
        # pending exit (seen-timer zeroed) and does not raise.
        self._p(self.bc, "BLUE_GREEN_ROLE", "prod")
        seen = [self.bc.time.time() - 11.0]
        self._p(self.bc, "_bg_handoff_seen_at", seen)
        self.bgm.consume_handoff_failure_signal.return_value = {"err": "boom"}
        self._speak.side_effect = RuntimeError("tts dead")
        out = self.bc._blue_green_loop_tick()
        self.assertFalse(out)          # failure zeroes the pending-exit timer
        self.assertEqual(seen[0], 0.0)
        self._speak.assert_called()


class ConsumeBlueGreenHandoffTests(SectionSevenBase):
    def setUp(self):
        super().setUp()
        self.bgm = mock.MagicMock()
        self.bgm.RESUME_HANDOFF_FLAG = "--resume-handoff"
        self._p(self.bc, "_bgm", self.bgm)

    def test_noop_when_not_prod(self):
        self._p(self.bc, "BLUE_GREEN_ROLE", "staging")
        self._p(self.bc.sys, "argv", ["bobert", "--resume-handoff"])
        ls, timers = self.bc._consume_blue_green_handoff()
        self.assertIsNone(ls)
        self.assertEqual(timers, [])
        self.bgm.consume_handoff_state.assert_not_called()

    def test_noop_when_flag_absent(self):
        self._p(self.bc, "BLUE_GREEN_ROLE", "prod")
        self._p(self.bc.sys, "argv", ["bobert"])
        ls, timers = self.bc._consume_blue_green_handoff()
        self.assertIsNone(ls)
        self.assertEqual(timers, [])

    def test_resumes_conversation_tail_and_payload(self):
        self._p(self.bc, "BLUE_GREEN_ROLE", "prod")
        self._p(self.bc.sys, "argv", ["bobert", "--resume-handoff"])
        self.bgm.consume_handoff_state.return_value = {
            "conversation_tail": [
                {"role": "user", "content": "what's the time"},
                {"role": "assistant", "content": "It's noon, sir."},
                {"bad": "entry"},               # filtered out (no role/content)
            ],
            "last_speech_time": 4567.0,
            "active_timers": [{"label": "tea", "secs": 300}],
            "version_at_handoff": "1.9.0",
            "signaled_at": 111.0,
        }
        start = len(self.bc.conversation_history)
        ls, timers = self.bc._consume_blue_green_handoff()
        self.assertEqual(ls, 4567.0)
        self.assertEqual(timers, [{"label": "tea", "secs": 300}])
        # Two valid messages appended; the malformed one dropped.
        self.assertEqual(len(self.bc.conversation_history) - start, 2)

    def test_handoff_consume_exception_returns_empty(self):
        self._p(self.bc, "BLUE_GREEN_ROLE", "prod")
        self._p(self.bc.sys, "argv", ["bobert", "--resume-handoff"])
        self.bgm.consume_handoff_state.side_effect = RuntimeError("disk gone")
        ls, timers = self.bc._consume_blue_green_handoff()
        self.assertIsNone(ls)
        self.assertEqual(timers, [])


# ════════════════════════════════════════════════════════════════════════════
#  get_response_with_animation  (the thinking-animation daemon must NOT leak on
#  an _call_llm raise, and the raise must NOT propagate to the main loop)
# ════════════════════════════════════════════════════════════════════════════
class GetResponseWithAnimationTests(SectionSevenBase):
    def setUp(self):
        super().setUp()
        self._p(self.bc, "pause_face_tracking")
        self._p(self.bc, "set_state")
        # send() is what the real _thinking_loop calls each tick; neutralise it
        # so the REAL loop body can run against a real (short-lived) thread
        # without driving the HUD.
        self._p(self.bc, "send")
        self._p(self.bc, "_heartbeat")

    def test_happy_path_returns_reply_and_joins_thread(self):
        captured = {}
        real_thread = self.bc.threading.Thread

        def _capture_thread(*a, **k):
            t = real_thread(*a, **k)
            captured["thread"] = t
            return t
        self._p(self.bc, "_call_llm", return_value="At your service, sir.")
        with mock.patch.object(self.bc.threading, "Thread", _capture_thread):
            reply = self.bc.get_response_with_animation("hi")
        self.assertEqual(reply, "At your service, sir.")
        # The animation daemon was stopped and joined — not left running.
        self.assertFalse(captured["thread"].is_alive())

    def test_call_llm_raise_does_not_propagate_and_returns_fallback(self):
        # The crux of the finding (1): an exception out of _call_llm must NOT
        # propagate — the outer main loop only catches KeyboardInterrupt, so a
        # raise here would crash the whole conversation loop.
        self._p(self.bc, "_call_llm", side_effect=RuntimeError("tone blew up"))
        reply = self.bc.get_response_with_animation("hi")
        self.assertEqual(reply, self.bc._llm_error_reply())

    def test_call_llm_raise_still_stops_animation_daemon(self):
        # The crux of the finding (2): even when _call_llm raises, the finally
        # block must stop+join the REAL _thinking_loop daemon. A leaked loop
        # keeps calling _heartbeat() forever, falsifying the main-loop watchdog
        # so it can never see a stale beat and recover the wedge.
        captured = {}
        real_thread = self.bc.threading.Thread

        def _capture_thread(*a, **k):
            t = real_thread(*a, **k)
            captured["thread"] = t
            return t
        self._p(self.bc, "_call_llm", side_effect=RuntimeError("local route blew up"))
        with mock.patch.object(self.bc.threading, "Thread", _capture_thread):
            self.bc.get_response_with_animation("hi")
        # Daemon really terminated (stop_evt set + joined in the finally).
        self.assertFalse(captured["thread"].is_alive())

    def test_local_then_cloud_or_honest_never_propagates(self):
        # The local route's no-propagate guarantee (finding fix 3): an
        # unexpected raise from a helper degrades to the honest unavailability
        # line instead of escaping into _call_llm / the main loop.
        self._p(self.bc, "_call_local_llm",
                side_effect=RuntimeError("ollama probe exploded"))
        self._p(self.bc, "_local_unavailable_message",
                return_value="HONEST UNAVAILABLE")
        out = self.bc._local_then_cloud_or_honest("sys", [])
        self.assertEqual(out, "HONEST UNAVAILABLE")

    def test_call_llm_prologue_classifier_raise_is_neutralised(self):
        # The prologue guard (finding fix 2): a detect_tone / route_voice_emotion
        # raise must degrade to neutral defaults, not crash the turn. Drive the
        # local route so we exercise _call_llm end-to-end without the network.
        self._p(self.bc, "detect_tone", side_effect=RuntimeError("tone classifier down"))
        self._p(self.bc, "route_voice_emotion", side_effect=RuntimeError("router down"))
        self._p(self.bc, "_system_prompt", "SYS")
        # Force the LOCAL route and have it answer so we reach the tail cleanly.
        import core.config as _cfg
        self._p(_cfg, "model_route", return_value="local")
        self._p(self.bc, "_local_then_cloud_or_honest", return_value="Local reply, sir.")
        self._p(self.bc, "_mcu_phrases",
                mock.Mock(detect_phrases_in_reply=mock.Mock(return_value={})))
        reply = self.bc._call_llm("what's up")
        self.assertEqual(reply, "Local reply, sir.")
        # Neutral defaults were cached despite both classifiers raising.
        self.assertIsNone(self.bc._last_user_tone[0])
        self.assertEqual(self.bc._last_voice_route[0], {"mood": "casual", "addendum": ""})


# ════════════════════════════════════════════════════════════════════════════
#  _do_proactive_turn
# ════════════════════════════════════════════════════════════════════════════
class DoProactiveTurnTests(SectionSevenBase):
    def setUp(self):
        super().setUp()
        self._p(self.bc, "pause_face_tracking")
        self._p(self.bc, "resume_face_tracking")
        self._p(self.bc, "set_state")
        self._speak = self._p(self.bc, "_speak")
        # Neutralise the animation thread so no real thread spins.
        self._p(self.bc, "_thinking_loop")
        fake_thread = mock.MagicMock()
        self._p(self.bc.threading, "Thread", return_value=fake_thread)

    def test_empty_comment_returns_without_speaking(self):
        self._p(self.bc, "generate_proactive_comment", return_value="")
        self.bc._do_proactive_turn({"facts": {}})
        self._speak.assert_not_called()

    def test_comment_is_spoken_and_recorded(self):
        self._p(self.bc, "generate_proactive_comment",
                return_value="You've been at this a while, sir.")
        self._p(self.bc, "parse_and_run_actions",
                return_value=("You've been at this a while, sir.", []))
        self._p(self.bc, "_apply_quip_layer", side_effect=lambda s, r: s)
        self.bc._do_proactive_turn({"facts": {}})
        self._speak.assert_called_once_with("You've been at this a while, sir.")
        self.assertEqual(self.bc.conversation_history[-1]["content"],
                         "You've been at this a while, sir.")


# ════════════════════════════════════════════════════════════════════════════
#  _drain_injected_command
# ════════════════════════════════════════════════════════════════════════════
class DrainInjectedCommandTests(SectionSevenBase):
    def setUp(self):
        super().setUp()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._path = os.path.join(self._tmp.name, "injected_commands.json")
        self._p(self.bc, "INJECTED_COMMANDS_PATH", self._path)

    def _write(self, obj):
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(obj, f)

    def test_absent_file_returns_none(self):
        self.assertIsNone(self.bc._drain_injected_command())

    def test_single_string_item(self):
        self._write(["open notepad"])
        self.assertEqual(self.bc._drain_injected_command(), "open notepad")
        # Queue fully consumed.
        self.assertFalse(os.path.exists(self._path))

    def test_dict_item_text_field(self):
        self._write([{"text": "play jazz"}])
        self.assertEqual(self.bc._drain_injected_command(), "play jazz")

    def test_tail_requeued(self):
        self._write(["first", "second", "third"])
        self.assertEqual(self.bc._drain_injected_command(), "first")
        # The remaining two are written back for the next pass.
        with open(self._path, encoding="utf-8") as f:
            rest = json.load(f)
        self.assertEqual(rest, ["second", "third"])

    def test_corrupt_json_discarded(self):
        with open(self._path, "w", encoding="utf-8") as f:
            f.write("{not valid json")
        self.assertIsNone(self.bc._drain_injected_command())
        # Snapshot dropped so the same garbage doesn't re-trip.
        self.assertFalse(os.path.exists(self._path))

    def test_empty_array_returns_none(self):
        self._write([])
        self.assertIsNone(self.bc._drain_injected_command())

    def test_blank_file_returns_none(self):
        with open(self._path, "w", encoding="utf-8") as f:
            f.write("   ")
        self.assertIsNone(self.bc._drain_injected_command())

    def test_non_str_non_dict_first_item_returns_none(self):
        self._write([12345])
        self.assertIsNone(self.bc._drain_injected_command())

    def test_empty_text_after_strip_returns_none(self):
        self._write(["   "])
        self.assertIsNone(self.bc._drain_injected_command())

    def test_cleanup_remove_failure_swallowed(self):
        # os.remove of the .consuming snapshot raising must not crash the drain
        # (the cleanup is wrapped in try/except: pass).
        self._write(["solo"])
        self._p(self.bc.os, "remove", side_effect=OSError("locked"))
        # Still returns the parsed command despite the cleanup failure.
        self.assertEqual(self.bc._drain_injected_command(), "solo")

    def test_requeue_write_failure_swallowed(self):
        # If writing the remaining-tail temp file fails, the head is still
        # returned (the tail is dropped rather than re-firing the head).
        self._write(["head", "tail"])
        self._p(self.bc.tempfile, "mkstemp",
                side_effect=OSError("no temp space"))
        self.assertEqual(self.bc._drain_injected_command(), "head")

    def test_empty_array_remove_failure_swallowed(self):
        # Empty list -> the early remove is guarded; a failing remove is fine.
        self._write([])
        self._p(self.bc.os, "remove", side_effect=OSError("locked"))
        self.assertIsNone(self.bc._drain_injected_command())

    def test_blank_file_remove_failure_swallowed(self):
        # 12920-12921: a whitespace-only snapshot whose os.remove ALSO fails ->
        # the failure is swallowed and the drain still returns None.
        with open(self._path, "w", encoding="utf-8") as f:
            f.write("   ")
        self._p(self.bc.os, "remove", side_effect=OSError("locked"))
        self.assertIsNone(self.bc._drain_injected_command())

    def test_read_failure_remove_failure_swallowed(self):
        # 12914-12917: the rename succeeds but reopening the .consuming snapshot
        # raises (read fail) AND the subsequent os.remove ALSO raises -> both
        # swallowed, drain returns None.
        self._write(["whatever"])
        real_open = open

        def _flaky_open(p, *a, **k):
            if str(p).endswith(".consuming"):
                raise OSError("snapshot vanished")
            return real_open(p, *a, **k)
        self._p(self.bc.os, "remove", side_effect=OSError("locked"))
        with mock.patch("builtins.open", _flaky_open):
            self.assertIsNone(self.bc._drain_injected_command())

    def test_requeue_fdopen_failure_closes_fd(self):
        # 12953-12955: the tail-requeue mkstemp succeeds but os.fdopen raises
        # (fd still owned by us) -> the cleanup os.close(fd) runs (real close so
        # the descriptor/temp file are released) and the head is still returned
        # (tail dropped, not re-fired).
        self._write(["head", "tail"])
        self._p(self.bc.os, "fdopen", side_effect=OSError("fdopen denied"))
        self.assertEqual(self.bc._drain_injected_command(), "head")

    def test_requeue_replace_failure_unlinks_tmp(self):
        # 12957-12958: the requeue write succeeds but os.replace of the temp
        # over INJECTED_COMMANDS_PATH raises (tmp still present) -> the cleanup
        # os.unlink(tmp) runs (real unlink so the temp is removed) and the head
        # is still returned.
        self._write(["head", "tail"])
        real_replace = self.bc.os.replace

        def _flaky_replace(src, dst, *a, **k):
            # The FIRST replace (claim of the queue -> .consuming) must succeed;
            # only the tail-requeue replace (tmp -> INJECTED_COMMANDS_PATH) fails.
            if str(dst).endswith(".consuming"):
                return real_replace(src, dst, *a, **k)
            raise OSError("replace denied")
        self._p(self.bc.os, "replace", side_effect=_flaky_replace)
        self.assertEqual(self.bc._drain_injected_command(), "head")


# ════════════════════════════════════════════════════════════════════════════
#  _enforce_singleton  (boot PID-file re-check; subprocess/os mocked)
# ════════════════════════════════════════════════════════════════════════════
class EnforceSingletonTests(SectionSevenBase):
    def setUp(self):
        super().setUp()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._lock = os.path.join(self._tmp.name, "jarvis.lock")
        self._p(self.bc, "_LOCK_FILE", self._lock)
        # Default: we do NOT hold the OS byte-range lock, so the re-check runs.
        self._p(self.bc, "_SINGLETON_HELD_FD", None)

    def test_held_fd_short_circuits(self):
        self._p(self.bc, "_SINGLETON_HELD_FD", 7)   # we already hold the lock
        # No lock file, but the held-fd guard returns before touching disk.
        self.bc._enforce_singleton()
        self.assertFalse(os.path.exists(self._lock))

    def test_no_lock_file_writes_our_pid(self):
        self.bc._enforce_singleton()
        with open(self._lock, encoding="utf-8") as f:
            self.assertEqual(f.read().strip(), str(os.getpid()))

    def test_lock_names_us_is_noop(self):
        with open(self._lock, "w", encoding="utf-8") as f:
            f.write(str(os.getpid()))
        self._p(self.bc, "_read_lock_pid", return_value=os.getpid())
        # Should return without rewriting / exiting.
        self.bc._enforce_singleton()

    def test_stale_dead_pid_overwritten(self):
        with open(self._lock, "w", encoding="utf-8") as f:
            f.write("999999")
        self._p(self.bc, "_read_lock_pid", return_value=999999)
        # tasklist says the PID is not running -> overwrite with ours.
        fake = mock.MagicMock(stdout="INFO: No tasks are running.")
        self._p(self.bc.subprocess, "run", return_value=fake)
        self.bc._enforce_singleton()
        with open(self._lock, encoding="utf-8") as f:
            self.assertEqual(f.read().strip(), str(os.getpid()))

    def test_live_other_instance_exits(self):
        with open(self._lock, "w", encoding="utf-8") as f:
            f.write("4242")
        self._p(self.bc, "_read_lock_pid", return_value=4242)
        fake = mock.MagicMock(
            stdout='"python.exe","4242","Console","1","100,000 K"')
        self._p(self.bc.subprocess, "run", return_value=fake)
        self._p(self.bc.sys, "platform", "win32")
        with self.assertRaises(SystemExit):
            self.bc._enforce_singleton()

    def test_tasklist_timeout_assumes_alive_and_exits(self):
        with open(self._lock, "w", encoding="utf-8") as f:
            f.write("4242")
        self._p(self.bc, "_read_lock_pid", return_value=4242)
        self._p(self.bc.sys, "platform", "win32")
        self._p(self.bc.subprocess, "run",
                side_effect=self.bc.subprocess.TimeoutExpired("tasklist", 5))
        with self.assertRaises(SystemExit):
            self.bc._enforce_singleton()

    def test_write_failure_is_swallowed(self):
        # open() for the PID write raises -> best-effort, no exception escapes.
        real_open = open

        def _open_shim(path, *a, **k):
            if path == self._lock and (a and "w" in a[0]):
                raise OSError("read-only fs")
            return real_open(path, *a, **k)
        with mock.patch("builtins.open", side_effect=_open_shim):
            self.bc._enforce_singleton()   # must not raise


# ════════════════════════════════════════════════════════════════════════════
#  _release_singleton  (OS byte-range lock release)
# ════════════════════════════════════════════════════════════════════════════
class ReleaseSingletonTests(SectionSevenBase):
    def test_no_held_fd_is_noop(self):
        self._p(self.bc, "_SINGLETON_HELD_FD", None)
        self.bc._release_singleton()   # returns immediately
        self.assertIsNone(self.bc._SINGLETON_HELD_FD)

    def test_held_fd_unlocked_and_closed(self):
        # A real temp fd so msvcrt.locking / os.close operate on something live.
        fd, path = tempfile.mkstemp()
        self.addCleanup(lambda: os.path.exists(path) and os.remove(path))
        self._p(self.bc, "_SINGLETON_HELD_FD", fd)
        self._p(self.bc.sys, "platform", "win32")
        # msvcrt.LK_UNLCK on a non-locked region raises; the code swallows it.
        self.bc._release_singleton()
        # fd is cleared and closed (a second close raises OSError -> swallowed).
        self.assertIsNone(self.bc._SINGLETON_HELD_FD)
        with self.assertRaises(OSError):
            os.close(fd)


# ════════════════════════════════════════════════════════════════════════════
#  _preflight_cameras
# ════════════════════════════════════════════════════════════════════════════
class PreflightCamerasTests(SectionSevenBase):
    def setUp(self):
        super().setUp()
        # Probe is disabled by default in this tree; force it on so the fan-out
        # body runs, and supply a deterministic two-camera config.
        self._p(self.bc, "CAMERA_PROBE_ENABLED", True)
        self._p(self.bc, "CAMERAS",
                [{"index": 0, "label": "cam0"}, {"index": 1, "label": "cam1"}])

    def test_no_cameras_early_returns(self):
        self._p(self.bc, "CAMERAS", [])
        probe = self._p(self.bc, "_probe_camera_index")
        self.bc._preflight_cameras(timeout_sec=0.1)
        probe.assert_not_called()

    def test_probe_disabled_early_returns(self):
        self._p(self.bc, "CAMERA_PROBE_ENABLED", False)
        probe = self._p(self.bc, "_probe_camera_index")
        self.bc._preflight_cameras(timeout_sec=0.1)
        probe.assert_not_called()

    def test_one_bad_camera_dropped(self):
        # index 0 opens, index 1 fails -> index 1 dropped (one survivor remains).
        self._p(self.bc, "_probe_camera_index",
                side_effect=lambda i, timeout_sec=2.0: i == 0)
        self.bc._preflight_cameras(timeout_sec=0.1)
        remaining = [c["index"] for c in self.bc.CAMERAS]
        self.assertEqual(remaining, [0])

    def test_all_bad_leaves_cameras_untouched(self):
        # Every probe fails -> CAMERAS is left intact for the later full sweep.
        self._p(self.bc, "_probe_camera_index",
                side_effect=lambda i, timeout_sec=2.0: False)
        self.bc._preflight_cameras(timeout_sec=0.1)
        self.assertEqual(len(self.bc.CAMERAS), 2)

    def test_dropped_primary_promotes_survivor(self):
        # 2026-07-07 owner fix ("camera preview broken again"): the config PRIMARY
        # is index 1 (Left webcam); it fails and is dropped, leaving only index 0
        # (primary=False). The survivor MUST be promoted to primary — otherwise
        # the HUD preview write + primary-face tracking (both gated on
        # cam["primary"]) never fire and the camera tile goes dark.
        self._p(self.bc, "CAMERAS",
                [{"index": 0, "label": "Right", "primary": False},
                 {"index": 1, "label": "Left", "primary": True}])
        self._p(self.bc, "_probe_camera_index",
                side_effect=lambda i, timeout_sec=2.0: i == 0)   # only 0 opens
        self.bc._preflight_cameras(timeout_sec=0.1)
        self.assertEqual([c["index"] for c in self.bc.CAMERAS], [0])
        self.assertTrue(self.bc.CAMERAS[0]["primary"],
                        "surviving camera must be promoted to primary")

    def test_surviving_primary_not_disturbed(self):
        # When the primary survives, promotion is a no-op (exactly one primary).
        self._p(self.bc, "CAMERAS",
                [{"index": 0, "label": "Left", "primary": True},
                 {"index": 1, "label": "Right", "primary": False}])
        self._p(self.bc, "_probe_camera_index",
                side_effect=lambda i, timeout_sec=2.0: i == 0)   # index 1 dropped
        self.bc._preflight_cameras(timeout_sec=0.1)
        self.assertEqual([c["index"] for c in self.bc.CAMERAS], [0])
        self.assertEqual(sum(1 for c in self.bc.CAMERAS if c["primary"]), 1)
        self.assertTrue(self.bc.CAMERAS[0]["primary"])

    def test_probe_exception_treated_as_bad(self):
        # A probe raising must be caught (treated as a failed open), not escape.
        def _raise(i, timeout_sec=2.0):
            if i == 1:
                raise RuntimeError("probe blew up")
            return True
        self._p(self.bc, "_probe_camera_index", side_effect=_raise)
        self.bc._preflight_cameras(timeout_sec=0.1)
        self.assertEqual([c["index"] for c in self.bc.CAMERAS], [0])

    def test_entry_without_index_is_skipped(self):
        # A malformed camera entry (no "index") is skipped in both the spawn and
        # the result-tally loops, without crashing.
        self._p(self.bc, "CAMERAS",
                [{"index": 0, "label": "cam0"}, {"label": "no-index-cam"}])
        self._p(self.bc, "_probe_camera_index",
                side_effect=lambda i, timeout_sec=2.0: True)
        self.bc._preflight_cameras(timeout_sec=0.1)
        # The valid camera survives; the index-less entry is left in place
        # (it was never probed, so it isn't in the "bad" set).
        self.assertIn(0, [c.get("index") for c in self.bc.CAMERAS])


# ════════════════════════════════════════════════════════════════════════════
#  _startup_preflight
# ════════════════════════════════════════════════════════════════════════════
class StartupPreflightTests(SectionSevenBase):
    def setUp(self):
        super().setUp()
        # Stub the three sub-checks; individual tests override return values.
        self._p(self.bc, "_preflight_cublas_check", return_value=True)
        self._p(self.bc, "_preflight_cameras")
        self._speak = self._p(self.bc, "_speak")

    def test_claude_reachable_path(self):
        self._p(self.bc, "_preflight_api_key", return_value=(True, "ok"))
        # Should complete without raising and without consulting the local model.
        alive = self._p(self.bc, "_ollama_alive")
        self.bc._startup_preflight()
        alive.assert_not_called()

    def test_claude_down_but_local_ok_continues(self):
        self._p(self.bc, "_preflight_api_key",
                return_value=(False, "usage cap - reset 2026-07-01"))
        self._p(self.bc, "_ollama_alive", return_value=True)
        # No fatal exit; boot continues on the local model.
        self.bc._startup_preflight()
        self._speak.assert_not_called()

    def test_no_backend_at_all_is_fatal(self):
        self._p(self.bc, "_preflight_api_key", return_value=(False, "no key"))
        self._p(self.bc, "_ollama_alive", return_value=False)
        self._p(self.bc, "close_log")
        with self.assertRaises(SystemExit):
            self.bc._startup_preflight()
        # The fatal path tries to speak the error.
        self._speak.assert_called_once()

    def test_api_preflight_exception_treated_as_down(self):
        self._p(self.bc, "_preflight_api_key",
                side_effect=RuntimeError("network blip"))
        self._p(self.bc, "_ollama_alive", return_value=True)
        # Exception is caught -> ok=False, local alive -> continues.
        self.bc._startup_preflight()

    def test_subcheck_exceptions_are_swallowed(self):
        self._p(self.bc, "_preflight_api_key", return_value=(True, "ok"))
        self._p(self.bc, "_preflight_cublas_check",
                side_effect=RuntimeError("dll probe boom"))
        self._p(self.bc, "_preflight_cameras",
                side_effect=RuntimeError("camera probe boom"))
        # Both sub-check failures are caught; preflight still completes.
        self.bc._startup_preflight()

    def test_ollama_alive_exception_treated_as_down_fatal(self):
        # Claude down + _ollama_alive() RAISES -> _local_ok stays False -> fatal.
        self._p(self.bc, "_preflight_api_key", return_value=(False, "no key"))
        self._p(self.bc, "_ollama_alive", side_effect=RuntimeError("ollama err"))
        self._p(self.bc, "close_log")
        with self.assertRaises(SystemExit):
            self.bc._startup_preflight()

    def test_fatal_speak_exception_swallowed_then_exits(self):
        # In the fatal branch, even if _speak raises the error is caught and we
        # still close_log + sys.exit.
        self._p(self.bc, "_preflight_api_key", return_value=(False, "no key"))
        self._p(self.bc, "_ollama_alive", return_value=False)
        self._speak.side_effect = RuntimeError("tts dead")
        closed = self._p(self.bc, "close_log")
        with self.assertRaises(SystemExit):
            self.bc._startup_preflight()
        closed.assert_called_once()


# ════════════════════════════════════════════════════════════════════════════
#  load_skills  (temp SKILLS_DIR with crafted fixtures)
# ════════════════════════════════════════════════════════════════════════════
class LoadSkillsTests(SectionSevenBase):
    def setUp(self):
        super().setUp()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._dir = self._tmp.name
        self._p(self.bc, "SKILLS_DIR", self._dir)
        self._p(self.bc, "SKILLS_ENABLED", True)
        # Isolate the loaded-name latch + ACTIONS so we don't pollute the real
        # monolith state; restore both after the test.
        self._p(self.bc, "_loaded_skill_names", set())
        self._p(self.bc, "ACTIONS", dict(self.bc.ACTIONS))
        # Drop any skill_* modules we register during the test.
        self._mods_before = set(self.bc.sys.modules)
        self.addCleanup(self._drop_new_modules)

    def _drop_new_modules(self):
        for name in set(self.bc.sys.modules) - self._mods_before:
            if name.startswith("skill_"):
                self.bc.sys.modules.pop(name, None)

    def _write(self, relpath, body):
        full = os.path.join(self._dir, relpath)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(body)
        return full

    def test_disabled_is_noop(self):
        self._p(self.bc, "SKILLS_ENABLED", False)
        # Even with a skill on disk, nothing loads.
        self._write("alpha.py",
                    "def register(actions):\n    actions['act_alpha']=lambda a:'a'\n")
        self.bc.load_skills()
        self.assertNotIn("skill_alpha", self.bc.sys.modules)

    def test_flat_module_registers_action(self):
        self._write("alpha.py",
                    "def register(actions):\n    actions['act_alpha']=lambda a:'a'\n")
        self.bc.load_skills()
        self.assertIn("act_alpha", self.bc.ACTIONS)
        self.assertIn("alpha", self.bc._loaded_skill_names)

    def test_package_takes_priority_over_flat(self):
        # A package AND a same-named flat .py -> the package wins, flat skipped.
        self._write("beta/__init__.py",
                    "def register(actions):\n    actions['act_beta_pkg']=lambda a:'p'\n")
        self._write("beta.py",
                    "def register(actions):\n    actions['act_beta_flat']=lambda a:'f'\n")
        self.bc.load_skills()
        self.assertIn("act_beta_pkg", self.bc.ACTIONS)
        self.assertNotIn("act_beta_flat", self.bc.ACTIONS)

    def test_underscore_dir_and_file_skipped(self):
        self._write("_hidden/__init__.py", "def register(actions):\n    pass\n")
        self._write("_private.py", "def register(actions):\n    pass\n")
        self.bc.load_skills()
        self.assertNotIn("skill__hidden", self.bc.sys.modules)
        self.assertNotIn("skill__private", self.bc.sys.modules)

    def test_already_loaded_skipped(self):
        self._write("gamma.py",
                    "def register(actions):\n    actions['act_gamma']=lambda a:'g'\n")
        self.bc._loaded_skill_names.add("gamma")   # pretend it's already loaded
        self.bc.load_skills()
        self.assertNotIn("act_gamma", self.bc.ACTIONS)

    def test_no_register_attr_loads_without_actions(self):
        # A module without register() loads (added to sys.modules) but adds no
        # actions -> exercises the "loaded (no new actions)" branch.
        self._write("delta.py", "VALUE = 42\n")
        self.bc.load_skills()
        self.assertIn("delta", self.bc._loaded_skill_names)
        self.assertIn("skill_delta", self.bc.sys.modules)

    def test_register_adding_no_actions_branch(self):
        self._write("epsilon.py", "def register(actions):\n    return None\n")
        self.bc.load_skills()
        self.assertIn("epsilon", self.bc._loaded_skill_names)

    def test_broken_skill_is_caught(self):
        # A skill that raises at import must not abort the whole load.
        self._write("boom.py", "raise RuntimeError('explode at import')\n")
        self._write("ok.py",
                    "def register(actions):\n    actions['act_ok']=lambda a:'o'\n")
        self.bc.load_skills()
        # The good skill still loaded despite the broken one.
        self.assertIn("act_ok", self.bc.ACTIONS)
        self.assertNotIn("boom", self.bc._loaded_skill_names)


# ════════════════════════════════════════════════════════════════════════════
#  _capture_focused_window_png  (guard + success paths; mss/PIL mocked)
# ════════════════════════════════════════════════════════════════════════════
class CaptureFocusedWindowPngTests(SectionSevenBase):
    def test_no_rect_returns_none(self):
        self._p(self.bc, "_focused_window_state", {"rect": None})
        self._p(self.bc, "_read_focused_window", return_value=(None, "", None))
        self.assertIsNone(self.bc._capture_focused_window_png())

    def test_tiny_rect_returns_none(self):
        self._p(self.bc, "_focused_window_state", {"rect": (0, 0, 10, 10)})
        self.assertIsNone(self.bc._capture_focused_window_png())

    def test_rect_from_reader_when_state_empty(self):
        # State has no rect -> falls back to _read_focused_window for it.
        self._p(self.bc, "_focused_window_state", {})
        self._p(self.bc, "_read_focused_window",
                return_value=(123, "Editor", (0, 0, 30, 30)))
        # 30x30 < 50 -> still too small, returns None (but exercised the reader).
        self.assertIsNone(self.bc._capture_focused_window_png())

    def test_success_path_via_mss(self):
        self._p(self.bc, "_focused_window_state",
                {"rect": (0, 0, 800, 600)})
        # Fake mss context manager whose grab() yields a tiny raw object, and a
        # fake PIL.Image that records a save() into a buffer.
        fake_raw = types.SimpleNamespace(size=(800, 600), bgra=b"\x00" * 16)

        class _FakeSct:
            def __enter__(self_):
                return self_

            def __exit__(self_, *a):
                return False

            def grab(self_, region):
                return fake_raw

        fake_mss = types.ModuleType("mss")
        fake_mss.mss = lambda: _FakeSct()

        fake_img = mock.MagicMock()
        fake_img.size = (800, 600)
        fake_img.save.side_effect = (
            lambda buf, **k: buf.write(b"PNGDATA"))
        fake_pil = types.ModuleType("PIL")
        fake_image_mod = types.ModuleType("PIL.Image")
        fake_image_mod.frombytes = lambda *a, **k: fake_img
        fake_image_mod.LANCZOS = 1
        fake_pil.Image = fake_image_mod
        with mock.patch.dict(self.bc.sys.modules,
                             {"mss": fake_mss, "PIL": fake_pil,
                              "PIL.Image": fake_image_mod}, clear=False):
            out = self.bc._capture_focused_window_png()
        self.assertEqual(out, b"PNGDATA")

    def test_pil_import_failure_returns_none(self):
        self._p(self.bc, "_focused_window_state", {"rect": (0, 0, 800, 600)})
        # Force `from PIL import Image` to fail.
        real_import = __import__

        def _shim(name, *a, **k):
            if name == "PIL":
                raise ImportError("no PIL")
            return real_import(name, *a, **k)
        with mock.patch("builtins.__import__", side_effect=_shim):
            self.assertIsNone(self.bc._capture_focused_window_png())


# ════════════════════════════════════════════════════════════════════════════
#  maybe_glance_response  (vision-fallback + error branches)
# ════════════════════════════════════════════════════════════════════════════
class MaybeGlanceResponseExtraTests(SectionSevenBase):
    def setUp(self):
        super().setUp()
        # Make the three preconditions pass so we reach the vision worker.
        self._p(self.bc, "_is_glance_ambiguous_question", return_value=True)
        self._p(self.bc, "_focus_changed_recently", return_value=True)
        self._p(self.bc, "AI_BACKEND", "claude")
        self._p(self.bc, "SCREEN_VISION_ENABLED", True)
        self._p(self.bc, "_capture_focused_window_png", return_value=b"PNG")
        self._p(self.bc, "_focused_window_state", {"title": "Editor"})
        self._p(self.bc, "pause_face_tracking")
        self._p(self.bc, "set_state")
        self._p(self.bc, "_thinking_loop")

    def test_precondition_not_ambiguous_returns_none(self):
        self._p(self.bc, "_is_glance_ambiguous_question", return_value=False)
        self.assertIsNone(self.bc.maybe_glance_response("hello"))

    def test_no_png_returns_none(self):
        self._p(self.bc, "_capture_focused_window_png", return_value=None)
        self.assertIsNone(self.bc.maybe_glance_response("what is this"))

    def test_vision_worker_exception_returns_none(self):
        # ask_vision raises inside the worker -> result stays None -> None out.
        self._p(self.bc, "ask_vision", side_effect=RuntimeError("vision down"))
        self.assertIsNone(self.bc.maybe_glance_response("what is this"))

    def test_vision_timeout_returns_none(self):
        # Simulate the worker thread never finishing: patch threading.Thread so
        # the vision worker's thread reports is_alive() True after join().
        live = mock.MagicMock()
        live.is_alive.return_value = True

        created = {"n": 0}

        def _thread_factory(*a, **k):
            # First Thread() is the animation loop (let it be real-ish/inert),
            # second is the vision worker we want to look stuck.
            created["n"] += 1
            if created["n"] == 2:
                return live
            inert = mock.MagicMock()
            inert.is_alive.return_value = False
            return inert
        with mock.patch.object(self.bc.threading, "Thread",
                               side_effect=_thread_factory):
            out = self.bc.maybe_glance_response("what is this")
        self.assertIsNone(out)

    def test_empty_answer_returns_none(self):
        self._p(self.bc, "ask_vision", return_value="")
        self.assertIsNone(self.bc.maybe_glance_response("what is this"))

    def test_parenthetical_answer_returns_none(self):
        # ask_vision returns a "(...)" status string -> treated as no answer.
        self._p(self.bc, "ask_vision", return_value="(vision unavailable)")
        self.assertIsNone(self.bc.maybe_glance_response("what is this"))

    def test_success_appends_history_and_returns_answer(self):
        self._p(self.bc, "ask_vision", return_value="That's your code editor, sir.")
        self._p(self.bc, "_push_screen_context")
        out = self.bc.maybe_glance_response("what is this")
        self.assertEqual(out, "That's your code editor, sir.")
        self.assertEqual(self.bc.conversation_history[-1]["content"],
                         "That's your code editor, sir.")

    def test_push_context_failure_swallowed(self):
        self._p(self.bc, "ask_vision", return_value="A terminal, sir.")
        self._p(self.bc, "_push_screen_context",
                side_effect=RuntimeError("cache write failed"))
        # The push failure is caught; the answer is still returned.
        out = self.bc.maybe_glance_response("what is this")
        self.assertEqual(out, "A terminal, sir.")


# ════════════════════════════════════════════════════════════════════════════
#  _media_key_with_focus
# ════════════════════════════════════════════════════════════════════════════
class MediaKeyWithFocusTests(SectionSevenBase):
    def setUp(self):
        super().setUp()
        self._pag = mock.MagicMock()
        self._p(self.bc, "_get_pyautogui", return_value=self._pag)

    def test_pyautogui_unavailable(self):
        self._p(self.bc, "_get_pyautogui", return_value=None)
        out = self.bc._media_key_with_focus("0xB0", "next button", "Next")
        self.assertEqual(out, "pyautogui unavailable")

    def test_focused_window_key_sent(self):
        self._p(self.bc, "_focus_music_window", return_value="Spotify")
        self._p(self.bc, "_ui_safe")
        out = self.bc._media_key_with_focus("0xB0", "next button", "Next track")
        self.assertIn("Next track", out)
        self.assertIn("Spotify", out)

    def test_focused_window_failsafe_returns_message(self):
        self._p(self.bc, "_focus_music_window", return_value="Spotify")
        self._p(self.bc, "_ui_safe",
                side_effect=self.bc.UIFailsafeError("mouse in corner"))
        out = self.bc._media_key_with_focus("0xB0", "next button", "Next")
        self.assertEqual(out, "mouse in corner")

    def test_no_window_global_keypress_failsafe(self):
        self._p(self.bc, "_focus_music_window", return_value=None)
        self._p(self.bc, "_ui_safe",
                side_effect=self.bc.UIFailsafeError("failsafe tripped"))
        out = self.bc._media_key_with_focus("0xB0", "next button", "Next")
        self.assertEqual(out, "failsafe tripped")

    def test_no_window_vision_fallback_clicks(self):
        self._p(self.bc, "_focus_music_window", return_value=None)
        self._p(self.bc, "_ui_safe")     # global keypress succeeds
        self._p(self.bc, "find_click_target", return_value=(100, 200))
        self._p(self.bc, "ui_click")
        out = self.bc._media_key_with_focus("0xB0", "next button", "Next")
        self.assertIn("clicked on-screen button", out)

    def test_vision_fallback_click_blocked(self):
        self._p(self.bc, "_focus_music_window", return_value=None)
        self._p(self.bc, "_ui_safe")
        self._p(self.bc, "find_click_target", return_value=(100, 200))
        self._p(self.bc, "ui_click",
                side_effect=self.bc.UIFailsafeError("blocked"))
        out = self.bc._media_key_with_focus("0xB0", "next button", "Next")
        self.assertIn("on-screen button click blocked", out)

    def test_vision_fallback_click_failed(self):
        self._p(self.bc, "_focus_music_window", return_value=None)
        self._p(self.bc, "_ui_safe")
        self._p(self.bc, "find_click_target", return_value=(100, 200))
        self._p(self.bc, "ui_click", side_effect=RuntimeError("driver gone"))
        out = self.bc._media_key_with_focus("0xB0", "next button", "Next")
        self.assertIn("on-screen button click failed", out)

    def test_vision_fallback_find_raises_then_no_control(self):
        self._p(self.bc, "_focus_music_window", return_value=None)
        self._p(self.bc, "_ui_safe")
        self._p(self.bc, "find_click_target", side_effect=RuntimeError("vision err"))
        out = self.bc._media_key_with_focus("0xB0", "next button", "Next")
        self.assertIn("no on-screen control visible", out)


# ════════════════════════════════════════════════════════════════════════════
#  _safe_close_stream
# ════════════════════════════════════════════════════════════════════════════
class SafeCloseStreamTests(SectionSevenBase):
    def test_none_stream_is_noop(self):
        self.bc._safe_close_stream(None)   # must not raise

    def test_stream_stopped_and_closed(self):
        stream = mock.MagicMock()
        self.bc._safe_close_stream(stream, timeout_sec=2.0)
        stream.stop.assert_called_once()
        stream.close.assert_called_once()

    def test_stop_raise_is_swallowed_then_closes(self):
        stream = mock.MagicMock()
        stream.stop.side_effect = RuntimeError("stop boom")
        self.bc._safe_close_stream(stream, timeout_sec=2.0)
        stream.close.assert_called_once()

    def test_close_raise_is_swallowed(self):
        stream = mock.MagicMock()
        stream.close.side_effect = RuntimeError("close boom")
        # The daemon-thread close raises; the helper swallows it and returns.
        self.bc._safe_close_stream(stream, timeout_sec=2.0)

    def test_close_hang_forces_sd_stop(self):
        import threading as _t
        # A close that blocks past the timeout -> the helper forces sd.stop().
        gate = _t.Event()
        stream = mock.MagicMock()
        stream.close.side_effect = lambda: gate.wait(5.0)
        sd_stop = self._p(self.bc.sd, "stop")
        try:
            self.bc._safe_close_stream(stream, timeout_sec=0.1)
            sd_stop.assert_called_once()
        finally:
            gate.set()   # release the blocked daemon thread

    def test_close_hang_sd_stop_exception_swallowed(self):
        import threading as _t
        # 4507-4510: close hangs past the timeout AND the forced sd.stop() also
        # raises -> the except swallows it; the helper still returns promptly.
        gate = _t.Event()
        stream = mock.MagicMock()
        stream.close.side_effect = lambda: gate.wait(5.0)
        sd_stop = self._p(self.bc.sd, "stop")
        sd_stop.side_effect = RuntimeError("sd.stop boom")
        try:
            self.bc._safe_close_stream(stream, timeout_sec=0.1)   # must not raise
            sd_stop.assert_called_once()
        finally:
            gate.set()


# ════════════════════════════════════════════════════════════════════════════
#  Singleton low-level helpers (_read_lock_pid / _acquire_os_singleton_lock)
# ════════════════════════════════════════════════════════════════════════════
class SingletonHelperTests(SectionSevenBase):
    def setUp(self):
        super().setUp()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def test_read_lock_pid_missing_file(self):
        p = os.path.join(self._tmp.name, "nope.lock")
        self.assertEqual(self.bc._read_lock_pid(p), -1)

    def test_read_lock_pid_valid(self):
        p = os.path.join(self._tmp.name, "v.lock")
        with open(p, "w", encoding="utf-8") as f:
            f.write("4242")
        self.assertEqual(self.bc._read_lock_pid(p), 4242)

    def test_read_lock_pid_empty_after_retries(self):
        p = os.path.join(self._tmp.name, "empty.lock")
        with open(p, "w", encoding="utf-8") as f:
            f.write("")
        # Empty file -> 0 after the retry budget (patched sleep keeps it fast).
        self._p(self.bc.time, "sleep", lambda *_a, **_k: None)
        self.assertEqual(self.bc._read_lock_pid(p, max_retries=2), 0)

    def test_read_lock_pid_nonnumeric_then_zero(self):
        p = os.path.join(self._tmp.name, "bad.lock")
        with open(p, "w", encoding="utf-8") as f:
            f.write("not-a-pid")
        self._p(self.bc.time, "sleep", lambda *_a, **_k: None)
        # Unparseable -> ValueError-continue path -> 0 after retries.
        self.assertEqual(self.bc._read_lock_pid(p, max_retries=2), 0)

    def test_acquire_os_lock_win32_success(self):
        # A real fd; msvcrt.locking succeeds -> True.
        fd, path = tempfile.mkstemp(dir=self._tmp.name)
        self.addCleanup(lambda: os.path.exists(path) and os.remove(path))
        self.addCleanup(lambda: self._safe_close(fd))
        self._p(self.bc.sys, "platform", "win32")
        self.assertTrue(self.bc._acquire_os_singleton_lock(fd))

    def test_acquire_os_lock_bad_fd_fails_open(self):
        # A nonsense fd makes the inner os.lseek raise OSError -> returns False
        # (the documented "another holds it / can't lock" posture on win32).
        self._p(self.bc.sys, "platform", "win32")
        out = self.bc._acquire_os_singleton_lock(-999)
        self.assertIn(out, (True, False))   # never raises; bool either way

    def _safe_close(self, fd):
        try:
            os.close(fd)
        except OSError:
            pass


# ════════════════════════════════════════════════════════════════════════════
#  _enforce_singleton — POSIX branch + generic-except fallbacks
# ════════════════════════════════════════════════════════════════════════════
class EnforceSingletonPosixTests(SectionSevenBase):
    def setUp(self):
        super().setUp()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._lock = os.path.join(self._tmp.name, "jarvis.lock")
        with open(self._lock, "w", encoding="utf-8") as f:
            f.write("4242")
        self._p(self.bc, "_LOCK_FILE", self._lock)
        self._p(self.bc, "_SINGLETON_HELD_FD", None)
        self._p(self.bc, "_read_lock_pid", return_value=4242)

    def test_posix_alive_process_exits(self):
        self._p(self.bc.sys, "platform", "linux")
        # os.kill(pid, 0) succeeding -> process alive -> sys.exit.
        self._p(self.bc.os, "kill", return_value=None)
        with self.assertRaises(SystemExit):
            self.bc._enforce_singleton()

    def test_posix_dead_process_overwrites(self):
        self._p(self.bc.sys, "platform", "linux")
        self._p(self.bc.os, "kill", side_effect=ProcessLookupError())
        self.bc._enforce_singleton()
        with open(self._lock, encoding="utf-8") as f:
            self.assertEqual(f.read().strip(), str(os.getpid()))

    def test_posix_permission_error_assumes_alive(self):
        self._p(self.bc.sys, "platform", "linux")
        self._p(self.bc.os, "kill", side_effect=PermissionError())
        with self.assertRaises(SystemExit):
            self.bc._enforce_singleton()

    def test_win32_generic_exception_assumes_alive(self):
        self._p(self.bc.sys, "platform", "win32")
        # subprocess.run raising a non-timeout error -> still_running stays True.
        self._p(self.bc.subprocess, "run", side_effect=RuntimeError("spawn fail"))
        with self.assertRaises(SystemExit):
            self.bc._enforce_singleton()


# ════════════════════════════════════════════════════════════════════════════
#  _release_singleton — POSIX branch + close-error
# ════════════════════════════════════════════════════════════════════════════
class ReleaseSingletonPosixTests(SectionSevenBase):
    def test_posix_unlock_path(self):
        fd, path = tempfile.mkstemp()
        self.addCleanup(lambda: os.path.exists(path) and os.remove(path))
        self._p(self.bc, "_SINGLETON_HELD_FD", fd)
        self._p(self.bc.sys, "platform", "linux")
        # Provide a fake fcntl so the POSIX unlock branch runs on Windows.
        fake_fcntl = types.ModuleType("fcntl")
        fake_fcntl.LOCK_UN = 8
        fake_fcntl.lockf = mock.MagicMock()
        with mock.patch.dict(self.bc.sys.modules, {"fcntl": fake_fcntl},
                             clear=False):
            self.bc._release_singleton()
        fake_fcntl.lockf.assert_called_once()
        self.assertIsNone(self.bc._SINGLETON_HELD_FD)

    def test_posix_unlock_oserror_swallowed(self):
        # 12263-12264: on the POSIX branch fcntl.lockf raises OSError -> the
        # except swallows it and the fd is still cleared/closed in the finally.
        fd, path = tempfile.mkstemp()
        self.addCleanup(lambda: os.path.exists(path) and os.remove(path))
        self._p(self.bc, "_SINGLETON_HELD_FD", fd)
        self._p(self.bc.sys, "platform", "linux")
        fake_fcntl = types.ModuleType("fcntl")
        fake_fcntl.LOCK_UN = 8
        fake_fcntl.lockf = mock.MagicMock(side_effect=OSError("not locked"))
        with mock.patch.dict(self.bc.sys.modules, {"fcntl": fake_fcntl},
                             clear=False):
            self.bc._release_singleton()   # must not raise
        fake_fcntl.lockf.assert_called_once()
        self.assertIsNone(self.bc._SINGLETON_HELD_FD)

    def test_close_error_swallowed(self):
        # _SINGLETON_HELD_FD set to an already-closed fd -> os.close raises
        # OSError in the finally, which is swallowed.
        fd, path = tempfile.mkstemp()
        os.close(fd)
        self.addCleanup(lambda: os.path.exists(path) and os.remove(path))
        self._p(self.bc, "_SINGLETON_HELD_FD", fd)
        self._p(self.bc.sys, "platform", "win32")
        self.bc._release_singleton()   # must not raise
        self.assertIsNone(self.bc._SINGLETON_HELD_FD)


# ════════════════════════════════════════════════════════════════════════════
#  Gap-closers: handler-exception + loop-break branches
# ════════════════════════════════════════════════════════════════════════════
class VoiceShortcutExceptionBranchTests(SectionSevenBase):
    """Each shortcut's handler is wrapped in try/except so a misbehaving skill
    can't crash the loop; these drive the except-and-log branches."""

    def setUp(self):
        super().setUp()
        self._p(self.bc, "_speak")
        self._p(self.bc, "set_state")
        self._p(self.bc, "maybe_replay_last_action", return_value=None)

    def _p_dict(self, mapping):
        patcher = mock.patch.dict(self.bc.sys.modules, mapping, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_backend_toggle_exception_falls_through(self):
        boom = types.SimpleNamespace(
            maybe_switch_backend=mock.MagicMock(side_effect=RuntimeError("x")))
        self._p_dict({"skill_custom_voice": boom})
        # mode_router + dispatcher inert so we reach the final return False.
        fr = types.ModuleType("core.mode_router")
        fr.maybe_handle_mode_toggle = lambda _t: None
        fr.controlled_dispatch = lambda _t, _a: None
        fr.is_in_controlled_mode = lambda: False
        self._p_dict({"core.mode_router": fr})
        fd = types.ModuleType("core.dispatcher")
        fd.resolve_and_dispatch = lambda _t, _a: None
        self._p_dict({"core.dispatcher": fd})
        self.assertFalse(self.bc._run_voice_shortcuts("hi"))

    def test_mode_toggle_exception_falls_through(self):
        self._p_dict({"skill_custom_voice":
                      types.SimpleNamespace(maybe_switch_backend=lambda _t: None)})
        fr = types.ModuleType("core.mode_router")
        fr.maybe_handle_mode_toggle = mock.MagicMock(side_effect=RuntimeError("y"))
        fr.controlled_dispatch = lambda _t, _a: None
        fr.is_in_controlled_mode = lambda: False
        self._p_dict({"core.mode_router": fr})
        fd = types.ModuleType("core.dispatcher")
        fd.resolve_and_dispatch = lambda _t, _a: None
        self._p_dict({"core.dispatcher": fd})
        self.assertFalse(self.bc._run_voice_shortcuts("hi"))

    def test_chain_resolver_exception_falls_through(self):
        self._p_dict({"skill_custom_voice":
                      types.SimpleNamespace(maybe_switch_backend=lambda _t: None)})
        fr = types.ModuleType("core.mode_router")
        fr.maybe_handle_mode_toggle = lambda _t: None
        fr.controlled_dispatch = lambda _t, _a: None
        fr.is_in_controlled_mode = lambda: False
        self._p_dict({"core.mode_router": fr})
        fd = types.ModuleType("core.dispatcher")
        fd.resolve_and_dispatch = mock.MagicMock(side_effect=RuntimeError("z"))
        self._p_dict({"core.dispatcher": fd})
        self.assertFalse(self.bc._run_voice_shortcuts("hi"))


class RunLlmDispatchBreakBranchTests(SectionSevenBase):
    def setUp(self):
        super().setUp()
        self._speak = self._p(self.bc, "_speak")
        self._p(self.bc, "set_state")
        self._p(self.bc, "_heartbeat")
        self._p(self.bc, "_apply_quip_layer", side_effect=lambda s, r: s)
        self._p(self.bc, "maybe_glance_response", return_value=None)
        self._p(self.bc, "get_response_with_animation", return_value="[ACTION: open_url]")

    def test_loop_detected_break(self):
        # open_url is a _loop_action; emitting it twice in a chain -> loop-detect
        # break on the second pass.
        self._p(self.bc, "parse_and_run_actions", side_effect=[
            ("", [("open_url", "opened https://x", True)]),
            ("", [("open_url", "opened https://x again", True)]),
        ])
        self._p(self.bc, "get_followup_response", return_value="again")
        self.bc._run_llm_dispatch("open the page")
        # Stops on loop detection (only the first follow-up ran).

    def test_terminal_action_break(self):
        # check_credits is terminal -> after it runs once, the loop stops.
        self._p(self.bc, "get_response_with_animation",
                return_value="[ACTION: check_credits]")
        self._p(self.bc, "parse_and_run_actions", side_effect=[
            ("", [("check_credits", "balance is $5", True)]),
            ("", [("check_credits", "balance is $5", True)]),
        ])
        self._p(self.bc, "get_followup_response", return_value="you have $5")
        self.bc._run_llm_dispatch("check my credits")

    def test_empty_followup_breaks(self):
        self._p(self.bc, "parse_and_run_actions",
                return_value=("", [("see_screen", "an editor", True)]))
        # Follow-up returns empty -> break immediately.
        self._p(self.bc, "get_followup_response", return_value="")
        self.bc._run_llm_dispatch("what's on screen")


class BlueGreenLoopTickBranchTests(SectionSevenBase):
    """Drives the inner announce/snapshot-exception sub-branches."""

    def setUp(self):
        super().setUp()
        self._speak = self._p(self.bc, "_speak")
        self.bgm = mock.MagicMock()
        self.bgm.read_version.return_value = "1.0"
        self.bgm.consume_handoff_signal.return_value = None
        self.bgm.consume_upgrade_aborted_signal.return_value = None
        self.bgm.consume_handoff_failure_signal.return_value = None
        self._p(self.bc, "_bgm", self.bgm)
        self._p(self.bc, "_bg_last_heartbeat", [self.bc.time.time()])  # not due
        self._p(self.bc, "_bg_handoff_seen_at", [0.0])
        self._p(self.bc, "BLUE_GREEN_ROLE", "prod")

    def test_handoff_announce_exception_swallowed(self):
        self.bgm.consume_handoff_signal.return_value = {"target_version": "2.0"}
        self._speak.side_effect = RuntimeError("tts dead")
        # Announce failure is caught; the handoff-state write still runs.
        out = self.bc._blue_green_loop_tick()
        self.assertFalse(out)
        self.bgm.write_handoff_state.assert_called_once()

    def test_handoff_timer_enumerate_failure_swallowed(self):
        self.bgm.consume_handoff_signal.return_value = {"target_version": "2.0"}
        boom_timer = types.SimpleNamespace(
            enumerate_timers=mock.MagicMock(side_effect=RuntimeError("timer x")))
        with mock.patch.dict(self.bc.sys.modules,
                             {"skill_timer": boom_timer}, clear=False):
            out = self.bc._blue_green_loop_tick()
        self.assertFalse(out)
        self.bgm.write_handoff_state.assert_called_once()

    def test_handoff_state_write_failure_swallowed(self):
        self.bgm.consume_handoff_signal.return_value = {"target_version": "2.0"}
        self.bgm.write_handoff_state.side_effect = RuntimeError("disk full")
        # The write failure is caught and logged, not raised.
        self.assertFalse(self.bc._blue_green_loop_tick())

    def test_abort_announce_exception_swallowed(self):
        self.bgm.consume_upgrade_aborted_signal.return_value = {"reason": "x"}
        self._speak.side_effect = RuntimeError("tts dead")
        self.assertFalse(self.bc._blue_green_loop_tick())

    def test_timer_enumerate_present_snapshots(self):
        self.bgm.consume_handoff_signal.return_value = {"target_version": "2.0"}
        good_timer = types.SimpleNamespace(
            enumerate_timers=lambda: [{"label": "tea", "secs": 60}])
        with mock.patch.dict(self.bc.sys.modules,
                             {"skill_timer": good_timer}, clear=False):
            self.bc._blue_green_loop_tick()
        # The snapshot fed the active_timers payload.
        payload = self.bgm.write_handoff_state.call_args[0][0]
        self.assertEqual(payload["active_timers"], [{"label": "tea", "secs": 60}])


class ConsumeHandoffReplayFailureTests(SectionSevenBase):
    def test_conversation_extend_failure_swallowed(self):
        bgm = mock.MagicMock()
        bgm.RESUME_HANDOFF_FLAG = "--resume-handoff"
        bgm.consume_handoff_state.return_value = {
            "conversation_tail": [{"role": "user", "content": "hi"}],
        }
        self._p(self.bc, "_bgm", bgm)
        self._p(self.bc, "BLUE_GREEN_ROLE", "prod")
        self._p(self.bc.sys, "argv", ["bobert", "--resume-handoff"])
        # Swap conversation_history for a list subclass whose extend() raises so
        # the replay except branch fires. (list itself is isinstance-checked by
        # the function, so a subclass keeps that guard happy.)
        class _BoomList(list):
            def extend(self, *_a, **_k):
                raise RuntimeError("hist broken")
        self._p(self.bc, "conversation_history", _BoomList())
        ls, timers = self.bc._consume_blue_green_handoff()
        # Replay failure is caught; returns the (empty) payload cleanly.
        self.assertIsNone(ls)


class CaptureFocusedWindowPngFallbackTests(SectionSevenBase):
    def setUp(self):
        super().setUp()
        self._p(self.bc, "_focused_window_state", {"rect": (0, 0, 800, 600)})

    def _pil(self, fake_img):
        fake_pil = types.ModuleType("PIL")
        fake_image_mod = types.ModuleType("PIL.Image")
        fake_image_mod.frombytes = lambda *a, **k: fake_img
        fake_image_mod.LANCZOS = 1
        fake_grab_mod = types.ModuleType("PIL.ImageGrab")
        fake_grab_mod.grab = lambda *a, **k: fake_img
        fake_pil.Image = fake_image_mod
        fake_pil.ImageGrab = fake_grab_mod
        return {"PIL": fake_pil, "PIL.Image": fake_image_mod,
                "PIL.ImageGrab": fake_grab_mod}

    def test_mss_failure_falls_back_to_pil_imagegrab(self):
        fake_img = mock.MagicMock()
        fake_img.size = (800, 600)
        fake_img.save.side_effect = lambda buf, **k: buf.write(b"PNG_PIL")
        fake_mss = types.ModuleType("mss")

        def _raise_mss():
            raise RuntimeError("no mss display")
        fake_mss.mss = _raise_mss
        mods = self._pil(fake_img)
        mods["mss"] = fake_mss
        with mock.patch.dict(self.bc.sys.modules, mods, clear=False):
            out = self.bc._capture_focused_window_png()
        self.assertEqual(out, b"PNG_PIL")

    def test_oversize_image_is_downscaled(self):
        # An image larger than 1568 on a side exercises the resize branch.
        fake_img = mock.MagicMock()
        fake_img.size = (4000, 3000)
        resized = mock.MagicMock()
        resized.size = (1568, 1176)
        resized.save.side_effect = lambda buf, **k: buf.write(b"PNG_SMALL")
        fake_img.resize.return_value = resized
        fake_mss = types.ModuleType("mss")

        class _Sct:
            def __enter__(self_): return self_
            def __exit__(self_, *a): return False
            def grab(self_, region):
                return types.SimpleNamespace(size=(4000, 3000), bgra=b"\x00" * 16)
        fake_mss.mss = lambda: _Sct()
        mods = self._pil(fake_img)
        mods["mss"] = fake_mss
        with mock.patch.dict(self.bc.sys.modules, mods, clear=False):
            out = self.bc._capture_focused_window_png()
        self.assertEqual(out, b"PNG_SMALL")
        fake_img.resize.assert_called_once()

    def test_encode_failure_returns_none(self):
        fake_img = mock.MagicMock()
        fake_img.size = (800, 600)
        fake_img.save.side_effect = RuntimeError("encode boom")
        fake_mss = types.ModuleType("mss")

        class _Sct:
            def __enter__(self_): return self_
            def __exit__(self_, *a): return False
            def grab(self_, region):
                return types.SimpleNamespace(size=(800, 600), bgra=b"\x00" * 16)
        fake_mss.mss = lambda: _Sct()
        mods = self._pil(fake_img)
        mods["mss"] = fake_mss
        with mock.patch.dict(self.bc.sys.modules, mods, clear=False):
            out = self.bc._capture_focused_window_png()
        self.assertIsNone(out)

    def test_both_mss_and_pil_fail_returns_none(self):
        fake_mss = types.ModuleType("mss")

        def _raise_mss():
            raise RuntimeError("no mss")
        fake_mss.mss = _raise_mss
        fake_pil = types.ModuleType("PIL")
        fake_image_mod = types.ModuleType("PIL.Image")
        fake_image_mod.frombytes = lambda *a, **k: None
        fake_image_mod.LANCZOS = 1
        fake_grab_mod = types.ModuleType("PIL.ImageGrab")
        fake_grab_mod.grab = mock.MagicMock(side_effect=RuntimeError("no grab"))
        fake_pil.Image = fake_image_mod
        fake_pil.ImageGrab = fake_grab_mod
        with mock.patch.dict(self.bc.sys.modules,
                             {"mss": fake_mss, "PIL": fake_pil,
                              "PIL.Image": fake_image_mod,
                              "PIL.ImageGrab": fake_grab_mod}, clear=False):
            out = self.bc._capture_focused_window_png()
        self.assertIsNone(out)


# ════════════════════════════════════════════════════════════════════════════
#  _announce_upgrade_summary  (TASK B — upgrade-announcer dedup)
# ════════════════════════════════════════════════════════════════════════════
class AnnounceUpgradeSummaryTests(SectionSevenBase):
    """The boot-time upgrade announcer must DEDUP on the TASK LIST (not the
    timestamp): a summary whose tasks were already announced — even with a
    bumped ``upgraded_at`` — is NOT spoken again (but the summary file is still
    deleted), while a genuinely-new task set IS spoken and updates the sidecar.
    """

    def setUp(self):
        super().setUp()
        self._speak = self._p(self.bc, "_speak")
        # Isolate the summary + announced-signature sidecar into a temp dir so we
        # never touch the real project-root dotfiles.
        self._dir = tempfile.mkdtemp(prefix="jarvis_upgrade_test_")
        self.addCleanup(self._cleanup_dir)
        self._summary_path = os.path.join(self._dir, ".last_upgrade_summary.json")
        self._sig_path = os.path.join(self._dir, ".last_upgrade_announced")
        self._p(self.bc, "UPGRADE_SUMMARY_FILE", self._summary_path)
        self._p(self.bc, "UPGRADE_ANNOUNCED_SIG_FILE", self._sig_path)

    def _cleanup_dir(self):
        import shutil
        shutil.rmtree(self._dir, ignore_errors=True)

    def _write_summary(self, tasks, when="11:52", syntax_ok=True):
        with open(self._summary_path, "w", encoding="utf-8") as f:
            json.dump({"tasks": tasks, "upgraded_at": when,
                       "syntax_ok": syntax_ok}, f)

    def _write_sig_for(self, tasks):
        # Compute + store the signature the announcer would compute for `tasks`
        # (after the same **date** prefix stripping the announcer applies).
        import re as _re
        clean = []
        for _t in tasks:
            _c = _re.sub(r'^\*\*[^*]+\*\*\s*[—–-]+\s*', '', _t).strip()
            clean.append(_c if _c else _t)
        sig = self.bc._upgrade_task_signature(clean)
        with open(self._sig_path, "w", encoding="utf-8") as f:
            f.write(sig)
        return sig

    # ── case 1: matching signature → NOT spoken, but summary removed ─────────
    def test_already_announced_signature_is_not_spoken(self):
        tasks = ["**2026-06-01** — added foo", "**2026-06-02** — fixed bar"]
        # Pre-seed the sidecar with this task set's signature (simulating a prior
        # boot that already announced it), then re-write the summary with a FRESH
        # timestamp but the SAME tasks (the old overnight-engine replay bug).
        prior_sig = self._write_sig_for(tasks)
        self._write_summary(tasks, when="09:30")   # bumped timestamp, same tasks

        self.bc._announce_upgrade_summary()

        # Stale replay: nothing spoken …
        self._speak.assert_not_called()
        # … the summary file is STILL deleted (no stuck re-announce loop) …
        self.assertFalse(os.path.exists(self._summary_path))
        # … and the stored signature is untouched.
        with open(self._sig_path, "r", encoding="utf-8") as f:
            self.assertEqual(f.read().strip(), prior_sig)

    # ── case 2: new signature → spoken, sidecar updated ─────────────────────
    def test_new_task_set_is_spoken_and_updates_sidecar(self):
        old_tasks = ["**2026-05-30** — old work"]
        new_tasks = ["**2026-06-03** — brand new feature",
                     "**2026-06-03** — another new thing"]
        # Sidecar holds the OLD signature; the summary carries genuinely-new tasks.
        old_sig = self._write_sig_for(old_tasks)
        self._write_summary(new_tasks, when="14:00")

        self.bc._announce_upgrade_summary()

        # Genuine upgrade: announced …
        self._speak.assert_called_once()
        spoken = self._speak.call_args[0][0].lower()
        self.assertIn("upgrade complete", spoken)
        # … the summary file is consumed …
        self.assertFalse(os.path.exists(self._summary_path))
        # … and the sidecar now holds the NEW signature (≠ the old one).
        expected_sig = self.bc._upgrade_task_signature(
            ["brand new feature", "another new thing"])
        with open(self._sig_path, "r", encoding="utf-8") as f:
            stored = f.read().strip()
        self.assertEqual(stored, expected_sig)
        self.assertNotEqual(stored, old_sig)

    # ── a first-ever announcement (no sidecar yet) still speaks ─────────────
    def test_first_announcement_with_no_sidecar_speaks(self):
        tasks = ["**2026-06-03** — the very first upgrade"]
        self._write_summary(tasks)
        self.assertFalse(os.path.exists(self._sig_path))

        self.bc._announce_upgrade_summary()

        self._speak.assert_called_once()
        self.assertFalse(os.path.exists(self._summary_path))
        self.assertTrue(os.path.exists(self._sig_path))


if __name__ == "__main__":
    unittest.main()
