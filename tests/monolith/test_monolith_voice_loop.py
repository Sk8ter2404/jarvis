"""Monolith side of the 2026-10-01 voice-loop fixes.

Each class drives the REAL bobert_companion code for one finding:

  * B001  the high-risk action confirmation: "Yeah." / "Sure." / "Jarvis,
          yes." confirm; "Yesterday..." / "Do it later" / "Go ahead and cancel
          it" never do; an unrelated reply cancels and routes on.
  * B008  the shutdown prompt hears Whisper's punctuated "Yes." / "No." and
          the wake-led "Jarvis, no.".
  * B002  Mute Mic is a capture-ENTRY rule: standby, get_mic_buffer and a
          capture muted mid-way keep nothing.
  * B009  a timer that fires in standby is spoken there; other queued lines
          wait for wake.
  * B010  "Jarvis, <command>" to a sleeping JARVIS runs the command; a bare
          "Jarvis" greets and admits exactly one un-prefixed reply.
  * B011  an action's capture on another thread takes the mic over from the
          main loop's listen - never two streams at once.
  * B061  dropped / ignored lines are logged by length, never by their words.
  * B062  an idle listen yields at once to a newly queued command.

No real audio device is touched: the mic is a fake InputStream (the
tests/monolith/test_monolith_tray_fixes.py pattern) and every queue file
lives in a temp dir.

    python -m unittest tests.monolith.test_monolith_voice_loop
"""
from __future__ import annotations

import contextlib
import inspect
import io
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith

_WAIT = 5.0


@requires_monolith
class _Base(MonolithGlobalsTestCase):
    def setUp(self):
        super().setUp()
        self.tmp = tempfile.mkdtemp(prefix="voice_loop_mono_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.spoken = []
        self._p(self.bc, "_speak",
                side_effect=lambda t, *a, **k: self.spoken.append(t))
        # Plain list cells the harness does not track: put them back.
        for name in ("_mic_muted", "_sleep_mode", "_standby_mode"):
            cell = getattr(self.bc, name)
            saved = cell[0]
            self.addCleanup(cell.__setitem__, 0, saved)
        self.bc._mic_muted[0] = False

    def _p(self, *args, **kwargs):
        patcher = mock.patch.object(*args, **kwargs)
        m = patcher.start()
        self.addCleanup(patcher.stop)
        return m

    def _quiet(self, fn, *a, **k):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            out = fn(*a, **k)
        return out, buf.getvalue()

    def _wait_for(self, pred, what):
        deadline = time.time() + _WAIT
        while time.time() < deadline:
            if pred():
                return
            time.sleep(0.01)
        self.fail(f"timed out waiting for {what}")


class _FakeMic:
    """A fake sd.InputStream factory that counts the streams OPEN at once.
    ``frames`` (optional) is a callable(push, stream) run on a thread when a
    stream starts; push(n, amp) delivers n frames of constant amplitude."""

    def __init__(self, np, frames=None):
        self.np = np
        self.frames = frames
        self.lock = threading.Lock()
        self.open_now = 0
        self.max_open = 0
        self.opened = 0
        mic = self

        class Stream:
            device = 1

            def __init__(self, *a, callback=None, **k):
                self.cb = callback
                self.closed = False

            def start(self):
                with mic.lock:
                    mic.open_now += 1
                    mic.opened += 1
                    mic.max_open = max(mic.max_open, mic.open_now)
                if mic.frames is not None:
                    cb, me = self.cb, self

                    def push(n, amp):
                        frame = mic.np.full((1024, 1), amp, dtype="float32")
                        for _ in range(n):
                            if me.closed:
                                return
                            cb(frame, 1024, None, None)
                    threading.Thread(target=mic.frames, args=(push, self),
                                     daemon=True, name="fake-mic").start()

            def close_(self):
                if not self.closed:
                    self.closed = True
                    with mic.lock:
                        mic.open_now -= 1
        self.Stream = Stream

    def safe_close(self, stream):
        if stream is not None:
            stream.close_()


class _CaptureBase(_Base):
    """record_speech / get_mic_buffer against _FakeMic."""

    def _mic(self, frames=None):
        bc = self.bc
        mic = _FakeMic(bc.np, frames)
        self._p(bc, "_mic_input_disabled", return_value=False)
        self._p(bc, "get_input_device", return_value=1)
        self._p(bc.sd, "InputStream", mic.Stream)
        self._p(bc, "_safe_close_stream", mic.safe_close)
        self._p(bc, "_note_live_capture", lambda *a, **k: None)
        self._p(bc, "_filler_capture_mark", lambda *a, **k: None)
        self._p(bc, "_input_backoff_wait", return_value=False)
        self._p(bc, "_process_capture_chunk",
                lambda data, sr, skip_ns=False: data)
        self._p(bc, "_spec_stt_should_snapshot", return_value=False)
        self._p(bc, "_write_hud_state")
        self._p(bc, "set_state")
        self._p(bc, "_heartbeat")
        self._p(bc, "pause_face_tracking")
        self._p(bc, "VAD_THRESHOLD", 0.008)
        self.inject_path = os.path.join(self.tmp, "injected_commands.json")
        self._p(bc, "INJECTED_COMMANDS_PATH", self.inject_path)
        self._p(bc, "PENDING_SPEECH_PATH",
                os.path.join(self.tmp, "pending_speech.json"))
        bc._utterance_in_progress[0] = False
        return mic


# ── B001: the action confirmation gate ─────────────────────────────────────
class ConfirmationGateTests(_Base):
    def setUp(self):
        super().setUp()
        bc = self.bc
        self.ran = []
        acts = dict(bc.ACTIONS)
        acts["vl_delete"] = lambda a: self.ran.append(a) or "ok"
        self._p(bc, "ACTIONS", acts)
        bc._pending_confirmation.clear()

    def _answer(self, text):
        bc = self.bc
        bc._pending_confirmation.clear()
        bc._pending_confirmation.append(("vl_delete", "X"))
        self.ran.clear()
        self.spoken.clear()
        out, _ = self._quiet(bc.handle_confirmation_response, text)
        return out

    def test_natural_yes_replies_confirm(self):
        for text in ("Yeah.", "Sure", "okay", "Yep, do it", "Jarvis, yes.",
                     "Yes."):
            with self.subTest(text=text):
                self.assertTrue(self._answer(text))
                self.assertEqual(self.ran, ["X"], "a plain yes was cancelled")
                self.assertNotIn("Cancelled.", self.spoken)
                self.assertEqual(self.bc._pending_confirmation, [])

    def test_lookalikes_never_run_the_action(self):
        for text in ("Yesterday we went to the store", "do it later",
                     "Go ahead and cancel it", "Confirmation number 5",
                     "Yes, but wait"):
            with self.subTest(text=text):
                self._answer(text)
                self.assertEqual(self.ran, [], "a look-alike confirmed it")
                self.assertEqual(self.bc._pending_confirmation, [])

    def test_a_no_or_hedged_reply_is_consumed_as_a_decline(self):
        for text in ("No.", "Jarvis, no", "do it later",
                     "Go ahead and cancel it"):
            with self.subTest(text=text):
                self.assertTrue(self._answer(text))
                self.assertIn("Cancelled.", self.spoken)

    def test_an_unrelated_reply_cancels_and_routes_on(self):
        self.assertFalse(self._answer("what's the weather tomorrow"))
        self.assertEqual(self.ran, [])
        self.assertEqual(self.bc._pending_confirmation, [])
        self.assertIn("Cancelled, sir.", self.spoken)

    def test_the_autocorrect_pick_hears_a_wake_led_yes(self):
        bc = self.bc
        picked = []
        acts = dict(bc.ACTIONS)
        acts["vl_first"] = lambda a: picked.append("first") or "ok"
        acts["vl_second"] = lambda a: picked.append("second") or "ok"
        self._p(bc, "ACTIONS", acts)
        self._p(bc, "_needs_confirmation", return_value=False)
        self._p(bc, "_jarvis_pushback", return_value=None)
        self._p(bc, "record_session_action")
        self._p(bc, "record_action_history")
        bc._pending_autocorrect_choice.clear()
        bc._pending_autocorrect_choice.append(
            {"primary": ("vl_first", ""), "secondary": ("vl_second", "")})
        out, _ = self._quiet(bc.handle_autocorrect_disambig_response,
                             "Jarvis, yes.")
        self.assertTrue(out)
        self.assertEqual(picked, ["first"])


# ── B008: the shutdown prompt ──────────────────────────────────────────────
class ShutdownPromptReplyTests(_Base):
    def setUp(self):
        super().setUp()
        bc = self.bc
        self._p(bc, "_shutdown_prompt_pending",
                {"armed": False, "expires_at": 0.0})
        self.overnight = self._p(bc, "_act_start_overnight_upgrade")
        self.shutdown = self._p(bc, "_act_shutdown_jarvis")

    def _reply(self, text):
        bc = self.bc
        bc._shutdown_prompt_pending["armed"] = True
        bc._shutdown_prompt_pending["expires_at"] = time.time() + 30
        self.overnight.reset_mock()
        self.shutdown.reset_mock()
        self.spoken.clear()
        out, _ = self._quiet(bc._handle_shutdown_prompt, text)
        return out

    def test_punctuated_yes_starts_overnight(self):
        for text in ("Yes.", "Yes, please.", "Sure.", "Okay.", "Jarvis, yes."):
            with self.subTest(text=text):
                self.assertTrue(self._reply(text))
                self.overnight.assert_called_once()
                self.shutdown.assert_not_called()
                self.assertNotIn("Shutdown cancelled.", self.spoken)

    def test_punctuated_no_shuts_down(self):
        for text in ("No.", "Nope.", "Jarvis, no.", "No, thanks."):
            with self.subTest(text=text):
                self.assertTrue(self._reply(text))
                self.shutdown.assert_called_once()
                self.overnight.assert_not_called()
                self.assertNotIn("Shutdown cancelled.", self.spoken)

    def test_a_hedged_yes_cancels_instead(self):
        self.assertFalse(self._reply("Yes, but later."))
        self.overnight.assert_not_called()
        self.shutdown.assert_not_called()
        self.assertIn("Shutdown cancelled.", self.spoken)


# ── B002: Mute Mic is a capture-entry rule ─────────────────────────────────
class MuteEntryRuleTests(_CaptureBase):
    def _standby(self):
        bc = self.bc
        bc._sleep_mode[0] = True
        bc._standby_mode[0] = True
        self._p(bc, "_speak_pending", return_value=False)
        self._p(bc, "set_state")
        self._p(bc, "_heartbeat")

    def test_standby_takes_no_capture_while_muted(self):
        bc = self.bc
        self._standby()
        bc._mic_muted[0] = True
        rec = self._p(bc, "record_speech")
        stt = self._p(bc, "_transcribe_capture")
        out, log = self._quiet(bc._handle_sleep_standby, None)
        self.assertIsNone(out)
        rec.assert_not_called()
        stt.assert_not_called()
        self.assertNotIn("ignored", log)
        self.assertTrue(bc._sleep_mode[0])

    def test_a_typed_wake_still_wakes_while_muted(self):
        bc = self.bc
        self._standby()
        bc._mic_muted[0] = True
        self._p(bc, "context_aware_greeting", return_value=("Yes, sir?", 1.0))
        self._p(bc, "OVERNIGHT_FLAG_FILE", os.path.join(self.tmp, "none"))
        self._p(bc, "_learn_gate_note_wake")
        self._quiet(bc._handle_sleep_standby, "Jarvis")
        self.assertFalse(bc._sleep_mode[0])
        self.assertEqual(self.spoken, ["Yes, sir?"])

    def test_get_mic_buffer_opens_nothing_while_muted(self):
        bc = self.bc
        mic = self._mic()
        bc._mic_muted[0] = True
        with mock.patch.dict(sys.modules, {"skill_wake_listener": None}):
            out, _ = self._quiet(bc.get_mic_buffer, 0.2)
        self.assertIsNone(out)
        self.assertEqual(mic.opened, 0, "a muted mic opened a stream")

    def test_a_mute_mid_buffer_keeps_nothing(self):
        bc = self.bc

        def frames(push, stream):
            push(3, 0.1)
            bc._mic_muted[0] = True      # the tray's Mute Mic, mid-capture
            for _ in range(40):
                push(1, 0.1)
                time.sleep(0.01)
        mic = self._mic(frames)
        with mock.patch.dict(sys.modules, {"skill_wake_listener": None}):
            out, _ = self._quiet(bc.get_mic_buffer, 1.0)
        self.assertIsNone(out, "audio heard before/after the mute was kept")
        self.assertEqual(mic.open_now, 0)


# ── B009: reminders in standby ─────────────────────────────────────────────
class StandbyReminderTests(_Base):
    def setUp(self):
        super().setUp()
        bc = self.bc
        self._p(bc, "__file__", os.path.join(self.tmp, "bobert_companion.py"))
        self.queue = os.path.join(self.tmp, "pending_speech.json")
        self._p(bc, "PENDING_SPEECH_PATH", self.queue)
        self._p(bc, "_audio_flap_flush")
        self._p(bc, "_heartbeat")
        self._p(bc, "set_state")
        self.tag = str(time.time())       # unique lines: no cross-test dedupe

    def _queued(self):
        try:
            with open(self.queue, encoding="utf-8") as f:
                return [e["message"] for e in json.load(f)]
        except OSError:
            return []

    def test_proactive_announce_stores_the_source(self):
        self._quiet(self.bc.proactive_announce, "Tea " + self.tag,
                    source="timer")
        with open(self.queue, encoding="utf-8") as f:
            self.assertEqual(json.load(f)[-1]["source"], "timer")

    def test_standby_speaks_the_owners_reminders_and_holds_the_rest(self):
        bc = self.bc
        bc._sleep_mode[0] = True
        bc._standby_mode[0] = True
        chat = "Banter line " + self.tag
        tea = "Reminder, sir - tea " + self.tag
        promise = "You asked me to say this " + self.tag
        self._quiet(bc.proactive_announce, chat, source="banter")
        self._quiet(bc.proactive_announce, tea, source="timer")
        self._quiet(bc.proactive_announce, promise, source="promise:chat")
        self._p(bc, "record_speech", return_value=None)
        self._quiet(bc._handle_sleep_standby, None)
        self.assertEqual(self.spoken, [tea, promise],
                         "a timer fired in standby was not spoken")
        self.assertEqual(self._queued(), [chat])
        self.assertTrue(bc._sleep_mode[0])
        # The wake drain (normal mode) speaks what standby held.
        self.spoken.clear()
        self._quiet(bc._speak_pending)
        self.assertEqual(self.spoken, [chat])
        self.assertEqual(self._queued(), [])


# ── B010: a standby wake that carries a command ────────────────────────────
class StandbyWakeCarryTests(_Base):
    def setUp(self):
        super().setUp()
        bc = self.bc
        from core.followup_window import FollowupWindow
        bc._sleep_mode[0] = True
        bc._standby_mode[0] = True
        self._p(bc, "_speak_pending", return_value=False)
        self._p(bc, "set_state")
        self._p(bc, "_heartbeat")
        self._p(bc, "context_aware_greeting", return_value=("Yes, sir?", 1.0))
        self._p(bc, "OVERNIGHT_FLAG_FILE", os.path.join(self.tmp, "none"))
        self._p(bc, "_learn_gate_note_wake")
        self._p(bc, "_standby_wake_detected", return_value=None)
        self._p(bc, "_audio_music_should_refuse_wake", return_value=False)
        self._p(bc, "_audio_music_feed")
        self._p(bc, "_device_speech_ignored", return_value=False)
        self._p(bc, "_dialogue_hold_ignored", return_value=False)
        self._p(bc, "_self_echo_ignored", return_value=False)
        self._p(bc, "_followup_window", FollowupWindow(0))
        self._p(bc, "_require_wake_runtime", True)

    def _spoken_wake(self, transcript):
        bc = self.bc
        self.audio = bc.np.zeros(bc.SAMPLE_RATE, dtype="float32")
        self._p(bc, "record_speech", return_value=self.audio)
        self.conf = {"no_speech_prob": 0.01, "avg_logprob": -0.2}
        self._p(bc, "_transcribe_capture",
                return_value=(transcript, self.conf))
        return self._quiet(bc._handle_sleep_standby, None)[0]

    def test_a_typed_wake_with_a_command_is_carried(self):
        bc = self.bc
        out, _ = self._quiet(bc._handle_sleep_standby,
                             "Jarvis, turn off the lights")
        self.assertIsNotNone(out, "the command was dropped behind a greeting")
        self.assertEqual(out[0], "Jarvis, turn off the lights")
        self.assertEqual(self.spoken, [])
        self.assertFalse(bc._sleep_mode[0])

    def test_a_spoken_wake_with_a_command_is_carried_with_its_audio(self):
        bc = self.bc
        out = self._spoken_wake("Jarvis, what time is it?")
        self.assertEqual(out, ("Jarvis, what time is it?", self.conf))
        self.assertIs(bc._last_capture_audio, self.audio)
        self.assertEqual(self.spoken, [])
        # The carried turn passes the normal-mode gate on its wake prefix.
        self.assertFalse(bc._should_refuse_background_audio(out[0])[0])

    def test_a_bare_wake_greets_and_admits_exactly_one_reply(self):
        bc = self.bc
        self.assertIsNone(self._spoken_wake("Jarvis."))
        self.assertEqual(self.spoken, ["Yes, sir?"])
        refuse, why = bc._should_refuse_background_audio("what's the weather")
        self.assertFalse(refuse, "the reply to 'Yes, sir?' was refused")
        self.assertEqual(why, "standby greeting reply")
        self.assertEqual(bc._should_refuse_background_audio("and the news"),
                         (True, "wake-word mode"))

    def test_the_greeting_admit_lapses(self):
        bc = self.bc
        self._quiet(bc._handle_sleep_standby, "Jarvis")
        bc._standby_greet_admit_until[0] = time.time() - 1
        self.assertTrue(bc._should_refuse_background_audio("hello")[0])

    def test_a_wake_or_greeting_phrase_only_greets(self):
        bc = self.bc
        for text in ("Jarvis, wake up", "hey JARVIS are you there?",
                     "Jarvis, good morning", "hey JARVIS wake up please"):
            with self.subTest(text=text):
                bc._sleep_mode[0] = True
                bc._standby_mode[0] = True
                self.spoken.clear()
                out, _ = self._quiet(bc._handle_sleep_standby, text)
                self.assertIsNone(out)
                self.assertEqual(self.spoken, ["Yes, sir?"])

    def test_a_standby_wake_opens_the_followup_window(self):
        bc = self.bc
        from core.followup_window import FollowupWindow
        self._p(bc, "_followup_window", FollowupWindow(45))
        self._quiet(bc._handle_sleep_standby, "Jarvis")
        self.assertTrue(bc._followup_window.admit())

    def test_the_main_loop_runs_the_carried_turn(self):
        src = inspect.getsource(self.bc.main)
        self.assertIn("_cap = _handle_sleep_standby(_injected_text)", src)
        at = src.index("_cap = _handle_sleep_standby(_injected_text)")
        self.assertLess(at, src.index("text, conf = _cap"))


# ── B011: one capture holds the mic at a time ──────────────────────────────
class OffThreadCaptureTests(_CaptureBase):
    def test_an_action_capture_takes_the_mic_from_the_idle_listen(self):
        bc = self.bc
        mic = self._mic()            # no frames: both captures idle-listen
        box = {}

        def action():                # a dashboard-run action's _listen()
            self._wait_for(lambda: bool(bc._record_speech_active[0]),
                           "the main loop's listen")
            box["audio"] = bc.record_speech(timeout=0.6)
            box["done"] = time.time()
        t = threading.Thread(target=action, name="web-action-test")
        t.start()
        t0 = time.time()
        audio, log = self._quiet(bc.record_speech, 5)    # the main loop
        took = time.time() - t0
        t.join(_WAIT)
        self.assertFalse(t.is_alive())
        self.assertIsNone(audio)
        self.assertLess(took, 3.0, "the main loop's listen kept the mic")
        self.assertIn("an action's capture needs the microphone", log)
        self.assertEqual(mic.max_open, 1, "two streams were open at once")
        self.assertEqual(mic.opened, 2)
        self.assertIsNone(bc._offthread_capture[0])
        self.assertFalse(bc._record_speech_active[0])

    def test_the_main_loop_opens_nothing_while_an_action_holds_the_mic(self):
        bc = self.bc
        mic = self._mic()
        release = threading.Event()
        holder = threading.Thread(target=release.wait, args=(_WAIT,))
        holder.start()
        self.addCleanup(holder.join, _WAIT)
        self.addCleanup(release.set)
        bc._offthread_capture[0] = holder
        t0 = time.time()
        audio, _ = self._quiet(bc.record_speech, 5, yield_to_work=True)
        self.assertIsNone(audio)
        self.assertLess(time.time() - t0, 2.0)
        self.assertEqual(mic.opened, 0)
        self.assertEqual(bc._capture_yield_reason[0], "mic")


# ── B061: dropped lines are logged by length only ──────────────────────────
class DroppedLineLogTests(_Base):
    def setUp(self):
        super().setUp()
        bc = self.bc
        bc._sleep_mode[0] = True
        bc._standby_mode[0] = True
        self._p(bc, "_speak_pending", return_value=False)
        self._p(bc, "set_state")
        self._p(bc, "_heartbeat")
        self._p(bc, "_standby_wake_detected", return_value=None)
        self._p(bc, "_audio_music_feed")
        self._p(bc, "_device_speech_ignored", return_value=False)
        self._p(bc, "_dialogue_hold_ignored", return_value=False)
        self._p(bc, "_self_echo_ignored", return_value=False)
        self._p(bc, "record_speech",
                return_value=bc.np.zeros(bc.SAMPLE_RATE, dtype="float32"))

    def _heard(self, transcript):
        self._p(self.bc, "_transcribe_capture", return_value=(transcript, {}))
        return self._quiet(self.bc._handle_sleep_standby, None)[1]

    def test_a_line_ignored_in_standby_is_not_logged(self):
        self._p(self.bc, "_audio_music_should_refuse_wake", return_value=False)
        log = self._heard("the meeting moved to Thursday at three")
        self.assertIn("[standby] ignored (", log)
        self.assertNotIn("Thursday", log)
        self.assertNotIn("meeting", log)

    def test_a_lyric_near_miss_is_not_logged(self):
        self._p(self.bc, "_audio_music_should_refuse_wake", return_value=True)
        log = self._heard("wake up little susie wake up")
        self.assertIn("wake-word ignored", log)
        self.assertNotIn("susie", log)

    def test_the_main_loop_drop_lines_carry_no_words(self):
        # The bg-audio and speech-filter drops live inline in main(): pin
        # that their log lines format the length, never a slice of the text.
        src = inspect.getsource(self.bc.main)
        for marker in ('[bg-audio] {_bg_why}', '[filter] dropped'):
            with self.subTest(marker=marker):
                at = src.index(marker)
                line = src[at:src.index(")\n", at)]
                self.assertNotIn("text[:", line)
                self.assertNotIn("snippet", line)
                self.assertIn("len(text", line)


# ── B062: an idle listen yields to queued work ─────────────────────────────
class IdleListenYieldTests(_CaptureBase):
    def _queue_inject_after(self, delay):
        def write():
            with open(self.inject_path, "w", encoding="utf-8") as f:
                json.dump([{"text": "what time is it"}], f)
        t = threading.Timer(delay, write)
        t.start()
        self.addCleanup(t.cancel)

    def test_a_typed_command_ends_the_idle_listen(self):
        bc = self.bc
        self._mic()
        self._queue_inject_after(0.4)
        t0 = time.time()
        audio, _ = self._quiet(bc.record_speech, 5, yield_to_work=True)
        took = time.time() - t0
        self.assertIsNone(audio)
        self.assertLess(took, 2.5, "the command waited out the listen")
        self.assertGreaterEqual(took, 0.35)
        self.assertEqual(bc._capture_yield_reason[0], "work")

    def test_an_in_turn_capture_never_yields(self):
        # A confirmation / the printer wizard must not be aborted by an
        # unrelated typed command.
        bc = self.bc
        self._mic()
        self._queue_inject_after(0.2)
        t0 = time.time()
        audio, _ = self._quiet(bc.record_speech, 1.2)
        self.assertIsNone(audio)
        self.assertGreaterEqual(time.time() - t0, 1.1)

    def test_an_utterance_in_progress_is_never_cut(self):
        bc = self.bc
        voiced = threading.Event()
        self._p(bc, "_prof", side_effect=lambda *a, **k: (
            voiced.set() if a and a[0] == "voiced" else None))

        def frames(push, stream):
            push(6, 0.2)                 # the owner starts talking
            voiced.wait(_WAIT)
            with open(self.inject_path, "w", encoding="utf-8") as f:
                json.dump([{"text": "typed meanwhile"}], f)
            for _ in range(8):           # still talking past a work check
                push(1, 0.2)
                time.sleep(0.06)
            push(40, 0.0)                # then silence ends the utterance
        self._mic(frames)
        audio, _ = self._quiet(bc.record_speech, 5, yield_to_work=True)
        self.assertIsNotNone(audio, "the owner's utterance was cut off")

    def test_capture_utterance_runs_no_proactive_turn_on_a_yield(self):
        bc = self.bc
        self._p(bc, "_get_realtime_session", return_value=None)
        self._p(bc, "resume_face_tracking")
        self._p(bc, "set_state")
        self._p(bc, "_heartbeat")
        proactive = self._p(bc, "should_be_proactive", return_value=True)
        self._p(bc, "_do_proactive_turn")
        for reason in ("work", "mic"):
            with self.subTest(reason=reason):
                pending = self._p(bc, "_speak_pending", return_value=False)

                def rec(timeout=None, **kw):
                    bc._capture_yield_reason[0] = reason
                    return None
                self._p(bc, "record_speech", side_effect=rec)
                out, _ = self._quiet(bc._capture_utterance, None, {})
                self.assertIsNone(out)
                proactive.assert_not_called()
                # "mic": nothing is spoken over the other capture (only the
                # drain at the top of the pass ran).
                if reason == "mic":
                    self.assertEqual(pending.call_count, 1)
                else:
                    self.assertEqual(pending.call_count, 2)


if __name__ == "__main__":
    unittest.main()
