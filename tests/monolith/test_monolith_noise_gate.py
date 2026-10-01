"""Monolith wiring for the noise-heard-as-speech gate (R10, 2026-09-29).

THE LIVE INCIDENT (session_2026-09-29_19-17-23.log, 19:18:55). Nobody home,
wake-word mode off, no media playing. JARVIS had just booted and spoken his
warm-restart greeting ("... shall I resume, or is there something else? At
your service."). Whisper turned room noise — peak RMS 0.0119 against the 0.008
VAD threshold — into "Bye.", and the main loop answered it as an owner turn
with a full LLM call. "bye" is in WHISPER_ALWAYS_ACCEPT, and is_valid_speech's
single-word shortcut ran before its hallucination check.

These tests build each turn's context with the REAL code that builds it live
— _note_owner_turn for the owner's last turn, the real _speak (only its
synthesis and device body are faked) for JARVIS's last line — then run the two
statements main() runs for a mic turn: _noise_verdict, then is_valid_speech
with its verdict. A tree without the gate runs is_valid_speech alone (what
main() did), so these tests fail there on their assertions, not on a name.

Synthetic lines only.

    python -m unittest tests.monolith.test_monolith_noise_gate
"""
from __future__ import annotations

import ast
import contextlib
import importlib.util
import inspect
import io
import json
import os
import time
import unittest
from unittest import mock

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith

OK_CONF = {"no_speech_prob": 0.30, "avg_logprob": -0.60}
GREETING = ("Welcome back, sir. When we left off you were tidying the "
            "workshop — shall I resume, or is there something else? At your "
            "service.")


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
        self._p(bc, "NOISE_FILTER_ENABLED", True, create=True)
        self._p(bc, "VAD_THRESHOLD", 0.008)
        # _speak reaches the (fake) device: not the staging / mute paths.
        self._p(bc, "_is_staging", lambda: False)
        self._p(bc, "_tts_layer", None)
        self._p(bc, "_session_start_time", time.time() - 3600)
        self._p(bc, "_sentence_tts_plan", lambda text: None)
        self._p(bc, "synthesise",
                side_effect=lambda t: (bc.np.zeros(2400, dtype="float32"),
                                       24000))
        self._p(bc, "set_state")
        self._p(bc, "_write_hud_state")
        self._p(bc, "_heartbeat")
        self.body = self._p(bc, ("_play_with_lipsync_body"
                                 if hasattr(bc, "_play_with_lipsync_body")
                                 else "play_with_lipsync"))
        bc._tts_muted[0] = False

    def _quiet(self, fn, *a, **k):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            out = fn(*a, **k)
        return out, buf.getvalue()

    def _jarvis_says(self, line):
        """The REAL _speak; returns when the line finished (monotonic)."""
        self._quiet(self.bc._speak, line)
        cell = getattr(self.bc, "_last_jarvis_line", None)
        return cell[0] if cell else time.monotonic()

    def _owner_spoke(self):
        """The REAL owner-turn stamp main() makes at 'You:'."""
        self.bc._note_owner_turn()
        return self.bc._last_owner_turn_at[0]

    def _turn(self, text, at, peak, conf=None, injected=False):
        """main()'s two statements for this transcript, at monotonic ``at``.
        Returns (answered, console output)."""
        bc = self.bc
        conf = dict(OK_CONF) if conf is None else conf
        self._p(bc, "_last_recording_peak", peak)
        gate = getattr(bc, "_noise_verdict", None)
        takes_reply = "reply" in inspect.signature(
            bc.is_valid_speech).parameters

        def _run():
            nv = gate(text, conf, injected, now=at) if gate else ""
            if nv == "noise":
                return False
            kw = {"reply": nv == "reply"} if takes_reply else {}
            return bc.is_valid_speech(text, conf, peak_rms=peak, **kw)[0]
        return self._quiet(_run)


class LiveIncidentTests(_Base):
    def test_noise_heard_as_bye_after_the_greeting_is_not_answered(self):
        # Fresh session: the owner has not spoken. The greeting asked a
        # question; eight seconds later the noise arrives at 1.49x the VAD
        # threshold.
        line_at = self._jarvis_says(GREETING)
        answered, log = self._turn("Bye.", at=line_at + 8.0, peak=0.0119)
        self.assertFalse(answered, "room noise was answered as the owner")
        self.assertIn("[noise] ignored", log)
        self.assertNotIn("bye", log.lower(), "the transcript was logged")

    def test_nobody_home_a_loud_thank_you_is_noise(self):
        answered, log = self._turn("Thank you.", at=time.monotonic(),
                                   peak=0.05)
        self.assertFalse(answered)
        self.assertIn("[noise] ignored (owner silent this session", log)
        self.assertNotIn("thank", log.lower())

    def test_whisper_doubt_drops_it_even_mid_conversation(self):
        self._owner_spoke()
        line_at = self._jarvis_says("It is seven forty-two, sir.")
        answered, log = self._turn(
            "Thank you.", at=line_at + 2.0, peak=0.2,
            conf={"no_speech_prob": 0.95, "avg_logprob": -0.4})
        self.assertFalse(answered)
        self.assertIn("[noise] ignored (whisper no_speech_prob 0.95", log)

    def test_a_stale_confirmation_is_not_confirmed_by_noise(self):
        bc = self.bc
        bc._pending_confirmation.append(("synthetic_risky_action", "x"))
        bc._last_owner_turn_at[0] = time.monotonic() - 3600.0
        answered, log = self._turn("Okay.", at=time.monotonic(), peak=0.009)
        self.assertFalse(answered, "an hour-old confirmation was answered "
                                   "by room noise")
        self.assertIn("[noise] ignored", log)


class LapsedPromptTests(_Base):
    """2026-10-01 merge audit: the confirmation TTL is lazy, so a lapsed
    prompt stays in _pending_confirmation until the next utterance - and
    _reply_prompt_pending kept counting it, holding the noise gate's "a
    prompt is waiting for an answer" exemption open for hours."""

    def _queue(self, age_s):
        bc = self.bc
        bc._queue_pending_confirmation("synthetic_risky_action", "x")
        bc._pending_confirmation_at[0] = time.monotonic() - age_s

    def test_only_an_answerable_confirmation_counts(self):
        bc = self.bc
        self.assertFalse(bc._reply_prompt_pending())
        self._queue(5.0)
        self.assertTrue(bc._reply_prompt_pending())
        bc._pending_confirmation_at[0] = (time.monotonic()
                                          - bc.CONFIRMATION_TTL_S - 1.0)
        self.assertFalse(bc._reply_prompt_pending(), "a lapsed prompt counted")
        # Unknown age (a hand-built queue, no stamp) still counts.
        bc._pending_confirmation_at[0] = 0.0
        self.assertTrue(bc._reply_prompt_pending())

    def test_a_lapsed_prompt_does_not_keep_a_hallucination_as_a_reply(self):
        bc = self.bc
        self._queue(3 * 3600.0)                    # queued three hours ago
        now = time.monotonic()
        bc._last_owner_turn_at[0] = now - 60.0     # but he spoke a minute ago
        answered, log = self._turn("Yeah.", at=now, peak=0.0090)
        self.assertFalse(answered, "a reply-shaped hallucination was kept as "
                                   "the answer to a prompt that had lapsed")
        self.assertIn("[noise] ignored", log)

    def test_a_fresh_prompt_still_keeps_the_reply(self):
        bc = self.bc
        self._queue(5.0)
        now = time.monotonic()
        bc._last_owner_turn_at[0] = now - 60.0
        answered, log = self._turn("Yeah.", at=now, peak=0.0090)
        self.assertTrue(answered)
        self.assertNotIn("[noise]", log)


class RealRepliesKeptTests(_Base):
    def test_thank_you_right_after_jarvis_answered_is_answered(self):
        # The owner's quiet desk mic: 1.2x the threshold is where his real
        # speech lands; the conversation makes it clearly a reply.
        self._owner_spoke()
        line_at = self._jarvis_says("It is seven forty-two, sir.")
        answered, log = self._turn("Thank you.", at=line_at + 2.5,
                                   peak=0.0096)
        self.assertTrue(answered, "the owner's thank-you was dropped")
        self.assertNotIn("[noise]", log)

    def test_bye_ending_the_conversation_is_answered(self):
        self._owner_spoke()
        line_at = self._jarvis_says("Anything else, sir?")
        for text in ("Bye.", "Bye bye!", "Thanks."):
            answered, log = self._turn(text, at=line_at + 3.0, peak=0.0090)
            self.assertTrue(answered, text)
            self.assertNotIn("[noise]", log)

    def test_an_answer_to_a_pending_prompt_is_answered(self):
        bc = self.bc
        bc._pending_confirmation.append(("synthetic_risky_action", "x"))
        now = time.monotonic()
        bc._last_owner_turn_at[0] = now - 60.0
        self._jarvis_says("Shall I go ahead, sir?")
        answered, log = self._turn("Yeah.", at=now + 45.0, peak=0.0090)
        self.assertTrue(answered)
        self.assertNotIn("[noise]", log)

    def test_typed_turns_are_never_checked(self):
        answered, log = self._turn("bye", at=time.monotonic(), peak=0.0,
                                   injected=True)
        self.assertTrue(answered)
        self.assertNotIn("[noise]", log)

    def test_ordinary_speech_is_untouched(self):
        answered, log = self._turn("turn off the lights",
                                   at=time.monotonic(), peak=0.0085)
        self.assertTrue(answered)
        self.assertNotIn("[noise]", log)


class RepetitionNoiseTests(_Base):
    """2026-09-30. Live session_2026-09-29_22-06-02.log 22:53:08: the owner
    mid-conversation, music in the room, peak RMS 0.0119. Whisper produced
    "I I I I I I I I I I I I I" and JARVIS answered "Very good, sir." — it is
    not a known hallucination PHRASE, so the R10 gate let it through."""

    LIVE = "I I I I I I I I I I I I I"

    def test_the_live_repetition_is_not_answered_mid_conversation(self):
        self._owner_spoke()
        line_at = self._jarvis_says("Of course, sir.")
        answered, log = self._turn(self.LIVE, at=line_at + 5.0, peak=0.0119)
        self.assertFalse(answered, "a repeated-word loop was answered")
        self.assertIn("[noise] ignored (1 distinct word in 13)", log)
        self.assertNotIn("I I", log, "the transcript was logged")

    def test_other_degenerate_shapes_are_not_answered(self):
        self._owner_spoke()
        line_at = self._jarvis_says("Of course, sir.")
        for text in ("you you you you", "Uh, um, uh, um.",
                     "Thank you. Thank you. Thank you. Thank you."):
            answered, log = self._turn(text, at=line_at + 3.0, peak=0.05)
            self.assertFalse(answered, text)
            self.assertIn("[noise] ignored (", log)

    def test_emphatic_stop_and_confirmation_words_are_answered(self):
        self._owner_spoke()
        line_at = self._jarvis_says("Shall I go ahead, sir?")
        for text in ("Stop, stop, stop!", "no no no", "yes yes",
                     "Yes, yes, yes."):
            answered, log = self._turn(text, at=line_at + 2.0, peak=0.0119)
            self.assertTrue(answered, text)
            self.assertNotIn("[noise]", log)

    def test_a_typed_repetition_is_never_filtered(self):
        answered, log = self._turn(self.LIVE, at=time.monotonic(), peak=0.0,
                                   injected=True)
        self.assertTrue(answered)
        self.assertNotIn("[noise]", log)

    def test_the_kill_switch_covers_it(self):
        self._p(self.bc, "NOISE_FILTER_ENABLED", False)
        answered, log = self._turn(self.LIVE, at=time.monotonic(),
                                   peak=0.0119)
        self.assertTrue(answered)
        self.assertNotIn("[noise]", log)


class GateBehaviourTests(_Base):
    def test_only_a_line_that_was_heard_opens_the_reply_window(self):
        bc = self.bc
        self.body.side_effect = RuntimeError("synthetic device fault")
        self._jarvis_says("It is seven forty-two, sir.")
        self.assertEqual(bc._last_jarvis_line[0], 0.0,
                         "a line that never played cannot be answered")
        self.body.side_effect = None
        bc._tts_muted[0] = True
        self._jarvis_says("It is seven forty-two, sir.")
        self.assertEqual(bc._last_jarvis_line[0], 0.0)
        bc._tts_muted[0] = False
        at = self._jarvis_says("Shall I?")
        self.assertGreater(at, 0.0)
        self.assertTrue(bc._last_jarvis_line[1], "a question was asked")

    def test_the_kill_switch_restores_the_old_behaviour(self):
        self._p(self.bc, "NOISE_FILTER_ENABLED", False)
        line_at = self._jarvis_says(GREETING)
        answered, log = self._turn("Bye.", at=line_at + 8.0, peak=0.0119)
        self.assertTrue(answered)
        self.assertNotIn("[noise]", log)

    def test_the_gate_fails_open(self):
        self._p(self.bc._speech_filter_mod, "hallucination_verdict",
                side_effect=RuntimeError("boom"))
        out, _ = self._quiet(self.bc._noise_verdict, "Bye.", OK_CONF)
        self.assertEqual(out, "")


class WiringTests(_Base):
    """Source-level: main() cannot run in a test."""

    def test_main_runs_the_gate_before_is_valid_speech_and_passes_its_verdict(self):
        src = inspect.getsource(self.bc.main)
        block = ("_nv = _noise_verdict(text, conf, _injected_text is not None)\n"
                 "                if _nv == \"noise\":\n"
                 "                    set_state(\"idle\")\n"
                 "                    continue\n")
        self.assertEqual(src.count("_noise_verdict("), 1)
        gate = src.index(block)
        self.assertLess(src.index("_bg_gate_for_turn("), gate)
        valid = src.index("valid, reason = is_valid_speech(text, conf,")
        self.assertLess(gate, valid)
        self.assertIn('reply=(_nv == "reply")', src[valid:valid + 200])
        self.assertLess(valid, src.index("_note_owner_turn()"))

    def test_speak_stamps_only_a_heard_line(self):
        src = inspect.getsource(self.bc._speak)
        stamp = src.index("_note_jarvis_line(spoken_text)")
        self.assertIn("if _speak_ok:", src[stamp - 80:stamp])

    def test_config_default_and_settings_wiring(self):
        root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        with open(os.path.join(root, "core", "config.py"),
                  encoding="utf-8") as fh:
            src = fh.read()
        lits = {}
        for node in ast.parse(src).body:
            if isinstance(node, ast.Assign) and isinstance(node.value,
                                                           ast.Constant):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name):
                        lits[tgt.id] = node.value.value
        self.assertIs(lits["NOISE_FILTER_ENABLED"], True)
        spec = importlib.util.spec_from_file_location(
            "sw_noise_gate", os.path.join(root, "tools", "settings_window.py"))
        sw = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sw)
        self.assertEqual(sw.SCHEMA["NOISE_FILTER_ENABLED"]["type"], "bool")
        self.assertIs(sw.SCHEMA["NOISE_FILTER_ENABLED"]["default"], True)
        with open(os.path.join(root, "tools", "user_settings.example.json"),
                  encoding="utf-8") as fh:
            self.assertIs(json.load(fh)["NOISE_FILTER_ENABLED"], True)


if __name__ == "__main__":
    unittest.main()
