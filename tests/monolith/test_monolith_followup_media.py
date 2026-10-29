"""FOLLOWUP_MEDIA_NO_EXTEND (3.0.6, scope B5): media audio inside the wake-word
follow-up window.

Live 2026-10-28 17:28 (wake-word mode, FOLLOWUP_WINDOW_S 20): "play vision
quest on the left monitor" (wake-led) opened the window; 16 s later "Yeah.",
heard over the YouTube video he had just started, scored 0.64 against his
voiceprint - "unsure", which the media gate lets through - and its admission
EXTENDED the window; 18 s after that the video's own 164-character narration
(0.68) was admitted inside the extended window and ran session_memory_recall.

Through the live entry points, in the main loop's order: _bg_gate_for_turn
(the background-audio gate, where admit() extends the window) and then
_media_voice_gate (where the voice is judged). Fakes only (the media gate
harness): no audio device, voiceprint or COM.

    python -m unittest tests.monolith.test_monolith_followup_media
"""
from __future__ import annotations

from unittest import mock

from core.followup_window import FollowupWindow
from tests.monolith.test_monolith_media_gate import _Base, _FakeVoiceId

WAKE_LINE = "Jarvis, play vision quest on the left monitor."
YEAH = "Yeah."
# Stand-in for the video's narration (its words are not kept here).
NARRATION = ("and that is the moment the whole series turns, because everything "
             "we saw in the first episodes was setting up this one reveal about "
             "the vision and his memory.")


class _Clock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t


class FollowupMediaTests(_Base):
    def setUp(self):
        super().setUp()
        self.clock = _Clock(0.0)
        self.win = FollowupWindow(20, clock=self.clock)
        for p in (mock.patch.object(self.bc, "_require_wake_runtime", True),
                  mock.patch.object(self.bc, "_followup_window", self.win),
                  mock.patch.object(self.bc, "_room_tv_gate", return_value=False)):
            p.start()
            self.addCleanup(p.stop)

    def _turn(self, text, at, *, score=0.64, name=None, peak=0.4):
        """One mic turn at ``at`` s: the bg gate, then (when it passed) the
        media gate. (refused_by_bg, dropped_by_media, printed lines)."""
        self.clock.t = at
        refuse, _why = self.bc._bg_gate_for_turn(text, False)
        if refuse:
            return True, False, []
        drop, lines, _vid = self._gate(text, peak=peak, vid=_FakeVoiceId(name, score))
        return False, drop, lines

    def test_the_live_1728_narration_is_no_longer_admitted(self):
        self.assertEqual(self._turn(WAKE_LINE, 0.0, name="owner", score=0.80)[:2],
                         (False, False))
        refused, dropped, lines = self._turn(YEAH, 16.0, score=0.64)
        self.assertEqual((refused, dropped), (False, False))      # "Yeah." still runs
        self.assertTrue(any("[follow-up] window not extended" in ln for ln in lines), lines)
        self.assertFalse(any("Yeah" in ln for ln in lines))       # numbers only
        refused, _d, _l = self._turn(NARRATION, 34.0, score=0.68)
        self.assertTrue(refused)                                   # window closed at 20 s

    def test_off_is_the_old_behaviour(self):
        with mock.patch.object(self.bc, "FOLLOWUP_MEDIA_NO_EXTEND", False, create=True):
            self._turn(WAKE_LINE, 0.0, name="owner", score=0.80)
            _r, _d, lines = self._turn(YEAH, 16.0, score=0.64)
            self.assertFalse(any("[follow-up]" in ln for ln in lines))
            refused, _d, _l = self._turn(NARRATION, 34.0, score=0.68)
            self.assertFalse(refused)                              # extended to 36 s

    def test_his_matched_voice_still_extends(self):
        self._turn(WAKE_LINE, 0.0, name="owner", score=0.80)
        self._turn("and make it full screen", 16.0, name="owner", score=0.81)
        refused, _d, _l = self._turn("turn it up a bit more please sir thanks", 34.0,
                                     name="owner", score=0.80)
        self.assertFalse(refused)

    def test_no_pc_audio_still_extends(self):
        self._turn(WAKE_LINE, 0.0, name="owner", score=0.80, peak=0.0)
        self._turn(YEAH, 16.0, score=0.64, peak=0.0)
        refused, _d, _l = self._turn("what about tomorrow", 34.0, score=0.64, peak=0.0)
        self.assertFalse(refused)

    def test_a_wake_led_line_over_media_is_unchanged(self):
        self._turn(WAKE_LINE, 0.0, name="owner", score=0.80)
        _r, dropped, lines = self._turn("Jarvis, what time is it?", 10.0, score=0.50)
        self.assertFalse(dropped)
        self.assertFalse(any("[follow-up]" in ln for ln in lines))
        self.assertAlmostEqual(self.win.remaining_s(), 20.0)       # re-opened in full

    def test_a_dropped_not_owner_line_does_not_hold_the_window_open(self):
        self._turn(WAKE_LINE, 0.0, name="owner", score=0.80)
        _r, dropped, _l = self._turn(NARRATION, 16.0, score=0.30)
        self.assertTrue(dropped)
        self.assertAlmostEqual(self.win.remaining_s(), 4.0)
