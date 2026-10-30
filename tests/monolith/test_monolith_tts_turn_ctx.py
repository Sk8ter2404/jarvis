"""An old owner turn no longer voices later lines (v3.0.10,
TTS_TURN_CONTEXT_TTL_S).

THE LIVE EVIDENCE (session_2026-10-28_20-18-06.log): his last turn of the
night (22:33:59) carried an emergency keyword and logged [tone] stressed. He
did not speak again until 09:52:27, and every line in between - the print-bed
line at 22:51, the memory recap and briefing at 07:21-07:22, the memory
alerts at 09:22 and 09:34 - logged tone=stressed, and the proactive ones were
voiced brisk_alert (+15 % rate, raised pitch, gain 1.10): detect_context_preset
still read the 22:33 sentence from _last_user_text. Nothing ever aged the
cells out.

  * within TTS_TURN_CONTEXT_TTL_S of his turn: the 3.0.9 behaviour;
  * older: his words, tone / voice mood and emotion no longer pick a preset;
  * TTS_TURN_CONTEXT_TTL_S 0, or cells no turn stamped: the 3.0.9 path.

Synthetic lines only.

    python tools/run_tests.py monolith_tts_turn_ctx
"""
from __future__ import annotations

import time
import unittest
from unittest import mock

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith

CRY = "help help, the printer is on fire"
LINE = "Sir, memory usage is at 90 percent."


@requires_monolith
class TurnContextTTLTests(MonolithGlobalsTestCase):
    def setUp(self):
        bc = self.bc
        if getattr(bc, "_tts_layer", None) is None:
            self.skipTest("core.tts not importable")
        self._p(bc, "TTS_TURN_CONTEXT_TTL_S", 300, create=True)
        self._cells = {}
        for name, value in (("_last_user_text", CRY),
                            ("_last_voice_route", {"mood": "stressed",
                                                   "addendum": ""}),
                            ("_last_user_tone", "frustrated"),
                            ("_last_emotion", None),
                            ("_last_wry", False),
                            ("_last_intent_override", None),
                            ("_last_mood", None)):
            cell = getattr(bc, name)
            self._cells[name] = cell[0]
            cell[0] = value
        self.addCleanup(self._restore)
        # No late-hour or vocal-stress context: only his words are in play.
        self._p(bc._tts_layer, "detect_late_hour", return_value=False)
        import core.audio_processor as ap
        self._p(ap, "recent_peak_rms", return_value=0.0)

    def _restore(self):
        for name, value in self._cells.items():
            getattr(self.bc, name)[0] = value

    def _p(self, *args, **kwargs):
        patcher = mock.patch.object(*args, **kwargs)
        m = patcher.start()
        self.addCleanup(patcher.stop)
        return m

    def _turn_ago(self, seconds):
        self.bc._last_turn_ctx_at[0] = time.monotonic() - seconds

    def _preset(self):
        bc = self.bc
        return bc._resolve_tts_preset(LINE, bc._synth_user_tone())[0]

    def test_a_line_right_after_his_turn_keeps_its_register(self):
        self._turn_ago(10)
        self.assertEqual(self.bc._synth_user_tone(), "stressed")
        self.assertEqual(self._preset(), "brisk_alert")

    def test_live_overnight_line_is_not_voiced_off_the_old_turn(self):
        self._turn_ago(11 * 3600)          # 22:33 -> 09:22
        self.assertIsNone(self.bc._synth_user_tone())
        self.assertNotEqual(self._preset(), "brisk_alert")

    def test_ttl_zero_is_the_3_0_9_path(self):
        self._p(self.bc, "TTS_TURN_CONTEXT_TTL_S", 0)
        self._turn_ago(11 * 3600)
        self.assertEqual(self.bc._synth_user_tone(), "stressed")
        self.assertEqual(self._preset(), "brisk_alert")

    def test_cells_no_turn_stamped_read_as_before(self):
        self.bc._last_turn_ctx_at[0] = 0.0
        self.assertEqual(self.bc._synth_user_tone(), "stressed")
        self.assertEqual(self._preset(), "brisk_alert")

    def test_a_briefing_intent_still_wins_either_way(self):
        for age in (10, 11 * 3600):
            self._turn_ago(age)
            got = self.bc._resolve_tts_preset_from(
                LINE, self.bc._synth_user_tone(), wry=False,
                intent_override="briefing", mood=None)[0]
            self.assertEqual(got, "briefing", age)

    def test_a_turn_that_records_his_words_stamps_the_context(self):
        # The replay-voice route records his words as a routed turn does.
        self._p(self.bc, "_act_replay_last_action", return_value="ok")
        self.bc._last_turn_ctx_at[0] = 0.0
        before = time.monotonic()
        self.assertEqual(self.bc.maybe_replay_last_action("do that again"),
                         "ok")
        self.assertGreaterEqual(self.bc._last_turn_ctx_at[0], before)
        self.assertTrue(self.bc._turn_ctx_fresh())


if __name__ == "__main__":
    unittest.main()
