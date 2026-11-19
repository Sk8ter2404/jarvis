"""Lines he summoned and the away question, through the REAL monolith drain
(JARVIS-Research/review_20261117_overhaul_audit.json, the proactive cluster,
proactive-02, -03 and -06).

  * proactive-03: 11 of 15 double claps got no "You rang, sir?" - the answer
    waited behind the 45 s room-talk hold and expired at its 10 s age.
  * proactive-02: the 11-17 06:35 arrival briefing was held behind his video
    and the room talk of getting ready, then "owner away" for hours.
  * proactive-06: "did anything happen while I was away" went through the
    same hold, so his direct question could be answered "still waiting".

Every voice, clock and sensor is faked; files live in temp dirs. Synthetic
lines only.

    python tools/run_tests.py monolith.test_monolith_summoned_lines_1117
"""
from __future__ import annotations

import contextlib
import io
import time
import types
import unittest
from unittest import mock

from core import owner_presence as op
from core import queue_receipts as qr
from tests._monolith_harness import requires_monolith
from tests.monolith.test_monolith_presence_hold import _Base

CLAP_LINE = "You rang, sir?"
BRIEFING = "[intent:briefing] Good morning, sir. Rain at four."


def _clap(age_s=2.0):
    e = _Base._e(CLAP_LINE, "clap", age_s)
    e["summoned"] = "clap"
    return e


def _arrival(age_s=60.0):
    e = _Base._e(BRIEFING, "arrival_v2", age_s)
    e["summoned"] = "arrival"
    return e


@requires_monolith
class _Summoned(_Base):
    def setUp(self):
        super().setUp()
        self.quiet = [True]
        self._p(self.bc, "_summoned_room_quiet",
                side_effect=lambda: self.quiet[0])
        self._p(self.bc, "CLAP_ACK_PASS_HOLDS_S", 10.0, create=True)
        self._p(self.bc, "ARRIVAL_PASS_HOLDS_S", 600, create=True)
        qr.reset()
        self.addCleanup(qr.reset)

    def _room_talk(self):
        self.bc._note_room_talk("so anyway I told him we'd be there by eight")

    def _media(self):
        self._p(self.bc, "MEDIA_HOLD_ENABLED", True, create=True)
        self._p(self.bc, "_media_hold_active", return_value=True)

    def _face(self):
        self.bc.last_face_seen = time.time() - 5


class ClapAnswerTests(_Summoned):
    def test_the_answer_passes_room_talk_when_he_is_seen_and_the_room_is_quiet(self):
        self._face()
        self._room_talk()
        self._write([_clap()])
        _, log = self._drain()
        self.assertEqual(self.spoke, [CLAP_LINE])
        self.assertIn("clap line passes the room talk hold", log)

    def test_his_voice_counts_as_seen(self):
        self._owner_spoke(20)
        self._room_talk()
        self._write([_clap()])
        self._drain()
        self.assertEqual(self.spoke, [CLAP_LINE])

    def test_polled_input_alone_does_not(self):
        # No face, no voice - only his mouse (an automation moves it too).
        self.input_age[0] = 3.0
        self._room_talk()
        self._write([_clap()])
        self._drain()
        self.assertEqual(self.spoke, [])

    def test_music_holds_it_as_before(self):
        self._face()
        self._room_talk()
        self.quiet[0] = False
        self._write([_clap()])
        self._drain()
        self.assertEqual(self.spoke, [])

    def test_the_empty_room_still_holds_it(self):
        # He left an hour ago: "owner away" is never passed.
        self.input_age[0] = 3600.0
        self._owner_spoke(3600)
        self._write([_clap()])
        _, log = self._drain()
        self.assertEqual(self.spoke, [])
        self.assertIn("holding queued speech: owner away", log)

    def test_his_video_still_holds_it(self):
        self._face()
        self._media()
        self._write([_clap()])
        self._drain()
        self.assertEqual(self.spoke, [])

    def test_flag_zero_is_the_old_hold(self):
        self._p(self.bc, "CLAP_ACK_PASS_HOLDS_S", 0)
        self._face()
        self._room_talk()
        self._write([_clap()])
        self._drain()
        self.assertEqual(self.spoke, [])

    def test_a_plain_line_is_not_summoned(self):
        self._face()
        self._room_talk()
        plain = self._e(CLAP_LINE, "clap", 2)         # no summoned mark
        self._write([plain])
        self._drain()
        self.assertEqual(self.spoke, [])

    def test_it_passes_the_gap_after_a_budget_deferral(self):
        self._face()
        self._owner_spoke(30)
        self.bc._last_owner_turn_at[0] = self.mono[0] - 30
        self.bc._pending_defer_gap[0] = self.mono[0] + 20
        self.bc._pending_defer_gap[1] = self.bc._last_owner_turn_at[0]
        self._write([self._e("Notice a, sir.", "src"), _clap()])
        _, log = self._drain()
        self.assertEqual(self.spoke, [CLAP_LINE])
        self.assertIn("passes the after the drain budget hold", log)
        self.assertEqual([e["message"] for e in self._read()],
                         ["Notice a, sir."])          # the rest still waits


class ArrivalBriefingTests(_Summoned):
    def test_it_passes_room_talk_while_he_is_here(self):
        self._owner_spoke(30)
        self._room_talk()
        self._write([_arrival()])
        self._drain()
        self.assertEqual(self.spoke, [BRIEFING])

    def test_it_passes_his_video_even_with_no_clean_room(self):
        self._owner_spoke(30)
        self._media()
        self.quiet[0] = False
        self._write([_arrival()])
        self._drain()
        self.assertEqual(self.spoke, [BRIEFING])

    def test_it_never_passes_owner_away(self):
        # 06:53: the face had been gone 5 min and he was leaving.
        self.input_age[0] = 3600.0
        self._owner_spoke(3600)
        self._write([_arrival()])
        _, log = self._drain()
        self.assertEqual(self.spoke, [])
        self.assertIn("holding queued speech: owner away", log)
        self.assertEqual(len(self._read()), 1)

    def test_it_never_passes_gaming(self):
        self._owner_spoke(30)
        self._p(self.bc, "_game_quiet_active", return_value=True)
        self._write([_arrival()])
        self._drain()
        self.assertEqual(self.spoke, [])

    def test_the_window_ends_it(self):
        self._owner_spoke(30)
        self._room_talk()
        self._write([_arrival(age_s=601)])
        self._drain()
        self.assertEqual(self.spoke, [])

    def test_flag_zero_is_the_old_hold(self):
        self._p(self.bc, "ARRIVAL_PASS_HOLDS_S", 0)
        self._owner_spoke(30)
        self._room_talk()
        self._write([_arrival()])
        self._drain()
        self.assertEqual(self.spoke, [])


class DrainReceiptTests(_Summoned):
    """The skill that queued a line hears what became of it."""

    def setUp(self):
        super().setUp()
        self.heard, self.dropped = [], []
        qr.register("arrival_v2", on_spoken=self.heard.append,
                    on_dropped=lambda e, why: self.dropped.append(why))
        self._owner_spoke(30)

    def test_spoken_is_reported(self):
        self._write([_arrival()])
        self._drain()
        self.assertEqual([e["source"] for e in self.heard], ["arrival_v2"])
        self.assertEqual(self.dropped, [])

    def test_too_old_is_reported_and_remembered(self):
        # 11-17 16:02:41: "waited 34031 s, limit 21600 s".
        self._write([_arrival(age_s=34031)])
        _, log = self._drain()
        self.assertEqual(self.spoke, [])
        self.assertIn("too old to say (arrival_v2", log)
        self.assertEqual(len(self.dropped), 1)
        self.assertIn("34031", self.dropped[0])
        self.assertEqual([m[2] for m in qr.missed()], ["briefing"])


class SummonedRoomQuietTests(_Base):
    """The real _summoned_room_quiet: every source fails CLOSED."""

    def setUp(self):
        super().setUp()
        bc = self.bc
        self._p(bc, "_pc_media_playing", return_value=False)
        self._p(bc, "_smtc_media_playing", return_value=False)
        self._p(bc, "_media_hold_active", return_value=False)
        self._p(bc, "_music_state", {"playing_at": None}, create=True)
        self._p(bc, "MEDIA_HOLD_ENABLED", True, create=True)
        self.detector = types.SimpleNamespace(
            music_seen_within=lambda s: False)
        self._mods = {"skill_standby_audio_detect": self.detector}
        patcher = mock.patch.dict("sys.modules", self._mods)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_a_quiet_room_is_quiet(self):
        self.assertTrue(self.bc._summoned_room_quiet())

    def test_each_source_makes_it_not_quiet(self):
        bc = self.bc
        with mock.patch.object(bc, "_pc_media_playing", return_value=True):
            self.assertFalse(bc._summoned_room_quiet())
        with mock.patch.object(bc, "_smtc_media_playing", return_value=True):
            self.assertFalse(bc._summoned_room_quiet())
        with mock.patch.object(bc, "_media_hold_active", return_value=True):
            self.assertFalse(bc._summoned_room_quiet())
        with mock.patch.dict(bc._music_state,
                             {"playing_at": time.monotonic() - 3}):
            self.assertFalse(bc._summoned_room_quiet())
        self.detector.music_seen_within = lambda s: True
        self.assertFalse(bc._summoned_room_quiet())

    def test_music_that_ended_long_ago_is_quiet(self):
        with mock.patch.dict(self.bc._music_state,
                             {"playing_at": time.monotonic() - 60}):
            self.assertTrue(self.bc._summoned_room_quiet())

    def test_a_failing_reading_is_not_quiet(self):
        with mock.patch.object(self.bc, "_pc_media_playing",
                               side_effect=RuntimeError("meter gone")):
            self.assertFalse(self.bc._summoned_room_quiet())


class AwayQuestionDrainTests(_Summoned):
    """proactive-06 through the real drain and the real shortcut."""

    ASK = "did you log anything while I was at school today?"

    def setUp(self):
        super().setUp()
        bc = self.bc
        self._p(bc, "AWAY_RECAP_ROUTE", True, create=True)
        self._p(bc, "_pending_drain_spoken", [None])
        self._p(bc, "_last_away_recap", [None])
        self._p(bc, "_append_turn")
        self._p(bc, "_last_turn_ctx_at", [0.0])

    def _ask(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            out = self.bc._away_recap_shortcut(self.ASK)
        return out, buf.getvalue()

    def test_he_is_asking_so_room_talk_and_away_do_not_hold_it(self):
        # A typed ask stamps no voice and the room is talking: the old drain
        # said "still waiting".
        self.input_age[0] = 3600.0
        self._owner_spoke(3600)
        self._room_talk()
        self._write([self._e("The print finished, sir.", "bambu", 600)])
        ok, log = self._ask()
        self.assertTrue(ok)
        self.assertIn("he asked what he missed - no presence hold", log)
        self.assertNotIn(op.AWAY_STILL_HELD_LINE, self.spoke)
        self.assertTrue(any("print finished" in s for s in self.spoke),
                        self.spoke)
        self.assertEqual(self._read(), [])

    def test_his_video_does_not_hold_it_either(self):
        self._owner_spoke(30)
        self._media()
        self._write([self._e("The print finished, sir.", "bambu", 600)])
        self._ask()
        self.assertTrue(any("print finished" in s for s in self.spoke),
                        self.spoke)

    def test_a_speech_hold_still_holds(self):
        self._p(self.bc, "_speech_hold_active", return_value=True)
        self._write([self._e("The print finished, sir.", "bambu", 600)])
        self._ask()
        self.assertFalse(any("print finished" in s for s in self.spoke))

    def test_what_went_unsaid_is_named(self):
        # 16:02:41: the briefing waited 34,031 s and was dropped.
        self._owner_spoke(30)
        self._write([_arrival(age_s=34031)])
        ok, log = self._ask()
        self.assertTrue(ok)
        self.assertIn("also missed: the morning briefing", log)
        self.assertEqual(len(self.spoke), 1)
        self.assertIn("the morning briefing", self.spoke[0])
        self.assertIn("went unsaid", self.spoke[0])
        # Told once: a second ask does not repeat it.
        self.spoke.clear()
        self._ask()
        self.assertNotIn("morning briefing", " ".join(self.spoke))


if __name__ == "__main__":
    unittest.main()
