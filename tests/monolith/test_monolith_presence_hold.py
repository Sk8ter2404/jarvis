"""Monolith wiring for the speech-queue presence hold (NEW #5, 2026-10-02).

THE LIVE EVIDENCE (session_2026-10-01_17-33-02.log / _19-43-10.log /
_21-48-36.log): the owner left the desk at about 18:27 (last physical input
18:27:01; the mic heard nothing until 19:43), yet three queued lines were
spoken into the empty room - the wellness "hour and a half" nudge at
19:05:30, the credits login nag at 19:33:46 and a GPU pulse at 19:36:35. At
21:16:10 a wellness nudge talked over people in the middle of a conversation
the desk mic was capturing (wake-word mode, so every line of it was dropped
at the background-audio gate). _speak_pending drained every source the
moment the loop came round; its only gate was focus mode.

  * away  -> hold every line except his own reminders (timer / schedule /
             promise) and the guard alert; the queue is not even claimed.
  * room talk (non-wake speech dropped in the last ROOM_TALK_HOLD_S) -> hold.
  * back  -> a status line older than PRESENCE_STALE_STATUS_S becomes one
             short "While you were away" recap; a stale break nudge expires.

Every test drives the REAL _speak_pending / helpers with the voice, the
input watcher and the clock faked; the queue lives in a temp dir. On a tree
without the hold the reproductions fail on their assertions (seams are
created when missing), not on a missing name. Synthetic lines only.

    python -m unittest tests.monolith.test_monolith_presence_hold
"""
from __future__ import annotations

import contextlib
import inspect
import io
import json
import os
import tempfile
import time
import unittest
from unittest import mock

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith

WELLNESS = ("You've been at it for an hour and a half, sir. "
            "Hydration, perhaps?")
CREDITS = ("Sir, the credits check needs a login - the console page asked "
           "me to sign in.")
PULSE = ("GPU pinned at 100 percent, sir. CPU 12 percent, memory 41 "
         "percent, 9 windows open.")
TIMER = "Reminder, sir — tea"
GUARD = "Sir, someone is at your desk."
MONO = 100_000.0          # the faked monotonic clock "now"


@requires_monolith
class _Base(MonolithGlobalsTestCase):
    def setUp(self):
        bc = self.bc
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.queue = os.path.join(self._tmp.name, "pending_speech.json")
        self._p(bc, "PENDING_SPEECH_PATH", self.queue)
        self._p(bc, "__file__", os.path.join(self._tmp.name,
                                             "bobert_companion.py"))
        self.spoke: list = []
        self._p(bc, "_speak", lambda msg, **k: self.spoke.append(msg))
        self._p(bc, "_heartbeat")
        self._p(bc, "_speech_hold_active", return_value=False)
        self._p(bc, "_audio_flap_flush")
        bc._recent_spoken_messages.clear()
        self.mono = [MONO]
        self._p(bc, "_proactive_mono", side_effect=lambda: self.mono[0],
                create=True)
        # The physical-input watcher (skills/_air_mouse_yield's low-level hook,
        # which ignores INJECTED events) is faked: no hook is installed and
        # the test machine's real mouse never leaks in.
        self.input_age = [float("inf")]
        age = self.input_age

        class _Watch:
            @staticmethod
            def seconds_since_real_input():
                return age[0]

            @staticmethod
            def install():
                return True
        self._p(bc, "_yield_watch_mod", return_value=_Watch, create=True)
        self._p(bc, "PRESENCE_HOLD_ENABLED", True, create=True)
        self._cell("_presence_gate_armed", True)
        self._cell("_last_room_talk_at", 0.0)
        bc._last_owner_voice_at[0] = 0.0
        bc.last_face_seen = 0.0

    # helpers -------------------------------------------------------------
    def _p(self, *args, **kwargs):
        patcher = mock.patch.object(*args, **kwargs)
        m = patcher.start()
        self.addCleanup(patcher.stop)
        return m

    def _cell(self, name, value):
        """Set a one-element state cell, creating it when the tree has none
        (so a pre-fix tree fails on an assertion, not on a missing name)."""
        cell = getattr(self.bc, name, None)
        if isinstance(cell, list) and cell:
            cell[0] = value
        else:
            self._p(self.bc, name, [value], create=True)

    def _write(self, entries):
        with open(self.queue, "w", encoding="utf-8") as f:
            json.dump(entries, f)

    def _read(self):
        if not os.path.exists(self.queue):
            return []
        with open(self.queue, encoding="utf-8") as f:
            return json.load(f)

    def _drain(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            out = self.bc._speak_pending()
        return out, buf.getvalue()

    def _owner_spoke(self, seconds_ago):
        self.bc._last_owner_voice_at[0] = self.mono[0] - seconds_ago

    @staticmethod
    def _e(msg, src, age_s=5.0):
        return {"ts": time.time() - age_s, "message": msg, "source": src}


# ── away: the 19:05 / 19:33 / 19:36 empty room ─────────────────────────────
class EmptyRoomTests(_Base):
    def setUp(self):
        super().setUp()
        # 19:05:30: his last physical input was 18:27:01, his last MIC turn
        # earlier still; no face on any camera.
        self.input_age[0] = 38 * 60 + 29
        self._owner_spoke(45 * 60)

    def test_live_empty_room_hears_none_of_the_three_lines(self):
        self._write([self._e(WELLNESS, "wellness"),
                     self._e(CREDITS, "credits"),
                     self._e(PULSE, "pulse")])
        before = os.path.getmtime(self.queue)
        spoke, log = self._drain()
        self.assertEqual(self.spoke, [], "spoke into the empty room")
        self.assertFalse(spoke)
        # Held, in order, and the queue was not even claimed / rewritten.
        self.assertEqual([e["message"] for e in self._read()],
                         [WELLNESS, CREDITS, PULSE])
        self.assertEqual(os.path.getmtime(self.queue), before)
        self.assertIn("[pending] holding queued speech: owner away", log)

    def test_an_empty_queue_holds_nothing_and_logs_nothing(self):
        spoke, log = self._drain()
        self.assertFalse(spoke)
        self.assertNotIn("holding queued speech", log)

    def test_the_hold_is_logged_once(self):
        self._write([self._e(WELLNESS, "wellness")])
        _, first = self._drain()
        _, second = self._drain()
        self.assertIn("holding queued speech", first)
        self.assertNotIn("holding queued speech", second)

    def test_his_own_timer_still_speaks_and_the_rest_waits(self):
        self._write([self._e(WELLNESS, "wellness"),
                     dict(self._e(TIMER, "timer"), dedupe_key="timer#1"),
                     self._e("Done as promised, sir.", "promise:memory"),
                     self._e("Time for your vitamins, sir.", "schedule")])
        self._drain()
        self.assertEqual(self.spoke, [TIMER, "Done as promised, sir.",
                                      "Time for your vitamins, sir."])
        self.assertEqual([e["message"] for e in self._read()], [WELLNESS])

    def test_the_guard_alert_is_meant_for_an_empty_room(self):
        self._write([self._e(GUARD, "guard")])
        self._drain()
        self.assertEqual(self.spoke, [GUARD])

    def test_a_typed_or_injected_turn_is_not_presence(self):
        # An autonomous session typing commands is not the owner at the desk.
        self.bc._last_owner_turn_at[0] = self.mono[0] - 5
        self._write([self._e(PULSE, "pulse")])
        self._drain()
        self.assertEqual(self.spoke, [])

    def test_a_stale_face_is_not_presence(self):
        self.bc.last_face_seen = time.time() - 30 * 60
        self._write([self._e(PULSE, "pulse")])
        self._drain()
        self.assertEqual(self.spoke, [])


class HereTests(_Base):
    def _queue_and_drain(self):
        self._write([self._e(PULSE, "pulse")])
        self._drain()
        return self.spoke

    def test_physical_input_is_presence(self):
        self.input_age[0] = 40.0
        self.assertEqual(self._queue_and_drain(), [PULSE])

    def test_a_sustained_face_is_presence(self):
        self.bc.last_face_seen = time.time() - 20
        self.assertEqual(self._queue_and_drain(), [PULSE])

    def test_his_voice_is_presence(self):
        self._owner_spoke(90)
        self.assertEqual(self._queue_and_drain(), [PULSE])

    def test_unarmed_gate_never_holds(self):
        # A bare import / a test / a failed boot start: no sensor armed, so
        # "nobody seen" means unknown - the drain behaves as it always did.
        self.bc._presence_gate_armed[0] = False
        self.assertEqual(self._queue_and_drain(), [PULSE])

    def test_kill_switch(self):
        self._p(self.bc, "PRESENCE_HOLD_ENABLED", False)
        self.assertEqual(self._queue_and_drain(), [PULSE])

    def test_a_broken_presence_check_fails_open(self):
        self._p(self.bc, "_owner_presence", side_effect=RuntimeError("boom"))
        self.assertEqual(self._queue_and_drain(), [PULSE])


# ── room talk: the 21:16:10 nudge over a live conversation ─────────────────
class RoomTalkTests(_Base):
    def setUp(self):
        super().setUp()
        self._owner_spoke(60)          # he is right here, talking to people

    def test_live_nudge_waits_while_the_room_talks_then_speaks(self):
        self.bc._note_room_talk("so anyway I told him we'd be there by eight")
        self._write([self._e(WELLNESS, "wellness")])
        _, log = self._drain()
        self.assertEqual(self.spoke, [], "talked over the conversation")
        self.assertIn("holding queued speech: room talk", log)
        # The room has been quiet for ROOM_TALK_HOLD_S: the line is spoken.
        self.mono[0] += self.bc.ROOM_TALK_HOLD_S + 1
        self._owner_spoke(60)
        self._drain()
        self.assertEqual(self.spoke, [WELLNESS])

    def test_a_timer_still_speaks_over_room_talk(self):
        self.bc._note_room_talk("we should order food before the game")
        self._write([self._e(TIMER, "timer")])
        self._drain()
        self.assertEqual(self.spoke, [TIMER])

    def test_a_lone_noise_word_is_not_room_talk(self):
        self.bc._note_room_talk("Thank you.")
        self._write([self._e(WELLNESS, "wellness")])
        self._drain()
        self.assertEqual(self.spoke, [WELLNESS])


# ── he is back: one short recap, not a stream of stale status ──────────────
class ReturnRecapTests(_Base):
    def setUp(self):
        super().setUp()
        self.input_age[0] = 3.0      # he just sat down

    def test_stale_status_becomes_one_recap_and_the_nudge_expires(self):
        fresh = "Your print is 40 percent done, sir."
        self._write([self._e(WELLNESS, "wellness", age_s=40 * 60),
                     self._e(CREDITS, "credits", age_s=32 * 60),
                     self._e(PULSE, "pulse", age_s=29 * 60),
                     self._e(fresh, "bambu", age_s=60)])
        # They waited because he was away (the recap says so only then).
        self.input_age[0] = float("inf")
        _, log = self._drain()
        self.assertEqual(self.spoke, [])
        self.assertIn("holding queued speech: owner away", log)
        self.input_age[0] = 3.0      # he sat down
        _, log = self._drain()
        self.assertEqual(len(self.spoke), 2, self.spoke)
        recap = self.spoke[0]
        self.assertTrue(recap.startswith("While you were away, sir:"), recap)
        self.assertIn("credits check needs a login", recap)
        self.assertIn("GPU pinned at 100 percent", recap)
        self.assertNotIn("?", recap)
        self.assertNotIn("Hydration", " ".join(self.spoke))
        self.assertEqual(self.spoke[1], fresh)
        self.assertIn("[pending] expired while away (wellness)", log)
        self.assertEqual(self._read(), [])

    def test_a_stale_briefing_is_spoken_whole(self):
        brief = "Good evening, sir. 12 voice interactions logged today."
        self._write([self._e(brief, "evening", age_s=45 * 60)])
        self._drain()
        self.assertEqual(self.spoke, [brief])

    def test_fresh_lines_are_spoken_as_they_are(self):
        self._write([self._e(PULSE, "pulse", age_s=120)])
        self._drain()
        self.assertEqual(self.spoke, [PULSE])


# ── the helpers and the wiring ─────────────────────────────────────────────
class HelperTests(_Base):
    def test_physical_input_comes_from_the_injected_input_aware_watcher(self):
        # Not GetLastInputInfo (the OS idle timer counts injected input — an
        # automation driving the PC looked like the owner at 19:05).
        self.input_age[0] = 12.0
        self.assertEqual(self.bc._physical_input_age_s(), 12.0)
        src = inspect.getsource(self.bc._physical_input_age_s)
        self.assertNotIn("GetLastInputInfo", src)
        self.assertIn("seconds_since_real_input", src)

    def test_no_watcher_means_unknown_input(self):
        self._p(self.bc, "_yield_watch_mod", return_value=None)
        self.assertEqual(self.bc._physical_input_age_s(), float("inf"))

    def test_a_broken_watcher_means_unknown_input(self):
        class _Bad:
            @staticmethod
            def seconds_since_real_input():
                raise OSError("hook gone")
        self._p(self.bc, "_yield_watch_mod", return_value=_Bad)
        self.assertEqual(self.bc._physical_input_age_s(), float("inf"))

    def test_watch_start_installs_the_watcher_and_arms(self):
        self.bc._presence_gate_armed[0] = False
        installs = []

        class _Watch:
            @staticmethod
            def install():
                installs.append(1)
                return True
        self._p(self.bc, "_yield_watch_mod", return_value=_Watch)
        self.assertTrue(self.bc._presence_watch_start())
        self.assertEqual(installs, [1])
        self.assertTrue(self.bc._presence_gate_armed[0])

    def test_watch_start_honours_the_kill_switch(self):
        self.bc._presence_gate_armed[0] = False
        self._p(self.bc, "PRESENCE_HOLD_ENABLED", False)
        watch = mock.MagicMock()
        self._p(self.bc, "_yield_watch_mod", return_value=watch)
        self.assertFalse(self.bc._presence_watch_start())
        watch.install.assert_not_called()
        self.assertFalse(self.bc._presence_gate_armed[0])


H2D = ("Slight problem, sir — your H2D appears to be unwell. Error code "
       "0300-0100 on layer 142.")


class UrgentTests(_Base):
    """Review repair (2026-10-02): a print failure or a printer error code
    waited behind the hold like a status line - while the owner's own video
    played, indefinitely. An entry queued urgent (proactive_announce(...,
    urgent=True)) is spoken through the away and room-talk holds."""

    def test_an_urgent_line_is_spoken_into_the_hold_and_the_rest_waits(self):
        self._write([self._e(WELLNESS, "wellness"),
                     dict(self._e(H2D, "bambu"), urgent=True)])
        self._drain()
        self.assertEqual(self.spoke, [H2D])
        self.assertEqual([e["source"] for e in self._read()], ["wellness"])

    def test_an_urgent_line_is_spoken_over_room_talk(self):
        self._owner_spoke(60)
        self.bc._note_room_talk("so anyway I told him we'd be there by eight")
        self._write([dict(self._e(H2D, "bambu"), urgent=True)])
        self._drain()
        self.assertEqual(self.spoke, [H2D])

    def test_the_same_line_without_the_flag_still_waits(self):
        self._write([self._e(H2D, "bambu")])
        self._drain()
        self.assertEqual(self.spoke, [])

    def test_proactive_announce_stores_the_flag(self):
        self.assertTrue(self.bc.proactive_announce(H2D, source="bambu",
                                                   urgent=True))
        self.assertTrue(self.bc.proactive_announce(WELLNESS,
                                                   source="wellness"))
        got = self._read()
        self.assertIs(got[0].get("urgent"), True)
        self.assertNotIn("urgent", got[1])


class RecapWordingTests(_Base):
    """Review repair (2026-10-02): every drain after a hold said "While you
    were away, sir" - after room talk too, when he never left."""

    def test_after_room_talk_the_recap_does_not_say_away(self):
        self._owner_spoke(60)
        self.bc._note_room_talk("so anyway I told him we'd be there by eight")
        self._write([self._e(PULSE, "pulse", age_s=30 * 60)])
        _, log = self._drain()
        self.assertIn("holding queued speech: room talk", log)
        self.mono[0] += self.bc.ROOM_TALK_HOLD_S + 1
        self._owner_spoke(60)
        self._drain()
        self.assertEqual(len(self.spoke), 1, self.spoke)
        self.assertTrue(self.spoke[0].startswith("Earlier, sir:"),
                        self.spoke[0])

    def test_after_he_was_away_it_does(self):
        self._write([self._e(PULSE, "pulse", age_s=30 * 60)])
        _, log = self._drain()
        self.assertIn("holding queued speech: owner away", log)
        self.input_age[0] = 3.0
        self._drain()
        self.assertTrue(self.spoke[0].startswith("While you were away, sir:"),
                        self.spoke)


class RoomTalkMediaTests(_Base):
    """Review repair (2026-10-02): in wake-word mode every lyric and video
    line of three words or more was dropped at the background-audio gate and
    stamped "room talk", so while his music or a video played every queued
    line waited until 45 s after it stopped. With the PC playing audio, a
    dropped line is room talk only in HIS voice; a line the speech filter
    rejects is never room talk."""

    LYRIC = "and the night is young and the lights are on"

    def setUp(self):
        super().setUp()
        import numpy as np
        self._owner_spoke(60)
        self._p(self.bc, "_last_capture_audio",
                np.zeros(16000 * 3, dtype=np.float32))
        self._p(self.bc, "_last_capture_sr", 16000)
        self._p(self.bc, "_room_talk_spawn",
                side_effect=lambda fn, *a: fn(*a), create=True)
        self.voice = self._p(self.bc, "_learn_voice_verdict",
                             return_value=(self.bc._learn_gate_mod.NOT_OWNER,
                                           0.43))

    def _talk(self, text, peak, conf=None):
        with mock.patch.object(self.bc, "_media_probe_result",
                               return_value=peak, create=True):
            if conf is None:
                self.bc._note_room_talk(text)
            else:
                self.bc._note_room_talk(text, conf)

    def test_media_playing_and_not_his_voice_is_not_room_talk(self):
        self._talk(self.LYRIC, peak=0.4)
        self._write([self._e(WELLNESS, "wellness")])
        self._drain()
        self.assertEqual(self.spoke, [WELLNESS])

    def test_his_own_voice_over_the_media_is_room_talk(self):
        self.voice.return_value = (self.bc._learn_gate_mod.OWNER, 0.81)
        self._talk("so anyway I told him we'd be there by eight", peak=0.4)
        self._write([self._e(WELLNESS, "wellness")])
        self._drain()
        self.assertEqual(self.spoke, [])

    def test_no_media_is_room_talk_without_a_voice_check(self):
        self._talk("so anyway I told him we'd be there by eight", peak=0.0)
        self.voice.assert_not_called()
        self._write([self._e(WELLNESS, "wellness")])
        self._drain()
        self.assertEqual(self.spoke, [])

    def test_a_line_the_speech_filter_rejects_is_not_room_talk(self):
        self._talk("so anyway I told him we'd be there by eight", peak=0.0,
                   conf={"no_speech_prob": 0.97, "avg_logprob": -2.4})
        self._write([self._e(WELLNESS, "wellness")])
        self._drain()
        self.assertEqual(self.spoke, [WELLNESS])


class PolledInputTests(_Base):
    """Review repair (2026-10-02). "Injected input never counts" held only
    for the low-level hook, and the hook is off by default
    (AIR_MOUSE_LL_HOOK_ENABLED False): every October live session logged
    "LL hook unavailable ... falling back to GetLastInputInfo polling". On
    that fallback the OS idle timer counts AFK-Helper / automation input as
    the owner's, so the 19:05 empty-room nudge would happen again. Driven
    through the REAL skills._air_mouse_yield in fallback mode, with the OS
    idle timer faked to "input 5 s ago" (the automation)."""

    def setUp(self):
        super().setUp()
        from skills import _air_mouse_yield as y
        self.y = y
        for name, val in (("_installed", True), ("_hook_ok", False),
                          ("_last_self_action_wall", float("-inf")),
                          ("_last_real_input", float("-inf")),
                          ("_last_hook_event", float("-inf"))):
            self._p(y, name, val)
        self._p(y, "_os_last_input_age_s", return_value=5.0)
        self._p(self.bc, "_yield_watch_mod", return_value=y)

    def test_polled_input_alone_is_not_presence(self):
        self.assertEqual(self.bc._physical_input_age_s(), float("inf"))
        self.assertFalse(self.bc._owner_present())

    def test_the_19_05_nudge_waits_in_the_empty_room(self):
        self._write([self._e(WELLNESS, "wellness")])
        _, log = self._drain()
        self.assertEqual(self.spoke, [], "an automation's input is not him")
        self.assertIn("holding queued speech: owner away", log)

    def test_polled_input_backed_by_his_voice_counts(self):
        self._owner_spoke(12 * 60)     # voice alone is stale by now
        self.assertLess(self.bc._physical_input_age_s(), 60.0)
        self.assertTrue(self.bc._owner_present())

    def test_polled_input_backed_by_a_face_counts(self):
        self.bc.last_face_seen = time.time() - 5 * 60
        self.assertLess(self.bc._physical_input_age_s(), 60.0)

    def test_old_backing_does_not_count(self):
        self._owner_spoke(60 * 60)
        self.bc.last_face_seen = time.time() - 60 * 60
        self.assertEqual(self.bc._physical_input_age_s(), float("inf"))

    def test_a_live_hook_needs_no_backing(self):
        now = time.monotonic()
        self._p(self.y, "_hook_ok", True)
        self._p(self.y, "_last_hook_event", now)
        self._p(self.y, "_last_real_input", now - 3.0)
        self.assertEqual(self.y.input_source(), "hook")
        self.assertLess(self.bc._physical_input_age_s(), 30.0)

    def test_input_source_names_the_fallback(self):
        self.assertEqual(self.y.input_source(), "poll")
        self._p(self.y, "_installed", False)
        self.assertEqual(self.y.input_source(), "none")


class WiringTests(_Base):
    """Source-level: main() cannot run in a test."""

    def test_the_bg_gate_drop_stamps_room_talk_before_it_continues(self):
        src = inspect.getsource(self.bc.main)
        drop = src.index("— ignoring non-wake ")
        stamp = src.index("_note_room_talk(text, conf)")
        self.assertLess(drop, stamp)
        self.assertLess(stamp, src.index("continue", stamp))
        self.assertLess(src.index("_bg_gate_for_turn("), stamp)

    def test_main_arms_the_hold_before_the_loop(self):
        src = inspect.getsource(self.bc.main)
        self.assertEqual(src.count("_presence_watch_start()"), 1)
        # Armed once, at the end of the boot, before the loop's first pass.
        self.assertLess(src.index("_start_boot_reprime()"),
                        src.index("_presence_watch_start()"))
        self.assertLess(src.index("_presence_watch_start()"),
                        src.index("if _blue_green_loop_tick():"))

    def test_config_defaults(self):
        import core.config as cfg
        self.assertIs(cfg.PRESENCE_HOLD_ENABLED, True)
        self.assertGreaterEqual(cfg.ROOM_TALK_HOLD_S, 30)
        self.assertLessEqual(cfg.ROOM_TALK_HOLD_S, 60)
        self.assertGreaterEqual(cfg.PRESENCE_STALE_STATUS_S, 300)
        # The live 19:05 empty room: 38.5 min since physical input, 45 min
        # since his voice - both far outside the windows.
        self.assertLess(cfg.OWNER_PRESENT_INPUT_WINDOW_S, 38 * 60)
        self.assertLess(cfg.OWNER_PRESENT_VOICE_WINDOW_S, 38 * 60)


if __name__ == "__main__":
    unittest.main()
