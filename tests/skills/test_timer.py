"""Logic tests for skills/timer.py.

Exemplar for per-skill logic tests: drive registered actions through the
isolation harness (no monolith), control threads so no real timers spawn, and
mock the cross-module speech enqueue. Covers pure functions, error paths,
happy paths + state, and the blue/green restore path.
"""
from __future__ import annotations

import contextlib
import io
import time
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated, no_background_threads


def _call_silently(fn, *a, **kw):
    """Invoke fn with stdout swallowed — the _fire closures print a 🔔 line
    that crashes on a cp1252 console when called directly outside the loader."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


class TimerSkillTests(unittest.TestCase):
    def setUp(self):
        # Fresh module + actions per test; timer keeps module-global state, and
        # re-exec gives a clean _timers/_next_id, but reset defensively.
        self.mod, self.actions = load_skill_isolated("timer")
        self.mod._timers.clear()
        self.mod._next_id[0] = 1

    # ── _parse_duration (pure) ───────────────────────────────────────────
    def test_parse_duration_units(self):
        p = self.mod._parse_duration
        self.assertEqual(p("30 seconds"), 30)
        self.assertEqual(p("5 minutes"), 300)
        self.assertEqual(p("2 hours"), 7200)
        self.assertEqual(p("1 day"), 86400)
        self.assertEqual(p("90 secs"), 90)
        self.assertEqual(p("1 hour 30 minutes"), 5400)  # compound

    def test_parse_duration_natural_forms(self):
        # The shapes the local LLM actually emits — bug 1 root cause was that
        # only the pipe form was accepted. These must all parse.
        p = self.mod._parse_duration
        self.assertEqual(p("10 min"), 600)          # abbrev unit
        self.assertEqual(p("10m"), 600)             # bare-letter unit after digit
        self.assertEqual(p("1h"), 3600)
        self.assertEqual(p("30s"), 30)
        self.assertEqual(p("five minutes"), 300)    # spelled-out number
        self.assertEqual(p("ten min"), 600)
        self.assertEqual(p("an hour"), 3600)        # 'an'/'a' → 1
        self.assertEqual(p("half an hour"), 1800)   # 'half a <unit>'
        self.assertEqual(p("an hour and a half"), 5400)  # '<unit> and a half'
        self.assertEqual(p("5"), 300)               # bare number → minutes
        self.assertEqual(p("set a timer for 5"), 300)  # 'for' noise stripped

    def test_parse_duration_clock_times(self):
        # Regression (2026-06-05, live): "set a reminder for 8 pm" fell through
        # to the bare-number rule and became an 8-MINUTE timer (the 'pm' was
        # dropped). A clock time must instead schedule for that ABSOLUTE time
        # (seconds until it, today or tomorrow), never 8 minutes. Asserted as
        # ranges (not exact) so it's timezone-independent across dev/CI.
        p = self.mod._parse_duration
        for clock in ("8 pm", "set a reminder for 8 pm to call mom",
                      "remind me at 7:30 am", "at 20:00", "12 pm"):
            r = p(clock)
            self.assertIsNotNone(r, f"clock time should parse: {clock!r}")
            self.assertGreater(r, 0)
            self.assertLessEqual(r, 86400)               # within a day
            self.assertNotEqual(r, 8 * 60, f"{clock!r} must not be an 8-min timer")
        # Durations and bare numbers are UNAFFECTED (no am/pm, no HH:MM colon).
        self.assertEqual(p("8"), 480)                    # bare 8 -> 8 minutes
        self.assertEqual(p("5 minutes"), 300)
        self.assertEqual(p("90 secs"), 90)

    def test_parse_duration_invalid(self):
        self.assertIsNone(self.mod._parse_duration("soon"))
        self.assertIsNone(self.mod._parse_duration(""))
        self.assertIsNone(self.mod._parse_duration("whenever"))
        self.assertIsNone(self.mod._parse_duration("minutes"))  # unit, no number

    # ── set_timer: BUG 1 — natural args set a REAL timer + truthful reply ──
    def test_set_timer_bare_duration_creates_real_timer(self):
        # "set a timer for 5 minutes" → LLM emits [ACTION: set_timer, 5 minutes].
        # The bug: the bare duration was rejected (no timer) yet JARVIS claimed
        # success. Now it must actually create the timer and confirm honestly.
        with no_background_threads():
            out = self.actions["set_timer"]("5 minutes")
        self.assertEqual(len(self.mod._timers), 1)   # a REAL timer exists
        self.assertIn("#1", out)
        self.assertIn("5m", out)

    def test_set_timer_unparseable_returns_honest_error_no_timer(self):
        # Genuinely unparseable → HONEST error, and NO timer created (so JARVIS
        # never claims a false success).
        out = self.actions["set_timer"]("whenever")
        self.assertEqual(len(self.mod._timers), 0)
        low = out.lower()
        self.assertIn("couldn't tell how long", low)
        # Must NOT look like a success confirmation.
        self.assertNotIn("timer #", low)
        self.assertNotIn("will remind you", low)

    def test_set_timer_natural_label_after_duration(self):
        with no_background_threads():
            out = self.actions["set_timer"]("5 minutes for tea")
        self.assertEqual(len(self.mod._timers), 1)
        self.assertIn("tea", out)
        # the stored message carries the label
        _, msg, _ = self.mod._timers[1]
        self.assertEqual(msg, "tea")

    def test_set_timer_label_then_duration(self):
        with no_background_threads():
            self.actions["set_timer"]("tea timer 5 minutes")
        _, msg, _ = self.mod._timers[1]
        self.assertEqual(msg, "tea")

    def test_set_timer_unlabeled_reply_omits_placeholder(self):
        # A bare duration gets a generic stored message but the spoken reply
        # should NOT quote the placeholder back.
        with no_background_threads():
            out = self.actions["set_timer"]("5 minutes")
        self.assertNotIn("your timer is up", out)

    def test_set_timer_empty_label_pipe_still_works(self):
        # Legacy 'duration | ' with an empty message must still set a timer
        # (was previously rejected as "needs a message").
        with no_background_threads():
            out = self.actions["set_timer"]("5 minutes | ")
        self.assertEqual(len(self.mod._timers), 1)
        self.assertIn("#1", out)

    # ── set_timer happy path (threads neutered) ──────────────────────────
    def test_set_timer_happy(self):
        with no_background_threads():
            out = self.actions["set_timer"]("5 minutes | check the oven")
        self.assertIn("#1", out)
        self.assertIn("5m", out)
        self.assertIn("check the oven", out)
        self.assertEqual(len(self.mod._timers), 1)

    def test_list_and_cancel(self):
        with no_background_threads():
            self.actions["set_timer"]("10 minutes | tea")
            self.actions["set_timer"]("20 minutes | walk")
        listed = self.actions["list_timers"]()
        self.assertIn("2 active timer", listed)
        self.assertIn("tea", listed)

        self.assertIn("cancelled timer #1", self.actions["cancel_timer"]("1"))
        self.assertEqual(len(self.mod._timers), 1)
        self.assertIn("cancelled", self.actions["cancel_timer"]("all"))
        self.assertEqual(len(self.mod._timers), 0)

    def test_cancel_unknown(self):
        self.assertIn("no timer", self.actions["cancel_timer"]("999"))

    def test_list_empty(self):
        self.assertEqual(self.actions["list_timers"](), "no active timers")

    # ── restore_timers (blue/green handoff) ──────────────────────────────
    def test_restore_past_timer_fires_immediately(self):
        with mock.patch.object(self.mod, "_enqueue_speech") as enq:
            n = self.mod.restore_timers(
                [{"id": 5, "message": "stretch", "fire_at": 1.0}])
        self.assertEqual(n, 1)
        enq.assert_called_once()
        self.assertIn("stretch", enq.call_args[0][0])

    def test_restore_future_timer_rearms(self):
        with no_background_threads():
            n = self.mod.restore_timers(
                [{"id": 7, "message": "later", "fire_at": time.time() + 9999}])
        self.assertEqual(n, 1)
        self.assertIn(7, self.mod._timers)

    def test_restore_rejects_garbage(self):
        self.assertEqual(self.mod.restore_timers("not a list"), 0)
        self.assertEqual(self.mod.restore_timers([{"bad": "entry"}]), 0)

    def test_restore_skips_non_dict_and_dup_id(self):
        # non-dict entry skipped; duplicate of an already-present id skipped.
        with no_background_threads():
            self.actions["set_timer"]("99 minutes | existing")  # claims id #1
        future = time.time() + 9999
        n = self.mod.restore_timers([
            "not-a-dict",
            {"id": 1, "message": "dup", "fire_at": future},  # id already live
        ])
        self.assertEqual(n, 0)

    def test_restore_rejects_blank_message_and_bad_id(self):
        future = time.time() + 9999
        n = self.mod.restore_timers([
            {"id": 3, "message": "", "fire_at": future},   # blank msg
            {"id": 0, "message": "zero-id", "fire_at": future},  # tid <= 0
            {"id": "NaN", "message": "x", "fire_at": future},    # int() raises
        ])
        self.assertEqual(n, 0)

    def test_restore_future_fire_callback_enqueues_and_pops(self):
        # Rearm a future timer (no real thread), then invoke its fire callback
        # directly to cover the closure body (152-155).
        future = time.time() + 9999
        with no_background_threads():
            self.mod.restore_timers([{"id": 8, "message": "yoga", "fire_at": future}])
        timer_obj, msg, _ = self.mod._timers[8]
        with mock.patch.object(self.mod, "_enqueue_speech") as enq:
            _call_silently(timer_obj.function)  # the _fire closure
        enq.assert_called_once()
        self.assertIn("yoga", enq.call_args[0][0])
        self.assertNotIn(8, self.mod._timers)  # popped itself

    def test_restore_fire_on_restore_exception_is_swallowed(self):
        # _enqueue_speech raising during fire-on-restore must not abort the loop;
        # the entry still counts as restored.
        with mock.patch.object(self.mod, "_enqueue_speech",
                               side_effect=RuntimeError("boom")):
            n = _call_silently(self.mod.restore_timers,
                               [{"id": 4, "message": "past", "fire_at": 1.0}])
        self.assertEqual(n, 1)

    # ── set_timer fire closure + format branches ─────────────────────────
    def test_set_timer_fire_callback(self):
        with no_background_threads():
            self.actions["set_timer"]("5 minutes | call mom")
        timer_obj, msg, _ = self.mod._timers[1]
        with mock.patch.object(self.mod, "_enqueue_speech") as enq:
            _call_silently(timer_obj.function)  # the _fire closure (182-185)
        enq.assert_called_once()
        self.assertIn("call mom", enq.call_args[0][0])
        self.assertNotIn(1, self.mod._timers)

    def test_set_timer_seconds_format(self):
        with no_background_threads():
            out = self.actions["set_timer"]("45 seconds | quick")
        self.assertIn("45s", out)  # secs < 60 branch (195)

    def test_set_timer_hours_format(self):
        with no_background_threads():
            out = self.actions["set_timer"]("2 hours 15 minutes | long")
        self.assertIn("2h 15m", out)  # secs >= 3600 branch (199)

    def test_set_timer_minutes_exact(self):
        with no_background_threads():
            out = self.actions["set_timer"]("10 minutes | exact")
        self.assertIn("10m", out)  # no leftover seconds branch

    # ── list_timers remaining-string branches ────────────────────────────
    def test_list_timers_remaining_formats(self):
        now = 1_000_000.0
        sentinel = mock.MagicMock()
        # Inject timers directly so we control fire_at precisely without threads.
        self.mod._timers[1] = (sentinel, "soon", now + 30)        # <60 → Ns
        self.mod._timers[2] = (sentinel, "mid", now + 90)         # <3600 → Nm Ns
        self.mod._timers[3] = (sentinel, "far", now + 7200)       # >=3600 → Nh Nm
        # Freeze the clock so remaining-time math is exact, not flaky.
        with mock.patch.object(self.mod.time, "time", return_value=now):
            out = self.actions["list_timers"]()
        self.assertIn("30s", out)
        self.assertIn("1m 30s", out)
        self.assertIn("2h 0m", out)

    # ── cancel_timer: BUG 1 — natural args, honest "no timers" ───────────
    def test_cancel_no_arg_cancels_the_running_one(self):
        # "cancel my timer" with one running → cancel it and say which.
        with no_background_threads():
            self.actions["set_timer"]("10 minutes | tea")
        out = self.actions["cancel_timer"]("")
        self.assertEqual(len(self.mod._timers), 0)
        self.assertIn("cancelled timer #1", out.lower())

    def test_cancel_loose_phrase_cancels_running(self):
        with no_background_threads():
            self.actions["set_timer"]("10 minutes | tea")
        out = self.actions["cancel_timer"]("my timer")
        self.assertEqual(len(self.mod._timers), 0)
        self.assertIn("cancelled", out.lower())

    def test_cancel_no_timers_is_honest(self):
        # No timers exist → honest message, NOT a false "cancelled".
        out = self.actions["cancel_timer"]("")
        low = out.lower()
        self.assertIn("no timers", low)
        self.assertNotIn("cancelled timer", low)

    def test_cancel_no_arg_multiple_cancels_soonest(self):
        # Several running, loose arg → cancel the soonest-to-fire and note
        # the rest remain.
        with no_background_threads():
            self.actions["set_timer"]("30 minutes | far")   # #1, later
            self.actions["set_timer"]("5 minutes | soon")   # #2, sooner
        out = self.actions["cancel_timer"]("the timer")
        self.assertIn("#2", out)                 # the soonest one
        self.assertIn(1, self.mod._timers)       # the later one survives
        self.assertNotIn(2, self.mod._timers)

    def test_cancel_by_label_substring(self):
        with no_background_threads():
            self.actions["set_timer"]("10 minutes | tea")
            self.actions["set_timer"]("20 minutes | laundry")
        out = self.actions["cancel_timer"]("the tea timer")
        self.assertIn("cancelled timer #1", out.lower())
        self.assertIn(2, self.mod._timers)

    def test_cancel_unmatched_label_is_honest(self):
        with no_background_threads():
            self.actions["set_timer"]("10 minutes | tea")
        out = self.actions["cancel_timer"]("the pizza timer")
        low = out.lower()
        self.assertIn("don't see a timer", low)
        self.assertEqual(len(self.mod._timers), 1)  # nothing cancelled


class TimerEnqueueSpeechTests(unittest.TestCase):
    """_enqueue_speech: proactive_announce route + atomic-file fallback."""

    def setUp(self):
        self.mod, self.actions = load_skill_isolated("timer")

    def test_enqueue_via_proactive_announce(self):
        fake_bc = mock.MagicMock()
        with mock.patch("importlib.import_module", return_value=fake_bc):
            self.mod._enqueue_speech("hello")
        fake_bc.proactive_announce.assert_called_once_with("hello", source="timer")

    def test_enqueue_falls_back_to_file_when_announcer_absent(self):
        import json as _json
        import os as _os
        import tempfile

        # import_module returns an object WITHOUT proactive_announce → fallback.
        bc_no_announce = mock.MagicMock(spec=[])
        with tempfile.TemporaryDirectory() as d:
            qpath = _os.path.join(d, "pending_speech.json")
            with mock.patch("importlib.import_module", return_value=bc_no_announce), \
                 mock.patch.object(self.mod, "_SPEECH_QUEUE", qpath):
                self.mod._enqueue_speech("file-route")
            with open(qpath, encoding="utf-8") as f:
                data = _json.load(f)
        self.assertEqual(data[-1]["message"], "file-route")

    def test_enqueue_appends_to_existing_and_ignores_corrupt(self):
        import json as _json
        import os as _os
        import tempfile

        with mock.patch("importlib.import_module", side_effect=ImportError):
            with tempfile.TemporaryDirectory() as d:
                qpath = _os.path.join(d, "pending_speech.json")
                # Pre-seed corrupt JSON → the read except resets data to [].
                with open(qpath, "w", encoding="utf-8") as f:
                    f.write("{not json")
                with mock.patch.object(self.mod, "_SPEECH_QUEUE", qpath):
                    self.mod._enqueue_speech("after-corrupt")
                with open(qpath, encoding="utf-8") as f:
                    data = _json.load(f)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["message"], "after-corrupt")

    def test_enqueue_write_failure_falls_back_to_print(self):
        # import_module raises (parent not loaded) AND the atomic write fails →
        # the print fallback fires instead of losing the reminder.
        with mock.patch("importlib.import_module", side_effect=ImportError), \
             mock.patch.object(self.mod, "_atomic_write_json",
                               side_effect=OSError("disk full")), \
             mock.patch.object(self.mod, "os") as fake_os:
            fake_os.path.exists.return_value = False
            # Should not raise despite the write failing.
            _call_silently(self.mod._enqueue_speech, "doomed")


class TimerEnumerateTests(unittest.TestCase):
    """enumerate_timers snapshot for blue/green handoff."""

    def setUp(self):
        self.mod, self.actions = load_skill_isolated("timer")
        self.mod._timers.clear()
        self.mod._next_id[0] = 1

    def test_enumerate_empty(self):
        self.assertEqual(self.mod.enumerate_timers(), [])

    def test_enumerate_snapshots_active(self):
        with no_background_threads():
            self.actions["set_timer"]("30 minutes | a")
            self.actions["set_timer"]("60 minutes | b")
        snap = self.mod.enumerate_timers()
        self.assertEqual(len(snap), 2)
        self.assertEqual(snap[0]["id"], 1)
        self.assertEqual(snap[0]["message"], "a")
        self.assertIsInstance(snap[0]["fire_at"], float)


if __name__ == "__main__":
    unittest.main()
