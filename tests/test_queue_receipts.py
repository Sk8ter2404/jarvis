"""Lines he summoned and lines he missed (2026-11-17 overhaul, the proactive
cluster: JARVIS-Research/review_20261117_overhaul_audit.json proactive-02,
-03 and -06).

  * core/owner_presence.SUMMONED_RULES / summoned_passes: his clap's answer
    passes the room-talk hold (and the gap after a budget deferral) for its
    few seconds while a face or his voice says he is here and the room is
    quiet; the arrival briefing passes the room-talk and media holds while
    he is present. Neither ever passes "owner away" or "gaming".
  * core/queue_receipts: the drain tells the skill that queued a line what
    became of it, and remembers what was dropped unsaid so his "did
    anything happen while I was away" can name it.
  * core/owner_presence.away_answer_line(missed=...): the answer names it.

Pure logic, no monolith. Synthetic lines.

    python tools/run_tests.py test_queue_receipts
"""
from __future__ import annotations

import time
import unittest

from core import owner_presence as op
from core import queue_receipts as qr


def _yes():
    return True


def _no():
    return False


def _boom():
    raise RuntimeError("sensor gone")


class SummonedPassesTests(unittest.TestCase):
    CLAP = {"ts": 0, "message": "You rang, sir?", "source": "clap",
            "summoned": "clap"}
    ARRIVAL = {"ts": 0, "message": "[intent:briefing] Good morning, sir.",
               "source": "arrival_v2", "summoned": "arrival"}

    def _passes(self, entry, hold, *, age=2.0, window=10.0, present=_yes,
                seen=_yes, quiet=_yes):
        return op.summoned_passes(entry, hold_kind=hold, age_s=age,
                                  window_s=window, present=present,
                                  seen_or_heard=seen, room_quiet=quiet)

    def test_kind_is_read_only_for_known_kinds(self):
        self.assertEqual(op.summoned_kind(self.CLAP), "clap")
        self.assertEqual(op.summoned_kind(self.ARRIVAL), "arrival")
        self.assertEqual(op.summoned_kind({"summoned": "banter"}), "")
        self.assertEqual(op.summoned_kind({"summoned": 3}), "")
        self.assertEqual(op.summoned_kind({"message": "x"}), "")
        self.assertEqual(op.summoned_kind("not an entry"), "")

    def test_clap_passes_room_talk_when_seen_and_quiet(self):
        self.assertTrue(self._passes(self.CLAP, "room talk"))
        self.assertTrue(self._passes(self.CLAP, "after the drain budget"))

    def test_clap_never_passes_away_media_or_gaming(self):
        for hold in ("owner away", "media playing", "gaming", ""):
            self.assertFalse(self._passes(self.CLAP, hold), hold)

    def test_clap_needs_his_face_or_voice_not_just_input(self):
        self.assertFalse(self._passes(self.CLAP, "room talk", seen=_no))

    def test_clap_waits_over_music(self):
        self.assertFalse(self._passes(self.CLAP, "room talk", quiet=_no))

    def test_arrival_passes_room_talk_and_media_while_present(self):
        for hold in ("room talk", "media playing"):
            self.assertTrue(self._passes(self.ARRIVAL, hold, age=60,
                                         window=600, seen=_no, quiet=_no),
                            hold)

    def test_arrival_never_passes_away_or_gaming(self):
        for hold in ("owner away", "gaming", "after the drain budget"):
            self.assertFalse(self._passes(self.ARRIVAL, hold, age=60,
                                          window=600), hold)

    def test_nobody_here_holds_both(self):
        self.assertFalse(self._passes(self.CLAP, "room talk", present=_no))
        self.assertFalse(self._passes(self.ARRIVAL, "room talk", age=60,
                                      window=600, present=_no))

    def test_the_window_ends_it_and_zero_is_off(self):
        self.assertFalse(self._passes(self.CLAP, "room talk", age=10.5))
        self.assertFalse(self._passes(self.CLAP, "room talk", window=0))
        self.assertFalse(self._passes(self.ARRIVAL, "media playing",
                                      age=601, window=600))
        self.assertFalse(self._passes(self.CLAP, "room talk",
                                      age=float("nan")))

    def test_a_plain_line_never_passes(self):
        plain = {"ts": 0, "message": "Memory at 91 percent, sir.",
                 "source": "system_monitor"}
        self.assertFalse(self._passes(plain, "room talk"))

    def test_a_failing_reading_holds_the_line(self):
        self.assertFalse(self._passes(self.CLAP, "room talk", quiet=_boom))
        self.assertFalse(self._passes(self.CLAP, "room talk", present=_boom))

    def test_readings_are_taken_only_when_needed(self):
        # The arrival kind never asks the room (a meter read is not free).
        self.assertTrue(self._passes(self.ARRIVAL, "room talk", age=1,
                                     window=600, seen=_boom, quiet=_boom))


def _row(msg, src, cls, at=None):
    return (time.time() if at is None else at,
            {"ts": 0, "message": msg, "source": src}, cls)


class QueueReceiptsTests(unittest.TestCase):
    def setUp(self):
        qr.reset()
        self.addCleanup(qr.reset)

    def test_the_hook_of_the_queuing_source_hears_both(self):
        heard, dropped = [], []
        qr.register("arrival_v2", on_spoken=heard.append,
                    on_dropped=lambda e, why: dropped.append((e, why)))
        line = {"ts": 1, "message": "Good morning, sir.",
                "source": "arrival_v2"}
        qr.note_spoken(line)
        qr.note_dropped(line, "briefing: waited 34031 s, limit 21600 s")
        qr.note_spoken({"source": "weather", "message": "x"})
        self.assertEqual(heard, [line])
        self.assertEqual(dropped, [(line, "briefing: waited 34031 s, "
                                          "limit 21600 s")])
        qr.unregister("arrival_v2")
        qr.note_spoken(line)
        self.assertEqual(len(heard), 1)

    def test_a_raising_hook_never_reaches_the_drain(self):
        def bad(*_a):
            raise ValueError("disk full")
        qr.register("arrival_v2", on_spoken=bad, on_dropped=bad)
        line = {"ts": 1, "message": "Good morning, sir.",
                "source": "arrival_v2"}
        qr.note_spoken(line)
        qr.note_dropped(line, "x")
        # ...and the drop is still remembered for his away question.
        self.assertEqual(len(qr.missed()), 1)

    def test_drops_are_remembered_but_not_moments_or_acks(self):
        qr.note_dropped({"ts": 1, "source": "arrival_v2",
                         "message": "[intent:briefing] Good morning, sir."})
        qr.note_dropped({"ts": 1, "source": "system_monitor",
                         "message": "Sir, memory usage is at 91 percent."})
        qr.note_dropped({"ts": 1, "source": "clap",
                         "message": "You rang, sir?"})
        qr.note_dropped({"ts": 1, "source": "ambient-listen",
                         "message": "Yes, listening, sir."})
        qr.note_dropped({"ts": 1, "source": "audio",
                         "message": "Sir, the mic has been completely silent "
                                    "for 30 seconds."})
        qr.note_dropped({"ts": 1, "source": "wellness",
                         "message": "Hydration, sir?"}, "expired while away",
                        remember=False)
        qr.note_dropped("not an entry")
        self.assertEqual([m[2] for m in qr.missed()], ["briefing", "ram"])

    def test_take_forgets_and_old_drops_age_out(self):
        now = time.time()
        qr.note_dropped({"ts": 1, "source": "weather",
                         "message": "Rain at four, sir."},
                        now=now - qr.MISSED_KEEP_S - 5)
        qr.note_dropped({"ts": 1, "source": "system_monitor",
                         "message": "Sir, memory usage is at 91 percent."},
                        now=now)
        self.assertEqual([m[2] for m in qr.take_missed(now)], ["ram"])
        self.assertEqual(qr.take_missed(now), [])

    def test_the_list_is_bounded(self):
        for n in range(qr.MISSED_CAP + 7):
            qr.note_dropped({"ts": 1, "source": f"src{n}",
                             "message": f"Notice {n}, sir."})
        rows = qr.missed()
        self.assertEqual(len(rows), qr.MISSED_CAP)
        self.assertEqual(rows[-1][1]["message"],
                         f"Notice {qr.MISSED_CAP + 6}, sir.")

    def test_the_phrase_names_briefings_and_counts_the_rest(self):
        rows = [_row("[intent:briefing] Good morning, sir.", "arrival_v2",
                     "briefing"),
                _row("[intent:briefing] Good morning, sir.", "daily",
                     "briefing")]
        rows += [_row("Sir, memory usage is at 91 percent.", "system_monitor",
                      "ram") for _ in range(4)]
        rows += [_row("Switched to the desk mic.", "audio", "mic_change"),
                 _row("Good evening, sir.", "evening", "briefing"),
                 _row("A parcel is out for delivery, sir.", "amazon",
                      "other")]
        self.assertEqual(
            qr.missed_phrase(rows),
            "the morning briefing, the evening briefing, four memory "
            "warnings, a microphone change and one other notice")
        self.assertEqual(qr.missed_phrase(rows[:1]), "the morning briefing")
        self.assertEqual(qr.missed_phrase([]), "")
        self.assertEqual(qr.missed_phrase(None), "")
        self.assertEqual(qr.missed_phrase([("bad",)]), "")


class AwayAnswerMissedTests(unittest.TestCase):
    MISSED = "the morning briefing and two memory warnings"

    def test_after_the_lines_it_names_what_went_unsaid(self):
        line = op.away_answer_line(spoke=True, missed=self.MISSED)
        self.assertEqual(line, op.AWAY_MISSED_LINE.format(missed=self.MISSED))
        self.assertIn("the morning briefing and two memory warnings", line)

    def test_with_nothing_waiting_it_says_so_and_names_them(self):
        line = op.away_answer_line(spoke=False, missed=self.MISSED)
        self.assertEqual(line,
                         op.AWAY_ONLY_MISSED_LINE.format(missed=self.MISSED))
        self.assertNotEqual(line, op.AWAY_NOTHING_WAITING_LINE)

    def test_a_repeated_recap_and_held_lines_keep_it_after(self):
        recap = "While you were away, sir: the print finished."
        self.assertEqual(
            op.away_answer_line(spoke=False, last_recap=recap,
                                last_recap_age_s=60, missed=self.MISSED),
            recap + " " + op.AWAY_MISSED_LINE.format(missed=self.MISSED))
        self.assertTrue(op.away_answer_line(
            spoke=False, still_queued=True, missed=self.MISSED).startswith(
                op.AWAY_STILL_HELD_LINE))

    def test_nothing_missed_is_the_old_answer(self):
        self.assertEqual(op.away_answer_line(spoke=True, missed=""), "")
        self.assertEqual(op.away_answer_line(spoke=False),
                         op.AWAY_NOTHING_WAITING_LINE)


if __name__ == "__main__":
    unittest.main()
