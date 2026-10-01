"""Monolith-tier regressions for the proactive-speech queue (2026-10-01).

Three defects in how bobert_companion drains pending_speech.json:

  * B042 — the drainer's duplicate check compared TEXT, and every unlabelled
    timer says the same "Reminder, sir — your timer is up". A second timer
    due within 60 s of the first (or in the same drain) was logged as a
    "suppressed duplicate" and never spoken. Entries now carry a
    ``dedupe_key`` (the timer's number) that the check compares instead.
  * B048 — a queued line that ASKED something (a pattern offer, the recap's
    closing "Shall I ...?") was spoken but never recorded in
    conversation_history, so even "JARVIS, yes" reached the LLM with no
    question to answer.
  * B099 — the queue was drained only by the normal-mode capture, so a timer
    that fired while JARVIS was in sleep / standby waited for the next
    "JARVIS" and then played late. The standby loop now speaks the entries
    the owner asked for (timer / promise / schedule) and leaves the rest.

Every test redirects PENDING_SPEECH_PATH (and proactive_announce's
__file__-derived queue path) into a temp dir — the live queue is never read.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest import mock

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith


@requires_monolith
class _QueueBase(MonolithGlobalsTestCase):
    def setUp(self):
        bc = self.bc
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.queue = os.path.join(self._tmp.name, "pending_speech.json")
        self._p(bc, "PENDING_SPEECH_PATH", self.queue)
        # proactive_announce derives its queue from bobert_companion.__file__.
        self._p(bc, "__file__", os.path.join(self._tmp.name, "bobert_companion.py"))
        self.spoke: list[str] = []
        self._p(bc, "_speak", lambda msg, **k: self.spoke.append(msg))
        self._p(bc, "_heartbeat")
        self._p(bc, "_speech_hold_active", return_value=False)
        self._p(bc, "_audio_flap_flush")
        bc._recent_spoken_messages.clear()

    def _p(self, *args, **kwargs):
        patcher = mock.patch.object(*args, **kwargs)
        m = patcher.start()
        self.addCleanup(patcher.stop)
        return m

    def _write(self, entries):
        with open(self.queue, "w", encoding="utf-8") as f:
            json.dump(entries, f)

    def _read(self):
        if not os.path.exists(self.queue):
            return []
        with open(self.queue, encoding="utf-8") as f:
            return json.load(f)


class TimerReminderDedupeTests(_QueueBase):
    MSG = "Reminder, sir — your timer is up"

    def test_two_unlabelled_timers_in_one_drain_are_both_spoken(self):
        self.bc.proactive_announce(self.MSG, source="timer", dedupe_key="timer#1")
        self.bc.proactive_announce(self.MSG, source="timer", dedupe_key="timer#2")
        self.assertTrue(self.bc._speak_pending())
        self.assertEqual(self.spoke, [self.MSG, self.MSG])

    def test_a_second_timer_inside_the_60s_window_is_spoken(self):
        # "set a 30 second timer" -> fires -> "again" -> fires ~35 s later.
        self.bc.proactive_announce(self.MSG, source="timer", dedupe_key="timer#1")
        self.bc._speak_pending()
        self.bc.proactive_announce(self.MSG, source="timer", dedupe_key="timer#2")
        self.bc._speak_pending()
        self.assertEqual(self.spoke, [self.MSG, self.MSG])

    def test_the_same_timer_entry_twice_is_still_spoken_once(self):
        # The dedupe's real job survives: one entry recovered twice (a crashed
        # batch, a looping writer) is spoken once.
        self._write([{"message": self.MSG, "dedupe_key": "timer#7"},
                     {"message": self.MSG, "dedupe_key": "timer#7"}])
        self.bc._speak_pending()
        self.assertEqual(self.spoke, [self.MSG])

    def test_unkeyed_identical_toasts_still_collapse(self):
        self._write([{"message": "New toast"}, {"message": "New toast"}])
        self.bc._speak_pending()
        self.assertEqual(self.spoke, ["New toast"])

    def test_entry_records_source_and_key(self):
        self.bc.proactive_announce(self.MSG, source="timer", dedupe_key="timer#3")
        self.bc.proactive_announce("plain", source="weather")
        a, b = self._read()
        self.assertEqual((a["source"], a["dedupe_key"]), ("timer", "timer#3"))
        self.assertEqual(b["source"], "weather")
        self.assertNotIn("dedupe_key", b)


class QueuedQuestionContextTests(_QueueBase):
    OFFER = "Shall I queue your usual mix, sir?"

    def test_a_spoken_offer_joins_the_conversation(self):
        hist = self.bc.conversation_history
        hist[:] = [{"role": "user", "content": "what time is it"},
                   {"role": "assistant", "content": "It's 3 PM, sir."}]
        self._write([{"message": "[intent:briefing] " + self.OFFER,
                      "source": "anticipation"}])
        self.bc._speak_pending()
        # Joined onto the trailing assistant turn — never two in a row.
        self.assertEqual(len(hist), 2)
        self.assertEqual(hist[-1]["role"], "assistant")
        self.assertTrue(hist[-1]["content"].endswith(self.OFFER))
        self.assertNotIn("[intent:", hist[-1]["content"])
        # What _call_llm sends next ("yes") carries the question with it.
        hist.append({"role": "user", "content": "yes"})
        self.bc._trim_conversation_history()
        self.assertIn(self.OFFER, " ".join(m["content"] for m in hist))

    def test_after_a_user_turn_the_offer_is_its_own_assistant_entry(self):
        hist = self.bc.conversation_history
        hist[:] = [{"role": "user", "content": "hello"}]
        self._write([{"message": self.OFFER}])
        self.bc._speak_pending()
        self.assertEqual(hist[-1], {"role": "assistant", "content": self.OFFER})

    def test_statements_are_not_added(self):
        hist = self.bc.conversation_history
        hist[:] = [{"role": "user", "content": "hello"},
                   {"role": "assistant", "content": "Hello, sir."}]
        self._write([{"message": "Your print is done, sir."}])
        self.bc._speak_pending()
        self.assertEqual(hist[-1]["content"], "Hello, sir.")


class StandbyReminderDrainTests(_QueueBase):
    def setUp(self):
        super().setUp()
        bc = self.bc
        self._p(bc, "set_state")
        self._p(bc, "_input_backoff_quiet", return_value=True)
        # The capture after the drain hears nothing.
        self._p(bc, "record_speech", return_value=None)
        bc._sleep_mode[0] = True
        bc._standby_mode[0] = True

    def test_timer_spoken_while_asleep_and_chatter_kept_for_the_wake(self):
        self._write([
            {"message": "Weather alert, sir.", "source": "weather"},
            {"message": "Reminder, sir — tea", "source": "timer",
             "dedupe_key": "timer#1"},
            {"message": "Promise kept, sir.", "source": "promise:memory"},
        ])
        self.bc._handle_sleep_standby(None)
        self.assertEqual(self.spoke, ["Reminder, sir — tea", "Promise kept, sir."])
        # The weather line is still queued for the wake word, unspoken.
        self.assertEqual([e["message"] for e in self._read()],
                         ["Weather alert, sir."])
        self.assertFalse(os.path.exists(self.queue + ".consuming"))
        self.assertTrue(self.bc._sleep_mode[0])   # still asleep

    def test_the_broken_schedule_diagnostic_waits_for_the_wake(self):
        # 2026-10-01 merge audit: "scheduler" (core/scheduler.py _announce,
        # the no-such-action diagnostic) rode in the standby set and spoke
        # overnight. A scheduled job's own line is tagged "schedule".
        self.assertNotIn("scheduler", self.bc._STANDBY_SPEAKABLE_SOURCES)
        diag = {"message": "Sir, scheduled job j1 tried to run 'x', but "
                           "there's no such action registered. Nothing ran.",
                "source": "scheduler"}
        job = {"message": "Time for your vitamins, sir.", "source": "schedule"}
        self._write([diag, job])
        self.bc._handle_sleep_standby(None)
        self.assertEqual(self.spoke, [job["message"]])
        self.assertEqual(self._read(), [diag])

    def test_nothing_owner_requested_leaves_the_queue_untouched(self):
        self._write([{"message": "Weather alert, sir.", "source": "weather"}])
        before = os.path.getmtime(self.queue)
        self.bc._handle_sleep_standby(None)
        self.assertEqual(self.spoke, [])
        self.assertEqual(os.path.getmtime(self.queue), before)
        self.assertEqual(len(self._read()), 1)

    def test_normal_drain_still_speaks_everything(self):
        self._write([{"message": "Weather alert, sir.", "source": "weather"},
                     {"message": "Reminder, sir — tea", "source": "timer"}])
        self.bc._speak_pending()
        self.assertEqual(len(self.spoke), 2)


class RequeueFailureNeverDropsTests(_QueueBase):
    """2026-10-01 merge audit: _speak_pending requeued its held (standby) and
    deferred (budget) entries AFTER speaking, ignored the False that
    _requeue_pending_speech returns, and deleted the claimed snapshot anyway -
    one refused os.replace dropped every held briefing, alert and offer.
    A failed requeue now shrinks the snapshot to the unspoken entries and
    leaves it for the next pass's orphan recovery."""

    WEATHER = {"message": "Weather alert, sir.", "source": "weather"}
    TIMER = {"message": "Reminder, sir — tea", "source": "timer",
             "dedupe_key": "timer#1"}
    OFFER = {"message": "Your print is done, sir.", "source": "device"}

    def _snapshot(self):
        with open(self.queue + ".consuming", encoding="utf-8") as f:
            return json.load(f)

    def test_standby_held_lines_survive_a_failed_requeue(self):
        bc = self.bc
        self._write([self.WEATHER, self.TIMER, self.OFFER])
        with mock.patch.object(bc, "_requeue_pending_speech",
                               return_value=False) as rq:
            self.assertTrue(
                bc._speak_pending(only_sources=bc._STANDBY_SPEAKABLE_SOURCES))
        rq.assert_called_once_with([self.WEATHER, self.OFFER])
        self.assertEqual(self.spoke, [self.TIMER["message"]])
        # Nothing on disk was lost: the claimed snapshot holds exactly the
        # two held lines - and not the timer that was already spoken.
        self.assertEqual(self._snapshot(), [self.WEATHER, self.OFFER])

        # The next standby pass recovers the snapshot into the live queue and
        # speaks nothing (neither held line is standby-speakable).
        self.assertFalse(
            bc._speak_pending(only_sources=bc._STANDBY_SPEAKABLE_SOURCES))
        self.assertEqual(self._read(), [self.WEATHER, self.OFFER])
        self.assertFalse(os.path.exists(self.queue + ".consuming"))

        # The wake drain speaks both held lines; the timer is not repeated
        # even with the recent-speech dedupe cleared (it is simply not queued).
        bc._recent_spoken_messages.clear()
        self.assertTrue(bc._speak_pending())
        self.assertEqual(self.spoke, [self.TIMER["message"],
                                      self.WEATHER["message"],
                                      self.OFFER["message"]])
        self.assertEqual(self._read(), [])

    def test_budget_deferred_tail_survives_a_failed_requeue(self):
        bc = self.bc
        self._p(bc, "_PENDING_DRAIN_BUDGET_S", 0.0)
        a, b, c = ({"message": f"Line {n}, sir."} for n in "abc")
        self._write([a, b, c])
        with mock.patch.object(bc, "_requeue_pending_speech",
                               return_value=False):
            bc._speak_pending()
        self.assertEqual(self.spoke, ["Line a, sir."])
        self.assertEqual(self._snapshot(), [b, c])
        bc._recent_spoken_messages.clear()
        bc._speak_pending()
        bc._speak_pending()
        self.assertEqual(self.spoke, ["Line a, sir.", "Line b, sir.",
                                      "Line c, sir."])

    def test_snapshot_rewrite_failure_keeps_the_whole_snapshot(self):
        # Last resort: when even the shrink fails, the claimed snapshot stays
        # whole - delayed, never lost; the spoken timer's dedupe key keeps it
        # from being said twice.
        bc = self.bc
        self._write([self.WEATHER, self.TIMER])
        with mock.patch.object(bc, "_requeue_pending_speech",
                               return_value=False),                 mock.patch.object(bc, "_rewrite_queue_snapshot",
                                  return_value=False):
            bc._speak_pending(only_sources=bc._STANDBY_SPEAKABLE_SOURCES)
        self.assertEqual(self._snapshot(), [self.WEATHER, self.TIMER])
        bc._speak_pending()
        self.assertEqual(self.spoke, [self.TIMER["message"],
                                      self.WEATHER["message"]])

    def test_recovery_io_error_keeps_the_orphan_and_skips_the_claim(self):
        # The retry used to discard an orphan on ANY error, I/O included -
        # so the same refused write that failed the requeue then lost it.
        bc = self.bc
        with open(self.queue + ".consuming", "w", encoding="utf-8") as f:
            json.dump([self.WEATHER], f)
        self._write([self.OFFER])
        with mock.patch.object(bc.tempfile, "mkstemp",
                               side_effect=PermissionError("in use")):
            self.assertFalse(bc._speak_pending())
        self.assertEqual(self.spoke, [])
        self.assertEqual(self._snapshot(), [self.WEATHER])   # not clobbered
        self.assertEqual(self._read(), [self.OFFER])
        bc._speak_pending()
        self.assertEqual(self.spoke, [self.WEATHER["message"],
                                      self.OFFER["message"]])

    def test_inject_drain_does_not_clobber_a_kept_orphan(self):
        bc = self.bc
        inj = os.path.join(self._tmp.name, "injected_commands.json")
        self._p(bc, "INJECTED_COMMANDS_PATH", inj)
        with open(inj + ".consuming", "w", encoding="utf-8") as f:
            json.dump(["orphan-cmd"], f)
        with open(inj, "w", encoding="utf-8") as f:
            json.dump(["live-cmd"], f)
        with mock.patch.object(bc.tempfile, "mkstemp",
                               side_effect=PermissionError("in use")):
            self.assertIsNone(bc._drain_injected_command())
        self.assertTrue(os.path.exists(inj + ".consuming"))
        self.assertEqual(bc._drain_injected_command(), "orphan-cmd")
        self.assertEqual(bc._drain_injected_command(), "live-cmd")


if __name__ == "__main__":
    unittest.main()
