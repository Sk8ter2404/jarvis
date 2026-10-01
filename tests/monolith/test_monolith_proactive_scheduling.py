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


if __name__ == "__main__":
    unittest.main()
