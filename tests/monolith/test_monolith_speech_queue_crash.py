"""Monolith-tier proof that a queued spoken line survives a crash + restart
exactly once - never lost, never doubled (2026-10-02).

A queued line vanished after a restart on 2026-08-20. v2.0.143 (tests stopped
claiming the LIVE queue) and v2.0.157 (a failed requeue keeps held lines,
orphan recovery keeps a snapshot it could not merge) were expected to have
fixed it, but nothing had ever simulated a crash at each point of the
pending_speech.json life cycle and then a restart.

A crash is a ``_Crash`` (a BaseException, so it escapes every
``except Exception`` in the drain exactly as a hard kill escapes everything):
whatever is on disk when it propagates is what a dead process leaves behind.
A restart clears the drain's in-memory state - above all the
_recent_spoken_messages dedupe, which spans one JARVIS lifetime only - and
drains again. The fake ``_speak`` counts a line only when it finishes, so a
line cut off mid-word is not "spoken".

Doing this found one gap: the claimed ``.consuming`` snapshot only shrank at
the END of a batch, so a crash after line 1 of 3 left all three on disk and
the restart said line 1 again (the in-memory dedupe that was meant to stop
that died with the process). _speak_pending now rewrites the snapshot after
each line it finishes.

Every test redirects PENDING_SPEECH_PATH (and proactive_announce's
__file__-derived queue path) into a temp dir - the live queue is never read.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest import mock

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith


class _Crash(BaseException):
    """The process dies here. BaseException, so no ``except Exception`` in
    the code under test can swallow it - like a hard kill."""


@requires_monolith
class _CrashBase(MonolithGlobalsTestCase):
    def setUp(self):
        bc = self.bc
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.queue = os.path.join(self._tmp.name, "pending_speech.json")
        self.snapshot = self.queue + ".consuming"
        self._p(bc, "PENDING_SPEECH_PATH", self.queue)
        # proactive_announce derives its queue from bobert_companion.__file__.
        self._p(bc, "__file__", os.path.join(self._tmp.name,
                                             "bobert_companion.py"))
        self.spoke: list[str] = []
        self.crash_on: set[str] = set()   # lines whose speech dies mid-word
        self.during: dict = {}            # line -> callback run while speaking
        self._p(bc, "_speak", side_effect=self._fake_speak)
        self._p(bc, "_heartbeat")
        self._p(bc, "_speech_hold_active", return_value=False)
        self._p(bc, "_audio_flap_flush")
        self._p(bc, "_PENDING_DRAIN_BUDGET_S", 600.0)
        bc._presence_gate_armed[0] = False   # the import default: no hold
        bc._recent_spoken_messages.clear()

    def _p(self, *args, **kwargs):
        patcher = mock.patch.object(*args, **kwargs)
        m = patcher.start()
        self.addCleanup(patcher.stop)
        return m

    def _fake_speak(self, msg, **_k):
        cb = self.during.pop(msg, None)
        if cb is not None:
            cb()
        if msg in self.crash_on:
            self.crash_on.discard(msg)
            raise _Crash(msg)
        self.spoke.append(msg)

    def _restart(self):
        """A new JARVIS process: only the files survive."""
        bc = self.bc
        bc._recent_spoken_messages.clear()
        bc._presence_hold_logged[0] = ""
        bc._presence_away_seen[0] = False

    def _write(self, entries):
        with open(self.queue, "w", encoding="utf-8") as f:
            json.dump(entries, f)

    def _load(self, path):
        if not os.path.exists(path):
            return []
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def _drain_until_empty(self, **kw):
        for _ in range(5):
            self.bc._speak_pending(**kw)
        self.assertEqual(self._load(self.queue), [])
        self.assertFalse(os.path.exists(self.snapshot),
                         "a claimed snapshot was left behind")

    @staticmethod
    def _lines(*names):
        return [{"message": f"Line {n}, sir.", "source": "weather"}
                for n in names]


class CrashBeforeSpeakTests(_CrashBase):
    def test_crash_after_enqueue_before_any_drain(self):
        self.assertTrue(self.bc.proactive_announce("Your print is done, sir.",
                                                   source="device"))
        # (crash: nothing in memory matters, the line is on disk)
        self._restart()
        self._drain_until_empty()
        self.assertEqual(self.spoke, ["Your print is done, sir."])

    def test_crash_after_the_claim_before_the_first_line(self):
        a, b = self._lines("a", "b")
        self._write([a, b])
        # _heartbeat ticks right before each line is spoken: dying there is
        # dying with the queue claimed (renamed to .consuming) and unspoken.
        with mock.patch.object(self.bc, "_heartbeat",
                               side_effect=_Crash("claimed")):
            with self.assertRaises(_Crash):
                self.bc._speak_pending()
        self.assertFalse(os.path.exists(self.queue))
        self.assertEqual(self._load(self.snapshot), [a, b])
        self._restart()
        self._drain_until_empty()
        self.assertEqual(self.spoke, ["Line a, sir.", "Line b, sir."])


class CrashMidSpeakTests(_CrashBase):
    def test_a_line_cut_off_mid_word_is_spoken_again_once(self):
        (a,) = self._lines("a")
        self._write([a])
        self.crash_on.add("Line a, sir.")
        with self.assertRaises(_Crash):
            self.bc._speak_pending()
        self.assertEqual(self.spoke, [])
        self._restart()
        self._drain_until_empty()
        self.assertEqual(self.spoke, ["Line a, sir."])

    def test_lines_finished_before_the_crash_are_not_said_again(self):
        # The gap: the snapshot only shrank at the END of a batch, so all
        # three lines were still on disk and the restart repeated line a.
        a, b, c = self._lines("a", "b", "c")
        self._write([a, b, c])
        self.crash_on.add("Line b, sir.")
        with self.assertRaises(_Crash):
            self.bc._speak_pending()
        self.assertEqual(self.spoke, ["Line a, sir."])
        self.assertEqual(self._load(self.snapshot), [b, c])
        self._restart()
        self._drain_until_empty()
        self.assertEqual(self.spoke, ["Line a, sir.", "Line b, sir.",
                                      "Line c, sir."])

    def test_a_line_queued_during_the_crashed_batch_is_spoken_once(self):
        a, b = self._lines("a", "b")
        self._write([a, b])
        self.during["Line a, sir."] = lambda: self.bc.proactive_announce(
            "Reminder, sir — tea", source="timer", dedupe_key="timer#1")
        self.crash_on.add("Line b, sir.")
        with self.assertRaises(_Crash):
            self.bc._speak_pending()
        self._restart()
        self._drain_until_empty()
        self.assertEqual(self.spoke, ["Line a, sir.", "Line b, sir.",
                                      "Reminder, sir — tea"])

    def test_a_suppressed_duplicate_is_not_spoken_after_the_restart(self):
        # The same entry twice in one snapshot is said once. Dying after the
        # duplicate was dropped must not bring it back once the in-memory
        # dedupe is gone.
        a, b = self._lines("a", "b")
        self._write([a, dict(a), b])
        self.crash_on.add("Line b, sir.")
        with self.assertRaises(_Crash):
            self.bc._speak_pending()
        self.assertEqual(self._load(self.snapshot), [b])
        self._restart()
        self._drain_until_empty()
        self.assertEqual(self.spoke, ["Line a, sir.", "Line b, sir."])

    def test_standby_crash_after_a_timer_does_not_repeat_it(self):
        bc = self.bc
        weather = {"message": "Weather alert, sir.", "source": "weather"}
        tea = {"message": "Reminder, sir — tea", "source": "timer",
               "dedupe_key": "timer#1"}
        eggs = {"message": "Reminder, sir — eggs", "source": "timer",
                "dedupe_key": "timer#2"}
        self._write([weather, tea, eggs])
        self.crash_on.add("Reminder, sir — eggs")
        with self.assertRaises(_Crash):
            bc._speak_pending(only_sources=bc._STANDBY_SPEAKABLE_SOURCES)
        self.assertEqual(self._load(self.snapshot), [weather, eggs])
        self._restart()
        bc._speak_pending(only_sources=bc._STANDBY_SPEAKABLE_SOURCES)
        self.assertEqual(self.spoke, ["Reminder, sir — tea",
                                      "Reminder, sir — eggs"])
        self._drain_until_empty()   # the wake drain
        self.assertEqual(self.spoke, ["Reminder, sir — tea",
                                      "Reminder, sir — eggs",
                                      "Weather alert, sir."])


class CrashAroundTheRequeueTests(_CrashBase):
    WEATHER = {"message": "Weather alert, sir.", "source": "weather"}
    TIMER = {"message": "Reminder, sir — tea", "source": "timer",
             "dedupe_key": "timer#1"}
    OFFER = {"message": "Your print is done, sir.", "source": "device"}

    def _standby(self):
        return self.bc._speak_pending(
            only_sources=self.bc._STANDBY_SPEAKABLE_SOURCES)

    def _assert_each_once_after_restart(self):
        self._restart()
        self._standby()
        self.assertEqual(self.spoke, [self.TIMER["message"]])
        self._drain_until_empty()
        self.assertEqual(self.spoke, [self.TIMER["message"],
                                      self.WEATHER["message"],
                                      self.OFFER["message"]])

    def test_failed_requeue_then_crash(self):
        self._write([self.WEATHER, self.TIMER, self.OFFER])
        with mock.patch.object(self.bc, "_requeue_pending_speech",
                               return_value=False):
            self._standby()
        # (crash: the process dies before the next pass)
        self._assert_each_once_after_restart()

    def test_crash_inside_the_requeue(self):
        # The process dies while the held lines are being written back, i.e.
        # before the end-of-drain shrink ever runs.
        self._write([self.WEATHER, self.TIMER, self.OFFER])
        with mock.patch.object(self.bc, "_requeue_pending_speech",
                               side_effect=_Crash("requeue")):
            with self.assertRaises(_Crash):
                self._standby()
        self.assertEqual(self._load(self.snapshot),
                         [self.WEATHER, self.OFFER])
        self._assert_each_once_after_restart()

    def test_crash_after_the_requeue_before_the_snapshot_is_removed(self):
        # The held lines are in the live queue AND still in the snapshot; the
        # restart's recovery merges both copies, and the drain says each once.
        real_remove = os.remove

        def remove(path, *a, **k):
            if str(path).endswith(".consuming"):
                raise _Crash("remove")
            return real_remove(path, *a, **k)

        self._write([self.WEATHER, self.TIMER, self.OFFER])
        with mock.patch.object(self.bc.os, "remove", side_effect=remove):
            with self.assertRaises(_Crash):
                self._standby()
        self.assertEqual(self._load(self.queue), [self.WEATHER, self.OFFER])
        self._assert_each_once_after_restart()

    def test_disk_refusing_every_write_repeats_rather_than_loses(self):
        # The one trade left: when the disk refuses the requeue AND every
        # snapshot rewrite, the claimed snapshot stays whole, so after a
        # restart the timer is said again. Repeated, never lost.
        self._write([self.WEATHER, self.TIMER])
        with mock.patch.object(self.bc, "_requeue_pending_speech",
                               return_value=False), \
                mock.patch.object(self.bc, "_rewrite_queue_snapshot",
                                  return_value=False):
            self._standby()
        self.assertEqual(self._load(self.snapshot), [self.WEATHER, self.TIMER])
        self._restart()
        self._drain_until_empty()
        self.assertEqual(self.spoke, [self.TIMER["message"],
                                      self.WEATHER["message"],
                                      self.TIMER["message"]])


if __name__ == "__main__":
    unittest.main()
