"""Queued speech: delivery failure and a damaged store (master item 37,
2026-10-13), the two cases test_monolith_speech_queue_crash does not force.

  * A line whose TTS raises mid-drain is NOT delivered, so it stays owed
    (bounded retries - a poison line must not wedge the queue forever) and
    the lines after it are still spoken, in order, exactly once.
  * A store torn mid-write (the array cut off inside a record) costs at most
    the damaged record: the intact records before it are still spoken, and a
    new announcement queued on top of it keeps them.

Real _speak_pending / proactive_announce run; the voice is faked and the
queue lives in a temp dir. Synthetic lines only.

    python -m unittest tests.monolith.test_monolith_speech_queue_failures
"""
from __future__ import annotations

import contextlib
import io
import json
import os

from tests.monolith.test_monolith_speech_queue_crash import _CrashBase
from tests._monolith_harness import requires_monolith


@requires_monolith
class _Base(_CrashBase):
    def setUp(self):
        super().setUp()
        self.fail_on: dict = {}   # line -> number of failures left (-1 = always)

    def _fake_speak(self, msg, **_k):
        left = self.fail_on.get(msg, 0)
        if left:
            if left > 0:
                self.fail_on[msg] = left - 1
            raise RuntimeError("tts down")
        self.spoke.append(msg)

    def _pass(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return self.bc._speak_pending()

    def _write_raw(self, text):
        with open(self.queue, "w", encoding="utf-8") as f:
            f.write(text)


class DeliveryFailureTests(_Base):
    def test_failed_line_is_kept_and_spoken_once_after_recovery(self):
        a, b, c = self._lines("a", "b", "c")
        self._write([a, b, c])
        self.fail_on["Line b, sir."] = 1
        self._pass()
        self.assertEqual(self.spoke, ["Line a, sir.", "Line c, sir."])
        self._restart()
        self._drain_until_empty()
        self.assertEqual(sorted(self.spoke), sorted(
            ["Line a, sir.", "Line b, sir.", "Line c, sir."]))
        self.assertEqual(self.spoke.count("Line b, sir."), 1)

    def test_failed_line_does_not_reorder_the_lines_after_it(self):
        a, b, c = self._lines("a", "b", "c")
        self._write([a, b, c])
        self.fail_on["Line a, sir."] = 1
        self._pass()
        self.assertEqual(self.spoke, ["Line b, sir.", "Line c, sir."])
        self._drain_until_empty()
        self.assertEqual(self.spoke[-1], "Line a, sir.")

    def test_a_line_that_always_fails_is_dropped_after_bounded_retries(self):
        a, b = self._lines("a", "b")
        self._write([a, b])
        self.fail_on["Line a, sir."] = -1
        for _ in range(10):
            self._restart()
            self._pass()
        self.assertEqual(self.spoke, ["Line b, sir."])
        self.assertEqual(self._load(self.queue), [])
        self.assertFalse(os.path.exists(self.snapshot))


class DamagedStoreTests(_Base):
    def _torn(self, intact):
        good = json.dumps(intact, indent=2).rstrip()[:-1]
        torn = json.dumps({"ts": 2.0, "message": "Line z, sir.",
                           "source": "weather"})[:-14]
        return good + ",\n" + torn

    def test_torn_store_speaks_the_intact_records_and_does_not_raise(self):
        a, b = self._lines("a", "b")
        self._write_raw(self._torn([a, b]))
        self._restart()
        self._drain_until_empty()
        self.assertEqual(self.spoke, ["Line a, sir.", "Line b, sir."])

    def test_garbage_store_is_survived_and_the_next_line_still_speaks(self):
        self._write_raw("\x00\x00 not json")
        self._pass()
        self.assertEqual(self.spoke, [])
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(self.bc.proactive_announce("After, sir.",
                                                       source="device"))
        self._drain_until_empty()
        self.assertEqual(self.spoke, ["After, sir."])

    def test_announce_onto_a_torn_store_keeps_the_intact_records(self):
        a, b = self._lines("a", "b")
        self._write_raw(self._torn([a]))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(self.bc.proactive_announce("Line b, sir.",
                                                       source="weather"))
        self._restart()
        self._drain_until_empty()
        self.assertEqual(self.spoke, ["Line a, sir.", "Line b, sir."])

    def test_torn_orphan_snapshot_keeps_its_intact_records(self):
        a, b = self._lines("a", "b")
        with open(self.snapshot, "w", encoding="utf-8") as f:
            f.write(self._torn([a]))
        self._write([b])
        self._restart()
        self._drain_until_empty()
        self.assertEqual(self.spoke, ["Line a, sir.", "Line b, sir."])
