"""Tests for core.llm_usage — the persisted month-to-date Claude token tally.

Fakes only: a temp dir for the file, a fake session tally, a fake clock and a
recording timer factory, so no real timer thread runs and nothing touches the
live data dir or the network.
"""
from __future__ import annotations

import ast
import json
import os
import re
import shutil
import tempfile
import time
import unittest
from unittest import mock

import core.llm_client as llm
import core.llm_usage as lu

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _local(y, mo, d, h=12):
    return time.mktime((y, mo, d, h, 0, 0, 0, 0, -1))


OCT = _local(2026, 10, 15)
NOV = _local(2026, 11, 2)


def _row(calls=1, inp=0, out=0, read=0, write=0):
    return {"calls": calls, "input": inp, "output": out,
            "cache_read": read, "cache_write": write}


class _Base(unittest.TestCase):
    def setUp(self):
        # A debounce timer another test scheduled must not fire into this one.
        stray = lu._timer[0]
        if stray is not None:
            stray.cancel()
        self.tmp = tempfile.mkdtemp(prefix="llm_usage_test_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.path = os.path.join(self.tmp, "llm_usage_month.json")
        self.session: dict = {}
        self.now = [1000.0]
        self.timers: list = []
        for name, value in (
                ("usage_path", lambda: self.path),
                ("_session_snapshot",
                 lambda: {m: dict(r) for m, r in self.session.items()}),
                ("_clock", lambda: self.now[0]),
                ("_start_timer", self._fake_timer),
                ("_flushed", {}), ("_last_attempt", [None]),
                ("_timer", [None]), ("_atexit_armed", [True])):
            p = mock.patch.object(lu, name, value)
            p.start()
            self.addCleanup(p.stop)

    def _fake_timer(self, delay, fn):
        self.timers.append((delay, fn))
        return object()

    def _write(self, data):
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(data if isinstance(data, str) else json.dumps(data))

    def _read(self):
        with open(self.path, encoding="utf-8") as f:
            return json.load(f)


class FlushTests(_Base):
    def test_missing_file_starts_fresh_with_counts_only(self):
        self.session = {"claude-sonnet-5-5": _row(3, 900, 120, 5000, 300)}
        self.assertTrue(lu.flush(now=OCT))
        data = self._read()
        self.assertEqual(set(data), {"month", "models", "updated"})
        self.assertEqual(data["month"], "2026-10")
        self.assertEqual(data["models"],
                         {"claude-sonnet-5-5": _row(3, 900, 120, 5000, 300)})

    def test_only_the_unflushed_delta_is_merged(self):
        # Another session already wrote this month.
        self._write({"month": "2026-10",
                     "models": {"claude-haiku-4-5": _row(5, 50, 5)}})
        self.session = {"claude-haiku-4-5": _row(2, 20, 2)}
        lu.flush(now=OCT)
        self.session = {"claude-haiku-4-5": _row(3, 30, 3),
                        "claude-sonnet-5-5": _row(1, 10, 1)}
        lu.flush(now=OCT)
        lu.flush(now=OCT)                      # nothing new: no double count
        self.assertEqual(self._read()["models"], {
            "claude-haiku-4-5": _row(8, 80, 8),
            "claude-sonnet-5-5": _row(1, 10, 1)})

    def test_month_rollover_keeps_last_month_under_previous(self):
        self._write({"month": "2026-10",
                     "models": {"claude-haiku-4-5": _row(7, 70, 7)},
                     "previous": {"month": "2026-09",
                                  "models": {"claude-haiku-4-5": _row(1)}}})
        self.session = {"claude-sonnet-5-5": _row(2, 20, 2)}
        lu.flush(now=NOV)
        data = self._read()
        self.assertEqual(data["month"], "2026-11")
        self.assertEqual(data["models"], {"claude-sonnet-5-5": _row(2, 20, 2)})
        self.assertEqual(data["previous"], {
            "month": "2026-10",
            "models": {"claude-haiku-4-5": _row(7, 70, 7)}})
        # Later in the same month the previous month is kept as it was.
        self.session = {"claude-sonnet-5-5": _row(3, 30, 3)}
        lu.flush(now=NOV + 3600)
        data = self._read()
        self.assertEqual(data["previous"]["month"], "2026-10")
        self.assertEqual(data["models"]["claude-sonnet-5-5"], _row(3, 30, 3))

    def test_corrupt_file_starts_fresh_without_raising(self):
        for junk in ("{not json", "[1, 2]", '{"month": 5}',
                     '{"month": "October", "models": {}}', "\x00\x01"):
            with self.subTest(junk=junk):
                self._write(junk)
                lu._flushed.clear()
                self.session = {"claude-haiku-4-5": _row(1, 10, 1)}
                self.assertIsNone(lu.month_usage(now=OCT))
                self.assertTrue(lu.flush(now=OCT))
                data = self._read()
                self.assertEqual(data["month"], "2026-10")
                self.assertEqual(data["models"],
                                 {"claude-haiku-4-5": _row(1, 10, 1)})
                self.assertNotIn("previous", data)

    def test_write_is_atomic_temp_file_then_replace(self):
        from core import atomic_io
        self._write({"month": "2026-10",
                     "models": {"claude-haiku-4-5": _row(1)}})
        seen = []
        real_replace = atomic_io._replace_with_retry

        def _spy(src, dst):
            with open(src, encoding="utf-8") as f:
                seen.append((src, dst, json.load(f)))
            real_replace(src, dst)

        self.session = {"claude-haiku-4-5": _row(1)}
        with mock.patch.object(atomic_io, "_replace_with_retry", _spy):
            self.assertTrue(lu.flush(now=OCT))
        src, dst, body = seen[0]
        self.assertEqual(dst, self.path)
        self.assertNotEqual(src, self.path)
        self.assertEqual(os.path.dirname(src), self.tmp)   # same volume
        self.assertEqual(body["models"]["claude-haiku-4-5"]["calls"], 2)
        self.assertEqual(os.listdir(self.tmp), ["llm_usage_month.json"])

    def test_failed_replace_leaves_the_old_file_and_no_temp(self):
        from core import atomic_io
        before = {"month": "2026-10", "models": {"claude-haiku-4-5": _row(4)}}
        self._write(before)
        self.session = {"claude-haiku-4-5": _row(1)}
        with mock.patch.object(atomic_io, "_replace_with_retry",
                               side_effect=OSError("disk full")):
            self.assertFalse(lu.flush(now=OCT))
        self.assertEqual(self._read(), before)
        self.assertEqual(os.listdir(self.tmp), ["llm_usage_month.json"])
        # Not marked flushed: the next write still carries the counts.
        self.assertTrue(lu.flush(now=OCT))
        self.assertEqual(self._read()["models"]["claude-haiku-4-5"]["calls"],
                         5)


class DebounceTests(_Base):
    def test_first_reply_flushes_now_then_at_most_every_interval(self):
        self.session = {"claude-haiku-4-5": _row(1, 10, 1)}
        lu.note_usage()
        lu.note_usage()                     # a write is already scheduled
        self.assertEqual([d for d, _ in self.timers], [0.0])
        self.timers.pop()[1]()              # the timer fires
        self.assertEqual(self._read()["models"]["claude-haiku-4-5"]["calls"],
                         1)
        self.now[0] += 10                   # 10 s later: wait out the rest
        self.session = {"claude-haiku-4-5": _row(2, 20, 2)}
        lu.note_usage()
        self.assertEqual([d for d, _ in self.timers], [50.0])
        self.now[0] += 50
        self.timers.pop()[1]()
        self.assertEqual(self._read()["models"]["claude-haiku-4-5"]["calls"],
                         2)
        self.assertEqual(self.timers, [])   # nothing pending: no reschedule

    def test_reply_landing_mid_write_gets_the_next_debounced_write(self):
        snaps = [{"claude-haiku-4-5": _row(1)}, {"claude-haiku-4-5": _row(2)}]
        lu.note_usage()
        with mock.patch.object(lu, "_session_snapshot",
                               side_effect=lambda: dict(snaps.pop(0))
                               if len(snaps) > 1 else dict(snaps[0])):
            self.timers.pop()[1]()
        self.assertEqual(self._read()["models"]["claude-haiku-4-5"]["calls"],
                         1)
        self.assertEqual([d for d, _ in self.timers], [lu.FLUSH_INTERVAL_S])

    def test_a_failing_disk_is_retried_once_an_interval_not_in_a_loop(self):
        self.session = {"claude-haiku-4-5": _row(1)}
        lu.note_usage()
        with mock.patch("core.atomic_io._atomic_write_json",
                        side_effect=PermissionError("read-only")):
            self.timers.pop()[1]()
        self.assertEqual([d for d, _ in self.timers], [lu.FLUSH_INTERVAL_S])

    def test_atexit_is_armed_once(self):
        with mock.patch.object(lu, "_atexit_armed", [False]), \
             mock.patch.object(lu.atexit, "register") as reg:
            lu.note_usage()
            lu.note_usage()
        reg.assert_called_once_with(lu.flush)

    def test_each_tallied_reply_notes_usage(self):
        class _U:
            input_tokens, output_tokens = 5, 1

        class _Msg:
            usage = _U()

        with mock.patch.object(llm, "session_usage", {}), \
             mock.patch.object(lu, "note_usage") as note:
            llm._record_session_usage("claude-haiku-4-5", _Msg())
            llm._record_session_usage("claude-haiku-4-5", object())  # no usage
        note.assert_called_once_with()

    def test_hard_exit_flushes_before_terminating(self):
        with open(os.path.join(_ROOT, "bobert_companion.py"),
                  encoding="utf-8") as fh:
            src = fh.read()
        fn = next(n for n in ast.parse(src).body
                  if isinstance(n, ast.FunctionDef) and n.name == "_hard_exit")
        body = ast.get_source_segment(src, fn)
        self.assertRegex(body, r"llm_usage(?: as \w+)?\n")
        self.assertLess(body.index(".flush()"),
                        body.index("_terminate_process_now(code)"))


class MonthUsageTests(_Base):
    def test_file_plus_unflushed_session_counts(self):
        self._write({"month": "2026-10",
                     "models": {"claude-sonnet-5-5": _row(5, 500, 50)}})
        self.session = {"claude-haiku-4-5": _row(2, 20, 2)}
        self.assertEqual(lu.month_usage(now=OCT), {
            "claude-sonnet-5-5": _row(5, 500, 50),
            "claude-haiku-4-5": _row(2, 20, 2)})
        lu.flush(now=OCT)                   # flushed: not counted twice
        self.assertEqual(lu.month_usage(now=OCT)["claude-haiku-4-5"],
                         _row(2, 20, 2))

    def test_none_without_a_tally_for_this_month(self):
        self.assertIsNone(lu.month_usage(now=OCT))               # missing
        self._write({"month": "2026-09",
                     "models": {"claude-haiku-4-5": _row(9)}})
        self.assertIsNone(lu.month_usage(now=OCT))               # last month


class NoTextStoredTests(_Base):
    PROSE = "my bank PIN is 4321, tell nobody"

    def _assert_counts_only(self, data):
        self.assertTrue(set(data) <= {"month", "models", "previous",
                                      "updated"}, data)
        self.assertRegex(data["month"], r"^\d{4}-\d{2}$")
        self.assertIsInstance(data["updated"], float)
        blocks = [data["models"]]
        if "previous" in data:
            self.assertEqual(set(data["previous"]), {"month", "models"})
            self.assertRegex(data["previous"]["month"], r"^\d{4}-\d{2}$")
            blocks.append(data["previous"]["models"])
        for models in blocks:
            for model, row in models.items():
                self.assertRegex(model, r"^[a-z0-9][a-z0-9._:\-]*$")
                self.assertEqual(set(row), set(lu._FIELDS))
                for v in row.values():
                    self.assertIs(type(v), int)

    def test_prompt_and_reply_text_never_reach_the_file(self):
        class _U:
            input_tokens, output_tokens = 40, 9
            cache_read_input_tokens = cache_creation_input_tokens = 0

        class _Block:
            type, text = "text", "Your PIN is safe with me: 4321."

        class _Msg:
            content, usage, stop_reason = [_Block()], _U(), "end_turn"

        class _Client:
            class messages:
                @staticmethod
                def create(**kwargs):
                    return _Msg()

        with mock.patch.object(llm, "session_usage", {}), \
             mock.patch.object(lu, "_session_snapshot",
                               llm.session_usage_snapshot):
            msg = llm.create_message(
                _Client(), model="claude-haiku-4-5", max_tokens=50,
                system="You are JARVIS. " + self.PROSE,
                messages=[{"role": "user", "content": self.PROSE}])
            self.assertIn("4321", llm.response_text(msg))
            # a model id that is really prose is filed under "other"
            llm._record_session_usage(self.PROSE, _Msg())
            self.assertTrue(lu.flush(now=OCT))
        with open(self.path, encoding="utf-8") as f:
            raw = f.read()
        for needle in ("4321", "PIN", "bank", "JARVIS", "safe"):
            self.assertNotIn(needle, raw)
        data = json.loads(raw)
        self._assert_counts_only(data)
        self.assertEqual(data["models"]["claude-haiku-4-5"]["input"], 40)
        self.assertEqual(data["models"]["other"]["calls"], 1)

    def test_text_in_a_hand_edited_file_is_dropped_on_rewrite(self):
        self._write({"month": "2026-10", "note": self.PROSE,
                     "models": {self.PROSE: _row(3),
                                "claude-haiku-4-5": {"calls": 1,
                                                     "summary": self.PROSE,
                                                     "input": "lots"}},
                     "previous": {"month": "2026-09", "text": self.PROSE,
                                  "models": {}}})
        self.session = {"claude-haiku-4-5": _row(1)}
        lu.flush(now=OCT)
        with open(self.path, encoding="utf-8") as f:
            raw = f.read()
        self.assertNotRegex(raw, re.escape("4321"))
        data = json.loads(raw)
        self._assert_counts_only(data)
        self.assertEqual(data["models"]["claude-haiku-4-5"], _row(2))


if __name__ == "__main__":
    unittest.main()
