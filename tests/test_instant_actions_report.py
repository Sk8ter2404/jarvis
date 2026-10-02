"""tools/instant_actions_report.py — the weekly INSTANT_ACTIONS score.

Every log here is SYNTHETIC and lives in a temp dir. The clock is pinned: NOW
is a fixed noon timestamp and row times are offsets from it, never
time.time() - N. "zebra" is a stand-in transcript word that a malformed row
may carry; the report must never print it.

Run: python tools/run_tests.py test_instant_actions_report
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest

from core import instant_actions as ia
from tools import instant_actions_report as rep

NOW = 1759406400.0          # 2026-10-02 12:00:00 UTC
DAY = 86400.0


def _rows():
    return [
        # inside the 7-day window
        ia.shadow_row("pause_music", ["pause_music"], True, ts=NOW - 1 * DAY),
        ia.shadow_row("pause_music", ["pause_music"], True, ts=NOW - 2 * DAY),
        ia.shadow_row("pause_music", ["media_playpause"], False,
                      ts=NOW - 3 * DAY),
        ia.shadow_row("volume_up", [], False, ts=NOW - 4 * DAY),
        ia.on_row("next_song", True, ts=NOW - 5 * DAY),
        ia.on_row("next_song", False, ts=NOW - 6 * DAY),
        # older than a week: counted per action, not in the pass rate
        ia.shadow_row("smart_home_control", ["smart_home_control"], True,
                      ts=NOW - 9 * DAY),
        ia.shadow_row("smart_home_control", ["control_device"], True,
                      ts=NOW - 20 * DAY),
    ]


GOLDEN = (
    "JARVIS instant-actions report (counts only; read-only)\n"
    "log: 8 row(s)\n"
    "  action               shadow  agree     on     ok\n"
    "  next_song                 0      0      2      1\n"
    "  pause_music               3      2      0      0\n"
    "  smart_home_control        2      2      0      0\n"
    "  volume_up                 1      0      0      0\n"
    "agreement with the brain (all shadow rows): 4/6 (66.7%)\n"
    "pass rate, last 7 days: 3/6 (50.0%)\n"
)


class ReportTests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, ia.LOG_NAME)
        for row in _rows():
            self.assertTrue(ia.append_row(self.path, row))

    def _main(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = rep.main(list(argv), now=NOW)
        return code, out.getvalue(), err.getvalue()

    def test_golden_report(self):
        code, out, _err = self._main("--log", self.path)
        self.assertEqual(code, 0)
        self.assertEqual(out, GOLDEN)

    def test_summary_counts(self):
        s = rep.summarize(ia.read_rows(self.path), NOW)
        self.assertEqual((s["shadow"], s["agree"]), (6, 4))
        self.assertEqual((s["window"], s["passed"]), (6, 3))
        self.assertEqual(s["per_action"]["next_song"],
                         {"shadow": 0, "agree": 0, "on": 2, "ok": 1})

    def test_the_window_is_configurable(self):
        s = rep.summarize(ia.read_rows(self.path), NOW, days=10)
        self.assertEqual((s["window"], s["passed"]), (7, 4))
        code, out, _err = self._main("--log", self.path, "--days", "30")
        self.assertEqual(code, 0)
        self.assertIn("pass rate, last 30 days: 5/8 (62.5%)", out)

    def test_future_rows_are_not_in_the_window(self):
        ia.append_row(self.path, ia.on_row("volume_up", True, ts=NOW + DAY))
        s = rep.summarize(ia.read_rows(self.path), NOW)
        self.assertEqual(s["window"], 6)

    def test_the_rotated_file_is_read_too(self):
        os.replace(self.path, self.path + ".1")
        ia.append_row(self.path, ia.on_row("volume_up", True, ts=NOW - 60))
        code, out, _err = self._main("--log", self.path)
        self.assertEqual(code, 0)
        self.assertIn("log: 9 row(s)", out)
        self.assertIn("pass rate, last 7 days: 4/7 (57.1%)", out)

    def test_an_empty_log_reports_no_rows(self):
        empty = os.path.join(self.tmp.name, "empty.jsonl")
        open(empty, "w", encoding="utf-8").close()
        code, out, _err = self._main("--log", empty)
        self.assertEqual(code, 0)
        self.assertIn("no instant-action turns logged yet", out)
        self.assertIn("agreement with the brain (all shadow rows): n/a", out)
        self.assertIn("pass rate, last 7 days: n/a", out)

    def test_no_log_file_is_not_a_clean_report(self):
        code, out, err = self._main(
            "--log", os.path.join(self.tmp.name, "missing.jsonl"))
        self.assertEqual(code, 2)
        self.assertEqual(out, "")
        self.assertIn("nothing to score", err)

    def test_it_prints_names_and_counts_only_and_writes_nothing(self):
        # A row written by hand with extra text: the report reads its counts
        # and prints none of the text.
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"ts": NOW - 60, "mode": "on",
                                 "action": "volume_up", "ok": True,
                                 "said": "turn the zebra up"}) + "\n")
        before = sorted(os.listdir(self.tmp.name))
        with open(self.path, "rb") as fh:
            raw = fh.read()
        code, out, err = self._main("--log", self.path)
        self.assertEqual(code, 0)
        self.assertNotIn("zebra", out + err)
        self.assertEqual(sorted(os.listdir(self.tmp.name)), before)
        with open(self.path, "rb") as fh:
            self.assertEqual(fh.read(), raw)

    def test_default_log_path_is_the_data_dir(self):
        self.assertEqual(os.path.basename(rep.default_log_path()),
                         ia.LOG_NAME)


if __name__ == "__main__":
    unittest.main()
