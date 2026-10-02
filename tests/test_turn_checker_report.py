"""tools/turn_checker_report.py — core/turn_checker verdict counts over JARVIS
session logs.

Every log here is SYNTHETIC (made-up turns, no real transcript). The golden
test pins the whole report for one session log that holds every verdict
kind; the rest pin the turn boundaries, the rebuilt registry and the two hard
rules: the tool never writes anything and never prints log text.

Run: python tools/run_tests.py test_turn_checker_report
"""
from __future__ import annotations

import builtins
import contextlib
import io
import os
import re
import shutil
import tempfile
import unittest
from unittest import mock

from tools import turn_checker_report as rep

# Ten owner turns. "zebra" / "quartz" / "lantern" are stand-in transcript
# words that must never reach the report.
SYNTH = (
    "[08:59:58]   [boot] starting up\n"
    "[08:59:59]   JARVIS: Good morning, sir. Systems online.\n"
    # 1. said_no_action
    "[09:00:01]   You:    set a five minute timer zebra\n"
    "[09:00:02]   JARVIS: I've set a timer for five minutes, sir.\n"
    "[09:00:03]   [turn-timing] kind=voice outcome=ok\n"
    # 2. command_no_action (screenshot is used by turn 3)
    "[09:00:10]   You:    take a screenshot\n"
    "[09:00:11]   JARVIS: Very good, sir.\n"
    "[09:00:12]   [turn-timing] kind=voice outcome=ok\n"
    # 3. ok: the action ran
    "[09:00:20]   You:    take a screenshot\n"
    "[09:00:21]   JARVIS: [ACTION: screenshot]\n"
    "[09:00:21]   [action] screenshot: saved quartz.png\n"
    "[09:00:22]   [turn-timing] kind=voice outcome=ok\n"
    # 4. made_up_action, escalates
    "[09:00:30]   You:    water the lantern plants\n"
    "[09:00:31]   JARVIS: [ACTION: water_plants] Watering them now, sir.\n"
    "[09:00:31]   [autocorrect] no match for 'water_plants' (best conf=0.31)\n"
    "[09:00:32]   [turn-timing] kind=voice outcome=ok\n"
    # 5. ok: autocorrect resolved the name and it ran
    "[09:00:40]   You:    set a 5 minute timer\n"
    "[09:00:41]   JARVIS: [ACTION: set_timr, 5 minutes]\n"
    "[09:00:41]   [autocorrect] 'set_timr' -> 'set_timer' (conf=0.91)\n"
    "[09:00:41]   [action] set_timer: Timer set for 5 minutes.\n"
    "[09:00:42]   [turn-timing] kind=voice outcome=ok\n"
    # 6. ok: the runtime asked "did you mean X or Y?"
    "[09:00:50]   You:    start zebra mode\n"
    "[09:00:51]   JARVIS: [ACTION: zebra_mode]\n"
    "[09:00:51]   [autocorrect] ambiguous 'zebra_mode' -> 'focus_mode' "
    "(0.80) vs 'game_mode' (0.78)\n"
    "[09:00:52]   [turn-timing] kind=voice outcome=ok\n"
    # 7. made_up_action that needs confirmation: never escalates
    "[09:01:00]   You:    send it and bolt the lantern door\n"
    "[09:01:01]   JARVIS: [ACTION: send_email, quartz] [ACTION: lock_door_bolt]\n"
    "[09:01:01]   [action] ⚠  REQUIRES CONFIRMATION: send_email(quartz) — "
    "say 'yes' to proceed\n"
    "[09:01:01]   [autocorrect] no match for 'lock_door_bolt' (best conf=0.20)\n"
    "[09:01:02]   [turn-timing] kind=voice outcome=ok\n"
    # 8. ok: chit-chat with a question back
    "[09:01:10]   You:    how are you\n"
    "[09:01:11]   JARVIS: Very well, sir. And you?\n"
    "[09:01:12]   [turn-timing] kind=voice outcome=ok\n"
    # 9. a fast path answered (no LLM): counted, not checked, though its
    #    reply would read as a claim
    "[09:01:15]   You:    run a system check\n"
    "[09:01:15]   [fast-path] self-check\n"
    "[09:01:16]   JARVIS: I've checked every subsystem, sir. All nominal.\n"
    "[09:01:17]   [turn-timing] kind=voice outcome=shortcut\n"
    # 10. ok: a proactive remark after it is not part of this turn
    "[09:01:20]   You:    thanks\n"
    "[09:01:21]   JARVIS: You're welcome, sir.\n"
    "[09:01:21]   JARVIS (spoken): Opening zebra now.\n"
    "[09:05:00]   \n"
    "[09:05:00]   [proactive]\n"
    "[09:05:00]   JARVIS: I've turned off the lights, sir.\n"
)

GOLDEN = """\
JARVIS turn-checker report (counts only; read-only)
logs: 1 session file(s), 10 turn(s), 1 on a fast path (no LLM, not checked)
  ok                      5
  said_no_action          1
  command_no_action       1
  made_up_action          2
  would escalate          3   (cloud allowed, confidence >= 0.8)
"""

NAME = "session_2026-10-02_08-59-58.log"


class _LogDir(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="jarvis_tcr_")
        self.addCleanup(shutil.rmtree, self.dir, True)

    def write(self, name, text):
        path = os.path.join(self.dir, name)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        return path

    def run_main(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = rep.main(list(args))
        return code, out.getvalue(), err.getvalue()


class GoldenTests(_LogDir):
    def test_golden_report_for_the_synthetic_log(self):
        self.write(NAME, SYNTH)
        code, out, _ = self.run_main(self.dir)
        self.assertEqual(code, 0)
        self.assertEqual(out, GOLDEN)

    def test_a_log_file_argument_reads_the_same(self):
        path = self.write(NAME, SYNTH)
        code, out, _ = self.run_main(path)
        self.assertEqual((code, out), (0, GOLDEN))

    def test_the_report_never_prints_log_text(self):
        self.write(NAME, SYNTH)
        _, out, err = self.run_main(self.dir)
        for words in ("zebra", "quartz", "lantern", "You:", "JARVIS:",
                      "sir", "screenshot", "water_plants", "timer",
                      self.dir, NAME):
            with self.subTest(words=words):
                self.assertNotIn(words, out + err)

    def test_the_tool_never_writes(self):
        self.write(NAME, SYNTH)
        real_open = builtins.open
        modes = []

        def guarded(file, mode="r", *a, **k):
            modes.append(mode)
            if any(c in mode for c in "wax+"):
                raise AssertionError(f"report opened {file!r} for {mode!r}")
            return real_open(file, mode, *a, **k)

        before = sorted(os.listdir(self.dir))
        with mock.patch("builtins.open", guarded), \
                mock.patch("os.makedirs", side_effect=AssertionError("mkdir")):
            code, _, _ = self.run_main(self.dir)
        self.assertEqual(code, 0)
        self.assertEqual(modes, ["rb"])
        self.assertEqual(sorted(os.listdir(self.dir)), before)

    def test_no_logs_is_an_error_with_nothing_on_stdout(self):
        code, out, err = self.run_main(os.path.join(self.dir, "missing"))
        self.assertEqual((code, out), (2, ""))
        self.assertIn("no session logs", err)

    def test_a_folder_reads_only_session_logs(self):
        self.write(NAME, SYNTH)
        self.write("notes.txt", "You:    take a screenshot\nJARVIS: Done.\n")
        _, out, _ = self.run_main(self.dir)
        self.assertEqual(out, GOLDEN)


class ParseTests(_LogDir):
    def turns(self):
        return rep.parse_log(self.write(NAME, SYNTH))

    def test_turn_boundaries(self):
        turns, _ = self.turns()
        self.assertEqual(len(turns), 10)
        # The boot greeting before the first "You:" belongs to no turn, and
        # the proactive remark after the last one is not its reply.
        self.assertEqual(turns[-1]["reply"], ["You're welcome, sir."])
        self.assertEqual(turns[0]["reply"],
                         ["I've set a timer for five minutes, sir."])

    def test_emitted_ran_asked_and_confirmation(self):
        turns, unknown = self.turns()
        self.assertEqual(turns[2]["emitted"], ["screenshot"])
        self.assertEqual(turns[2]["ran"], {"screenshot"})
        self.assertEqual(turns[4]["ran"], {"set_timer"})
        self.assertTrue(turns[5]["asked"])
        self.assertTrue(turns[6]["confirm"])
        self.assertEqual(turns[6]["emitted"], ["send_email", "lock_door_bolt"])
        self.assertEqual(turns[6]["ran"], set())
        self.assertEqual(unknown, {"water_plants", "set_timr", "zebra_mode",
                                   "lock_door_bolt"})

    def test_registry_leaves_out_what_autocorrect_handled(self):
        turns, unknown = self.turns()
        kinds = [v.kind if v else None for v, _c in rep.verdicts(turns, unknown)]
        self.assertEqual(kinds, [
            "said_no_action", "command_no_action", "ok", "made_up_action",
            "ok", "ok", "made_up_action", "ok", None, "ok"])

    def test_a_fast_path_turn_is_not_checked(self):
        turns, _ = self.turns()
        self.assertTrue(turns[8]["fast_path"])
        self.assertFalse(any(t["fast_path"] for t in turns[:8] + turns[9:]))
        # Its reply alone would be flagged, which is why it is skipped.
        t = turns[8]
        v = rep.turn_checker.check_turn(t["user"], " ".join(t["reply"]),
                                        t["emitted"], t["ran"], {"x"})
        self.assertEqual(v.kind, "said_no_action")
        # A fast path that FAILED falls through to the LLM: still checked.
        path = self.write("session_2026-10-02_10-00-00.log", (
            "[10:00:01]   You:    what's my name\n"
            "[10:00:01]   [fast-path] failed: KeyError\n"
            "[10:00:02]   JARVIS: I'm not sure, sir.\n"))
        self.assertFalse(rep.parse_log(path)[0][0]["fast_path"])

    def test_a_pushback_needs_confirmation(self):
        path = self.write(NAME, (
            "[09:00:01]   You:    close every window\n"
            "[09:00:02]   JARVIS: [ACTION: close_all_windows]\n"
            "[09:00:02]   [pushback] gray-zone close -> 'Every one, sir?'\n"))
        turns, _ = rep.parse_log(path)
        self.assertTrue(turns[0]["confirm"])
        self.assertEqual(turns[0]["ran"], set())

    def test_a_failed_action_counts_as_ran(self):
        path = self.write(NAME, (
            "[09:00:01]   You:    take a screenshot\n"
            "[09:00:02]   JARVIS: [ACTION: screenshot] Done, sir.\n"
            "[09:00:02]   [action] screenshot failed [io]: disk full\n"))
        turns, _ = rep.parse_log(path)
        self.assertEqual(turns[0]["ran"], {"screenshot"})

    def test_action_token_regex_is_the_monoliths(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, "bobert_companion.py"),
                  encoding="utf-8", errors="replace") as fh:
            src = fh.read()
        m = re.search(r'^_ACTION_RE = re\.compile\(r"(.+)", re\.IGNORECASE\)$',
                      src, re.M)
        self.assertIsNotNone(m)
        self.assertEqual(rep.ACTION_TOKEN_RE.pattern, m.group(1))
        self.assertTrue(rep.ACTION_TOKEN_RE.flags & re.IGNORECASE)


if __name__ == "__main__":
    unittest.main()
