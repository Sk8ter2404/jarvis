"""The warm-restart greeting ("When we left off you were working on ...").

THE LIVE FAILURES (2026-10-01), each the first thing JARVIS said after a restart:
  * 17:33:37  "...working on Fix open_url/browser window placement — new windows
              sometimes spawn off-screen to the r…" (a to-do cut mid-word, a code
              identifier read aloud);
  * 19:44:30  "...working on jarvis, accelo, entry, token, bambu, hbo max, ..." (the
              last "command" was Whisper reading the hotwords hint back);
  * 21:49:25  "...working on jarvis plays skrillex essentials on youtube" (a raw media
              command, wake word and all, 6 minutes after a deliberate restart with
              the music already playing);
  * 22:35:41  "...working on jarvis, can you tell me how much it costs to run you" (a
              question is not work).
  * 23:09:43  a session summary cut mid-word ("...for the time, whic…").

Pinned here:
  * within SESSION_RESUME_QUIET_S of the last session (a restart) there is no
    auto-greeting; the verbal ask (force) still answers;
  * last commands that are hotword echoes, list-shaped, media / system / window
    commands or questions are skipped (the next one back may be used);
  * the wake word is stripped, and a command is phrased "you'd asked me to ...";
  * to-do lines cut at a clause, code identifiers made speakable, and any cut
    lands on a word boundary.

Generic stand-in names only.

    python -m unittest tests.monolith.test_monolith_session_resume
"""
from __future__ import annotations

import os
import shutil
import tempfile
import time
import unittest
from unittest import mock

from tests._monolith_harness import (MonolithGlobalsTestCase, load_monolith,
                                     requires_monolith)

WARM = 40 * 60          # 40 min after the last session: past the restart quiet


@requires_monolith
class _Base(MonolithGlobalsTestCase):
    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def setUp(self):
        bc = self.bc
        self._patches = [
            mock.patch.object(bc, "_last_session_end_ts",
                              return_value=time.time() - WARM),
            mock.patch.object(bc, "_last_n_user_commands", return_value=[]),
            mock.patch.object(bc, "_last_queued_task_line", return_value=""),
            mock.patch.object(bc.pattern_memory, "get_session_summaries",
                              return_value=[]),
            mock.patch.object(bc._stt_vocab, "live_hotwords",
                              side_effect=lambda fb: "Zorblat, Entry, Token, Flemwick"),
        ]
        for p in self._patches:
            p.start()
            self.addCleanup(p.stop)
        # An empty logs folder of our own: the restart quiet also reads the
        # previous process's session log (2026-10-02), never the real one.
        self.logs = tempfile.mkdtemp(prefix="jarvis-resume-logs-")
        self.addCleanup(shutil.rmtree, self.logs, True)
        for p in (mock.patch.object(bc, "LOGS_DIR", self.logs),
                  mock.patch.object(bc, "_log_file_path",
                                    os.path.join(self.logs, "session_now.log"),
                                    create=True)):
            p.start()
            self.addCleanup(p.stop)
        orig_latch = list(bc._session_resume_done)
        self.addCleanup(lambda: bc._session_resume_done.__setitem__(
            slice(None), orig_latch))

    def _resume(self, commands=(), task="", summary="", age=WARM, force=False):
        bc = self.bc
        with mock.patch.object(bc, "_last_session_end_ts",
                               return_value=time.time() - age), \
                mock.patch.object(bc, "_last_n_user_commands",
                                  return_value=list(commands)), \
                mock.patch.object(bc, "_last_queued_task_line", return_value=task), \
                mock.patch.object(bc.pattern_memory, "get_session_summaries",
                                  return_value=([{"summary": summary}] if summary else [])):
            return bc._build_session_resume(force=force)


class QuickRestartTests(_Base):
    def test_no_auto_greeting_minutes_after_a_restart(self):
        # Live 21:49:25: 6 minutes after a deliberate restart.
        text, details = self._resume(["Jarvis, do the Zorblat skit"], age=6 * 60)
        self.assertEqual(text, "")
        self.assertTrue(details.get("quick_restart"))

    def test_the_verbal_ask_still_answers_after_a_restart(self):
        text, _details = self._resume(["Jarvis, do the Zorblat skit"], age=6 * 60,
                                      force=True)
        self.assertIn("zorblat skit", text)

    def test_past_the_quiet_the_greeting_is_back(self):
        text, details = self._resume(["Jarvis, do the Zorblat skit"], age=20 * 60)
        self.assertIn("Welcome back", text)
        self.assertFalse(details.get("quick_restart"))

    def test_maybe_greeting_says_why_it_kept_quiet(self):
        bc = self.bc
        bc._session_resume_done[0] = False
        with mock.patch.object(bc, "_last_session_end_ts",
                               return_value=time.time() - 5 * 60), \
                mock.patch("builtins.print") as pr:
            out = bc.maybe_session_resume_greeting()
        self.assertEqual(out, "")
        self.assertTrue(any("[session_resume]" in str(c.args[0]) and "restart" in str(c.args[0])
                            for c in pr.call_args_list if c.args))


class ProcessEndTests(_Base):
    """Review repair (2026-10-02). Live 19:44:30 the greeting fired 80 s
    after the previous process ended (19:43:09): the quiet window was
    measured from the last session-summary checkpoint (17:29:03, 2.3 h
    earlier), and with no usable command the greeting read out the summary
    itself - "you were working on The user provides a series of fragmented
    commands ..."."""

    _LIVE_SUMMARY = ("The user provides a series of fragmented commands and "
                     "repetitive prompts to the AI, including requests to "
                     "change the model. The assistant answers each.")

    def _previous_log(self, ended_ago_s, name="session_prev.log"):
        path = os.path.join(self.logs, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write("=== previous session ===" + chr(10))
        t = time.time() - ended_ago_s
        os.utime(path, (t, t))
        return path

    def test_80_s_after_the_previous_process_there_is_no_greeting(self):
        self._previous_log(80)
        text, details = self._resume(age=int(2.3 * 3600),
                                     summary=self._LIVE_SUMMARY)
        self.assertEqual(text, "")
        self.assertTrue(details.get("quick_restart"))

    def test_the_current_processes_own_log_does_not_count(self):
        own = os.path.join(self.logs, "session_now.log")
        with open(own, "w", encoding="utf-8") as f:
            f.write("this process" + chr(10))
        self._previous_log(3 * 3600)
        _text, details = self._resume(["Jarvis, do the Zorblat skit"],
                                      age=int(3 * 3600))
        self.assertFalse(details.get("quick_restart"))

    def test_a_long_gap_since_the_previous_process_still_greets(self):
        self._previous_log(40 * 60)
        text, _details = self._resume(["Jarvis, do the Zorblat skit"],
                                      age=int(2.3 * 3600))
        self.assertIn("Welcome back", text)

    def test_a_third_person_summary_is_never_what_he_was_working_on(self):
        for summary in (self._LIVE_SUMMARY,
                        "User asked about the weather and a timer.",
                        "The assistant explained the GPU load.",
                        "JARVIS reported the print status."):
            with self.subTest(summary=summary[:30]):
                text, details = self._resume(age=int(2.3 * 3600),
                                             summary=summary)
                self.assertNotIn("working on The", text)
                self.assertNotIn(summary.split(".")[0][:25], text)
                self.assertEqual(details.get("work"), "")

    def test_a_first_person_summary_still_works(self):
        text, _details = self._resume(
            age=int(2.3 * 3600),
            summary="Wiring the Zorblat sensor to the rover. Then a break.")
        self.assertIn("Wiring the Zorblat sensor to the rover", text)


class CommandChoiceTests(_Base):
    def test_hotword_read_back_is_not_work(self):
        # Live 19:44:30 shape.
        text, details = self._resume(
            ["JARVIS, Zorblat, Entry, Token, Flemwick, HBO Max, Quonset, 365, ESA,"])
        self.assertEqual(details["work"], "")
        self.assertNotIn("zorblat", text.lower())
        self.assertIn("I'm afraid", text)

    def test_list_shaped_command_is_not_work_even_without_hotwords(self):
        bc = self.bc
        with mock.patch.object(bc._stt_vocab, "live_hotwords", side_effect=lambda fb: ""):
            _text, details = self._resume(["JARVIS, JARVIS, JARVIS, JARVIS, JARVIS,"])
            self.assertEqual(details["work"], "")
            _text, details = self._resume(["Jarvis, Quonset, Brindle, Wexley, Ombra, Tisk"])
            self.assertEqual(details["work"], "")

    def test_media_command_is_not_work(self):
        # Live 21:49:25 shape (Whisper heard "play" as "plays").
        for cmd in ("Jarvis plays Brindle Essentials on YouTube.",
                    "Jarvis, pause the music", "Jarvis, next song",
                    "Jarvis, turn the volume down", "Jarvis, open HBO Max.",
                    "Jarvis put it on the main monitor.", "Jarvis, restart yourself",
                    "go to sleep", "Jarvis, set a timer for ten minutes"):
            with self.subTest(cmd=cmd):
                _text, details = self._resume([cmd])
                self.assertEqual(details["work"], "")

    def test_a_question_is_not_work(self):
        # Live 22:35:41 shape.
        for cmd in ("Jarvis, Can you tell me how much it costs to run you?",
                    "Jarvis, what time is it?", "how's my print doing"):
            with self.subTest(cmd=cmd):
                _text, details = self._resume([cmd])
                self.assertEqual(details["work"], "")

    def test_wake_word_stripped_and_phrased_as_a_request(self):
        text, details = self._resume(["Jarvis, do the Zorblat skit with Flemwick."])
        self.assertEqual(details["work"], "do the zorblat skit with flemwick")
        self.assertIn("you'd asked me to do the zorblat skit with flemwick", text)
        self.assertNotIn("jarvis", text.lower().replace("welcome back", ""))
        self.assertIn("At your service", text)

    def test_hey_jarvis_is_stripped_too(self):
        _text, details = self._resume(["Hey Jarvis, use the Kinect again"])
        self.assertEqual(details["work"], "use the kinect again")

    def test_an_older_real_request_is_used_when_the_newest_is_skipped(self):
        _text, details = self._resume(["Jarvis plays Brindle Essentials on YouTube.",
                                       "Jarvis, what time is it?",
                                       "Jarvis, do the Zorblat skit"])
        self.assertEqual(details["work"], "do the zorblat skit")


class CutTests(_Base):
    def test_todo_line_cut_at_the_clause_with_speakable_identifiers(self):
        # Live 17:33:37 shape.
        line = ("- [ ] **2026-10-01 15:00** — Fix open_url/browser window placement — "
                "new windows sometimes spawn off-screen to the right of the right "
                "monitor instead of centered on an available display")
        self.assertEqual(self.bc._summarise_task_line(line),
                         "Fix open url browser window placement")
        text, details = self._resume(task=line)
        self.assertIn("working on Fix open url browser window placement —", text)
        self.assertNotIn("…", text)

    def test_backticks_and_call_parens_are_not_read_aloud(self):
        line = "- [ ] **2026-10-01 15:00** [bug] — make `see_screen()` honour the monitor"
        self.assertEqual(self.bc._summarise_task_line(line),
                         "make see screen honour the monitor")

    def test_summary_cut_lands_on_a_word_boundary(self):
        # Live 23:09:43 shape: "...for the time, whic…".
        summary = ("Reviewed the history of the construction site while checking the "
                   "time, which was provided after a long detour through the logs. More.")
        _text, details = self._resume(summary=summary)
        work = details["work"]
        self.assertLessEqual(len(work), 90)
        first = summary.split(".", 1)[0]
        stem = work.rstrip("…").rstrip()
        self.assertTrue(first.startswith(stem), work)
        nxt = first[len(stem):len(stem) + 1]
        self.assertIn(nxt, (" ", ",", ""), f"cut mid-word: {work!r}")

    def test_word_cut_never_ends_on_a_dangling_small_word(self):
        cut = self.bc._cut_at_word(
            "Refactored the entire audio capture and playback pipeline including "
            "the noise cancellation stages and the barge-in watchdog", 60)
        self.assertTrue(cut.endswith("…"))
        last = cut.rstrip("…").split()[-1].lower()
        self.assertNotIn(last, {"and", "the", "a", "to", "of", "with", "for", "in", "on"})

    def test_short_text_is_untouched(self):
        self.assertEqual(self.bc._cut_at_word("fix tray.py crash on boot", 90),
                         "fix tray.py crash on boot")

    def test_never_raises_on_junk(self):
        self.assertEqual(self.bc._cut_at_word("", 90), "")
        self.assertEqual(self.bc._cut_at_word(None, 90), "")
        self.assertEqual(self.bc._resume_command_phrase(None), "")
        self.assertEqual(self.bc._resume_command_phrase(""), "")


if __name__ == "__main__":
    unittest.main()
