"""Head hold in parse_and_run_actions (v3 W02, SPEECH_PLAN_HEAD_HOLD).

THE LIVE TURNS (session_2026-10-26_20-02-43, "it's not playing"):

  20:05:40  "I'm afraid I've no way to check if the video is actually playing
            without running a scan, sir. [ACTION: screenshot]" - voiced, then
            the check it said it could not do ran.
  20:05:48  "I've taken a look at the screen, sir - it appears the video
            hasn't loaded or is simply stuck on a black frame. [ACTION:
            local_describe_screen]" - voiced before the look; the vision
            result then said otherwise.
  20:06:02  "I've looked at the screen locally, sir - ... it appears you've
            only managed to pull up the search results. [ACTION:
            click_on_screen, ...]" - a read-back of the real look: KEPT.

Result hold already holds the prose AFTER a held token; the head before it is
now judged by core.speech_plan.hold_head. 'shadow' (the default) only logs
"[head-hold] would drop"; 'on' drops; 'off' is the old path.

Drives the REAL parse_and_run_actions / _run_llm_dispatch_body with stub
actions and a recording _speak (the answer-first harness). No audio, no LLM.

    python -m unittest tests.monolith.test_monolith_head_hold
"""
from __future__ import annotations

import contextlib
import io

from tests._monolith_harness import requires_monolith
from tests.monolith.test_monolith_answer_first import _Base

_USER = "Jarvis, it's not playing."
_R_200540 = ("[intent:briefing] I'm afraid I've no way to check if the video "
             "is actually playing without running a scan, sir. "
             "[ACTION: screenshot]")
_R_200548 = ("[intent:briefing] I've taken a look at the screen, sir \u2014 it "
             "appears the video hasn't loaded or is simply stuck on a black "
             "frame. [ACTION: local_describe_screen]")
_R_200602 = ("[intent:briefing] I've looked at the screen locally, sir \u2014 "
             "the video is sitting right there on your top monitor, but it "
             "appears you've only managed to pull up the search results. "
             "[ACTION: click_on_screen, the first video]")
_SHOT = "screenshot saved to C:\\shots\\shot.png"
_LOOK = ("A YouTube search results page on the top monitor; no video is "
         "playing.")
_CLICK = "Playing 'the first video' on the top monitor, sir."


@requires_monolith
class _HeadBase(_Base):
    def setUp(self):
        super().setUp()
        bc = self.bc
        self._p(bc, "MISSION_NARRATION_ENABLED", False)
        bc.ACTIONS["screenshot"] = lambda a="": _SHOT
        bc.ACTIONS["local_describe_screen"] = lambda a="": _LOOK
        bc.ACTIONS["click_on_screen"] = lambda a="": _CLICK
        bc.ACTIONS["system_pulse"] = lambda a="": "CPU 12%, memory 40%, sir."
        self._prev_frame = getattr(bc._turn_grounding, "frame", None)
        self.addCleanup(self._restore_frame)

    def _restore_frame(self):
        self.bc._turn_grounding.frame = self._prev_frame

    def _mode(self, mode):
        self._p(self.bc, "SPEECH_PLAN_HEAD_HOLD", mode)

    def _open_turn(self, ran=(), readings=()):
        self.bc._turn_grounding.frame = {"user_text": _USER, "ran": set(ran),
                                         "readings": list(readings)}

    def _parse(self, reply):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cleaned, results = self.bc.parse_and_run_actions(reply)
        return cleaned, results, buf.getvalue()


class ModeTests(_HeadBase):
    def test_the_default_is_shadow(self):
        from core import config
        self.assertEqual(config.SPEECH_PLAN_HEAD_HOLD, "shadow")
        self.assertIn("SPEECH_PLAN_HEAD_HOLD", self.bc._TURN_FLAG_KEYS)

    def test_shadow_logs_and_changes_nothing(self):
        self._mode("off")
        self._open_turn()
        off, off_results, off_printed = self._parse(_R_200540)
        self._mode("shadow")
        self._open_turn()
        cleaned, results, printed = self._parse(_R_200540)
        self.assertEqual(cleaned, off)
        self.assertIn("no way to check", cleaned)
        self.assertEqual(results, off_results)
        self.assertIn("[head-hold] would drop sentence 1/1", printed)
        self.assertIn("[ACTION: screenshot] - inability", printed)
        # The log line names the sentence's place, never its words.
        line = [ln for ln in printed.splitlines() if "[head-hold]" in ln][0]
        self.assertNotIn("no way", line)
        self.assertNotIn("[head-hold]", off_printed)

    def test_on_drops_the_inability(self):
        self._mode("on")
        self._open_turn()
        cleaned, results, printed = self._parse(_R_200540)
        self.assertEqual(cleaned, "")
        self.assertEqual([n for n, _r, _i in results], ["screenshot"])
        self.assertIn("[head-hold] not speaking sentence 1/1", printed)

    def test_off_is_the_old_path(self):
        self._mode("off")
        self._open_turn()
        cleaned, _r, printed = self._parse(_R_200548)
        self.assertIn("it appears the video hasn't loaded", cleaned)
        self.assertNotIn("[head-hold]", printed)

    def test_an_unknown_value_reads_as_off(self):
        self._mode("yes please")
        self._open_turn()
        cleaned, _r, printed = self._parse(_R_200548)
        self.assertIn("it appears the video hasn't loaded", cleaned)
        self.assertNotIn("[head-hold]", printed)


class OnTests(_HeadBase):
    def setUp(self):
        super().setUp()
        self._mode("on")

    def test_200548_the_finding_before_the_look_is_dropped(self):
        self._open_turn(ran={"screenshot"}, readings=[_SHOT])
        cleaned, _r, printed = self._parse(_R_200548)
        self.assertEqual(cleaned, "")
        self.assertIn("[ACTION: local_describe_screen] - finding", printed)

    def test_200602_a_read_back_after_the_look_is_kept(self):
        self._open_turn(ran={"screenshot", "local_describe_screen"},
                        readings=[_SHOT, _LOOK])
        cleaned, _r, printed = self._parse(_R_200602)
        self.assertIn("only managed to pull up the search results", cleaned)
        self.assertNotIn("[head-hold]", printed)

    def test_the_moot_offer_is_dropped(self):
        self._open_turn()
        cleaned, _r, printed = self._parse(
            "[intent:briefing] Shall I run the numbers, sir? "
            "[ACTION: system_pulse]")
        self.assertEqual(cleaned, "")
        self.assertIn("moot offer", printed)

    def test_an_acknowledgement_is_kept(self):
        self._open_turn()
        cleaned, _r, printed = self._parse(
            "Let me take a look, sir. [ACTION: see_screen, read this page] "
            "It shows a text editor.")
        self.assertEqual(cleaned, "Let me take a look, sir.")
        self.assertNotIn("[head-hold]", printed)

    def test_a_side_effect_only_reply_keeps_its_head(self):
        self._open_turn()
        cleaned, _r, printed = self._parse(
            "It appears you'd like it louder, sir. [ACTION: volume_up]")
        self.assertEqual(cleaned, "It appears you'd like it louder, sir.")
        self.assertNotIn("[head-hold]", printed)

    def test_a_claim_another_action_of_the_reply_grounds_is_kept(self):
        # Review 2026-10-27: open_app ran before the held look, so the
        # confirmation is true and stays (in shadow: no "would drop" line).
        self.bc.ACTIONS["open_app"] = lambda a="": "Opened chrome"
        self.bc.ACTIONS["see_screen"] = lambda a="": "A browser window."
        self._open_turn()
        cleaned, results, printed = self._parse(
            "I've opened Chrome, sir. [ACTION: open_app, chrome] "
            "[ACTION: see_screen]")
        self.assertEqual([n for n, _r, _i in results],
                         ["open_app", "see_screen"])
        self.assertEqual(cleaned, "I've opened Chrome, sir.")
        self.assertNotIn("[head-hold]", printed)

    def test_a_check_clause_progress_line_is_kept(self):
        self.bc.ACTIONS["see_screen"] = lambda a="": "A browser window."
        self._open_turn()
        cleaned, _r, printed = self._parse(
            "Let me see what your screen is showing, sir. "
            "[ACTION: see_screen]")
        self.assertEqual(cleaned,
                         "Let me see what your screen is showing, sir.")
        self.assertNotIn("[head-hold]", printed)

    def test_an_offer_of_another_step_is_kept(self):
        self.bc.ACTIONS["see_screen"] = lambda a="": "A browser window."
        self._open_turn()
        head = ("Pulling it up now. If you'd like, I can also check the "
                "weather?")
        cleaned, _r, printed = self._parse(head + " [ACTION: see_screen]")
        self.assertEqual(cleaned, head)
        self.assertNotIn("[head-hold]", printed)

    def test_outside_a_turn_nothing_changes(self):
        self.bc._turn_grounding.frame = None
        cleaned, _r, printed = self._parse(_R_200540)
        self.assertIn("no way to check", cleaned)
        self.assertNotIn("[head-hold]", printed)

    def test_a_head_the_cloud_stream_voiced_is_never_dropped(self):
        self._open_turn()
        self.bc._stream_spoken_prefix[0] = (
            "I'm afraid I've no way to check if the video is actually "
            "playing without running a scan, sir.")
        cleaned, _r, printed = self._parse(_R_200540)
        self.assertIn("no way to check", cleaned)
        self.assertNotIn("[head-hold]", printed)

    def test_a_fault_keeps_the_text(self):
        from core import speech_plan
        self._open_turn()
        self._p(speech_plan, "hold_head", side_effect=RuntimeError("boom"))
        cleaned, _r, printed = self._parse(_R_200540)
        self.assertIn("no way to check", cleaned)
        self.assertIn("[head-hold] skipped (RuntimeError)", printed)

    def test_an_ambiguous_deferral_runs_over_all_the_prose(self):
        bc = self.bc
        self._open_turn()
        prose = ("Which one did you mean, sir? It appears the price is about "
                 "forty dollars.")
        results = [("read_scren", "⚠  AMBIGUOUS: read_screen vs see_screen "
                    "— awaiting clarification", False)]
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            out = bc._head_hold_apply(prose, results, [], frozenset(), [],
                                      "on")
        self.assertEqual(out, "Which one did you mean, sir?")
        self.assertIn("before an autocorrect question - finding",
                      buf.getvalue())

    def test_the_hold_of_a_failed_check_keeps_the_inability(self):
        self.bc.ACTIONS["screenshot"] = lambda a="": "could not capture screen"
        self._open_turn()
        cleaned, _r, _printed = self._parse(_R_200540)
        self.assertIn("no way to check", cleaned)


class DispatchReplayTests(_HeadBase):
    """10-26 20:05:40 -> 20:06:02 through _run_llm_dispatch_body."""

    def _replay(self):
        # _run_llm_dispatch opens the turn ledger around the body.
        self._open_turn()
        self.followup.side_effect = [_R_200548, _R_200602, ""]
        self._run(_R_200540, text=_USER)
        return " ".join(str(s) for s in self.spoken)

    def test_on_speaks_only_what_came_after_a_real_look(self):
        self._mode("on")
        spoken = self._replay()
        self.assertNotIn("no way to check", spoken)
        self.assertNotIn("hasn't loaded", spoken)
        self.assertIn("only managed to pull up the search results", spoken)
        self.assertIn(_CLICK, spoken)
        self.assertIn("hasn't loaded", self._history_text(),
                      "only the audio goes; the reply is remembered")

    def test_shadow_speaks_what_off_speaks(self):
        self._mode("off")
        off = self._replay()
        self.spoken.clear()
        self._mode("shadow")
        self.assertEqual(self._replay(), off)
        self.assertIn("hasn't loaded", off)
