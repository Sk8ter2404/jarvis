"""v3.0.11 lane T wiring in the monolith: the post-STT Teams fix, the Teams hotword,
the teams_unread shortcut, the prompt, and the idle remark's prompt and drop."""
import unittest
from unittest import mock

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith


class _Base(MonolithGlobalsTestCase):
    def _p(self, *args, **kwargs):
        patcher = mock.patch.object(*args, **kwargs)
        m = patcher.start()
        self.addCleanup(patcher.stop)
        return m


@requires_monolith
class SttTeamsTests(_Base):
    LIVE = "Jarvis, any messages I missed on TV while I was gone?"

    def test_parakeet_path_fixes_and_tags(self):
        self._p(self.bc, "STT_REPLACEMENTS", {})
        self._p(self.bc, "STT_REPLACEMENTS_PARAKEET", {})
        self._p(self.bc, "STT_TEAMS_FIX", True)
        with mock.patch("builtins.print") as out:
            got = self.bc._parakeet_post_text(self.LIVE)
        self.assertEqual(got, "Jarvis, any messages I missed on Teams while I was gone?")
        logged = " ".join(str(c) for c in out.call_args_list)
        self.assertIn("[stt] teams-fix", logged)
        self.assertNotIn("while I was gone", logged)

    def test_flag_off_leaves_it(self):
        self._p(self.bc, "STT_TEAMS_FIX", False)
        self.assertEqual(self.bc._stt_fix_mishearings(self.LIVE), self.LIVE)

    def test_a_tv_line_stays_and_the_youtube_fix_still_runs(self):
        self._p(self.bc, "STT_TEAMS_FIX", True)
        self.assertEqual(self.bc._stt_fix_mishearings("what's on TV"), "what's on TV")
        self.assertEqual(self.bc._stt_fix_mishearings("open you two"), "open YouTube")


@requires_monolith
class WhisperTeamsTests(_Base):
    def _transcribe(self, hotword_flag, said="any messages on TV"):
        import numpy as np
        seg = mock.Mock(text=said, no_speech_prob=0.1, avg_logprob=-0.2)
        stt = mock.Mock()
        stt.transcribe.return_value = (iter([seg]), mock.Mock(no_speech_prob=0.1))
        self._p(self.bc, "_ensure_whisper")
        self._p(self.bc, "_stt", stt)
        self._p(self.bc, "_stt_engine", "faster_whisper")
        self._p(self.bc, "STT_HOTWORDS", "", create=True)
        self._p(self.bc, "STT_REPLACEMENTS", {}, create=True)
        self._p(self.bc, "STT_TEAMS_FIX", True)
        self._p(self.bc, "STT_TEAMS_HOTWORD", hotword_flag)
        with mock.patch("builtins.print"):
            text, _conf = self.bc.transcribe(np.zeros(8, dtype=np.float32))
        return text, stt.transcribe.call_args.kwargs["hotwords"]

    def test_the_hotword_reaches_whisper_and_the_text_is_fixed(self):
        self.assertEqual(self._transcribe(True), ("any messages on Teams", "Teams"))

    def test_flag_off_no_hotword(self):
        self.assertEqual(self._transcribe(False), ("any messages on Teams", None))


@requires_monolith
class TeamsShortcutTests(_Base):
    def setUp(self):
        super().setUp()
        bc = self.bc
        self._speak = self._p(bc, "_speak")
        self._p(bc, "set_state")
        self._p(bc, "_append_turn")
        self._p(bc, "FAST_PATHS_ENABLED", True)
        self._p(bc, "TEAMS_UNREAD_ACTION", True)
        self.action = mock.Mock(return_value="3 unread Teams chats, sir.")
        patcher = mock.patch.dict(bc.ACTIONS, {"teams_unread": self.action})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_the_live_line_is_answered_by_the_action(self):
        with mock.patch("builtins.print"):
            hit = self.bc._run_teams_unread_shortcut(
                "Jarvis, any messages I missed on Teams while I was gone?")
        self.assertTrue(hit)
        self.action.assert_called_once()
        self._speak.assert_called_once_with("3 unread Teams chats, sir.")

    def test_plain_email_and_ordinary_lines_fall_through(self):
        for line in ("any email?", "what's on TV", "open Teams",
                     "any good movies on TV"):
            with self.subTest(line=line):
                self.assertFalse(self.bc._run_teams_unread_shortcut(line))
        self.action.assert_not_called()

    def test_flag_off_falls_through(self):
        self._p(self.bc, "TEAMS_UNREAD_ACTION", False)
        self.assertFalse(self.bc._run_teams_unread_shortcut("any Teams messages"))
        self.action.assert_not_called()

    def test_with_no_action_registered_the_llm_answers(self):
        del self.bc.ACTIONS["teams_unread"]
        self.assertFalse(self.bc._run_teams_unread_shortcut("any Teams messages"))

    def test_a_raising_action_says_so(self):
        self.action.side_effect = RuntimeError("boom")
        with mock.patch("builtins.print"):
            self.assertTrue(self.bc._run_teams_unread_shortcut("who messaged me"))
        self._speak.assert_called_once_with("I couldn't read Teams just now, sir.")

    def test_spoken_as_written_and_a_status_action(self):
        self.assertIn("teams_unread", self.bc.SPEAK_RESULT_VERBATIM_ACTIONS)
        self.assertNotIn("teams_unread", self.bc.INFORMATIVE_ACTIONS)
        self.assertTrue(self.bc._action_fit.is_status_action("teams_unread"))

    def test_action_topic_keeps_it_off_an_email_line(self):
        miss = self.bc._action_fit.action_topic_missing
        self.assertTrue(miss("teams_unread", "any email?"))
        self.assertTrue(miss("teams_unread", "any unread email"))
        self.assertEqual(miss("teams_unread", "any messages I missed"), "")
        self.assertEqual(miss("teams_unread", "anything on Teams"), "")

    def test_cheat_sheet_lists_it_only_when_registered(self):
        self.assertIn("teams_unread", self.bc._teams_unread_hint())
        del self.bc.ACTIONS["teams_unread"]
        self.assertEqual(self.bc._teams_unread_hint(), "")


class PromptTests(unittest.TestCase):
    def test_the_prompt_lists_it_and_the_refusal_rule_excludes_messages(self):
        from core import prompts
        self.assertIn("teams_unread", prompts.PC_CONTROL_PROMPT)
        self.assertIn("Trigger phrases: 'any messages I missed'", prompts.PC_CONTROL_PROMPT)
        rules = prompts.PC_CONTROL_SAFETY_RULES
        self.assertIn("[ACTION: teams_unread]", rules)
        self.assertIn("never state a message count no action gave you", rules)
        self.assertIn("NO SUCH CAPABILITY EXISTS", rules)


@requires_monolith
class IdleRemarkTests(_Base):
    def setUp(self):
        super().setUp()
        bc = self.bc
        self._p(bc, "_llm_quick_goes_local", return_value=False)
        self._p(bc, "_system_prompt", "BASE")
        self._p(bc, "_proactive_recent", [])
        self.llm = self._p(bc, "_llm_quick", return_value="Octopuses have three hearts, sir.")

    def _run(self, reply=None):
        if reply is not None:
            self.llm.return_value = reply
        with mock.patch("builtins.print") as out:
            line = self.bc.generate_proactive_comment(now=1_800_000_000.0)
        return line, " ".join(str(c) for c in out.call_args_list)

    def test_a_fact_is_spoken_and_the_prompt_asks_for_one(self):
        self._p(self.bc, "PROACTIVE_GROUNDED_ONLY", True)
        line, _log = self._run()
        self.assertEqual(line, "Octopuses have three hearts, sir.")
        kw = self.llm.call_args.kwargs
        self.assertIn("exactly SKIP", kw["system"])
        self.assertIn("Never mention the silence", kw["system"])
        self.assertNotIn("check-in", kw["system"])
        self.assertNotIn("quiet for a while", kw["system"])
        self.assertNotIn("There has been silence", kw["user"])

    def test_the_live_remarks_are_dropped(self):
        self._p(self.bc, "PROACTIVE_GROUNDED_ONLY", True)
        for reply in ("[intent:observation] If I may observe, sir, it's a "
                      "remarkably productive morning so far.",
                      "[intent:observation] You've been quite still for some "
                      "time, sir; I trust you're contemplating something.",
                      "[intent:observation] I couldn't help but notice the "
                      "silence, sir; it's rather uncharacteristic.",
                      "SKIP", "[intent:dry_wit] SKIP"):
            with self.subTest(reply=reply):
                line, log = self._run(reply)
                self.assertEqual(line, "")
                self.assertIn("[proactive] dropped", log)

    def test_flag_off_is_the_old_prompt_and_no_drop(self):
        self._p(self.bc, "PROACTIVE_GROUNDED_ONLY", False)
        line, _log = self._run("You've been quite still for some time, sir.")
        self.assertEqual(line, "You've been quite still for some time, sir.")
        kw = self.llm.call_args.kwargs
        self.assertIn("a brief check-in", kw["system"])
        self.assertIn("There has been silence", kw["user"])


if __name__ == "__main__":
    unittest.main()
