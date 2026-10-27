"""parse_and_run_actions._runner with ACTION_ALIASES (v3 W04, 2026-10-27).

Pinned against the real monolith and the real corrector (embeddings off):
  * on: an invented name with a row in core/action_aliases is rewritten
    exactly before the fuzzy layer, logged "[alias] x -> y", and the rewritten
    action runs with the model's argument; a registered name is never
    rewritten; an argument-needing row with no argument is not applied;
  * on: the verb-clash guard stops the silent send_email -> read_email route
    (unknown action instead); off: the v2.0.231 route, unchanged;
  * on: "Interpreting that as ..." / "Did you mean ... or ...?" speak words,
    not backticked names; off: the old strings byte for byte;
  * every alias target the monolith registers is not _autocorrect_protected,
    not confirmation-gated, not self-terminating, not a destructive-replay
    action; every target is not protected by name;
  * the shipped default is on and the key is in the [turn-flags] tuple;
  * the unknown-action follow-up question (report 05 left it open): an
    "unknown action" result IS a failure, so a follow-up round starts and the
    model may name run_shell itself there. run_shell is never an alias or an
    autocorrect target, and a destructive command is still held for a yes.
"""
from __future__ import annotations

import contextlib
import io
import unittest
from unittest import mock

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith


class _Base(MonolithGlobalsTestCase):
    def _p(self, *args, **kwargs):
        patcher = mock.patch.object(*args, **kwargs)
        m = patcher.start()
        self.addCleanup(patcher.stop)
        return m

    def setUp(self):
        bc = self.bc
        self.spoken = []
        self._p(bc, "_speak", lambda text, *a, **k: self.spoken.append(text))
        self._p(bc, "_write_hud_state", lambda **k: None)
        self._p(bc, "record_session_action", lambda *a, **k: None)
        self._p(bc, "record_action_history", lambda *a, **k: None)
        self._p(bc, "record_action_error", lambda *a, **k: None)
        self._p(bc, "PC_CONTROL_ENABLED", True)
        self._p(bc._cmd_autocorrect, "_embed_disabled", True)
        self.see = mock.Mock(return_value="the screen shows a search page")
        self.playpause = mock.Mock(return_value="toggled playback")
        self.yt = mock.Mock(return_value="playing on YouTube")
        self.read_email = mock.Mock(return_value="no new mail")
        self.shot = mock.Mock(return_value="screenshot saved")
        self.acts = {
            "see_screen": self.see,
            "media_playpause": self.playpause,
            "youtube_play": self.yt,
            "read_email": self.read_email,
            "screenshot": self.shot,
            "volume_mute": mock.Mock(return_value="muted"),
            "volume_up": mock.Mock(return_value="louder"),
            "get_time": mock.Mock(return_value="noon"),
        }
        self._p(bc, "ACTIONS", self.acts)

    def _dispatch(self, reply, user_text="Jarvis, do it.", aliases=True):
        bc = self.bc
        self._p(bc, "ACTION_ALIASES", aliases)
        prev = bc._begin_turn_grounding(user_text)
        out = io.StringIO()
        try:
            with contextlib.redirect_stdout(out):
                res = bc.parse_and_run_actions(reply)
        finally:
            bc._end_turn_grounding(prev)
        self.log = out.getvalue()
        return res


@requires_monolith
class AliasRewriteTests(_Base):
    def test_read_screen_runs_see_screen(self):
        _cleaned, results = self._dispatch(
            "[ACTION: read_screen, monitor:middle]",
            user_text="read me the results")
        self.see.assert_called_once_with("monitor:middle")
        self.assertEqual(results[0][0], "see_screen")
        self.assertIn("[alias] read_screen -> see_screen", self.log)
        self.assertFalse(any("Did you mean" in s for s in self.spoken))

    def test_pause_runs_media_playpause(self):
        self._dispatch("[ACTION: pause]", user_text="pause the video")
        self.playpause.assert_called_once()

    def test_play_video_needs_an_argument(self):
        self._dispatch("[ACTION: play_video, lofi beats]",
                       user_text="play some lofi beats")
        self.yt.assert_called_once_with("lofi beats")
        self.yt.reset_mock()
        _c, results = self._dispatch("[ACTION: play_video]",
                                     user_text="play a video")
        self.yt.assert_not_called()
        self.assertNotIn("[alias]", self.log)

    def test_a_registered_name_is_never_rewritten(self):
        own_pause = mock.Mock(return_value="paused the print")
        self.acts["pause"] = own_pause
        self._dispatch("[ACTION: pause]", user_text="pause")
        own_pause.assert_called_once()
        self.playpause.assert_not_called()
        self.assertNotIn("[alias]", self.log)

    def test_off_never_rewrites(self):
        _c, results = self._dispatch("[ACTION: pause]", user_text="pause",
                                     aliases=False)
        self.playpause.assert_not_called()
        self.assertNotIn("[alias]", self.log)


@requires_monolith
class VerbClashDispatchTests(_Base):
    def test_send_email_no_longer_reads_mail(self):
        _c, results = self._dispatch("[ACTION: send_email, to bob]",
                                     user_text="email bob")
        self.read_email.assert_not_called()
        self.assertEqual(results[0][1], "unknown action: send_email")
        self.assertIn("verb clash", self.log)
        self.assertFalse(any("Interpreting" in s for s in self.spoken))

    def test_off_keeps_the_old_route(self):
        self._dispatch("[ACTION: send_email, to bob]", user_text="email bob",
                       aliases=False)
        self.read_email.assert_called_once()
        self.assertIn("Interpreting that as `read_email`, sir.", self.spoken)


@requires_monolith
class FriendlyLabelSpeechTests(_Base):
    def test_interpreting_line_in_words(self):
        self._dispatch("[ACTION: screen_shot]", user_text="take a screenshot")
        self.shot.assert_called_once()
        self.assertIn("Interpreting that as screenshot, sir.", self.spoken)
        self.assertFalse(any("`" in s for s in self.spoken), self.spoken)

    def test_interpreting_line_off_is_unchanged(self):
        self._dispatch("[ACTION: screen_shot]", user_text="take a screenshot",
                       aliases=False)
        self.assertIn("Interpreting that as `screenshot`, sir.", self.spoken)

    def test_did_you_mean_in_words(self):
        # set_volume is not registered here, so volume_set has no alias
        # target and the corrector's volume_mute / volume_up question stands.
        _c, results = self._dispatch("[ACTION: volume_set, 30]",
                                     user_text="volume to 30")
        self.assertIn("Did you mean volume mute or volume up, sir?",
                      self.spoken)
        self.assertTrue(results[0][1].startswith("⚠  AMBIGUOUS"))
        # The owner can still answer by saying the words back.
        pend = self.bc._pending_autocorrect_choice
        self.assertEqual(pend[0]["primary"][0], "volume_mute")

    def test_did_you_mean_off_is_unchanged(self):
        self._dispatch("[ACTION: volume_set, 30]", user_text="volume to 30",
                       aliases=False)
        self.assertIn("Did you mean `volume_mute` or `volume_up`, sir?",
                      self.spoken)


@requires_monolith
class AliasTargetsAreSafeTests(MonolithGlobalsTestCase):
    def test_targets_pass_every_monolith_gate(self):
        bc = self.bc
        from core import action_aliases as aa
        for dst in sorted(set(aa.ALIASES.values())):
            with self.subTest(dst=dst):
                self.assertFalse(bc._autocorrect_name_protected(dst))
                self.assertFalse(bc._needs_confirmation(dst, ""))
                self.assertNotIn(dst, bc._DESTRUCTIVE_REPLAY_ACTIONS)
                self.assertNotIn(dst, bc._FIRE_AND_EXIT_ACTIONS)
                fn = bc.ACTIONS.get(dst)
                if fn is not None:           # monolith-registered target
                    self.assertFalse(bc._autocorrect_protected(dst))
                    self.assertFalse(bc._self_terminating_target(dst, fn))

    def test_shipped_default_and_turn_flag(self):
        from core import config
        self.assertIs(config.ACTION_ALIASES, True)
        self.assertIn("ACTION_ALIASES", self.bc._TURN_FLAG_KEYS)

    def test_broken_setting_is_off(self):
        bc = self.bc
        for bad in ("yes", 1, None):
            with self.subTest(bad=bad), \
                    mock.patch.object(bc, "ACTION_ALIASES", bad):
                self.assertFalse(bc._action_aliases_on())


@requires_monolith
class UnknownActionFollowUpTests(MonolithGlobalsTestCase):
    """Report 05 section 8: can an invented name become run_shell?"""

    def test_unknown_action_is_a_failure_so_a_follow_up_runs(self):
        self.assertTrue(
            self.bc._action_result_failed("unknown action: move_file"))

    def test_run_shell_is_never_an_alias_or_autocorrect_target(self):
        bc = self.bc
        from core import action_aliases as aa
        self.assertNotIn("run_shell", aa.ALIASES.values())
        self.assertTrue(bc._autocorrect_protected("run_shell"))

    def test_a_destructive_follow_up_shell_command_is_held(self):
        bc = self.bc
        for cmd in (r'Remove-Item -Recurse -Force "$env:USERPROFILE\Downloads\*"',
                    r'del /s /q "%USERPROFILE%\Downloads\*"'):
            with self.subTest(cmd=cmd):
                self.assertIsNotNone(bc._jarvis_pushback("run_shell", cmd))


if __name__ == "__main__":
    unittest.main()
