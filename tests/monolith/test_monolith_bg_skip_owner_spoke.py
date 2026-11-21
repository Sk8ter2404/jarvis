"""v3.0.27 (overhaul reliability-07, the part v3.0.26 left open):
BG_SKIP_IF_OWNER_SPOKE. A background brain call already running when the
owner speaks no longer delivers its spoken / queued line over or right after
his turn. v3.0.26 only kept one from STARTING then.

Pinned: an accepted owner turn moves the core/bg_delivery mark; the proactive
remark and the screen remark are dropped when a turn of his landed during
their call (and delivered when none did); the one-time background changelog
summary is held until the conversation is quiet instead, never dropped;
the flag off restores the v3.0.26 behaviour; the learning batcher is wired
to the mark and the flag (data jobs are never skipped - see
tests/test_learn_batch.OwnerSpokeSkipTests).

Everything external is mocked: no model, no audio, no queue file."""
from __future__ import annotations

import contextlib
import threading
import unittest
from unittest import mock

from tests._monolith_harness import (
    MonolithGlobalsTestCase, load_monolith, requires_monolith)


@requires_monolith
class BgSkipTests(MonolithGlobalsTestCase):
    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def _p(self, *a, **k):
        p = mock.patch.object(*a, **k)
        m = p.start()
        self.addCleanup(p.stop)
        return m

    def setUp(self):
        super().setUp()
        bc = self.bc
        self._p(bc, "BG_SKIP_IF_OWNER_SPOKE", True, create=True)
        self._p(bc, "_turn_in_progress", [False])
        self._p(bc, "_utterance_in_progress", [False])

    def _he_speaks(self):
        """What the main loop does when his transcript is accepted."""
        self.bc._note_owner_turn()
        self.bc._turn_in_progress[0] = False

    # -- the helper --------------------------------------------------------
    def test_default_is_on_and_logged_in_turn_flags(self):
        import core.config as cfg
        self.assertIs(cfg.BG_SKIP_IF_OWNER_SPOKE, True)
        self.assertIn("BG_SKIP_IF_OWNER_SPOKE", self.bc._TURN_FLAG_KEYS)

    def test_an_accepted_turn_moves_the_mark(self):
        m = self.bc._bg_delivery.mark()
        self._he_speaks()
        self.assertNotEqual(self.bc._bg_delivery.mark(), m)

    def test_skip_after_a_turn_and_not_without_one(self):
        bc = self.bc
        m = bc._bg_delivery.mark()
        with mock.patch("builtins.print"):
            self.assertFalse(bc._bg_skip_delivery("x", m))
            self._he_speaks()
            self.assertTrue(bc._bg_skip_delivery("x", m))

    def test_a_turn_or_capture_in_progress_skips(self):
        bc = self.bc
        m = bc._bg_delivery.mark()
        bc._utterance_in_progress[0] = True
        self._p(bc, "_capture_is_sustained_music", return_value=False)
        with mock.patch("builtins.print"):
            self.assertTrue(bc._bg_skip_delivery("x", m))

    def test_a_music_capture_is_not_him(self):
        bc = self.bc
        m = bc._bg_delivery.mark()
        bc._utterance_in_progress[0] = True
        self._p(bc, "_capture_is_sustained_music", return_value=True)
        self.assertFalse(bc._bg_skip_delivery("x", m))

    def test_flag_off_never_skips(self):
        bc = self.bc
        self._p(bc, "BG_SKIP_IF_OWNER_SPOKE", False, create=True)
        m = bc._bg_delivery.mark()
        self._he_speaks()
        bc._turn_in_progress[0] = True
        self.assertFalse(bc._bg_skip_delivery("x", m))

    # -- screen remark -----------------------------------------------------
    def _screen_llm(self, he_speaks: bool):
        bc = self.bc
        self._p(bc, "_llm_quick_goes_local", return_value=False)
        self._p(bc._lt, "background_work",
                lambda *_a, **_k: contextlib.nullcontext())

        def llm(*_a, **_k):
            if he_speaks:
                self._he_speaks()
            return "Bold choice of film."
        self._p(bc, "_llm_quick", llm)
        with mock.patch("builtins.print") as pr:
            out = bc._screen_remark_llm(" addendum", "user")
        return out, [str(c.args[0]) for c in pr.call_args_list if c.args]

    def test_screen_remark_dropped_when_he_spoke_during_the_call(self):
        out, lines = self._screen_llm(True)
        self.assertEqual(out, "")
        self.assertTrue(any("[bg-skip] screen_remark" in l for l in lines))

    def test_screen_remark_kept_when_he_did_not(self):
        out, _lines = self._screen_llm(False)
        self.assertEqual(out, "Bold choice of film.")

    def test_screen_remark_flag_off_keeps_it(self):
        self._p(self.bc, "BG_SKIP_IF_OWNER_SPOKE", False, create=True)
        out, _lines = self._screen_llm(True)
        self.assertEqual(out, "Bold choice of film.")

    # -- proactive remark --------------------------------------------------
    def _proactive(self, he_speaks: bool, injected: bool = False):
        bc = self.bc
        for name in ("pause_face_tracking", "resume_face_tracking",
                     "set_state", "_thinking_loop", "_reset_see_screen_budget"):
            self._p(bc, name)
        self._p(bc, "_pending_work_signature", return_value=("sig",))
        self._p(bc, "_inject_arrived_since", return_value=injected)

        def gen(*_a, **_k):
            if he_speaks:
                self._he_speaks()
            return "Rain is due by four."
        self._p(bc, "generate_proactive_comment", gen)
        run = self._p(bc, "parse_and_run_actions",
                      return_value=("Rain is due by four.", []))
        hist = []
        self._p(bc, "conversation_history", hist)
        self._p(bc, "_speak")
        with mock.patch("builtins.print"):
            try:
                bc._do_proactive_turn({})
            except Exception:
                pass        # only the dispatch decision is pinned here
        return run, hist

    def test_proactive_remark_dropped_when_he_spoke(self):
        run, hist = self._proactive(True)
        run.assert_not_called()
        self.assertEqual(hist, [])

    def test_proactive_remark_dropped_when_a_command_was_typed(self):
        run, hist = self._proactive(False, injected=True)
        run.assert_not_called()
        self.assertEqual(hist, [])

    def test_proactive_remark_goes_on_when_he_did_not(self):
        run, _hist = self._proactive(False)
        run.assert_called_once()

    # -- background changelog summary ---------------------------------------
    def _changelog_bg(self, he_speaks: bool, quiet_after: int = 0):
        """Run the background changelog with the summary call faked;
        ``_conversation_active`` reads True ``quiet_after`` times, then
        False. Returns (announce mock, active-poll count, printed lines)."""
        bc = self.bc
        done = threading.Event()

        def llm(*_a, **_k):
            if he_speaks:
                self._he_speaks()
            return "Sir, since we last spoke, things changed."
        self._p(bc, "_llm_quick", llm)
        self._p(bc, "BOOT_CHANGELOG_BACKGROUND", True, create=True)
        self._p(bc, "_CHANGELOG_HOLD_POLL_S", 0.0)
        polls = [0]

        def active(*_a, **_k):
            polls[0] += 1
            return polls[0] <= quiet_after
        self._p(bc, "_conversation_active", active)
        announce = self._p(bc, "proactive_announce",
                           side_effect=lambda *a, **k: done.set())
        with mock.patch("builtins.print") as pr:
            bc._changelog_announce("1.0", "1.1", "- a change", [])
            self.assertTrue(done.wait(5))
            for t in threading.enumerate():
                if t.name == "boot-changelog":
                    t.join(5)
        lines = [str(c.args[0]) for c in pr.call_args_list if c.args]
        return announce, polls[0], lines

    def test_background_changelog_held_not_dropped_when_he_spoke(self):
        # The version is stamped announced right after the thread starts:
        # a dropped summary would never be said, so it waits for quiet.
        announce, polls, lines = self._changelog_bg(True, quiet_after=3)
        announce.assert_called_once()
        self.assertEqual(polls, 4)
        self.assertTrue(any("[bg-hold] boot_changelog_summary" in l
                            for l in lines))

    def test_background_changelog_not_held_when_he_did_not_speak(self):
        announce, polls, lines = self._changelog_bg(False, quiet_after=3)
        announce.assert_called_once()
        self.assertEqual(polls, 0)
        self.assertFalse(any("[bg-hold]" in l for l in lines))

    def test_background_changelog_flag_off_no_hold(self):
        self._p(self.bc, "BG_SKIP_IF_OWNER_SPOKE", False, create=True)
        announce, polls, _lines = self._changelog_bg(True, quiet_after=3)
        announce.assert_called_once()
        self.assertEqual(polls, 0)

    def test_background_changelog_hold_is_capped(self):
        self._p(self.bc, "_CHANGELOG_HOLD_MAX_S", 0.0)
        announce, polls, _lines = self._changelog_bg(True, quiet_after=99)
        announce.assert_called_once()
        self.assertEqual(polls, 1)

    # -- learning batcher wiring -------------------------------------------
    def test_learn_batcher_gets_the_mark_and_the_flag(self):
        bc = self.bc
        self._p(bc, "_learn_batcher_state", [None])
        b = bc._learn_batcher()
        self.assertIs(b._owner_mark, bc._bg_delivery.mark)
        self.assertTrue(b._skip_spoken())
        self._p(bc, "BG_SKIP_IF_OWNER_SPOKE", False, create=True)
        self.assertFalse(b._skip_spoken())


if __name__ == "__main__":
    unittest.main()
