"""Regression tests for the 2026-10-01 actions/control fix batch (actions-b).

Each class names the finding it pins. Every test fails on the pre-fix code
(122b746 / v2.0.153) and passes now:

  B032  "do that again" no longer re-fires a draft send: a held send_* (or its
        confirm_pending_draft alias) is never written to the replay history.
  B033  a deferred confirmation lapses after CONFIRMATION_TTL_S instead of
        waiting forever, "Jarvis, yes" confirms instead of cancelling, and a
        drain that loses the race with the gesture thread stops cleanly.
  B035  "shuffle my road trip playlist" reaches the Library > Playlists flow
        end to end (skill fallback -> real apple_music router).
  B036  the Apple Music playlist flow pins every vision step to the player's
        monitor, like the search flow.
  B038  boot re-arms the timers the previous process saved (prod only).
  B039  the media window JARVIS records is the one it OPENED, never the
        owner's own browser window that happens to match the title first.
  B041  the live web-player title (LRM + NO-BREAK SPACE) counts as Apple Music.
  B093  confirm_pending_draft - and any handler that IS a send_* handler -
        goes through the draft read-back gate, on every path that runs one.

(B030 / B031 / B032's regex live in test_monolith_sec5.py next to the tests
they changed.)

Monolith-tier (full-deps): run locally; skip on the light-deps CI runner.
    python -B -m unittest tests.monolith.test_monolith_actions_b_fixes
"""
from __future__ import annotations

import sys
import time
import types
import unittest
from unittest import mock

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith


class _Win:
    """A pygetwindow-like window: title, native handle, activate()."""

    def __init__(self, hwnd: int, title: str):
        self._hWnd = hwnd
        self.title = title
        self.activated = 0

    def activate(self):
        self.activated += 1


def _fake_gw(*snapshots):
    """A fake pygetwindow whose getAllWindows() returns each list in turn
    (the last one repeats)."""
    gw = types.ModuleType("pygetwindow")
    calls = {"n": 0}

    def _all():
        i = min(calls["n"], len(snapshots) - 1)
        calls["n"] += 1
        return list(snapshots[i])
    gw.getAllWindows = _all
    return gw


# ════════════════════════════════════════════════════════════════════════════
#  B093 — every path that runs a draft sender goes through the read-back gate
# ════════════════════════════════════════════════════════════════════════════
@requires_monolith
class DraftSendGateTests(MonolithGlobalsTestCase):
    def setUp(self):
        bc = self.bc
        self.sender = mock.Mock(return_value="sent the email")
        acts = dict(bc.ACTIONS)
        # email_triage's real shape: ONE sender under three names.
        acts["send_draft"] = self.sender
        acts["send_pending_draft"] = self.sender
        acts["confirm_pending_draft"] = self.sender
        self.acts = acts
        for name, value in (("ACTIONS", acts),
                            ("_speak", lambda *a, **k: None),
                            ("_write_hud_state", lambda **k: None),
                            ("record_session_action", lambda *a, **k: None),
                            ("_cmd_autocorrect", None),
                            ("PC_CONTROL_ENABLED", True),
                            ("_needs_confirmation", lambda n, a: False),
                            ("_jarvis_pushback", lambda n, a: None)):
            p = mock.patch.object(bc, name, value)
            p.start()
            self.addCleanup(p.stop)
        self.gate = mock.patch.object(
            bc._draft_preview_gate, "run_with_gate",
            return_value="Holding the draft, sir.")
        self.run_with_gate = self.gate.start()
        self.addCleanup(self.gate.stop)

    def test_confirm_pending_draft_is_read_back_before_sending(self):
        _cleaned, results = self.bc.parse_and_run_actions(
            "[ACTION: confirm_pending_draft]")
        self.run_with_gate.assert_called_once()
        self.assertEqual(self.run_with_gate.call_args.args[0],
                         "confirm_pending_draft")
        self.sender.assert_not_called()
        self.assertEqual(results[0][1], "Holding the draft, sir.")

    def test_any_alias_of_a_send_handler_is_gated(self):
        # A future skill alias the name rule has never heard of.
        self.acts["deliver_reply"] = self.sender
        self.bc.parse_and_run_actions("[ACTION: deliver_reply]")
        self.run_with_gate.assert_called_once()
        self.sender.assert_not_called()

    def test_a_confirmed_deferred_send_is_still_read_back(self):
        bc = self.bc
        bc._pending_confirmation.clear()
        bc._pending_confirmation.append(("send_draft", ""))
        self.assertTrue(bc.handle_confirmation_response("yes"))
        self.run_with_gate.assert_called_once()
        self.sender.assert_not_called()

    def test_ordinary_actions_are_not_gated(self):
        other = mock.Mock(return_value="ok")
        self.acts["open_notes_x"] = other
        self.bc.parse_and_run_actions("[ACTION: open_notes_x, hi]")
        other.assert_called_once_with("hi")
        self.run_with_gate.assert_not_called()


# ════════════════════════════════════════════════════════════════════════════
#  B032 — a draft send never enters the replay history
# ════════════════════════════════════════════════════════════════════════════
@requires_monolith
class ReplayHistoryDraftSendTests(MonolithGlobalsTestCase):
    def test_draft_sends_are_not_recorded(self):
        bc = self.bc
        sender = mock.Mock()
        acts = dict(bc.ACTIONS)
        acts["send_draft"] = sender
        acts["confirm_pending_draft"] = sender
        with mock.patch.object(bc, "ACTIONS", acts):
            bc._action_history.clear()
            bc.record_action_history("send_draft", "", "Holding the draft, sir.")
            bc.record_action_history("confirm_pending_draft", "", "sent")
            self.assertEqual(list(bc._action_history), [])
            bc.record_action_history("set_timer", "5 minutes", "timer #1 set")
            self.assertEqual([e["action"] for e in bc._action_history],
                             ["set_timer"])


# ════════════════════════════════════════════════════════════════════════════
#  B033 — confirmations lapse; "Jarvis, yes" confirms; a lost race is safe
# ════════════════════════════════════════════════════════════════════════════
@requires_monolith
class PendingConfirmationLifetimeTests(MonolithGlobalsTestCase):
    def setUp(self):
        bc = self.bc
        self.ran = []
        self.spoken = []
        acts = dict(bc.ACTIONS)
        acts["wipe_thing_x"] = lambda a: self.ran.append(a) or "wiped"
        for name, value in (("ACTIONS", acts),
                            ("_speak", lambda t, *a, **k: self.spoken.append(t)),
                            ("_write_hud_state", lambda **k: None),
                            ("record_session_action", lambda *a, **k: None),
                            ("record_action_history", lambda *a, **k: None),
                            ("_cmd_autocorrect", None),
                            ("PC_CONTROL_ENABLED", True),
                            ("_needs_confirmation",
                             lambda n, a: n == "wipe_thing_x"),
                            ("_jarvis_pushback", lambda n, a: None)):
            p = mock.patch.object(bc, name, value)
            p.start()
            self.addCleanup(p.stop)
        bc._pending_confirmation.clear()

    def _defer(self):
        _cleaned, results = self.bc.parse_and_run_actions(
            "[ACTION: wipe_thing_x, everything]")
        self.assertIn("REQUIRES CONFIRMATION", results[0][1])
        self.assertEqual(self.bc._pending_confirmation,
                         [("wipe_thing_x", "everything")])
        self.spoken.clear()

    def test_an_unanswered_prompt_lapses_instead_of_running_later(self):
        bc = self.bc
        self._defer()
        later = time.monotonic() + 3600.0      # an hour later
        with mock.patch.object(bc.time, "monotonic", return_value=later):
            consumed = bc.handle_confirmation_response("yes")
        self.assertFalse(consumed, "a lapsed prompt must not eat the turn")
        self.assertEqual(self.ran, [])
        self.assertEqual(bc._pending_confirmation, [])
        self.assertTrue(any("lapsed" in s for s in self.spoken), self.spoken)

    def test_a_fresh_prompt_still_runs_on_yes(self):
        self._defer()
        self.assertTrue(self.bc.handle_confirmation_response("yes"))
        self.assertEqual(self.ran, ["everything"])

    def test_wake_word_led_yes_confirms_instead_of_cancelling(self):
        # Wake-word mode only lets "Jarvis, yes" through; it used to fail
        # startswith("yes") and CANCEL the action the owner had approved.
        self._defer()
        self.assertTrue(self.bc.handle_confirmation_response("Jarvis, yes."))
        self.assertEqual(self.ran, ["everything"])
        self.assertNotIn("Cancelled.", self.spoken)

    def test_wake_word_led_no_still_cancels(self):
        self._defer()
        self.assertTrue(self.bc.handle_confirmation_response("Jarvis, no"))
        self.assertEqual(self.ran, [])
        self.assertIn("Cancelled.", self.spoken)

    def test_drain_that_loses_the_gesture_race_stops_cleanly(self):
        bc = self.bc

        class _Raced(list):
            """Non-empty when checked, empty by the time it is popped - the
            Kinect gesture thread cleared it in between (a SWIPE cancel, or a
            raised hand expiring a lapsed prompt; a gesture never confirms)."""
            def pop(self, *a):
                self.clear()
                raise IndexError("pop from empty list")

        raced = _Raced([("wipe_thing_x", "everything")])
        with mock.patch.object(bc, "_pending_confirmation", raced):
            self.assertTrue(bc.handle_confirmation_response("yes"))
        self.assertEqual(self.ran, [])

    def test_prompt_age_reflects_the_prompt(self):
        # The age _expire_pending_confirmation and _reply_prompt_pending read.
        bc = self.bc
        self.assertIsNone(bc.pending_confirmation_age())
        self._defer()
        age = bc.pending_confirmation_age()
        self.assertIsNotNone(age)
        self.assertLess(age, bc.CONFIRMATION_TTL_S)
        self.assertFalse(hasattr(bc, "GESTURE_CONFIRM_MAX_AGE_S"),
                         "a raised hand confirms nothing, at any age")


# ════════════════════════════════════════════════════════════════════════════
#  B035 — "shuffle my X playlist" reaches the playlist flow end to end
# ════════════════════════════════════════════════════════════════════════════
@requires_monolith
class ShufflePlaylistEndToEndTests(MonolithGlobalsTestCase):
    def test_shuffle_playlist_fallback_reaches_library_playlists(self):
        from tests._skill_harness import load_skill_isolated
        bc = self.bc
        mod, _acts = load_skill_isolated("itunes_library")
        with mock.patch.object(mod.itunes_bridge, "get_client",
                               return_value=(None, "iTunes is gone")), \
                mock.patch.dict(sys.modules, {"__main__": bc}), \
                mock.patch.object(bc, "_apple_music_play_playlist",
                                  return_value="playing 'road trip'") as flow, \
                mock.patch.object(bc, "_streaming_auto_play",
                                  return_value="a random song") as song:
            out = mod.play_playlist("shuffle road trip")
        flow.assert_called_once_with("road trip")
        song.assert_not_called()
        self.assertEqual(out, "playing 'road trip'")


# ════════════════════════════════════════════════════════════════════════════
#  B036 — the playlist flow's vision is pinned to the player's monitor
# ════════════════════════════════════════════════════════════════════════════
@requires_monolith
class PlaylistFlowMonitorPinTests(MonolithGlobalsTestCase):
    def test_every_vision_step_uses_the_player_monitor(self):
        bc = self.bc
        win = _Win(4242, "Apple Music - Web Player - Google Chrome")
        with mock.patch.object(bc, "_open_url_in_browser", return_value="chrome"), \
                mock.patch.object(bc, "_find_browser_window_matching",
                                  return_value=win), \
                mock.patch.object(bc, "_ensure_window_visible_maximized",
                                  return_value=True), \
                mock.patch.object(bc, "_monitor_name_for_window",
                                  return_value="left"), \
                mock.patch.object(bc.time, "sleep"), \
                mock.patch.multiple(bc, SCREEN_VISION_ENABLED=True,
                                    UI_AUTOMATION_ENABLED=True), \
                mock.patch.object(bc, "_vision_click_backend_available",
                                  return_value=True), \
                mock.patch.object(bc, "_streaming_find_with_retry",
                                  side_effect=[None, (12, 13)]) as fwr, \
                mock.patch.object(bc, "find_click_target",
                                  side_effect=[(1, 1), (2, 2)]) as fct, \
                mock.patch.object(bc, "ui_click"), \
                mock.patch.object(bc, "_streaming_play_and_verify",
                                  return_value="playing 'workout'") as pv, \
                mock.patch.dict(bc._JARVIS_MEDIA_WINDOW_HWND, {}, clear=True):
            out = bc._apple_music_play_playlist("workout")
            self.assertEqual(bc._JARVIS_MEDIA_WINDOW_HWND.get("apple_music"), 4242)
        self.assertEqual(out, "playing 'workout'")
        self.assertEqual(fwr.call_count, 2)
        self.assertEqual(fct.call_count, 2)
        for call in fwr.call_args_list + fct.call_args_list:
            self.assertEqual(call.kwargs.get("monitor"), "left", call)
        # ...and the play / verify step inherits it through cfg.
        self.assertEqual(pv.call_args.args[0].get("vision_monitor"), "left")


# ════════════════════════════════════════════════════════════════════════════
#  B039 — record the window JARVIS opened, never the owner's
# ════════════════════════════════════════════════════════════════════════════
@requires_monolith
class MediaWindowIsTheNewOneTests(MonolithGlobalsTestCase):
    OWNER = "Apple Music - Web Player - Google Chrome"   # his own, on top
    NEW = "Apple Music - Web Player - Google Chrome"

    def test_exclude_skips_windows_that_existed_before(self):
        bc = self.bc
        owner, new = _Win(111, self.OWNER), _Win(222, self.NEW)
        gw = _fake_gw([owner, new])
        with mock.patch.dict(sys.modules, {"pygetwindow": gw}):
            self.assertIs(bc._find_browser_window_matching(["apple music"]), owner)
            self.assertIs(bc._find_browser_window_matching(
                ["apple music"], exclude={111}), new)
            self.assertIsNone(bc._find_browser_window_matching(
                ["apple music"], exclude={111, 222}))

    def test_search_flow_records_and_maximizes_only_the_new_window(self):
        bc = self.bc
        owner, new = _Win(111, self.OWNER), _Win(222, self.NEW)
        # Before the open only the owner's window exists; after it, his window
        # is STILL first in z-order (the new one was refused the foreground).
        gw = _fake_gw([owner], [owner, new])
        with mock.patch.dict(sys.modules, {"pygetwindow": gw}), \
                mock.patch.object(bc, "_apple_music_resolve_track",
                                  return_value=None), \
                mock.patch.object(bc, "_open_url_in_browser",
                                  return_value="chrome"), \
                mock.patch.object(bc, "_ensure_window_visible_maximized",
                                  return_value=True) as maxi, \
                mock.patch.object(bc, "_monitor_name_for_window",
                                  return_value=None), \
                mock.patch.object(bc.time, "sleep"), \
                mock.patch.multiple(bc, SCREEN_VISION_ENABLED=False,
                                    UI_AUTOMATION_ENABLED=False), \
                mock.patch.dict(bc._JARVIS_MEDIA_WINDOW_HWND, {}, clear=True):
            bc._streaming_auto_play("apple_music", "thriller")
            self.assertEqual(bc._JARVIS_MEDIA_WINDOW_HWND.get("apple_music"), 222)
        maxi.assert_called_once_with(222)
        self.assertEqual(owner.activated, 0, "the owner's window was touched")

    def test_no_new_window_records_nothing(self):
        bc = self.bc
        owner = _Win(111, self.OWNER)
        gw = _fake_gw([owner])           # the open added a TAB, no new window
        with mock.patch.dict(sys.modules, {"pygetwindow": gw}), \
                mock.patch.object(bc, "_apple_music_resolve_track",
                                  return_value=None), \
                mock.patch.object(bc, "_open_url_in_browser",
                                  return_value="chrome:webbrowser"), \
                mock.patch.object(bc, "_ensure_window_visible_maximized") as maxi, \
                mock.patch.object(bc.time, "sleep"), \
                mock.patch.multiple(bc, SCREEN_VISION_ENABLED=False,
                                    UI_AUTOMATION_ENABLED=False), \
                mock.patch.dict(bc._JARVIS_MEDIA_WINDOW_HWND, {}, clear=True):
            bc._streaming_auto_play("apple_music", "thriller")
            self.assertNotIn("apple_music", bc._JARVIS_MEDIA_WINDOW_HWND)
        maxi.assert_not_called()


# ════════════════════════════════════════════════════════════════════════════
#  B041 — the real web-player title is recognised
# ════════════════════════════════════════════════════════════════════════════
@requires_monolith
class AppleMusicNbspTitleTests(MonolithGlobalsTestCase):
    def test_live_title_with_lrm_and_nbsp_counts_as_apple_music(self):
        bc = self.bc
        title = "\u200eApple\xa0Music - Web Player - Google Chrome"
        gw = _fake_gw([_Win(5, title)])
        saved = bc._apple_music_last_seen[0]
        self.addCleanup(bc._apple_music_last_seen.__setitem__, 0, saved)
        bc._apple_music_last_seen[0] = 0.0          # cold cache
        with mock.patch.dict(sys.modules, {"pygetwindow": gw}):
            self.assertTrue(bc._apple_music_chrome_active())


# ════════════════════════════════════════════════════════════════════════════
#  B038 — boot re-arms the timers the previous process saved
# ════════════════════════════════════════════════════════════════════════════
@requires_monolith
class BootTimerRestoreTests(MonolithGlobalsTestCase):
    def _skill(self, n):
        mod = types.ModuleType("skill_timer")
        mod.restore_persisted_timers = mock.Mock(return_value=n)
        return mod

    def test_prod_boot_re_arms_saved_timers(self):
        bc = self.bc
        skill = self._skill(2)
        with mock.patch.dict(sys.modules, {"skill_timer": skill}), \
                mock.patch.object(bc, "_is_staging", return_value=False):
            self.assertEqual(bc._restore_persisted_timers_at_boot(), 2)
        skill.restore_persisted_timers.assert_called_once_with()

    def test_staging_never_fires_the_owners_reminders(self):
        bc = self.bc
        skill = self._skill(2)
        with mock.patch.dict(sys.modules, {"skill_timer": skill}), \
                mock.patch.object(bc, "_is_staging", return_value=True):
            self.assertEqual(bc._restore_persisted_timers_at_boot(), 0)
        skill.restore_persisted_timers.assert_not_called()

    def test_missing_skill_or_failing_restore_never_raises(self):
        bc = self.bc
        with mock.patch.dict(sys.modules, {"skill_timer": None}), \
                mock.patch.object(bc, "_is_staging", return_value=False):
            self.assertEqual(bc._restore_persisted_timers_at_boot(), 0)
        skill = self._skill(0)
        skill.restore_persisted_timers.side_effect = OSError("disk")
        with mock.patch.dict(sys.modules, {"skill_timer": skill}), \
                mock.patch.object(bc, "_is_staging", return_value=False):
            self.assertEqual(bc._restore_persisted_timers_at_boot(), 0)


if __name__ == "__main__":
    unittest.main()
