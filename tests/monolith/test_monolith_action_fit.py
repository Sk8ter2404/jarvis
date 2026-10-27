"""parse_and_run_actions._runner with ACTION_FIT_MODE (v3 W11, 2026-10-27).

THE LIVE TURNS (brain_eval_20261020 cases):
  c006  "Jarvis, get out of the way for just a second." -> game_mode_on
        (swapped the brain to the 12b, stopped the cameras / Kinect);
  c030  "Jarvis, turn on all smart home devices." -> list_smart_home_devices
        only; nothing was turned on;
  c032  "Jarvis, the headset is on though. I'm using it right now." -> the
        same which_mic_is_active that had just run.

Pinned against the real monolith with stub actions:
  * shadow (the shipped default): the action RUNS; "[action-fit] would
    block" is logged; the result is the handler's own;
  * on: the handler is NOT called; the result is a "refused (action-fit)"
    failure (so the follow-up round chooses again);
  * off: no check, no log line, the handler runs - the old path;
  * the check sees the name AFTER the W04 alias step (list_devices ->
    list_smart_home_devices);
  * confirmation still wins: a confirmation-gated action is queued for a
    yes exactly as before, and never reaches the guard;
  * the previous turn is read from the replay history before the turn's
    start; a dispatch with no owner words (proactive) is never judged;
  * a fitting action runs untouched in 'on';
  * the shipped default is 'shadow' and the key is in the [turn-flags]
    tuple; a broken value reads as off.

    python -m unittest tests.monolith.test_monolith_action_fit
"""
from __future__ import annotations

import collections
import contextlib
import io
import time
from unittest import mock

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith

_C006 = "Jarvis, get out of the way for just a second."
_C030 = "Jarvis, turn on all smart home devices."
_C032 = "Jarvis, the headset is on though. I'm using it right now."


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
        self._p(bc, "MISSION_NARRATION_ENABLED", False)
        self._p(bc, "MID_TASK_STATUS_ENABLED", False)
        self._p(bc, "_jarvis_pushback", return_value=None)
        self._p(bc._cmd_autocorrect, "_embed_disabled", True)
        self.history = collections.deque(maxlen=5)
        self._p(bc, "_action_history", self.history)
        self.game_on = mock.Mock(return_value="Game mode engaged, sir.")
        self.lister = mock.Mock(return_value="Fan, lamp, desk light.")
        self.which_mic = mock.Mock(return_value="The Blue mic is active.")
        self.control = mock.Mock(return_value="Turned on 3 devices.")
        self.acts = {
            "game_mode_on": self.game_on,
            "list_smart_home_devices": self.lister,
            "which_mic_is_active": self.which_mic,
            "smart_home_control": self.control,
        }
        self._p(bc, "ACTIONS", self.acts)

    def _dispatch(self, reply, user_text, mode="shadow"):
        bc = self.bc
        self._p(bc, "ACTION_FIT_MODE", mode)
        prev = bc._begin_turn_grounding(user_text) if user_text is not None \
            else None
        out = io.StringIO()
        try:
            with contextlib.redirect_stdout(out):
                res = bc.parse_and_run_actions(reply)
        finally:
            if user_text is not None:
                bc._end_turn_grounding(prev)
        self.log = out.getvalue()
        return res


@requires_monolith
class ShadowTests(_Base):
    def test_c006_runs_and_logs(self):
        _c, results = self._dispatch("Certainly, sir. [ACTION: game_mode_on]",
                                     _C006)
        self.game_on.assert_called_once()
        self.assertIn("[action-fit] would block game_mode_on", self.log)
        self.assertEqual(results[0], ("game_mode_on",
                                      "Game mode engaged, sir.",
                                      results[0][2]))

    def test_c030_runs_and_logs(self):
        self._dispatch("Very good. [ACTION: list_smart_home_devices]", _C030)
        self.lister.assert_called_once()
        self.assertIn("[action-fit] would block list_smart_home_devices",
                      self.log)


@requires_monolith
class OnTests(_Base):
    def test_c006_is_refused_without_running(self):
        _c, results = self._dispatch("On it, sir. [ACTION: game_mode_on]",
                                     _C006, mode="on")
        self.game_on.assert_not_called()
        name, res, _info = results[0]
        self.assertEqual(name, "game_mode_on")
        self.assertTrue(res.startswith("refused (action-fit): not run"))
        self.assertTrue(self.bc._action_result_failed(res))
        self.assertIn("[action-fit] blocked game_mode_on", self.log)

    def test_c030_is_refused_without_running(self):
        _c, results = self._dispatch(
            "Very good. [ACTION: list_smart_home_devices]", _C030, mode="on")
        self.lister.assert_not_called()
        self.assertIn("turn on all smart home devices", results[0][1])

    def test_c032_uses_the_previous_turn(self):
        self.history.append({"action": "which_mic_is_active", "arg": "",
                             "result": "The Blue mic is active.",
                             "at": time.time() - 20})
        _c, results = self._dispatch("On it. [ACTION: which_mic_is_active]",
                                     _C032, mode="on")
        self.which_mic.assert_not_called()
        self.assertIn("ran last turn", results[0][1])

    def test_c032_without_a_previous_turn_runs(self):
        self._dispatch("On it. [ACTION: which_mic_is_active]", _C032,
                       mode="on")
        self.which_mic.assert_called_once()

    def test_after_the_alias_step(self):
        self._p(self.bc, "ACTION_ALIASES", True)
        _c, results = self._dispatch("Very good. [ACTION: list_devices]",
                                     _C030, mode="on")
        self.lister.assert_not_called()
        self.assertEqual(results[0][0], "list_smart_home_devices")
        self.assertIn("refused (action-fit)", results[0][1])

    def test_fitting_action_runs(self):
        _c, results = self._dispatch(
            "Certainly. [ACTION: smart_home_control, turn on all]", _C030,
            mode="on")
        self.control.assert_called_once_with("turn on all")
        self.assertNotIn("[action-fit]", self.log)

    def test_lookup_beside_the_control_runs(self):
        self._dispatch("Certainly. [ACTION: list_smart_home_devices] "
                       "[ACTION: smart_home_control, turn on all]", _C030,
                       mode="on")
        self.lister.assert_called_once()
        self.control.assert_called_once()

    def test_proactive_dispatch_is_never_judged(self):
        self._dispatch("[ACTION: game_mode_on]", None, mode="on")
        self.game_on.assert_called_once()
        self.assertNotIn("[action-fit]", self.log)

    def test_confirmation_still_wins(self):
        bc = self.bc
        self._p(bc, "_needs_confirmation",
                side_effect=lambda n, a: n == "game_mode_on")
        queued = self._p(bc, "_queue_pending_confirmation")
        _c, results = self._dispatch("[ACTION: game_mode_on]", _C006,
                                     mode="on")
        queued.assert_called_once_with("game_mode_on", "")
        self.game_on.assert_not_called()
        self.assertIn("REQUIRES CONFIRMATION", results[0][1])
        self.assertNotIn("[action-fit]", self.log)


@requires_monolith
class OffTests(_Base):
    def test_off_is_the_old_path(self):
        _c, results = self._dispatch("On it, sir. [ACTION: game_mode_on]",
                                     _C006, mode="off")
        self.game_on.assert_called_once()
        self.assertNotIn("[action-fit]", self.log)
        self.assertEqual(results[0][1], "Game mode engaged, sir.")

    def test_broken_value_is_off(self):
        for bad in ("yes", 1, None, True):
            with self.subTest(bad=bad):
                self.game_on.reset_mock()
                self._dispatch("[ACTION: game_mode_on]", _C006, mode=bad)
                self.game_on.assert_called_once()
                self.assertNotIn("[action-fit]", self.log)


@requires_monolith
class SettingTests(MonolithGlobalsTestCase):
    def test_shipped_default_and_turn_flag(self):
        from core import config
        self.assertEqual(config.ACTION_FIT_MODE, "shadow")
        self.assertIn("ACTION_FIT_MODE", self.bc._TURN_FLAG_KEYS)

    def test_turn_records_its_start(self):
        bc = self.bc
        prev = bc._begin_turn_grounding("hello")
        try:
            t0 = bc._turn_grounding.frame.get("t0")
            self.assertIsInstance(t0, float)
            self.assertLessEqual(abs(time.time() - t0), 5.0)
        finally:
            bc._end_turn_grounding(prev)

    def test_prior_actions_outside_a_turn_is_empty(self):
        bc = self.bc
        with mock.patch.object(bc._turn_grounding, "frame", None,
                               create=True):
            self.assertEqual(bc._action_fit_prior_actions(), [])
