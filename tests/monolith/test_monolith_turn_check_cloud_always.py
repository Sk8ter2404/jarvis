"""TURN_CHECK_CLOUD_ALWAYS (2026-10-28, backlog "keep the Claude safety net
after the boost ends").

With chat routed local and no Claude boost running, TURN_CHECK_MODE 'on'
could not retry a failed turn: _turn_check_after_chain passed
cloud_allowed=False, and _claude_oneshot refused the call anyway. The new
setting (default False - each caught turn is a paid Claude call) keeps both
gates open while an API key exists.

Two layers, both through the live code:
  * a whole turn through _run_llm_dispatch (stub brain, stub actions, the
    Claude call stubbed) - the retry happens only with the setting on;
  * the real _claude_oneshot gate with a stub client - the setting opens it
    for the turn checker's retry only, never for another caller.

No real audio, no LLM, no network. The key is a dummy value.
"""
from __future__ import annotations

import os
import unittest
from unittest import mock

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith
from tests.monolith.test_monolith_integration_1002 import _BothFeaturesBase

_DUMMY_KEY = {"ANTHROPIC_API_KEY": "dummy-test-key"}


@requires_monolith
class CloudAlwaysTurnTests(_BothFeaturesBase):
    """Chat routed local, the boost over: the owner's setup after the boost
    window."""

    TURN_CHECK = "on"

    def setUp(self):
        super().setUp()
        self.cloud.return_value = False                 # chat cloud: no
        self._p(self.bc, "_cloud_boost_on", return_value=False)
        env = mock.patch.dict(os.environ, _DUMMY_KEY)
        env.start()
        self.addCleanup(env.stop)
        self.llm.return_value = "Right away, sir."      # a claim, no token
        self.oneshot.return_value = "[ACTION: volume_up] Volume up, sir."

    def _flag(self, on):
        p = mock.patch.object(self.bc, "TURN_CHECK_CLOUD_ALWAYS", on,
                              create=True)
        p.start()
        self.addCleanup(p.stop)

    def test_ships_off(self):
        from core import config as cfg
        self.assertIs(cfg.TURN_CHECK_CLOUD_ALWAYS, False)

    def test_off_no_retry_after_the_boost(self):
        self._flag(False)
        self._run("volume up")
        self.oneshot.assert_not_called()
        self.assertEqual(self.calls["volume_up"], [])
        self.assertIs(self.tc_rows()[0]["cloud_allowed"], False)

    def test_on_the_retry_still_runs(self):
        self._flag(True)
        self._run("volume up")
        self.assertEqual(self.oneshot.call_count, 1)
        kw = self.oneshot.call_args.kwargs
        self.assertIs(kw.get("cloud_always"), True)
        self.assertEqual(kw.get("caller"), "turn_check_retry")
        self.assertEqual(self.calls["volume_up"], [""])
        self.assertIn("Volume up, sir.", self.spoken)
        row = self.tc_rows()[0]
        self.assertIs(row["cloud_allowed"], True)
        self.assertIs(row["escalated"], True)

    def test_on_without_a_key_no_retry(self):
        self._flag(True)
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}):
            self._run("volume up")
        self.oneshot.assert_not_called()


@requires_monolith
class CloudAlwaysOneshotGateTests(MonolithGlobalsTestCase):
    """The real _claude_oneshot gate, a stub client behind it."""

    def setUp(self):
        bc = self.bc
        for name, val in (("AI_BACKEND", "ollama"),):
            p = mock.patch.object(bc, name, val)
            p.start()
            self.addCleanup(p.stop)
        for name, kw in (("_cloud_boost_on", {"return_value": False}),
                         ("_note_turn_brain", {})):
            p = mock.patch.object(bc, name, **kw)
            p.start()
            self.addCleanup(p.stop)
        self.client = mock.Mock()
        self.client.complete.return_value = "fine, sir"
        p = mock.patch.object(bc, "_llm_client", self.client)
        p.start()
        self.addCleanup(p.stop)
        env = mock.patch.dict(os.environ, _DUMMY_KEY)
        env.start()
        self.addCleanup(env.stop)

    def _call(self, **kw):
        return self.bc._claude_oneshot(
            "system", [{"role": "user", "content": "hello"}], **kw)

    def _flag(self, on):
        p = mock.patch.object(self.bc, "TURN_CHECK_CLOUD_ALWAYS", on,
                              create=True)
        p.start()
        self.addCleanup(p.stop)

    def test_off_the_retry_is_refused_without_the_boost(self):
        self._flag(False)
        self.assertIsNone(self._call(caller="turn_check_retry", boost=True,
                                     cloud_always=True))
        self.client.complete.assert_not_called()

    def test_on_the_retry_reaches_claude(self):
        self._flag(True)
        self.assertEqual(self._call(caller="turn_check_retry", boost=True,
                                    cloud_always=True), "fine, sir")
        self.assertEqual(self.client.complete.call_args.kwargs.get("caller"),
                         "turn_check_retry")

    def test_on_never_widens_another_caller(self):
        self._flag(True)
        self.assertIsNone(self._call(caller="other", boost=True))
        self.client.complete.assert_not_called()


if __name__ == "__main__":   # pragma: no cover
    unittest.main()
