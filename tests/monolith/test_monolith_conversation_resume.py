"""The conversation across a quick restart, through the monolith's real save
and boot paths (2026-11-17, brain-10).

Live 2026-11-17: three restarts between 16:59 and 17:05 each came back with
an empty conversation, while "[session_resume] no greeting: restart 0 min
after the last session" called it the same session. _act_restart now saves
the tail (_save_conversation_for_restart) through the blue/green handoff
file with kind "restart", and the next prod start puts it back
(_resume_conversation_after_restart) when the owner's last turn is at most
CONVERSATION_RESUME_MAX_AGE_S (900 s) ago. These tests drive both halves
against a temp handoff file at the live defaults; synthetic phrases only.

    python -m unittest tests.monolith.test_monolith_conversation_resume
"""
from __future__ import annotations

import contextlib
import inspect
import io
import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from unittest import mock

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(os.path.dirname(_HERE))
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from tests._monolith_harness import (  # noqa: E402
    MonolithGlobalsTestCase, requires_monolith,
)


def _talk(pairs):
    out = []
    for i in range(pairs):
        out += [{"role": "user", "content": f"open the widget panel {i}"},
                {"role": "assistant",
                 "content": f"[ACTION: widget_show] Panel {i} is up, sir."}]
    return out


@requires_monolith
class ConversationResumeTests(MonolithGlobalsTestCase):

    def setUp(self):
        super().setUp()
        bc = self.bc
        self.assertIsNotNone(bc._bgm)
        self.d = tempfile.mkdtemp(prefix="conv_resume_")
        self.addCleanup(shutil.rmtree, self.d, True)
        self.f = os.path.join(self.d, "handoff.json")
        self._p(bc._bgm, "HANDOFF_FILE", self.f)
        self._p(bc, "BLUE_GREEN_ROLE", "prod")
        self._p(sys, "argv", ["bobert_companion.py"])
        bc._guest_mode.set_on(False)
        bc._session_opening_turns[:] = []
        bc._session_opening_ts[:] = []
        bc._session_opening_lost[0] = False
        self.addCleanup(lambda: bc._session_opening_turns.clear())
        self.addCleanup(lambda: bc._session_opening_ts.clear())

    def _p(self, *args, **kwargs):
        patcher = mock.patch.object(*args, **kwargs)
        m = patcher.start()
        self.addCleanup(patcher.stop)
        return m

    def _run(self, fn, *a):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            res = fn(*a)
        return res, out.getvalue()

    def _talked(self, seconds_ago, history):
        bc = self.bc
        bc.conversation_history[:] = [dict(m) for m in history]
        bc._last_convo_activity[0] = time.monotonic() - seconds_ago
        bc._last_owner_turn_at[0] = time.monotonic() - seconds_ago

    def _new_process(self):
        """What a fresh start sees: no conversation, no turn, no record."""
        bc = self.bc
        bc.conversation_history[:] = []
        bc._last_convo_activity[0] = 0.0
        bc._last_owner_turn_at[0] = 0.0
        bc._restored_last_turn_at[0] = 0.0
        bc._session_opening_turns[:] = []
        bc._session_opening_ts[:] = []
        bc._session_opening_lost[0] = False
        bc._handoff_resumed[0] = False

    def test_a_restart_seconds_after_talking_keeps_the_conversation(self):
        bc = self.bc
        self.assertIs(bc.CONVERSATION_RESUME_ON_RESTART, True)
        self._talked(20, _talk(4) + [{"role": "user",
                                      "content": "restart yourself"}])
        bc._session_opening_turns[:] = ["open the widget panel 0"]
        bc._session_opening_ts[:] = [time.time() - 300]
        n, out = self._run(bc._save_conversation_for_restart, "restart")
        self.assertEqual(n, 8)
        self.assertIn("[conversation-resume] saved 8 message(s) for the "
                      "restart (last turn", out)
        with open(self.f, encoding="utf-8") as fh:
            saved = json.load(fh)
        self.assertEqual(saved["kind"], "restart")
        self.assertNotIn("restart yourself", json.dumps(saved))
        self._new_process()
        n, out = self._run(bc._resume_conversation_after_restart)
        self.assertEqual(n, 8)
        self.assertEqual(bc.conversation_history, _talk(4))
        self.assertRegex(out, r"\[conversation-resume\] restored 8 "
                              r"message\(s\) from before the restart "
                              r"\(last turn \d+ s ago\)")
        # "What was the first thing I asked you" still answers from the
        # real start of the session.
        self.assertEqual(bc._session_opening_turns,
                         ["open the widget panel 0"])
        self.assertFalse(os.path.exists(self.f), "taken once")
        # A second restart before any new turn carries the same last turn.
        first = saved["last_turn_at"]
        n, _out = self._run(bc._save_conversation_for_restart, "restart")
        self.assertEqual(n, 8)
        with open(self.f, encoding="utf-8") as fh:
            self.assertAlmostEqual(json.load(fh)["last_turn_at"], first,
                                   delta=1.0)

    def test_never_across_a_long_gap(self):
        bc = self.bc
        # Save side: nothing goes to disk for a turn already too old.
        self._talked(1000, _talk(2))
        n, out = self._run(bc._save_conversation_for_restart, "restart")
        self.assertEqual(n, 0)
        self.assertFalse(os.path.exists(self.f))
        self.assertIn("nothing saved: the last turn was 17 min ago", out)
        # Boot side: a payload whose last turn is over 900 s old.
        from core import conversation_resume as cr
        now = time.time()
        bc._bgm.write_handoff_state(cr.build_payload(_talk(2), now - 901,
                                                     now - 60))
        self._new_process()
        n, out = self._run(bc._resume_conversation_after_restart)
        self.assertEqual(n, 0)
        self.assertEqual(bc.conversation_history, [])
        self.assertIn("[conversation-resume] not restored: the last turn "
                      "was 15 min ago (over 15 min)", out)

    def test_what_saves_nothing(self):
        bc = self.bc
        self._talked(10, _talk(2))
        bc._guest_mode.set_on(True)
        try:
            n, out = self._run(bc._save_conversation_for_restart, "restart")
        finally:
            bc._guest_mode.set_on(False)
        self.assertEqual(n, 0)
        self.assertIn("guest mode: nothing saved", out)
        with mock.patch.object(bc, "BLUE_GREEN_ROLE", "staging"):
            self.assertEqual(bc._save_conversation_for_restart(), 0)
        with mock.patch.object(bc, "CONVERSATION_RESUME_ON_RESTART", False):
            self.assertEqual(bc._save_conversation_for_restart(), 0)
        with mock.patch.object(bc, "_bgm", None):
            self.assertEqual(bc._save_conversation_for_restart(), 0)
        self._new_process()                     # no turn at all
        bc.conversation_history[:] = _talk(2)
        self.assertEqual(bc._save_conversation_for_restart(), 0)
        self.assertFalse(os.path.exists(self.f))

    def test_a_secret_never_crosses(self):
        bc = self.bc
        self._talked(10, _talk(1) + [
            {"role": "user", "content": "my bank password is lemon-tree"},
            {"role": "assistant", "content": "Noted, sir."}] + _talk(1))
        bc._session_opening_turns[:] = ["my bank password is lemon-tree"]
        bc._session_opening_ts[:] = [time.time() - 60]
        self._run(bc._save_conversation_for_restart, "restart")
        with open(self.f, encoding="utf-8") as fh:
            text = fh.read()
        self.assertNotIn("lemon-tree", text)
        self.assertTrue(json.loads(text)["session_opening_lost"])

    def test_what_restores_nothing(self):
        bc = self.bc
        now = time.time()
        # A blue/green payload is left for --resume-handoff.
        bc._bgm.write_handoff_state({"conversation_tail": _talk(1),
                                     "signaled_at": now})
        self._new_process()
        n, out = self._run(bc._resume_conversation_after_restart)
        self.assertEqual(n, 0)
        self.assertTrue(os.path.exists(self.f))
        self.assertIn("not a fresh restart handoff", out)
        os.remove(self.f)
        # Nothing on disk: a fresh start says nothing.
        n, out = self._run(bc._resume_conversation_after_restart)
        self.assertEqual((n, out), (0, ""))
        from core import conversation_resume as cr

        def _payload():
            bc._bgm.write_handoff_state(cr.build_payload(_talk(1), now - 5,
                                                         now))
        _payload()
        for why, ctx in (
                ("blue/green relaunch", mock.patch.object(
                    sys, "argv", ["x", bc._bgm.RESUME_HANDOFF_FLAG])),
                ("staging", mock.patch.object(bc, "BLUE_GREEN_ROLE",
                                              "staging")),
                ("off", mock.patch.object(
                    bc, "CONVERSATION_RESUME_ON_RESTART", False)),
                ("swap already resumed", mock.patch.object(
                    bc, "_handoff_resumed", [True]))):
            with self.subTest(why=why), ctx:
                self.assertEqual(bc._resume_conversation_after_restart(), 0)
                self.assertTrue(os.path.exists(self.f))
        bc.conversation_history[:] = _talk(1)
        self.assertEqual(bc._resume_conversation_after_restart(), 0)
        self.assertTrue(os.path.exists(self.f))

    def test_boot_restores_after_the_log_opens_before_the_greeting(self):
        src = inspect.getsource(self.bc.main)
        i = src.index("_resume_conversation_after_restart()")
        self.assertGreater(i, src.index("setup_logging()"))
        self.assertGreater(i, src.index("_consume_blue_green_handoff()"))
        self.assertLess(i, src.index("maybe_session_resume_greeting()"))


if __name__ == "__main__":
    unittest.main()
