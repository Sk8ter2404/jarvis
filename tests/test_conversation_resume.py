"""The conversation across a quick restart (2026-11-17, brain-10).

Live 2026-11-17: three restarts between 16:59 and 17:05 (two through the
tray's hardened teardown) each came back with an empty conversation, while
"[session_resume] no greeting: restart 0 min after the last session" called
it the same session. core/conversation_resume decides what may cross;
blue_green_manager.consume_handoff_state(kind=...) takes only a restart's
payload and leaves a blue/green one for its own consumer; core.actions.
_act_restart saves the tail before it spawns the successor.

Light tier (no monolith). Synthetic phrases only.

    python -m unittest tests.test_conversation_resume
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

import blue_green_manager as bgm  # noqa: E402
from core import conversation_resume as cr  # noqa: E402
from core import config as cfg  # noqa: E402
from core.memory_guards import _is_secret_fact  # noqa: E402


def _u(text):
    return {"role": "user", "content": text}


def _a(text):
    return {"role": "assistant", "content": text}


def _talk(pairs):
    out = []
    for i in range(pairs):
        out += [_u(f"open the widget panel {i}"),
                _a(f"[ACTION: widget_show] Panel {i} is up, sir.")]
    return out


class LiveDefaultsTests(unittest.TestCase):
    def test_the_shipped_settings(self):
        self.assertIs(cfg.CONVERSATION_RESUME_ON_RESTART, True)
        self.assertEqual(cfg.CONVERSATION_RESUME_MAX_AGE_S, 900)
        self.assertEqual(cfg.CONVERSATION_RESUME_MAX_MESSAGES, 10)
        self.assertEqual(cr.DEFAULT_MAX_AGE_S, cfg.CONVERSATION_RESUME_MAX_AGE_S)
        self.assertEqual(cr.DEFAULT_MAX_MESSAGES,
                         cfg.CONVERSATION_RESUME_MAX_MESSAGES)
        # The age window is the one the welcome-back greeting already calls
        # "the same session" (bobert_companion.SESSION_RESUME_QUIET_S).
        src = open(os.path.join(_PROJECT, "bobert_companion.py"),
                   encoding="utf-8").read()
        self.assertIn("SESSION_RESUME_QUIET_S = 15 * 60", src)
        # The handoff file's own TTL (teardown -> boot) is the tighter bound.
        self.assertLessEqual(bgm.HANDOFF_STATE_TTL_SECONDS,
                             cfg.CONVERSATION_RESUME_MAX_AGE_S)


class TailTests(unittest.TestCase):
    def tail(self, history, **kw):
        kw.setdefault("is_secret", _is_secret_fact)
        return cr.tail_for_restart(history, cfg.CONVERSATION_RESUME_MAX_MESSAGES,
                                   **kw)

    def test_the_newest_messages_owner_first_reply_last(self):
        hist = _talk(8) + [_u("restart yourself")]
        tail = self.tail(hist)
        self.assertEqual(len(tail), 10)
        self.assertEqual(tail, hist[-11:-1], "the restart ask stays behind")
        self.assertEqual(tail[0]["role"], "user")
        self.assertEqual(tail[-1]["role"], "assistant")
        # Plain copies.
        tail[0]["content"] = "changed"
        self.assertNotEqual(hist[-11]["content"], "changed")

    def test_a_secret_turn_takes_its_replies_with_it(self):
        hist = (_talk(1)
                + [_u("the wifi password is hunter-two"),
                   _a("Noted, sir: hunter-two."),
                   _a("Anything else?")]
                + _talk(1)
                + [_u("what is my token"), _a("I couldn't say, sir.")])
        tail = self.tail(hist)
        text = json.dumps(tail)
        self.assertNotIn("hunter-two", text)
        self.assertNotIn("Anything else", text)
        self.assertNotIn("token", text)
        self.assertEqual(tail, _talk(1) + _talk(1))
        # A reply that reads like a credential goes on its own.
        tail = self.tail([_u("read the router card"),
                          _a("The access code is 1234, sir."),
                          _u("thanks"), _a("Of course, sir.")])
        self.assertEqual(tail, [_u("read the router card"), _u("thanks"),
                                _a("Of course, sir.")])

    def test_bounds(self):
        long = "x" * 5000
        tail = self.tail([_u("read me the page"), _a(long)])
        self.assertLessEqual(len(tail[1]["content"]),
                             cr.MAX_MESSAGE_CHARS + 10)
        self.assertTrue(tail[1]["content"].endswith("[...]"))
        big = []
        for i in range(5):
            big += [_u(f"q{i}"), _a(f"{i}" + "y" * 1990)]
        tail = self.tail(big)
        self.assertLessEqual(sum(len(m["content"]) for m in tail),
                             cr.MAX_TOTAL_CHARS)
        self.assertEqual(tail[0]["role"], "user")
        self.assertEqual(tail[-1]["content"][0], "4")

    def test_junk_and_nothing(self):
        self.assertEqual(self.tail(None), [])
        self.assertEqual(self.tail([]), [])
        self.assertEqual(self.tail([_u("only an ask")]), [])
        self.assertEqual(self.tail([_a("only a reply")]), [])
        self.assertEqual(self.tail(
            ["str", None, {"role": "system", "content": "x"},
             {"role": "user", "content": ["block"]}, _u("  "),
             _u("hello"), _a("Hello, sir.")]),
            [_u("hello"), _a("Hello, sir.")])
        self.assertEqual(cr.tail_for_restart(_talk(2), 0), [])

        def _boom(_t):
            raise RuntimeError
        # A filter that cannot tell keeps the message home.
        self.assertEqual(cr.tail_for_restart(_talk(2), 10, _boom), [])


class AcceptTests(unittest.TestCase):
    NOW = 1_800_000_000.0

    def payload(self, age, tail=None, **kw):
        p = cr.build_payload(_talk(3) if tail is None else tail,
                             self.NOW - age, self.NOW - age + 5, **kw)
        return p

    def accept(self, p):
        return cr.accept(p, self.NOW, cfg.CONVERSATION_RESUME_MAX_AGE_S,
                         cfg.CONVERSATION_RESUME_MAX_MESSAGES, _is_secret_fact)

    def test_a_recent_restart_comes_back(self):
        msgs, why = self.accept(self.payload(40))
        self.assertEqual(msgs, _talk(3))
        self.assertEqual(why, "")
        self.assertAlmostEqual(cr.age_s(self.payload(40), self.NOW), 40.0)

    def test_never_across_a_long_gap(self):
        msgs, why = self.accept(self.payload(901))
        self.assertEqual(msgs, [])
        self.assertIn("15 min ago (over 15 min)", why)
        self.assertEqual(self.accept(self.payload(899))[0], _talk(3))

    def test_what_is_refused(self):
        self.assertEqual(self.accept(None)[0], [])
        self.assertEqual(self.accept({"conversation_tail": _talk(1),
                                      "last_turn_at": self.NOW})[1],
                         "not a restart handoff")
        p = self.payload(10)
        p["last_turn_at"] = None
        self.assertEqual(self.accept(p)[1], "no last-turn time")
        self.assertIn("future", self.accept(self.payload(-600))[1])
        self.assertEqual(self.accept(self.payload(10, tail=[]))[1],
                         "nothing to restore")

    def test_read_back_is_filtered_again(self):
        # A payload written by an older build (or edited on disk) still
        # never puts a secret or an oversized tail into the conversation.
        tail = _talk(8) + [_u("my password is swordfish"), _a("Noted.")]
        msgs, _why = self.accept(self.payload(5, tail=tail))
        self.assertNotIn("swordfish", json.dumps(msgs))
        self.assertLessEqual(len(msgs), cfg.CONVERSATION_RESUME_MAX_MESSAGES)

    def test_payload_shape(self):
        p = cr.build_payload(_talk(1), 100.0, 105.0, ["hello"], [99.0],
                             False, "restart")
        self.assertEqual(p["kind"], "restart")
        self.assertEqual(p["signaled_at"], 105.0)
        self.assertEqual(p["last_turn_at"], 100.0)
        self.assertEqual(p["session_opening_turns"], ["hello"])
        json.dumps(p)


class HandoffKindTests(unittest.TestCase):
    """consume_handoff_state(kind="restart") takes only a restart payload and
    leaves a blue/green one in place for --resume-handoff."""

    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="handoff_kind_")
        self.addCleanup(shutil.rmtree, self.d, True)
        self.f = os.path.join(self.d, "handoff.json")
        p = mock.patch.object(bgm, "HANDOFF_FILE", self.f)
        p.start()
        self.addCleanup(p.stop)

    def test_a_restart_payload_is_taken_once(self):
        now = time.time()
        self.assertTrue(bgm.write_handoff_state(
            cr.build_payload(_talk(1), now - 30, now)))
        got = bgm.consume_handoff_state(kind="restart")
        self.assertEqual(got["conversation_tail"], _talk(1))
        self.assertFalse(os.path.exists(self.f))
        self.assertIsNone(bgm.consume_handoff_state(kind="restart"))

    def test_a_blue_green_payload_is_left_for_its_consumer(self):
        now = time.time()
        bg = {"conversation_tail": _talk(1), "signaled_at": now,
              "active_timers": []}
        self.assertTrue(bgm.write_handoff_state(bg))
        self.assertIsNone(bgm.consume_handoff_state(kind="restart"))
        self.assertTrue(os.path.exists(self.f))
        self.assertEqual(bgm.consume_handoff_state()["active_timers"], [])
        self.assertFalse(os.path.exists(self.f))

    def test_the_ttl_still_applies(self):
        old = time.time() - bgm.HANDOFF_STATE_TTL_SECONDS - 5
        bgm.write_handoff_state(cr.build_payload(_talk(1), old, old))
        self.assertIsNone(bgm.consume_handoff_state(kind="restart"))
        self.assertFalse(os.path.exists(self.f), "a stale one is cleared")


class ActRestartSavesFirstTests(unittest.TestCase):
    """core.actions._act_restart saves the tail BEFORE the successor's
    thread exists (the successor reads it at boot)."""

    def test_saved_before_the_spawn_thread(self):
        import core.actions as A
        order = []
        fake = mock.Mock()
        fake.__file__ = os.path.join(tempfile.gettempdir(), "bc.py")
        fake._save_conversation_for_restart.side_effect = \
            lambda reason: order.append(("save", reason))

        def _thread(target=None, daemon=None):
            order.append(("thread",))
            return mock.Mock()
        with mock.patch.object(A, "_bc", return_value=fake), \
                mock.patch.object(threading, "Thread", side_effect=_thread):
            out = A._act_restart("")
        self.assertIn("Restarting now", out)
        self.assertEqual(order, [("save", "restart"), ("thread",)])

    def test_a_failing_save_never_stops_the_restart(self):
        import core.actions as A
        fake = mock.Mock()
        fake.__file__ = os.path.join(tempfile.gettempdir(), "bc.py")
        fake._save_conversation_for_restart.side_effect = OSError("disk")
        with mock.patch.object(A, "_bc", return_value=fake), \
                mock.patch.object(threading, "Thread") as mt, \
                mock.patch("builtins.print"):
            out = A._act_restart("")
        mt.return_value.start.assert_called_once()
        self.assertIn("Restarting now", out)


if __name__ == "__main__":
    unittest.main()
