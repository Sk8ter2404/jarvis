"""CLOUD_BOOST_UNTIL (2026-10-12): hard + screen-reading questions go to Claude
until a date, then everything returns to local-first by itself. Empty = off."""
from __future__ import annotations

import datetime as dt
import unittest
from unittest import mock

from core import cloud_boost as cb
from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith

D = dt.date(2026, 11, 5)


class BoostWindowTests(unittest.TestCase):
    def setUp(self):
        cb._state["announced"] = None

    def test_empty_and_garbage_are_off_and_silent(self):
        log = mock.Mock()
        for v in ("", None, "  ", "soon", "2026-13-40"):
            self.assertFalse(cb.boost_active(v, D, log=log))
        log.assert_not_called()

    def test_on_through_the_end_date(self):
        log = mock.Mock()
        self.assertTrue(cb.boost_active("2026-11-05", dt.date(2026, 10, 12), log=log))
        self.assertTrue(cb.boost_active("2026-11-05", D, log=log))
        self.assertEqual(log.call_count, 1)         # one ON line, not per call

    def test_expires_the_day_after_with_one_log_line(self):
        log = mock.Mock()
        cb.boost_active("2026-11-05", D, log=log)
        nxt = dt.date(2026, 11, 6)
        self.assertFalse(cb.boost_active("2026-11-05", nxt, log=log))
        self.assertFalse(cb.boost_active("2026-11-05", nxt, log=log))
        self.assertEqual(log.call_count, 2)         # ON once, expired once
        self.assertIn("expired", log.call_args.args[0])


class HardTurnTests(unittest.TestCase):
    def test_hard(self):
        for t in ("think hard about whether I should switch my router setup",
                  "please look it up and tell me what the latest firmware is",
                  "explain step by step how a heat pump works in winter",
                  " ".join(["word"] * 45),
                  "what is the best way to cool the server rack? and also how "
                  "do I keep the noise down at night? then what fans to buy"):
            self.assertTrue(cb.is_hard_turn(t), t)

    def test_not_hard(self):
        for t in ("", "what time is it", "open notepad", "turn the lights off",
                  "play some music please", "how are you doing today sir"):
            self.assertFalse(cb.is_hard_turn(t), t)


@requires_monolith
class BoostRoutingTests(MonolithGlobalsTestCase):
    def _env(self, until):
        bc = self.bc
        cb._state["announced"] = None
        return (mock.patch.dict("os.environ", {"ANTHROPIC_API_KEY": "k"}),
                mock.patch.object(bc, "CLOUD_BOOST_UNTIL", until, create=True),
                mock.patch.object(bc, "AI_BACKEND", "ollama"),
                mock.patch.object(bc, "_note_turn_brain"))

    def _run(self, until, hard, local="local answer", today=dt.date(2026, 10, 12)):
        bc = self.bc
        e1, e2, e3, e4 = self._env(until)
        with e1, e2, e3, e4, \
             mock.patch("core.cloud_boost._dt") as fake, \
             mock.patch.object(bc, "_call_local_llm", return_value=local) as loc, \
             mock.patch.object(bc, "_claude_create"), \
             mock.patch.object(bc, "_llm_client", None), \
             mock.patch.object(bc, "_claude_reply_text", return_value="cloud answer"):
            fake.date.today.return_value = today
            fake.date.fromisoformat = dt.date.fromisoformat
            out = bc._local_then_cloud_or_honest("sys", [{"role": "user", "content": "q"}],
                                                 **({"hard": True} if hard else {}))
            return out, loc

    def test_hard_turn_goes_to_claude_when_boost_on(self):
        out, loc = self._run("2026-11-05", hard=True)
        self.assertEqual(out, "cloud answer")
        loc.assert_not_called()

    def test_easy_turn_stays_local_when_boost_on(self):
        out, loc = self._run("2026-11-05", hard=False)
        self.assertEqual(out, "local answer")

    def test_off_when_empty(self):
        out, loc = self._run("", hard=True)
        self.assertEqual(out, "local answer")

    def test_expired_goes_back_to_local(self):
        out, loc = self._run("2026-11-05", hard=True, today=dt.date(2026, 11, 6))
        self.assertEqual(out, "local answer")

    def test_failed_local_falls_to_claude_on_ollama_backend_only_with_boost(self):
        out, _ = self._run("2026-11-05", hard=False, local=None)
        self.assertEqual(out, "cloud answer")
        out, _ = self._run("", hard=False, local=None)
        self.assertNotEqual(out, "cloud answer")

    def test_vision_goes_cloud_only_while_boosted(self):
        bc = self.bc
        e1, e2, e3, e4 = self._env("2026-11-05")
        with e1, e2, e3, e4, \
             mock.patch("core.cloud_boost._dt") as fake, \
             mock.patch.object(bc, "_local_vision_usable", return_value=True):
            fake.date.today.return_value = dt.date(2026, 10, 12)
            fake.date.fromisoformat = dt.date.fromisoformat
            self.assertFalse(bc._vision_goes_local())
            fake.date.today.return_value = dt.date(2026, 11, 6)
            self.assertTrue(bc._vision_goes_local())
