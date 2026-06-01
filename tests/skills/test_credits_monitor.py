"""Logic tests for skills/credits_monitor.py.

Reads the Claude credit balance by vision-scraping the off-screen billing page.
We never open Chrome or call a real VLM: bobert_companion is faked via
_import_companion, so _read_credits_via_vision exercises only the
capture→ask_vision→parse pipeline. We cover:

  • the $X.XX / LOGIN_REQUIRED / NOT_FOUND answer parsing (incl. comma grouping)
  • capture / vision failure → graceful (None, status)
  • _check_and_maybe_alert — low-balance alert + cooldown + login nudge
  • the check_credits action — success / login / unreadable, plus the
    "already in progress" lock
  • _save_state redirected to a tempfile (no real credits_state.json write)

The background monitor thread never starts (harness neuters threads).
"""
from __future__ import annotations

import os
import tempfile
import time
import types
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


def _fake_companion(*, capture=("PNGBYTES", "ok"), vision="BALANCE: $42.50"):
    """A fake bobert_companion exposing _open_url_offscreen_capture + ask_vision."""
    bc = types.ModuleType("bobert_companion")
    bc._open_url_offscreen_capture = mock.MagicMock(return_value=capture)
    bc.ask_vision = mock.MagicMock(return_value=vision)
    # proactive_announce is consulted by _enqueue_speech; make it a no-op that
    # reports success so nothing falls through to a real file write.
    bc.proactive_announce = mock.MagicMock(return_value=True)
    return bc


class CreditsVisionParseTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("credits_monitor")

    def _read(self, *, capture=("PNG", "ok"), vision="BALANCE: $42.50"):
        bc = _fake_companion(capture=capture, vision=vision)
        with mock.patch.object(self.mod, "_import_companion", return_value=bc):
            return self.mod._read_credits_via_vision()

    def test_parses_dollar_amount(self):
        dollars, raw = self._read(vision="BALANCE: $42.50")
        self.assertAlmostEqual(dollars, 42.50)

    def test_parses_amount_with_comma_grouping(self):
        dollars, _ = self._read(vision="BALANCE: $1,234.56")
        self.assertAlmostEqual(dollars, 1234.56)

    def test_login_required(self):
        dollars, raw = self._read(vision="LOGIN_REQUIRED")
        self.assertIsNone(dollars)
        self.assertEqual(raw, "login_required")

    def test_not_found(self):
        dollars, raw = self._read(vision="NOT_FOUND")
        self.assertIsNone(dollars)
        self.assertIn("NOT_FOUND", raw)

    def test_capture_failure_returns_status(self):
        dollars, raw = self._read(capture=(None, "spawn_failed"))
        self.assertIsNone(dollars)
        self.assertIn("capture_failed", raw)

    def test_unparseable_vision_answer(self):
        dollars, raw = self._read(vision="the balance is unclear")
        self.assertIsNone(dollars)
        self.assertIn("unclear", raw)


class CreditsSaveStateTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("credits_monitor")

    def test_save_state_writes_json(self):
        fd, p = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        self.addCleanup(lambda: os.path.exists(p) and os.remove(p))
        with mock.patch.object(self.mod, "_STATE_FILE", p):
            self.mod._save_state(12.34, "BALANCE: $12.34")
        import json
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        self.assertAlmostEqual(data["balance"], 12.34)
        self.assertIn("checked_at", data)


class CreditsAlertTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("credits_monitor")
        # Make sure the check lock is free and timestamps are reset.
        self.mod._last_alert_at[0] = 0.0
        self.mod._last_login_alert_at[0] = 0.0

    def test_low_balance_queues_alert(self):
        with mock.patch.object(self.mod, "_read_credits_via_vision",
                               return_value=(2.0, "BALANCE: $2.00")), \
             mock.patch.object(self.mod, "_save_state"), \
             mock.patch.object(self.mod, "_enqueue_speech") as enq:
            self.mod._check_and_maybe_alert()
        enq.assert_called_once()
        self.assertIn("2.00 dollars", enq.call_args.args[0])

    def test_healthy_balance_no_alert(self):
        with mock.patch.object(self.mod, "_read_credits_via_vision",
                               return_value=(50.0, "BALANCE: $50.00")), \
             mock.patch.object(self.mod, "_save_state"), \
             mock.patch.object(self.mod, "_enqueue_speech") as enq:
            self.mod._check_and_maybe_alert()
        enq.assert_not_called()

    def test_low_balance_respects_cooldown(self):
        self.mod._last_alert_at[0] = time.time()  # alerted just now
        with mock.patch.object(self.mod, "_read_credits_via_vision",
                               return_value=(1.0, "BALANCE: $1.00")), \
             mock.patch.object(self.mod, "_save_state"), \
             mock.patch.object(self.mod, "_enqueue_speech") as enq:
            self.mod._check_and_maybe_alert()
        enq.assert_not_called()

    def test_login_required_nudges_once_per_window(self):
        with mock.patch.object(self.mod, "_read_credits_via_vision",
                               return_value=(None, "login_required")), \
             mock.patch.object(self.mod, "_save_state"), \
             mock.patch.object(self.mod, "_enqueue_speech") as enq:
            self.mod._check_and_maybe_alert()
        enq.assert_called_once()
        self.assertIn("login", enq.call_args.args[0].lower())


class CreditsActionTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("credits_monitor")

    def test_check_credits_success(self):
        with mock.patch.object(self.mod, "_read_credits_via_vision",
                               return_value=(42.50, "BALANCE: $42.50")), \
             mock.patch.object(self.mod, "_save_state"):
            out = self.actions["check_credits"]("")
        self.assertIn("$42.50", out)

    def test_check_credits_login_required(self):
        with mock.patch.object(self.mod, "_read_credits_via_vision",
                               return_value=(None, "login_required")), \
             mock.patch.object(self.mod, "_save_state"):
            out = self.actions["check_credits"]("")
        self.assertIn("login", out.lower())

    def test_check_credits_unreadable(self):
        with mock.patch.object(self.mod, "_read_credits_via_vision",
                               return_value=(None, "vision said gibberish")), \
             mock.patch.object(self.mod, "_save_state"):
            out = self.actions["check_credits"]("")
        self.assertIn("couldn't read balance", out.lower())

    def test_check_credits_lock_blocks_concurrent(self):
        self.assertTrue(self.mod._check_lock.acquire(blocking=False))
        try:
            out = self.actions["check_credits"]("")
        finally:
            self.mod._check_lock.release()
        self.assertIn("already in progress", out.lower())


if __name__ == "__main__":
    unittest.main()
