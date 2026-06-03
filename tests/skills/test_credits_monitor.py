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

    def test_check_credits_read_raises(self):
        with mock.patch.object(self.mod, "_read_credits_via_vision",
                               side_effect=RuntimeError("vision exploded")), \
             mock.patch.object(self.mod, "_save_state"):
            out = self.actions["check_credits"]("")
        self.assertIn("credits check failed", out.lower())

    def test_check_credits_screenshot_failed_status(self):
        # NOTE: _read_credits_via_vision never actually returns the legacy
        # 'screenshot_failed' status (capture failures come back as
        # 'capture_failed: ...'), so this action branch is otherwise dead. We
        # drive it explicitly to lock in the documented message.
        with mock.patch.object(self.mod, "_read_credits_via_vision",
                               return_value=(None, "screenshot_failed")), \
             mock.patch.object(self.mod, "_save_state"):
            out = self.actions["check_credits"]("")
        self.assertIn("couldn't screenshot", out.lower())


class CreditsImportCompanionTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("credits_monitor")

    def test_import_companion_returns_injected_module(self):
        bc = _fake_companion()
        with mock.patch.dict("sys.modules", {"bobert_companion": bc}):
            self.assertIs(self.mod._import_companion(), bc)


class CreditsVisionFailureBranchTests(unittest.TestCase):
    """The defensive failure branches inside _read_credits_via_vision that the
    sibling tests don't hit: capture call raising, ask_vision raising, and a
    float-parse failure on an otherwise-matching answer."""

    def setUp(self):
        self.mod, self.actions = load_skill_isolated("credits_monitor")

    def test_open_capture_raises(self):
        bc = _fake_companion()
        bc._open_url_offscreen_capture = mock.MagicMock(
            side_effect=RuntimeError("chrome spawn failed"))
        with mock.patch.object(self.mod, "_import_companion", return_value=bc):
            dollars, raw = self.mod._read_credits_via_vision()
        self.assertIsNone(dollars)
        self.assertIn("open_failed", raw)

    def test_ask_vision_raises(self):
        bc = _fake_companion()
        bc.ask_vision = mock.MagicMock(side_effect=RuntimeError("vlm down"))
        with mock.patch.object(self.mod, "_import_companion", return_value=bc):
            dollars, raw = self.mod._read_credits_via_vision()
        # ask_vision raising → answer becomes 'vision_failed: ...' → unparseable.
        self.assertIsNone(dollars)
        self.assertIn("vision_failed", raw)

    def test_float_parse_failure_returns_none(self):
        # Force a regex match whose group(1) can't be float()'d, exercising the
        # defensive ValueError guard (the live regex can't normally produce
        # this, but the guard must hold if it ever did).
        bc = _fake_companion(vision="BALANCE: $1.2.3")
        fake_match = mock.MagicMock()
        fake_match.group.return_value = "1.2.3"  # float() will reject this
        with mock.patch.object(self.mod, "_import_companion", return_value=bc), \
             mock.patch.object(self.mod.re, "search", return_value=fake_match):
            dollars, raw = self.mod._read_credits_via_vision()
        self.assertIsNone(dollars)


class CreditsEnqueueSpeechTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("credits_monitor")

    def test_routes_through_proactive_announce(self):
        bc = _fake_companion()
        with mock.patch.object(self.mod, "_import_companion", return_value=bc):
            self.mod._enqueue_speech("low credits sir")
        bc.proactive_announce.assert_called_once()
        self.assertEqual(bc.proactive_announce.call_args.kwargs.get("source"),
                         "credits")

    def test_fallback_writes_to_speech_queue(self):
        # proactive_announce unavailable (import raises) → atomic write to the
        # queue file, which we redirect to a tempfile.
        fd, p = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        os.remove(p)  # start absent so the append-create path runs
        self.addCleanup(lambda: os.path.exists(p) and os.remove(p))
        with mock.patch.object(self.mod, "_import_companion",
                               side_effect=RuntimeError("no companion")), \
             mock.patch.object(self.mod, "_SPEECH_QUEUE", p):
            self.mod._enqueue_speech("queued alert sir")
        import json
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data[-1]["message"], "queued alert sir")

    def test_fallback_appends_to_existing_queue(self):
        fd, p = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        self.addCleanup(lambda: os.path.exists(p) and os.remove(p))
        import json
        with open(p, "w", encoding="utf-8") as f:
            json.dump([{"ts": 1.0, "message": "old"}], f)
        with mock.patch.object(self.mod, "_import_companion",
                               side_effect=RuntimeError("no companion")), \
             mock.patch.object(self.mod, "_SPEECH_QUEUE", p):
            self.mod._enqueue_speech("new alert")
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual([d["message"] for d in data], ["old", "new alert"])

    def test_fallback_ignores_corrupt_existing_queue(self):
        # An existing but non-JSON queue file → the read except resets to []
        # and the new message still lands.
        fd, p = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        self.addCleanup(lambda: os.path.exists(p) and os.remove(p))
        with open(p, "w", encoding="utf-8") as f:
            f.write("{not valid json")
        with mock.patch.object(self.mod, "_import_companion",
                               side_effect=RuntimeError("no companion")), \
             mock.patch.object(self.mod, "_SPEECH_QUEUE", p):
            self.mod._enqueue_speech("after corruption")
        import json
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual([d["message"] for d in data], ["after corruption"])

    def test_fallback_atomic_write_failure_is_logged(self):
        with mock.patch.object(self.mod, "_import_companion",
                               side_effect=RuntimeError("no companion")), \
             mock.patch.object(self.mod, "_atomic_write_json",
                               side_effect=OSError("disk full")), \
             mock.patch("os.path.exists", return_value=False):
            import contextlib
            import io
            with contextlib.redirect_stdout(io.StringIO()) as buf:
                self.mod._enqueue_speech("doomed")
        self.assertIn("speech-queue write failed", buf.getvalue())


class CreditsSaveStateFailureTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("credits_monitor")

    def test_save_state_swallows_write_error(self):
        # open() raising must not propagate out of _save_state.
        with mock.patch("builtins.open", side_effect=OSError("read-only fs")):
            self.mod._save_state(9.0, "BALANCE: $9.00")  # must not raise


class CreditsCheckCycleBranchTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("credits_monitor")
        self.mod._last_alert_at[0] = 0.0
        self.mod._last_login_alert_at[0] = 0.0

    def test_skips_when_lock_already_held(self):
        # A concurrent check holds the lock → this cycle returns immediately
        # without reading the balance.
        self.assertTrue(self.mod._check_lock.acquire(blocking=False))
        try:
            with mock.patch.object(self.mod, "_read_credits_via_vision") as rd:
                self.mod._check_and_maybe_alert()
            rd.assert_not_called()
        finally:
            self.mod._check_lock.release()

    def test_read_error_is_swallowed(self):
        with mock.patch.object(self.mod, "_read_credits_via_vision",
                               side_effect=RuntimeError("boom")), \
             mock.patch.object(self.mod, "_save_state") as save:
            import contextlib
            import io
            with contextlib.redirect_stdout(io.StringIO()) as buf:
                self.mod._check_and_maybe_alert()
        # Errored before _save_state; logged to console.
        save.assert_not_called()
        self.assertIn("background check error", buf.getvalue())

    def test_login_nudge_respects_window(self):
        # A recent login nudge suppresses a second within the 12h window.
        self.mod._last_login_alert_at[0] = time.time()
        with mock.patch.object(self.mod, "_read_credits_via_vision",
                               return_value=(None, "login_required")), \
             mock.patch.object(self.mod, "_save_state"), \
             mock.patch.object(self.mod, "_enqueue_speech") as enq:
            self.mod._check_and_maybe_alert()
        enq.assert_not_called()


class CreditsRegisterTests(unittest.TestCase):
    def test_duplicate_monitor_thread_is_skipped(self):
        # Simulate an already-running 'credits-monitor' OS thread so register()
        # takes the "skip duplicate" branch instead of starting another loop.
        import threading

        fake_thread = mock.MagicMock()
        fake_thread.name = "credits-monitor"
        fake_thread.is_alive.return_value = True
        with mock.patch.object(threading, "enumerate",
                               return_value=[fake_thread]):
            mod, actions = load_skill_isolated("credits_monitor")
        self.assertIn("check_credits", actions)


if __name__ == "__main__":
    unittest.main()
