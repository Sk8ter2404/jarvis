"""v3.0.8 F3: a browser-agent task runs in the background.

Live 2026-10-28 (scope F3): browse_for / browser_task held the voice loop for
~77 s ("[watchdog] main loop stalled 68 s") while the agent's paid cloud steps
ran. With BROWSER_TASK_BACKGROUND on, the action returns at once with the
skill's BACKGROUND_STARTED_REPLY, which is voiced word for word (no read-back
round), the result is announced ONCE through proactive_announce when the run
ends, and a spoken stop / "never mind" cancels the run (nothing is announced
after). The 3.0.5 hand-off guard still asks instead of browsing.

Drives the REAL _run_llm_dispatch_body -> parse_and_run_actions -> the REAL
skills/browser_agent.py actions (loaded as skill_browser_agent, the live
loader's name) with only _submit faked (a plain concurrent Future, no browser,
no network), and the REAL _run_voice_shortcuts for the stop. Neutral text.

    python -m unittest tests.monolith.test_monolith_v308_browser_background
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from unittest import mock

from tests._monolith_harness import requires_monolith
from tests._skill_harness import load_skill_isolated
from tests.monolith.test_monolith_answer_first import _Base

_RESULT = ("1. Trail Boot A (https://shop.example/a) - light and waterproof. "
           "2. Trail Boot B www.shop.example/b - warm.")
_ANNOUNCED = ("The browser task is done, sir. 1. Trail Boot A - light and "
              "waterproof. 2. Trail Boot B - warm.")


@requires_monolith
class _BgBase(_Base):
    MOCK_ANNOUNCE = True

    def setUp(self):
        super().setUp()
        bc = self.bc
        mods = mock.patch.dict(sys.modules)
        mods.start()
        self.addCleanup(mods.stop)
        self.assertIs(sys.modules.get("bobert_companion"), bc)
        self.mod, _ = load_skill_isolated("browser_agent", actions=bc.ACTIONS)
        self.assertIs(sys.modules["skill_browser_agent"], self.mod)
        self.addCleanup(self._reset_state)
        import core.config as cfg
        self._p(cfg, "BROWSER_TASK_BACKGROUND", True, create=True)
        self._p(self.mod, "is_available", return_value=True)
        self._p(self.mod, "_ssrf_guard", return_value=None)
        self._p(self.mod, "_BG_INLINE_WAIT_S", 0.05)
        if self.MOCK_ANNOUNCE:
            self.announce = self._p(bc, "proactive_announce",
                                    return_value=True)
        self.futs: list = []
        self.submit = self._p(self.mod, "_submit", side_effect=self._fake_submit)

    def _reset_state(self):
        st = self.mod._state
        for k in ("current_fut", "current_task", "bg_run", "bg_inline",
                  "bg_cancelled", "agent", "loop", "browser"):
            st[k] = None

    def _fake_submit(self, task, max_steps, headless):
        # Left PENDING, as run_coroutine_threadsafe's future stays while the
        # coroutine runs (so a cancel() takes, as live).
        fut = concurrent.futures.Future()
        with self.mod._lock:
            self.mod._state["current_fut"] = fut
            self.mod._state["current_task"] = task
        self.futs.append(fut)
        return fut


class BackgroundRunTests(_BgBase):
    def test_live_turn_returns_at_once_and_announces_once(self):
        t0 = time.monotonic()
        printed = self._run(
            "Right away, sir. [ACTION: browse_for, best trail boots]",
            text="look up the best trail boots online")
        self.assertLess(time.monotonic() - t0, 3.0)
        self.assertIn(self.mod.BACKGROUND_STARTED_REPLY, self.spoken)
        self.followup.assert_not_called()          # no read-back round
        self.assertNotIn("Reading results", printed)
        self.announce.assert_not_called()          # still running
        self.assertTrue(self.mod.background_running())
        # The run ends: announced once, URLs left out, deduped by run.
        self.futs[0].set_result(_RESULT)
        self.announce.assert_called_once()
        args, kwargs = self.announce.call_args
        self.assertEqual(args[0], _ANNOUNCED)
        self.assertEqual(kwargs.get("source"), "browser")
        self.assertTrue(str(kwargs.get("dedupe_key")).startswith(
            "browser-task:"))
        run_id = self.mod._state["bg_run"]
        self.mod._on_bg_done(self.futs[0], run_id)  # a second callback
        self.announce.assert_called_once()

    def test_browser_task_alias_goes_to_the_background_too(self):
        self._run("[ACTION: browser_task, find a trail map of the park]",
                  text="find a trail map of the park")
        self.assertIn(self.mod.BACKGROUND_STARTED_REPLY, self.spoken)
        self.followup.assert_not_called()

    def test_a_fast_run_is_answered_inline_and_never_announced(self):
        def _done_submit(task, max_steps, headless):
            fut = self._fake_submit(task, max_steps, headless)
            fut.set_result("Browser agent failed, sir — the cloud API is "
                           "capped.")
            return fut
        self.submit.side_effect = _done_submit
        self._run("[ACTION: browse_for, trail boots]", text="look it up")
        self.assertNotIn(self.mod.BACKGROUND_STARTED_REPLY, self.spoken)
        self.announce.assert_not_called()

    def test_stop_cancels_the_run_and_nothing_is_announced(self):
        self._run("[ACTION: browse_for, best trail boots]",
                  text="look up the best trail boots online")
        for words in ("stop the music", "cancel the timer"):
            self.assertFalse(self.bc._browser_bg_stop_shortcut(words), words)
        self.spoken.clear()
        handled = self.bc._run_voice_shortcuts("Never mind.")
        self.assertTrue(handled)
        self.assertEqual(self.spoken, ["Stopped the browser task, sir."])
        self.assertTrue(self.futs[0].cancelled())
        self.assertFalse(self.mod.background_running())
        self.announce.assert_not_called()

    def test_e_stop_and_stop_words_cancel(self):
        for words in ("Jarvis, stop.", "E-stop", "cancel the browser task",
                      "stop browsing"):
            with self.subTest(words=words):
                self.spoken.clear()
                self._run("[ACTION: browse_for, trail boots]", text="look")
                self.assertTrue(self.bc._browser_bg_stop_shortcut(words))
                self.assertTrue(self.futs[-1].cancelled())
        self.announce.assert_not_called()

    def test_no_run_no_stop_claim(self):
        self.assertFalse(self.bc._browser_bg_stop_shortcut("stop"))

    def test_hand_off_guard_still_asks(self):
        self._run("[ACTION: browser_task, pass that over]",
                  text="pass that over")
        self.submit.assert_not_called()
        self.assertNotIn(self.mod.BACKGROUND_STARTED_REPLY, self.spoken)

    def test_setting_off_is_the_old_blocking_wait(self):
        import core.config as cfg
        self._p(cfg, "BROWSER_TASK_BACKGROUND", False, create=True)

        def _done_submit(task, max_steps, headless):
            fut = self._fake_submit(task, max_steps, headless)
            fut.set_result(_RESULT)
            return fut
        self.submit.side_effect = _done_submit
        self.followup.side_effect = ["Trail Boot A looks best, sir.", ""]
        self._run("[ACTION: browse_for, best trail boots]", text="look it up")
        self.followup.assert_called()               # the old read-back
        self.assertNotIn(self.mod.BACKGROUND_STARTED_REPLY, self.spoken)
        self.announce.assert_not_called()


class BackgroundResultHistoryTests(_BgBase):
    """F3: the announced result reaches conversation_history, so a follow-up
    ("what was the price again?", "open that one") has its context. Real
    proactive_announce -> the real pending-speech file -> the real drain."""

    MOCK_ANNOUNCE = False

    def setUp(self):
        super().setUp()
        bc = self.bc
        tmp = tempfile.mkdtemp(prefix="bg_hist_")
        self.addCleanup(shutil.rmtree, tmp, True)
        self.path = os.path.join(tmp, "pending_speech.json")
        self._p(bc, "PENDING_SPEECH_PATH", self.path)
        # The writer derives its queue file from the monolith's own location.
        self._p(bc, "__file__", os.path.join(tmp, "bobert_companion.py"))
        self._p(bc, "_speak", side_effect=self._say_ok)
        self._p(bc, "_mark_speech_spoken")
        self._p(bc, "_heartbeat")
        bc.conversation_history[:] = [
            {"role": "user", "content": "find the cheapest trail boots"}]
        self.addCleanup(bc.conversation_history.clear)

    def _say_ok(self, t, *a, **k):
        self.spoken.append(t)
        return True

    def test_result_is_in_history_after_the_drain(self):
        bc = self.bc
        self._run("[ACTION: find_cheapest, trail boots]",
                  text="find the cheapest trail boots")
        self.assertTrue(self.mod.background_running())
        self.futs[0].set_result(_RESULT)           # done-callback announces
        self.assertTrue(os.path.exists(self.path))
        self.spoken.clear()
        bc._speak_pending()
        self.assertEqual(self.spoken, [_ANNOUNCED])    # URLs stay unspoken
        joined = " ".join(str(m.get("content"))
                          for m in bc.conversation_history
                          if m.get("role") == "assistant")
        self.assertIn("Trail Boot A", joined)
        self.assertIn("https://shop.example/a", joined)   # kept for "open it"
        self.assertIn("www.shop.example/b", joined)
        self.assertEqual(bc.conversation_history[-1]["role"], "assistant")

    def test_history_text_is_capped_and_optional(self):
        bc = self.bc
        # A plain statement without history_text is still not recorded.
        before = len(bc.conversation_history)
        self.assertTrue(bc.proactive_announce("A plain statement, sir.",
                                              source="skill"))
        bc._speak_pending()
        self.assertEqual(len(bc.conversation_history), before)
        long_result = "word " * 2000
        self.assertTrue(bc.proactive_announce(
            "Spoken part.", source="browser", history_text=long_result))
        with open(self.path, encoding="utf-8") as f:
            entry = json.load(f)[-1]
        self.assertLessEqual(len(entry["history_text"]),
                             bc._HISTORY_TEXT_MAX_CHARS)

    def test_agent_text_is_capped_for_history(self):
        txt = self.mod._bg_history_text("x", "word " * 1000)
        self.assertLessEqual(len(txt), self.mod._BG_HISTORY_CHARS + 60)
        self.assertEqual(self.mod._bg_history_text(
            "Browser agent failed, sir.", "boom"), "Browser agent failed, sir.")


if __name__ == "__main__":   # pragma: no cover
    unittest.main()
