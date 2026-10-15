"""Brain caller tags, monolith side (2026-10-15): the declared caller reaches
the gate, the ledger and the log lines on the REAL local path.

The mechanism is core.local_traffic's (background_work / slot / GATE); these
tests drive the monolith's real _llm_quick / _call_local_llm / ask_vision with
only the HTTP layer faked (the test_monolith_r5_tagging recipe).

  1. A call that DECLARES itself background (``background=True``), made from a
     thread with no job, waits behind the owner's turn with BACKGROUND_TAG_STRICT
     on, then runs; with the switch off it runs at once and the log says it
     would have waited.
  2. A call that declares only a caller (foreground) never waits.
  3. The ledger row and the "[local-llm] served via" line carry the caller and
     the foreground / background side, with Ollama's own token counts.
  4. The cloud-first branch of _llm_quick hands the caller down to the client
     layer (so a cap-fallback to the local model is still the same caller).
  5. A background call made from the main loop never fails over to another
     model (it would unload the brain).
  6. ask_vision passes the caller down to the local vision entry point.
"""
from __future__ import annotations

import contextlib
import io
import unittest

from tests._monolith_harness import requires_monolith
from tests.monolith.test_monolith_r5_tagging import _Base


@requires_monolith
class DeclaredBackgroundTests(_Base):
    def setUp(self):
        super().setUp()
        self.bc._lt.CALLS.reset()
        self.addCleanup(self.bc._lt.CALLS.reset)

    def _quick(self, **kw):
        return lambda: self.bc._llm_quick("sys", "user", max_tokens=20, **kw)

    def test_declared_background_waits_out_the_turn(self):
        out, log = self._held_during(
            self.bc._turn_in_progress,
            self._quick(caller="proactive_remark", background=True))
        self.assertEqual(out, "urgent")
        self.assertEqual(len(self.posted), 1)
        self.assertIn("[bg-local] defer proactive_remark (turn)", log)
        self.assertRegex(log, r"\[bg-local\] run proactive_remark after "
                              r"\d+ ms \(released\)")

    def test_declared_background_runs_at_once_with_the_switch_off(self):
        self._p(self.bc, "BACKGROUND_TAG_STRICT", False)
        self.bc._turn_in_progress[0] = True
        with contextlib.redirect_stdout(io.StringIO()) as log:
            th, out = self._thread(
                self._quick(caller="screen_remark", background=True))
            th.join(5)
        self.assertFalse(th.is_alive())
        self.assertEqual(out.get("value"), "urgent")
        self.assertEqual(len(self.posted), 1)
        self.assertEqual(log.getvalue().count(
            "[bg-local] shadow screen_remark would defer (turn)"), 1)

    def test_foreground_declared_caller_never_waits(self):
        self.bc._turn_in_progress[0] = True
        with contextlib.redirect_stdout(io.StringIO()) as log:
            th, out = self._thread(self._quick(caller="phone_reply"))
            th.join(5)
        self.assertFalse(th.is_alive(), "a foreground call waited")
        self.assertEqual(out.get("value"), "urgent")
        self.assertNotIn("[bg-local]", log.getvalue())

    def test_ledger_and_served_via_line_carry_the_caller(self):
        bc = self.bc
        with contextlib.redirect_stdout(io.StringIO()) as log:
            th, out = self._thread(
                self._quick(caller="learn_from_turn", background=True))
            th.join(5)
            th2, out2 = self._thread(self._quick(caller="phone_reply"))
            th2.join(5)
        text = log.getvalue()
        self.assertRegex(text, r"served via gemma-test pe=600/\S+ ev=2 "
                               r"caller=learn_from_turn bg")
        self.assertRegex(text, r"served via gemma-test pe=600/\S+ ev=2 "
                               r"caller=phone_reply\s*$|caller=phone_reply\n")
        snap = bc._lt.CALLS.snapshot()
        bgrow, fgrow = snap["learn_from_turn"], snap["phone_reply"]
        self.assertTrue(bgrow["background"])
        self.assertFalse(fgrow["background"])
        self.assertEqual((bgrow["calls"], bgrow["input"], bgrow["output"]),
                         (1, 600, 2))
        self.assertEqual(bgrow["route"], {"local": 1})

    def test_cloud_first_branch_hands_the_caller_down(self):
        import core.config as cfg
        bc = self.bc
        self._p(cfg, "model_route", return_value="auto")
        self._p(cfg, "AMBIENT_LEARNING_FORCE_LOCAL", False)
        self._p(bc, "AI_BACKEND", "claude")
        seen = {}

        def _create(purpose, client=None, **kw):
            seen.update(kw, purpose=purpose)
            raise RuntimeError("cap")       # force the local fallback
        self._p(bc, "_claude_create", side_effect=_create)
        with contextlib.redirect_stdout(io.StringIO()) as log:
            th, out = self._thread(
                self._quick(caller="session_checkpoint", background=True))
            th.join(5)
        self.assertEqual(seen["caller"], "session_checkpoint")
        self.assertIs(seen["background"], True)
        self.assertEqual(out.get("value"), "urgent")      # local fallback
        self.assertIn("caller=session_checkpoint bg", log.getvalue())

    def test_background_call_on_the_main_loop_never_fails_over(self):
        bc = self.bc
        self._p(bc, "_local_empty_streak", [99])
        self.assertTrue(bc._local_failover_allowed())
        with bc._lt.call_scope("proactive_remark", True):
            self.assertFalse(bc._local_failover_allowed())
        with bc._lt.call_scope("voice_turn"):
            self.assertTrue(bc._local_failover_allowed())

    def test_ask_vision_passes_the_caller_to_local_vision(self):
        import core.config as cfg
        bc = self.bc
        self._p(cfg, "model_route", return_value="local")
        self._p(bc, "SCREEN_VISION_ENABLED", True)
        self._p(bc, "_cloud_boost_on", return_value=False)
        seen = {}

        def _vision(q, imgs, **kw):
            seen.update(kw)
            return "a desk"
        self._p(bc, "_call_local_vision", side_effect=_vision)
        ans = bc.ask_vision("what?", b"png", caller="credits_monitor",
                            background=True)
        self.assertEqual(ans, "[local-vision] a desk")
        self.assertEqual(seen, {"caller": "credits_monitor",
                                "background": True})

    def test_served_via_suffix_marks_only_declared_background(self):
        bc = self.bc
        st = {"prompt_eval_count": 5, "prompt_eval_ms": 9, "eval_count": 2}
        self.assertTrue(bc._served_via_suffix(
            st, ("x_y", True)).endswith("caller=x_y bg"))
        self.assertTrue(bc._served_via_suffix(
            st, ("x_y", False)).endswith("caller=x_y"))
        # undeclared: the old frame-derived tag, never ' bg'
        self.assertNotIn(" bg", bc._served_via_suffix(st))
        self.assertNotIn(" bg", bc._served_via_suffix(st, (bc._lt.UNTAGGED, True)))


if __name__ == "__main__":
    unittest.main()
