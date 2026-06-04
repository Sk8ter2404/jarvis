"""Regression tests for runtime bugs JARVIS itself queued from live sessions.

2026-06-02:

  1. Printer-status actions (check_print / how_is_the_print / print_details)
     returned a result that was logged but never SPOKEN, because they weren't in
     INFORMATIVE_ACTIONS — so the result->speech follow-up loop never fired
     (unlike check_credits, which is listed and speaks correctly).

  2. Vision-click targeting overshot on a >100%-scaled multi-monitor rig:
     find_click_target added a NATIVE-pixel offset to a LOGICAL monitor origin
     without scaling, so clicks landed too far right/down.

2026-06-03 ("you didn't speak it"):

  3. version_info (and the system_pulse status family) returned a finished,
     user-facing answer ("I'm on version 1.20.4, last updated …, sir.") that was
     logged but never SPOKEN: the action wasn't in INFORMATIVE_ACTIONS (so no
     follow-up LLM turn) and its result isn't a failure (so the failure
     follow-up didn't fire either) — the user only heard the "One moment, sir."
     preamble. Unlike the printer fix, these results are already perfect
     sentences, so they are now spoken VERBATIM via SPEAK_RESULT_VERBATIM_ACTIONS
     + _speak_verbatim_results(), with dedup so an inlined answer isn't
     double-spoken and side-effect/failure results are left alone.

Monolith-tier (full-deps): run locally; skip on the light-deps CI runner.
    python -m unittest tests.monolith.test_monolith_runtime_bugfixes
"""
from __future__ import annotations

import io
import unittest
from unittest import mock

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith


@requires_monolith
class PrinterStatusInformativeTests(MonolithGlobalsTestCase):
    def test_printer_status_actions_are_informative(self):
        # Without these in INFORMATIVE_ACTIONS the dispatch follow-up loop breaks
        # immediately and the printer status is logged but never voiced.
        for name in ("check_print", "how_is_the_print", "print_details"):
            self.assertIn(name, self.bc.INFORMATIVE_ACTIONS,
                          f"{name} must be informative so its result is spoken")

    def test_check_credits_still_informative(self):
        # The reference behaviour we're matching — guard against accidental removal.
        self.assertIn("check_credits", self.bc.INFORMATIVE_ACTIONS)


@requires_monolith
class FindClickTargetScalingTests(MonolithGlobalsTestCase):
    @staticmethod
    def _png(w: int, h: int) -> bytes:
        from PIL import Image
        buf = io.BytesIO()
        Image.new("RGB", (w, h), (0, 0, 0)).save(buf, format="PNG")
        return buf.getvalue()

    def test_native_pixels_scaled_to_logical_before_adding_origin(self):
        """A target at the native-pixel CENTRE of a 3840x2160 capture of a
        2560x1440-LOGICAL monitor (150% scale) whose origin is (-2560, 0) must
        click at the LOGICAL centre (-1280, 720) — not the un-scaled (-640, 1080)
        the old code produced."""
        bc = self.bc
        png_lowres = self._png(1568, 882)     # Pass-1 downscale (max_dim 1568)
        png_native = self._png(3840, 2160)    # Pass-2 full-res (native pixels)

        def fake_shot(monitor=None, max_dim=1568):
            return png_lowres if max_dim <= 1568 else png_native

        def fake_vision(_desc, _png, w, h):
            # Pass-1 (full image) -> centre; Pass-2 (the 500x500 crop) -> centre.
            return (784, 441) if (w, h) == (1568, 882) else (250, 250)

        with mock.patch.dict(bc.MONITORS, {"qa": (-2560, 0, 2560, 1440)}), \
             mock.patch.object(bc, "take_screenshot", side_effect=fake_shot), \
             mock.patch.object(bc, "_query_vision_for_coords", side_effect=fake_vision):
            pt = bc.find_click_target("a sidebar item", monitor="qa")

        self.assertIsNotNone(pt)
        # native centre (1920,1080) x (2560/3840, 1440/2160) -> (1280,720); + origin
        self.assertAlmostEqual(pt[0], -1280, delta=2)
        self.assertAlmostEqual(pt[1], 720, delta=2)
        # And definitively NOT the old buggy native-added coordinate.
        self.assertNotEqual((pt[0], pt[1]), (-640, 1080))

    def test_no_scale_when_native_equals_logical(self):
        """At 100% scale (native == logical) the scaling is a no-op, so an
        un-scaled single-monitor setup can't regress."""
        bc = self.bc
        png_lowres = self._png(1280, 720)
        png_native = self._png(2560, 1440)   # == logical below

        def fake_shot(monitor=None, max_dim=1568):
            return png_lowres if max_dim <= 1568 else png_native

        def fake_vision(_desc, _png, w, h):
            return (640, 360) if (w, h) == (1280, 720) else (250, 250)

        with mock.patch.dict(bc.MONITORS, {"qa": (0, 0, 2560, 1440)}), \
             mock.patch.object(bc, "take_screenshot", side_effect=fake_shot), \
             mock.patch.object(bc, "_query_vision_for_coords", side_effect=fake_vision):
            pt = bc.find_click_target("x", monitor="qa")

        self.assertIsNotNone(pt)
        # native == logical -> the returned point equals the native refined coord
        # plus the (0,0) origin, i.e. no scale distortion.
        self.assertTrue(0 <= pt[0] <= 2560 and 0 <= pt[1] <= 1440)


@requires_monolith
class VerbatimResultSpokenTests(MonolithGlobalsTestCase):
    """Bug 3: an informational action whose result is a finished sentence
    (version_info / system_pulse) must be SPOKEN, exactly once, without an LLM
    round-trip — and without double-speaking or speaking side-effect/failure
    results.
    """

    VER = "I'm on version 1.20.4, last updated Saturday morning at 7:03 AM, sir."

    # ── membership ──────────────────────────────────────────────────────────
    def test_version_family_in_verbatim_set(self):
        for name in ("version_info", "what_version", "when_updated"):
            self.assertIn(name, self.bc.SPEAK_RESULT_VERBATIM_ACTIONS,
                          f"{name} must speak its result verbatim")

    def test_status_family_in_verbatim_set(self):
        for name in ("system_pulse", "check_system", "status_report"):
            self.assertIn(name, self.bc.SPEAK_RESULT_VERBATIM_ACTIONS)

    def test_verbatim_set_excludes_side_effect_actions(self):
        # Side-effect actions must NEVER verbatim-speak their result (the inline
        # reply already confirms them) — guards against a careless future add.
        for name in ("play_music", "volume_up", "set_timer", "launch_app",
                     "weather_briefing"):
            self.assertNotIn(name, self.bc.SPEAK_RESULT_VERBATIM_ACTIONS)

    # ── _speak_verbatim_results() unit behaviour ────────────────────────────
    def test_helper_speaks_informational_result(self):
        bc = self.bc
        spoken = []
        with mock.patch.object(bc, "_speak",
                               side_effect=lambda t, *a, **k: spoken.append(t)):
            handled = bc._speak_verbatim_results(
                [("version_info", self.VER, False)], already_spoken="One moment, sir.")
        self.assertEqual(spoken, [self.VER])
        self.assertEqual(handled, {"version_info"})

    def test_helper_dedupes_already_spoken(self):
        bc = self.bc
        spoken = []
        with mock.patch.object(bc, "_speak",
                               side_effect=lambda t, *a, **k: spoken.append(t)):
            # The inline reply already contained the answer (case-insensitively).
            handled = bc._speak_verbatim_results(
                [("version_info", self.VER, False)],
                already_spoken=f"Certainly. {self.VER.upper()}")
        self.assertEqual(spoken, [], "must not re-speak an already-voiced answer")
        self.assertEqual(handled, set())

    def test_helper_skips_failures(self):
        bc = self.bc
        spoken = []
        with mock.patch.object(bc, "_speak",
                               side_effect=lambda t, *a, **k: spoken.append(t)):
            handled = bc._speak_verbatim_results(
                [("version_info", "could not read version info: boom", False)])
        self.assertEqual(spoken, [], "raw failures stay with the failure follow-up")
        self.assertEqual(handled, set())

    def test_helper_ignores_non_verbatim_actions(self):
        bc = self.bc
        spoken = []
        with mock.patch.object(bc, "_speak",
                               side_effect=lambda t, *a, **k: spoken.append(t)):
            handled = bc._speak_verbatim_results(
                [("play_music", "playing Take Five by Dave Brubeck", True)])
        self.assertEqual(spoken, [])
        self.assertEqual(handled, set())

    # ── end-to-end through _run_llm_dispatch ────────────────────────────────
    def _dispatch_capture(self, reply, actions):
        """Run _run_llm_dispatch with a canned LLM reply + stub actions, and
        return the list of strings handed to _speak."""
        bc = self.bc
        spoken = []
        with mock.patch.object(bc, "get_response_with_animation",
                               return_value=reply), \
             mock.patch.object(bc, "maybe_glance_response", return_value=None), \
             mock.patch.object(bc, "_speak",
                               side_effect=lambda t, *a, **k: spoken.append(t)), \
             mock.patch.object(bc, "_apply_quip_layer",
                               side_effect=lambda s, r: s), \
             mock.patch.object(bc, "get_followup_response",
                               side_effect=lambda info: ""), \
             mock.patch.dict(bc.ACTIONS, actions), \
             mock.patch.object(bc, "PC_CONTROL_ENABLED", True):
            bc._run_llm_dispatch("what version are you on?")
        return spoken

    def test_dispatch_speaks_version_result_exactly_once(self):
        # THE BUG: preamble was spoken, version answer was dropped.
        spoken = self._dispatch_capture(
            "One moment, sir. [ACTION: version_info]",
            {"version_info": lambda a="": self.VER})
        self.assertIn("One moment, sir.", spoken)
        self.assertEqual(sum(1 for s in spoken if self.VER in s), 1,
                         f"version answer must be spoken exactly once: {spoken}")

    def test_dispatch_does_not_double_speak_inlined_answer(self):
        # When the LLM already inlined the answer, speak it once, not twice.
        spoken = self._dispatch_capture(
            f"{self.VER} [ACTION: version_info]",
            {"version_info": lambda a="": self.VER})
        self.assertEqual(sum(1 for s in spoken if self.VER in s), 1,
                         f"inlined answer double-spoken: {spoken}")

    def test_dispatch_side_effect_result_not_verbatim_spoken(self):
        # play_music is in INFORMATIVE_ACTIONS but NOT the verbatim set — its
        # result must not be read aloud as a second confirmation.
        spoken = self._dispatch_capture(
            "Playing your jazz playlist, sir. [ACTION: play_music, jazz]",
            {"play_music": lambda a="": "playing Take Five by Dave Brubeck"})
        self.assertFalse(any("Take Five" in s for s in spoken),
                         f"side-effect music result was verbatim-spoken: {spoken}")


if __name__ == "__main__":
    unittest.main()
