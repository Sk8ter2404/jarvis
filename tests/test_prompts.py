"""Content guards for the cloud action-routing prompt (core/prompts.py).

These are static-string assertions, not LLM round-trips: they pin the routing
*guidance* the model is given, which is the lever that actually decides where a
phrase like "what's on my calendar" gets dispatched. core.prompts is stdlib-only
string constants, so this stays in the fast import-light tier (no monolith boot).
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import unittest

from core.prompts import PC_CONTROL_PROMPT

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_TESTS_DIR)
_TOOLS = os.path.join(_PROJECT_ROOT, "tools")


def _load_tool(name: str):
    """Import a tools/*.py module by path (tools/ is not a package)."""
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_TOOLS, name + ".py"))
    assert spec and spec.loader, f"could not build import spec for {name}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod      # MUST precede exec for dataclass (3.14)
    spec.loader.exec_module(mod)
    return mod


def _section(head: str, span: int = 1200) -> str:
    """The prompt text starting at an ALL-CAPS section head."""
    i = PC_CONTROL_PROMPT.find(head)
    assert i != -1, f"section {head!r} vanished from the prompt"
    return PC_CONTROL_PROMPT[i:i + span]


class CalendarRoutingPromptTests(unittest.TestCase):
    """Regression: 'what is on my calendar' mis-routed to morning_briefing
    because morning_briefing was the only action whose description mentioned
    'calendar', and there was no calendar action documented at all. That detour
    also re-exposed a Celsius leak baked into the briefing's weather line. The
    prompt must now expose a dedicated calendar action and steer bare
    calendar/schedule questions to it."""

    def test_calendar_action_is_documented(self):
        self.assertIn("calendar_today", PC_CONTROL_PROMPT)
        self.assertIn("[ACTION: calendar_today]", PC_CONTROL_PROMPT)

    def test_calendar_trigger_phrases_present(self):
        for phrase in ("what's on my calendar", "what's on my schedule"):
            self.assertIn(phrase, PC_CONTROL_PROMPT)

    def test_calendar_section_precedes_morning_briefing(self):
        # Ordering matters for a token-greedy planner: the calendar action must
        # be introduced before the MORNING BRIEFING block so the model meets the
        # right handler first for a bare schedule question.
        cal = PC_CONTROL_PROMPT.find("calendar_today")
        brief = PC_CONTROL_PROMPT.find("MORNING BRIEFING")
        self.assertNotEqual(cal, -1)
        self.assertNotEqual(brief, -1)
        self.assertLess(cal, brief)

    def test_calendar_section_disambiguates_from_briefing(self):
        # The guidance explicitly tells the planner to prefer the calendar read
        # over morning_briefing for a plain calendar lookup.
        cal = PC_CONTROL_PROMPT.find("CALENDAR (read the user")
        self.assertNotEqual(cal, -1)
        section = PC_CONTROL_PROMPT[cal:cal + 400]
        self.assertIn("morning_briefing", section)


class ShippedPromptActionInvariantTests(unittest.TestCase):
    """THE doc-truth invariant: an action the SHIPPED prompt teaches must have a
    handler in a file that actually ships.

    2026-08-20: core/prompts.py documented answer_call / decline_call /
    vip_priority_handler, whose only handlers live in skills/teams_screener.py —
    gitignored (.gitignore "Personal skills kept permanently LOCAL"). tools/
    build_release.py exports `git ls-files`, so every public install shipped a
    prompt teaching three actions the dispatcher answers with "unknown action".

    tools/audit_codebase.py has this exact check (check_prompt_action_-
    consistency) but feeds it walk_py_files(), which enumerates the LOCAL
    filesystem — so the auditor audits the owner's tree, where the gitignored
    skill is present, and the class was invisible. This test closes that hole by
    scanning only TRACKED files.

    Both rules are imported from their ONE home rather than re-implemented:
    prompt extraction from tools/audit_codebase._extract_prompt_actions, the
    registration forms from tools/registration_scan.scan_registrations (see its
    module docstring — that rule previously rotted as four diverging copies)."""

    @classmethod
    def setUpClass(cls):
        cls.audit = _load_tool("audit_codebase")
        cls.rs = _load_tool("registration_scan")

    def _tracked_py(self) -> list[str]:
        try:
            out = subprocess.run(
                ["git", "ls-files", "-z", "*.py"], cwd=_PROJECT_ROOT,
                capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.SubprocessError) as exc:
            self.skipTest(f"git unavailable: {exc}")
        if out.returncode != 0:
            self.skipTest("not a git checkout — cannot tell shipped from local")
        paths = [p for p in out.stdout.split("\0")
                 if p and not p.startswith("tests/")]
        if not paths:
            self.skipTest("git ls-files returned nothing")
        return paths

    def test_every_documented_action_ships_a_handler(self):
        src_path = os.path.join(_PROJECT_ROOT, "core", "prompts.py")
        with open(src_path, encoding="utf-8") as fh:
            documented = {a for a in self.audit._extract_prompt_actions(fh.read())
                          if not a.startswith("_")}
        self.assertGreater(len(documented), 100,
                           "prompt-action extraction found almost nothing — the "
                           "extractor drifted, so this guard would pass blind")
        registered: set[str] = set()
        unparseable: list[str] = []
        for rel in self._tracked_py():
            path = os.path.join(_PROJECT_ROOT, rel.replace("/", os.sep))
            try:
                with open(path, encoding="utf-8", errors="replace") as fh:
                    registered |= set(self.rs.scan_registrations(
                        fh.read(), filename=rel))
            except SyntaxError:
                unparseable.append(rel)   # the syntax gate owns this failure
            except OSError:
                pass
        self.assertGreater(len(registered), 300,
                           f"registration scan found almost nothing "
                           f"(unparseable={unparseable[:5]}) — guard is blind")
        missing = sorted(documented - registered)
        self.assertEqual(
            missing, [],
            "core/prompts.py teaches action(s) with no handler in any TRACKED "
            f"file: {missing}. Public installs get 'unknown action'. Move the "
            "block into that skill's module-level PROMPT_EXAMPLES (see "
            "skills/trip_planner.py) so it is documented only where it runs.")


class BargeInPromptHonestyTests(unittest.TestCase):
    """The prompt used to open 'BARGE-IN (already implemented — you have this
    capability)', describe the RMS/headset watcher, and close with 'just confirm
    it works and tell them to try interrupting you' — while
    bobert_companion.BARGE_IN_ENABLED is hard-disabled (PortAudio
    use-after-free) and the only live path needs the wake-word DETECTOR, which
    does not autostart. tools/settings_window.py already carried the honest
    contract and tests/test_audit_2026_07_21.py pinned THAT copy; the prompt
    copy rotted beside it. Talking over JARVIS does nothing."""

    def setUp(self):
        self.block = _section("BARGE-IN")

    def test_does_not_order_the_model_to_affirm_it(self):
        for lie in ("already implemented", "just confirm it works",
                    "try interrupting you"):
            self.assertNotIn(lie, PC_CONTROL_PROMPT)

    def test_names_the_wake_detector_prerequisite(self):
        self.assertIn("wake_listener_start", self.block)
        self.assertIn("does NOT autostart", self.block)

    def test_does_not_promise_the_dead_loudness_path(self):
        low = self.block.lower()
        self.assertNotIn("sustained speech", low)
        self.assertNotIn("rms=", low)


class GestureSwipePromptHonestyTests(unittest.TestCase):
    """'SWIPE cancels/stops' was false: request_tts_interrupt rejects every
    gesture source unless JARVIS_GESTURE_BARGE_IN=1, off by default since the
    2026-07-15 phantom-swipe process deaths. The owner's own session logs show
    nine swipes rejected mid-speech in one morning."""

    def setUp(self):
        self.block = _section("  gestures_on / gestures_off", 900)

    def test_swipe_is_not_advertised_as_stopping_speech(self):
        self.assertNotIn("SWIPE cancels/stops", PC_CONTROL_PROMPT)
        self.assertIn("does NOT stop your", self.block)

    def test_names_the_optin_env_var(self):
        self.assertIn("JARVIS_GESTURE_BARGE_IN", self.block)


class AirMousePromptHonestyTests(unittest.TestCase):
    """The air-mouse engage gate is HEIGHT-only (skills/kinect_air_mouse.py sets
    every forward-reach threshold to 0.0 = non-gating, and engage_decision()
    returns False whenever lift_ok is False), and calibration walks the user
    through raise-then-lower. The prompt still taught the removed reach-to-engage
    contract, contradicting the VERBATIM-spoken strings the same actions
    return."""

    def setUp(self):
        self.block = _section("  AIR-MOUSE (", 1400)

    def test_teaches_the_height_gate(self):
        self.assertIn("ABOVE YOUR SHOULDER", self.block)
        self.assertIn("LOWER the hand to release", self.block)

    def test_does_not_teach_reach_to_engage(self):
        self.assertNotIn("EXTEND an arm", PC_CONTROL_PROMPT)
        self.assertNotIn("reach-to-engage", PC_CONTROL_PROMPT)
        # the AIR CONTROL block below it IS genuinely reach-to-engage and must
        # keep saying so — this guard is scoped to the air-mouse block.
        self.assertIn("reach a hand out toward the Kinect", PC_CONTROL_PROMPT)


class TeamsSectionHonestyTests(unittest.TestCase):
    """The tracked prompt must not teach the gitignored screener's actions, and
    its header must not promise a VIP screener the shipped tree cannot provide
    (the only tracked handler under it is check_teams — the vision-based unread
    sweep in skills/teams_nudge.py)."""

    def test_screener_actions_are_not_in_the_tracked_prompt(self):
        for name in ("answer_call", "decline_call", "vip_priority_handler"):
            self.assertNotIn(name, PC_CONTROL_PROMPT)

    def test_header_describes_what_ships(self):
        self.assertNotIn("TEAMS CALL SCREENING", PC_CONTROL_PROMPT)
        self.assertIn("MICROSOFT TEAMS", PC_CONTROL_PROMPT)
        self.assertIn("check_teams", PC_CONTROL_PROMPT)


class OvernightUpgradePromptHonestyTests(unittest.TestCase):
    """start_overnight_upgrade is in NEITHER speak set, so what the owner hears
    is the model's own prose — written from this prompt. The prompt promised
    'Generates new features immediately, then keeps cycling while idle', but the
    engine thread only starts when OVERNIGHT_UPGRADE_ENABLED is on, and that has
    shipped False since 2026-05-30 (and is False in the owner's settings). The
    action sets a run-now Event that nothing is listening to, so 'goodnight'
    produced a promise of overnight work that never happened."""

    def setUp(self):
        self.block = _section("  start_overnight_upgrade", 900)

    def test_does_not_promise_unconditional_generation(self):
        self.assertNotIn("Generates new features immediately", PC_CONTROL_PROMPT)

    def test_states_the_flag_and_the_standby_half(self):
        self.assertIn("PAUSED by default", self.block)
        self.assertIn("do NOT promise new features", self.block)


if __name__ == "__main__":
    unittest.main()
