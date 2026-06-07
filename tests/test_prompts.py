"""Content guards for the cloud action-routing prompt (core/prompts.py).

These are static-string assertions, not LLM round-trips: they pin the routing
*guidance* the model is given, which is the lever that actually decides where a
phrase like "what's on my calendar" gets dispatched. core.prompts is stdlib-only
string constants, so this stays in the fast import-light tier (no monolith boot).
"""
from __future__ import annotations

import unittest

from core.prompts import PC_CONTROL_PROMPT


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


if __name__ == "__main__":
    unittest.main()
