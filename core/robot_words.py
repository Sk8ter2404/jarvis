"""core/robot_words.py - the one "robot word" pattern (v3.0.9).

The owner's robot skill lives in a private folder outside the repo, so public
code never names the device. Two places need to know when his words name it:

  * core/onscreen_refs (developer to-dos, "we need to give the robot more
    voice lines" - DEV_TODO_ROBOT_STRICT, v3.0.8);
  * core/action_fit rule 6 (the robot's speak action fits only when his
    words make the ROBOT the speaker - ACTION_FIT_ROBOT_SPEAK_ENFORCE,
    v3.0.9; live 2026-10-29 18:54:35 "Jarvis, say <line>." ran it).

Both import THIS pattern; it lives here, not in either of them, so neither
imports the other (no import cycle). Pure stdlib, no state.
"""
from __future__ import annotations

__all__ = ["BOT_COMMON_WORDS", "ROBOT_WORD_STRICT"]

# Common English words that end in "bot" and are not a robot.
BOT_COMMON_WORDS = (r"(?:abbott?|sabot|turbot|jabot|talbott?|cabot|chatbot|"
                    r"chabot)")
# "robot", "bot" on its own, a two-word "<word> bot" / "<word>-bot" (not
# "chat bot"), or a one-word "<word>bot" that is not a common English word;
# never a household "robot vacuum / mower ...". A doubled final t from the
# recogniser still counts. Lower-case letters: compile it with re.IGNORECASE,
# and put a \b in front of it.
ROBOT_WORD_STRICT = (
    r"(?:(?!chat\b)[a-z]+(?:\s+|\s*-\s*)bott?\b|robot\b|bott?\b"
    r"|(?!" + BOT_COMMON_WORDS + r"\b)[a-z]+bott?\b)"
    r"(?!\s+(?:vacuums?|mops?|mowers?|lawn\s*mowers?|cleaners?|toys?)\b)")
