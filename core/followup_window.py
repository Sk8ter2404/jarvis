"""core/followup_window.py — the wake-word follow-up window.

Wake-word mode (config ``REQUIRE_WAKE_MODE``) makes every turn begin with
"JARVIS", which is right in a noisy room and exhausting in a real back-and-forth.
This window relaxes it for a short time AFTER the user has addressed JARVIS
directly: follow-ups inside the window need no wake word, and each one that is
admitted extends the window. When the conversation stops, the window lapses and
overheard room crosstalk is gated out again.

Ported 2026-09-28 from a local-only patch on the Dell edge node (where it ran at
45 s). On main it is OFF by default (``FOLLOWUP_WINDOW_S = 0``) so the desktop's
wake-word behaviour is unchanged; an install opts in through
``data/user_settings.json``.

KNOWN RISK — read before raising the window
-------------------------------------------
Every ADMITTED utterance extends the window, and admission happens at the
background-audio gate, before the speech filter or the LLM has judged it. So in
exactly the room that needed wake-word mode, one overheard line inside the window
extends it for the next, and a steady source such as a TV can hold it open
indefinitely. The window is only as strict as the room is quiet. This port keeps
the Dell's semantics unchanged; capping the total extension is a policy choice
left to the owner.

3.0.6 MITIGATION (media audio). Live 2026-10-28 17:28: "play vision quest on
the left monitor" opened the window, a "Yeah." heard over the video he had
just started was admitted and extended it, and 18 s later a 164-character
narration from that video was admitted too and ran session_memory_recall.
Both voices scored 0.64-0.68 - "unsure", which the media gate lets through
because his own commands over media land there as well. So the media gate
can now TAKE BACK the extension an admitted utterance made (take_back_admit)
when the PC was playing audio and the voice was not confirmed as his (see
core.media_gate.keeps_followup_extension, setting FOLLOWUP_MEDIA_NO_EXTEND).
The utterance itself still runs; it just cannot hold the window open. A
wake-word turn (note_addressed) is unchanged.

3.0.7 GENERALISED (any unsure voice, setting FOLLOWUP_UNSURE_NO_EXTEND).
Live 2026-10-28 19:08-19:12 the PC audio was MUTED, so the media rule above
never ran, and room speech scoring 0.61-0.67 against his voiceprint was
admitted line after line, each one extending the window, until he said
"You're supposed to be quiet." His own commands score as low (0.61-0.73 on
some that session), so a voice score alone cannot refuse a follow-up.
Instead:

  * an admitted follow-up extends the window only when its voice MATCHED an
    enrolled speaker (keeps_extension) - or when voice-ID is unavailable
    (nobody enrolled), which keeps today's behaviour. UNSURE and NOT_OWNER
    are taken back, as is an admitted line that never reached an answer
    (noise, filter, unclear or media drops: four low-confidence captures in
    19:09:28-19:09:39 each pushed the window 20 s further);
  * with FOLLOWUP_UNSURE_COMMANDS_ONLY, an unsure-voice follow-up that is
    neither a command nor a question to JARVIS (addressed_form) is dropped
    silently - unless JARVIS just asked something or a prompt waits for an
    answer. His first follow-up command in an unsure voice still runs, and
    so does a short continuation of the last request ("And tomorrow.",
    "Same for the kitchen.", "Ten minutes instead.", "A bit quieter.",
    "Are you there"): his live scores of 0.61-0.72 are as often him as not,
    and carrying on from the last request is the window's main use
    (continuation_form, addressed_form). A lone statement still is not.
  * a drop that speaks a cue inviting a repeat ("I didn't catch that")
    keeps the window the follow-up opened, so the repeat needs no wake word
    (the owner is answering JARVIS); silent drops are still taken back.

Pure stdlib, no I/O, injectable clock — testable on the light-deps CI runner.
"""
from __future__ import annotations

import re
import time

# Voice verdicts (the strings core.learn_gate uses; repeated here so this
# module stays import-free).
_OWNER = "owner"
_UNAVAILABLE = "unavailable"


def keeps_extension(voice) -> bool:
    """Whether a follow-up in this voice (a core.learn_gate verdict) may
    EXTEND the window: a matched enrolled voice, or no voice evidence at all
    (voice-ID unavailable = today's behaviour). UNSURE / NOT_OWNER / anything
    unknown does not. Never raises."""
    try:
        return voice in (_OWNER, _UNAVAILABLE)
    except Exception:
        return True


# ── command or question to JARVIS (FOLLOWUP_UNSURE_COMMANDS_ONLY) ──────────
# The form, not the words' meaning: an imperative verb first, a question
# (a "?" or a wh-word first), "I need / want you to ...", or the name anywhere.
# A statement, a fragment, a lone farewell or a crude remark is none of these.
# An auxiliary first WITHOUT a "?" ("Should the ...", "Did a ...") is a
# statement or a sung line, not a question: the decoder writes the "?". The
# exceptions are the _AUX_YOU / _AUX_THING forms below ("Are you there").
_NAME_RE = re.compile(r"\bjarvis\b", re.IGNORECASE)
_WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")
_LEAD_SOFT = frozenset({
    "please", "okay", "ok", "hey", "so", "now", "alright", "and", "then",
    "also", "just", "um", "uh", "oh", "yo", "sir", "actually", "right",
    "well", "hmm", "ah",
})
_QUESTION_WORDS = frozenset({
    "what", "what's", "whats", "where", "where's", "when", "when's", "why",
    "how", "how's", "who", "who's", "whose", "whom", "which",
})
COMMAND_VERBS = frozenset({
    "add", "answer", "ask", "be", "bring", "calculate", "call", "cancel",
    "change", "check", "choose", "clear", "click", "close", "come",
    "connect", "continue", "convert", "copy", "decrease", "define", "delete",
    "describe", "dim", "disable", "disconnect", "do", "don't", "dont",
    "email", "enable", "end", "exit", "explain", "find", "finish", "forget",
    "forward", "get", "give", "go", "google", "help", "hide", "hold",
    "increase", "install", "keep", "kill", "launch", "let", "let's", "lets",
    "list", "load", "lock", "look", "louder", "lower", "make", "maximise",
    "maximize", "message", "minimise", "minimize", "move", "mute", "never",
    "next", "note", "open", "paste", "pause", "pick", "play", "previous",
    "print", "pull", "push", "put", "quieter", "quit", "raise", "read",
    "reboot", "record", "refresh", "reload", "remember", "remind", "remove",
    "repeat", "restart", "resume", "rewind", "run", "save", "say",
    "schedule", "screenshot", "scroll", "search", "select", "send", "set",
    "show", "shuffle", "shut", "silence", "skip", "sleep", "spell", "start",
    "stay", "stop", "summarise", "summarize", "switch", "take", "tell",
    "text", "toggle", "translate", "try", "turn", "type", "undo", "unlock",
    "unmute", "unpause", "update", "use", "volume", "wait", "wake", "write",
    "zoom",
})
_WANT_RE = re.compile(
    r"^(?:i|we)(?:'d| would)?\s+(?:want|need|like)\s+(?:you|to)\b")
# A question with the "?" left off: an auxiliary first, then "you" (talking
# to JARVIS) or a bare "it / that / there" (about the thing in hand: "Is it
# going to rain tomorrow"). "Should the ..." / "Did a ..." stay statements.
_AUX_YOU = frozenset({
    "are", "do", "did", "does", "have", "has", "is", "was", "were", "should",
    "shall", "may", "might", "must",
})
_AUX_THING = frozenset({
    "is", "was", "does", "did", "will", "would", "can", "could", "has",
    "should",
})
_THING_WORDS = frozenset({"it", "that", "there", "this"})
# Short continuations of the last request ("And tomorrow.", "Same for the
# kitchen.", "Ten minutes instead.", "A bit quieter."). Kept short and
# shaped so an overheard sentence does not qualify.
_CONT_MAX_WORDS = 5
_CONT_LEAD = frozenset({"and", "same"})
_COMPARATIVES = frozenset({
    "quieter", "louder", "softer", "lower", "higher", "brighter", "dimmer",
    "darker", "faster", "slower", "warmer", "cooler", "bigger", "smaller",
})
_DEGREE_WORDS = frozenset({
    "a", "bit", "little", "lot", "much", "slightly", "even", "way", "just",
    "somewhat", "tad",
})


def continuation_form(words) -> bool:
    """Whether these lowercase words (leading filler NOT stripped) are a
    SHORT continuation of the last request: led by "and" / "same", ending in
    "instead", or a degree + comparative ("a bit quieter"). At most
    _CONT_MAX_WORDS words. Never raises (False)."""
    try:
        words = list(words or ())
        if not words or len(words) > _CONT_MAX_WORDS:
            return False
        if words[0] in _CONT_LEAD and len(words) > 1:
            return True
        if words[-1] == "instead" and len(words) > 1:
            return True
        if words[-1] in _COMPARATIVES and all(
                w in _DEGREE_WORDS for w in words[:-1]):
            return True
        return False
    except Exception:
        return False


def addressed_form(text) -> str:
    """"named", "question", "command", "continuation" or "" for one
    transcript: whether it has the FORM of something said to JARVIS (see the
    block above). Used only for a follow-up whose voice is unsure. Never
    raises ("")."""
    try:
        raw = str(text or "").strip()
        if not raw:
            return ""
        if _NAME_RE.search(raw):
            return "named"
        if raw.rstrip(" \t\"'”’)]").endswith("?"):
            return "question"
        words = _WORD_RE.findall(raw.lower().replace("’", "'"))
        if continuation_form(words):
            return "continuation"
        n = 0
        while words and words[0] in _LEAD_SOFT and n < 3:
            words.pop(0)
            n += 1
        if not words:
            return ""
        if words[0] in _QUESTION_WORDS:
            return "question"
        if words[0] in COMMAND_VERBS:
            return "command"
        if (len(words) > 1 and words[1] == "you"
                and words[0] in ("can", "could", "would", "will")):
            return "question"        # "Can you turn it off." (no "?")
        if len(words) > 1 and (
                (words[1] == "you" and words[0] in _AUX_YOU)
                or (words[1] in _THING_WORDS and words[0] in _AUX_THING)):
            return "question"        # "Are you there" / "Is it going to rain"
        if _WANT_RE.match(" ".join(words)):
            return "command"
        return ""
    except Exception:
        return ""


class FollowupWindow:
    """Tracks whether a no-wake-word follow-up is currently admissible.

    ``window_s <= 0`` disables it entirely: ``admit()`` is then always False and
    the caller's gate behaves exactly as it did before this module existed.
    """

    def __init__(self, window_s: float = 0.0, clock=time.time) -> None:
        try:
            self.window_s = max(0.0, float(window_s))
        except (TypeError, ValueError):
            self.window_s = 0.0          # a malformed setting disables, never widens
        self._clock = clock
        self._until = 0.0
        # (deadline before, deadline after) of the last admit() that extended
        # the window, until the turn is judged (take_back_admit) or a new
        # gate decision starts (forget_admit / note_addressed).
        self._last_admit = None

    @property
    def enabled(self) -> bool:
        return self.window_s > 0.0

    def note_addressed(self) -> None:
        """The user just addressed JARVIS by its wake word: (re)open the window."""
        self._last_admit = None
        if self.enabled:
            self._until = self._clock() + self.window_s

    def admit(self) -> bool:
        """True if a no-wake-word utterance is admissible right now.

        An admitted utterance EXTENDS the window (see KNOWN RISK above).
        """
        if not self.enabled:
            return False
        now = self._clock()
        if now < self._until:
            self._last_admit = (self._until, now + self.window_s)
            self._until = now + self.window_s
            return True
        self._last_admit = None
        return False

    def forget_admit(self) -> None:
        """A new gate decision starts: an earlier turn's extension can no
        longer be taken back. Never raises."""
        self._last_admit = None

    def take_back_admit(self) -> bool:
        """Undo the extension the last admit() made, once: the window keeps
        the deadline it had before that utterance (it may still be open, it
        is just not pushed further). False when there is nothing to take
        back (no admit since the last forget_admit / note_addressed, or the
        window moved since). Never raises."""
        try:
            memo, self._last_admit = self._last_admit, None
            if not memo:
                return False
            before, after = memo
            if self._until != after:
                return False
            self._until = before
            return True
        except Exception:
            return False

    def remaining_s(self) -> float:
        """Seconds left in the window (0.0 when closed or disabled)."""
        if not self.enabled:
            return 0.0
        return max(0.0, self._until - self._clock())
