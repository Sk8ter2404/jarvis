"""core/recall_actions.py - ground a "wait, what just happened?" turn in the
actions that really ran (v3.0.10, RECALL_ACTION_GROUNDING).

Live 2026-10-29 18:54:35 he asked JARVIS to say a line; the robot's speak
action ran and answered that the robot had said it. At 18:54:51 he asked,
surprised, whether it had been sent to the robot ("... sent it over to the
<robot>?"), and the model answered that it was "a bit of a
misinterpretation" - a denial of an action that ran 14 s earlier. The
[ACTION:] token was in the conversation history, but nothing put the result
in front of the model, and no check stops a denial.

When sir's line is a recall or surprise question (is_recall_question), the
turn's prompt gets one short note listing the actions that ran in the last
minute and what they answered, with the instruction to say plainly what ran.
Pure: no I/O, no monolith import. The monolith passes in its replay history
(``{"action", "arg", "result", "at"}`` dicts) and the clock. Never raises.
"""
from __future__ import annotations

import re
import time

from core import repeat_line as _repeat_line

RECALL_WINDOW_S = 60.0
RECALL_MAX_ENTRIES = 3
RECALL_MAX_CHARS = 300

# "did you ...", "have you ..." - the turn checker's own recall-ask pattern,
# read from there so the two cannot drift (a local copy only if that import
# ever fails).
try:
    from core.turn_checker import _RECALL_ASK_RE
except Exception:  # pragma: no cover - core.turn_checker is in-tree
    _RECALL_ASK_RE = re.compile(
        r"^(?:(?:hey|ok(?:ay)?|so|and|um+|uh+|well|jarvis)[,\s]+)*"
        r"(?:(?:when|what|where|why|how|which)\s+)?"
        r"(?:did|didn'?t|have|haven'?t|had)\s+you\b")

# A surprised opener about what just happened: "what the ...", "why did
# ...", "did that / it / he ...", "was that ...", "where did ...", "who said
# ...". A bare "what" counts only on its own ("what?", "what, ..."), never
# "what's the weather".
_SURPRISE_OPEN_RE = re.compile(
    r"^(?:(?:hey|ok(?:ay)?|so|and|um+|uh+|well|jarvis|sir)[,\s]+)*"
    r"(?:what(?=\s*[?!,.]|\s*$)|what the|why did|"
    r"why'd|did (?:that|it|he|she|they)|was that|where did|"
    r"who (?:did|said))\b")
# "Wait", "huh", "hold on", "hang on" also open commands ("hold on, turn it
# down", "wait, pause it"; v3.0.10 review), so after one of them the line
# counts only with a "?", a past-tense verb or a recall opener after it
# ("Wait, what?", "hold on, was that the robot").
_SOFT_OPEN_RE = re.compile(
    r"^(?:(?:hey|ok(?:ay)?|so|and|um+|uh+|well|jarvis|sir)[,\s]+)*"
    r"(?:wait|huh|hold on|hang on)\b[,.!\s]*")
# "Did you know octopuses have three hearts?" is trivia, not recall.
_DID_YOU_KNOW_RE = re.compile(
    r"^(?:(?:hey|ok(?:ay)?|so|and|um+|uh+|well|jarvis|sir)[,\s]+)*"
    r"did\s+you\s+know\b")
# A question about something that happened: "?" and a past-tense action verb
# ("They sent it over to the robot?", "it said that?").
_PAST_VERB_RE = re.compile(
    r"\b(?:sent|said|ran|played|went|happened|did|told|spoke|opened|closed|"
    r"paused|turned|started|stopped|deleted|posted)\b")


def _low(text) -> str:
    t = str(text or "").replace("\u2019", "'").replace("\u2018", "'")
    return re.sub(r"\s+", " ", t).strip().lower()


def is_recall_question(user_text) -> bool:
    """True when sir's line asks about what JARVIS just did, or reacts to it
    with surprise (see the patterns above). Never raises."""
    try:
        t = _low(user_text)
        if not t or len(t) > 240:
            return False
        if _DID_YOU_KNOW_RE.match(t):
            return False
        # "What did you say?" (REPEAT_LAST_LINE, v3.0.11; live 2026-11-05
        # 14:57:42) wants his last sentence, not a retelling of what ran.
        if _repeat_line.repeat_line_on() and _repeat_line.is_repeat_ask(t):
            return False
        if _RECALL_ASK_RE.match(t) or _SURPRISE_OPEN_RE.match(t):
            return True
        soft = _SOFT_OPEN_RE.match(t)
        if soft:
            rest = t[soft.end():]
            return ("?" in t or not rest
                    or _PAST_VERB_RE.search(rest) is not None
                    or _RECALL_ASK_RE.match(rest) is not None
                    or _SURPRISE_OPEN_RE.match(rest) is not None)
        return "?" in t and _PAST_VERB_RE.search(t) is not None
    except Exception:
        return False


def _clip(text, n: int) -> str:
    s = " ".join(str(text or "").split())
    return s if len(s) <= n else s[:max(0, n - 3)].rstrip() + "..."


def recent_actions(history, now=None, window_s: float = RECALL_WINDOW_S,
                   max_entries: int = RECALL_MAX_ENTRIES) -> list:
    """The newest ``max_entries`` real actions (no "_" synthetic names) of
    ``history`` that ran in the last ``window_s`` seconds, oldest first.
    Never raises ([])."""
    try:
        t = time.time() if now is None else float(now)
        out = []
        for e in list(history or ()):
            if not isinstance(e, dict):
                continue
            name = str(e.get("action") or "").strip()
            at = e.get("at")
            if not name or name.startswith("_") or not isinstance(
                    at, (int, float)):
                continue
            if 0.0 <= t - float(at) <= window_s:
                out.append(e)
        return out[-max_entries:] if max_entries > 0 else []
    except Exception:
        return []


def recall_action_note(user_text, history, now=None,
                       window_s: float = RECALL_WINDOW_S,
                       max_entries: int = RECALL_MAX_ENTRIES,
                       max_chars: int = RECALL_MAX_CHARS) -> str:
    """The prompt note for a recall / surprise question: the actions that ran
    in the last ``window_s`` seconds with their results (the list capped at
    ``max_chars``), and the instruction to confirm plainly what ran. "" when
    the line is not such a question or nothing ran. Never raises ("")."""
    try:
        if not is_recall_question(user_text):
            return ""
        t = time.time() if now is None else float(now)
        entries = recent_actions(history, t, window_s, max_entries)
        if not entries:
            return ""
        lines = []
        used = 0
        for e in reversed(entries):          # newest first under the cap
            name = str(e.get("action") or "").strip()
            arg = _clip(e.get("arg"), 60)
            res = _clip(e.get("result"), 80)
            age = max(0, int(round(t - float(e.get("at")))))
            line = (f"- {name}({arg!r}) -> {res!r}, {age} s ago" if arg
                    else f"- {name} -> {res!r}, {age} s ago")
            if lines and used + len(line) > max_chars:
                break
            lines.append(_clip(line, max_chars))
            used += len(line)
        lines.reverse()
        return ("\n\n[Recent actions you ran in the last "
                f"{int(window_s)} s:\n" + "\n".join(lines) + "\n"
                "If he is asking about what just happened, say plainly "
                "what ran and what it did; never deny it or call it a "
                "misunderstanding. If his line is a new request, just do "
                "it.]")
    except Exception:
        return ""
