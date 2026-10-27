"""Chain loop guard (v3 W21, 2026-10-27): ping-pong and per-action budget.

The follow-up loop in ``bobert_companion._run_llm_dispatch_body`` already
stops a chain when the same action fails in a second round, when a round adds
no new result, and when an ``_loop_actions`` member (open_url, web_search,
...) comes round again. What it never noticed is ALTERNATION: live 2026-10-19
JARVIS ran web_search, see_screen, web_search, see_screen ... (6 calls in
35 s), every result a little different from the last, so no repeat guard
fired. ``LoopGuard`` adds that, plus a budget for the "poll" actions that
look at the screen. It is pure (no I/O, no clock, no imports from the
monolith) and the monolith only asks it.

``LoopGuard.check(name, arg)`` records one call of the current owner turn and
returns ``None`` or a short reason string:

* ``repeat``     the same (name, normalised arg) the third time;
* ``ping-pong``  the last four calls are A, B, A, B (two different actions,
                 each time with a similar argument);
* ``poll``       a screen-look action (see_screen / read_screen /
                 local_describe_screen ...) for the fourth time.

Input-driving actions (click, type, press, hotkey, scroll, wait ...) are
exempt from all three: filling a form is click, type, click, type, and
reading a page is scroll, see_screen, scroll, see_screen - neither is a
loop. Confirmation, risk and shutdown paths are not consulted here and this
module never blocks an action: the monolith calls it between follow-up
rounds only, and only the setting ``LOOP_GUARD_MODE`` decides whether a
verdict logs ("shadow") or ends the chain ("on").

The idea (a loop guard with hashed repeats, an alternation window and a
per-tool budget) comes from OpenJarvis ``agents/loop_guard.py`` (Apache-2.0);
this is a re-implementation, nothing is copied.
"""
from __future__ import annotations

import re

MODES = ("off", "shadow", "on")

# Screen-look actions: a chain may take a few looks, not a stream of them.
POLL_ACTIONS = frozenset({
    "see_screen", "read_screen", "local_describe_screen", "recall_screen",
    "last_screen", "previous_screen", "screen_history", "look_at_screen",
})
POLL_BUDGET = 3

# Actions that drive the screen one step at a time. Doing the same or
# alternating ones is how a form gets filled; never a loop by themselves.
EXEMPT_ACTIONS = frozenset({
    "click", "click_on_screen", "find_on_screen",
    "local_click_target_by_description", "double_click", "right_click",
    "type", "type_text", "press", "press_key", "hotkey", "key", "scroll",
    "scroll_up", "scroll_down", "wait", "sleep", "pause_for", "drag",
    "move_mouse", "mouse_move",
})

REPEAT_LIMIT = 3          # the same (name, arg) this many times in a chain
WINDOW = 4                # A, B, A, B
_SIMILAR = 0.34           # token overlap that makes two arguments "similar"

_WORD = re.compile(r"[a-z0-9']+")


def normalise_mode(value) -> str:
    """``value`` as 'off' | 'shadow' | 'on'; anything else reads as 'off'
    (the old behaviour). Never raises."""
    try:
        mode = str(value).strip().lower()
    except Exception:
        return "off"
    return mode if mode in MODES else "off"


def _norm_name(name) -> str:
    return str(name or "").strip().lower()


def _norm_arg(arg) -> str:
    return " ".join(_WORD.findall(str(arg or "").lower()))[:160]


def _similar(a: str, b: str) -> bool:
    """Two normalised arguments name (about) the same thing."""
    if a == b:
        return True
    if not a or not b:
        return False
    if a in b or b in a:
        return True
    ta, tb = set(a.split()), set(b.split())
    return bool(ta and tb) and len(ta & tb) / len(ta | tb) >= _SIMILAR


class LoopGuard:
    """One owner turn's follow-up chain. Make a new one per turn."""

    def __init__(self) -> None:
        self._calls: list[tuple[str, str]] = []   # (name, normalised arg)
        self._counts: dict[tuple[str, str], int] = {}
        self._polls = 0

    def __len__(self) -> int:
        return len(self._calls)

    def check(self, name, arg="") -> "str | None":
        """Record the call; the reason this chain looks like a loop, or
        None. Never raises (a fault reads as 'no loop')."""
        try:
            n, a = _norm_name(name), _norm_arg(arg)
            if not n or n.startswith("_"):
                return None
            self._calls.append((n, a))
            del self._calls[:-WINDOW * 2]
            if n in EXEMPT_ACTIONS:
                # Kept in the window so a step between two look-ups breaks
                # an A, B, A, B run; never counted or judged itself.
                return None
            key = (n, a)
            self._counts[key] = self._counts.get(key, 0) + 1
            if n in POLL_ACTIONS:
                self._polls += 1
                if self._polls > POLL_BUDGET:
                    return f"poll budget {n} x{self._polls}"
            if self._counts[key] >= REPEAT_LIMIT:
                return f"repeat {n} x{self._counts[key]}"
            return self._ping_pong()
        except Exception:
            return None

    def _ping_pong(self) -> "str | None":
        if len(self._calls) < WINDOW:
            return None
        (n0, a0), (n1, a1), (n2, a2), (n3, a3) = self._calls[-WINDOW:]
        if n0 != n2 or n1 != n3 or n0 == n1:
            return None
        if n0 in EXEMPT_ACTIONS or n1 in EXEMPT_ACTIONS:
            return None
        for n, x, y in ((n0, a0, a2), (n1, a1, a3)):
            if n not in POLL_ACTIONS and not _similar(x, y):
                return None
        return f"ping-pong {n0}/{n1}"
