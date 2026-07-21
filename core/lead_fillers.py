"""lead_fillers — the ONE canonical lead-in filler stripper.

Utterances arrive with polite lead-ins ("could you", "please") and wake-word
prefixes ("JARVIS, ", "hey jarvis") that must be peeled off before intent
matching. This rule used to exist as two drifting copies: core/mode_router.py
was fixed (comma variants + loop-until-stable) while core/dispatcher.py kept
the old list (no comma variants, single pass), so Controlled mode refused
"JARVIS, take a screenshot" (2026-07-21 audit, stale-duplicates dimension).

Both modules now import THIS implementation. Do not fork it again — add new
fillers here, and give every wake-word filler ("... jarvis ") its comma
sibling ("... jarvis, "); tests/test_mode_router.py enforces both invariants.
"""

from __future__ import annotations

LEAD_FILLERS = (
    "could you ", "can you ", "would you ", "please ",
    "jarvis ", "jarvis, ", "hey jarvis ", "hey jarvis, ",
    "ok jarvis ", "ok jarvis, ", "okay jarvis ", "okay jarvis, ",
    "go ahead and ", "i need you to ", "i'd like you to ",
)


def strip_lead_filler(s: str) -> str:
    # Strip STACKED lead-ins ("JARVIS, please switch to agent mode") by looping
    # until stable — a single pass left the second filler in place and broke the
    # toggle match (caught by tests/test_mode_router). Each pass removes at most
    # one filler, then re-checks the shortened head.
    cur = s.strip()
    changed = True
    while changed:
        changed = False
        low = cur.lower()
        for f in LEAD_FILLERS:
            if low.startswith(f):
                cur = cur[len(f):].lstrip()
                changed = True
                break
    return cur
