"""Canonical failure-marker substrings for action-result classification.

Actions in :mod:`core.actions` (and the in-monolith handlers) return free-text
result strings rather than raising on a soft failure — e.g.
``"no window matching 'Spotify'"`` or ``"could not capture screen"``. Several
call sites need to decide, after the fact, whether such a string represents a
failed action so they can trigger a follow-up / surface it to the user.

This tuple is the single source of truth for those substrings. A result is a
failure if it contains ANY marker, matched **case-insensitively** (callers
lower-case both sides). Keep markers lower-case here; the comparison is
case-insensitive either way.

Consumers (keep this list in sync if you add one):
  • bobert_companion.py — ``_is_failure`` inside the follow-up loop, so a failed
    action auto-triggers a re-prompt instead of going silently unreported.
  • core/dispatcher.py — ``_is_failure_result`` for the multi-step command chain
    resolver's consolidated confirmation.

Both previously held their own hand-mirrored copy of this tuple (dispatcher's
even carried a "Mirror of bobert_companion._is_failure's marker list" note);
they only differed in the casing of ``refused`` (immaterial under the
case-insensitive match). Extracted here so the two can no longer drift.
"""
from __future__ import annotations

# Substrings that flag a free-text action result as a failure (case-insensitive).
FAILURE_MARKERS: tuple[str, ...] = (
    "could not",
    "failed",
    "refused",
    "no tracks found",
    "no window matching",
    "unknown ",
    "format:",
    # ── 2026-07-07 bug-hunt: contraction forms of the "could not"/"cannot" that
    # skills actually emit. Before this, a result phrased "I couldn't reach the
    # printer" / "I can't see the webcam" / "OBS didn't answer" matched NO marker,
    # so it was neither spoken verbatim (the guard didn't flag it) NOR routed to
    # the failure follow-up — the failure line was doubly dropped. These are
    # matched as plain lowercased substrings (see core/dispatcher._is_failure_result
    # and bobert_companion._is_failure), so each must be a form that only appears
    # in genuine failure phrasings. "couldn't"/"can't"/"didn't"/"wouldn't" clear
    # that bar (a full-repo scan of returned result strings found only real
    # failures). "won't" is DELIBERATELY EXCLUDED: it appears in by-design honest,
    # non-error refusals ("I won't use the '<x>' profile, sir", "I won't expose the
    # web interface without a token, sir", the browser SSRF refusal) that the
    # verbatim speak-set intentionally voices — flagging it would swallow those.
    "couldn't",
    "can't",
    "didn't",
    "wouldn't",
)


# ── Terminal failures (2026-10-02) ──────────────────────────────────────────
# A failure no retry can fix, already phrased for the owner: live 12:00:40,
# close_window on Task Manager (an elevated window; Windows' UIPI refuses a
# normal-integrity WM_CLOSE) answered "could not close", and the follow-up
# loop went round close_window / see_screen with new arguments until the depth
# cap. An action returns TERMINAL_FAILURE_PREFIX + the owner-facing line
# instead. The prefix carries "failed", so every FAILURE_MARKERS consumer above
# still reads a failure (the chain resolver never confirms it as done); the
# monolith speaks the line verbatim (_verbatim_result_text) and ends the
# follow-up chain on it.
TERMINAL_FAILURE_PREFIX = "failed (final): "


def terminal_failure_text(result) -> str:
    """The owner-facing line of a terminal failure result, else ""."""
    if not isinstance(result, str) or not result.startswith(
            TERMINAL_FAILURE_PREFIX):
        return ""
    return result[len(TERMINAL_FAILURE_PREFIX):].strip()
