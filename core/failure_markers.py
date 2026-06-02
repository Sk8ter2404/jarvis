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
)
