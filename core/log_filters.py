"""Logger-level filters for two known third-party log lines (2026-09-30).

Measured on the live session log: two library warnings that carry no
information for the owner and drown the lines that do.

  * ``phonemizer`` - "words count mismatch on 100.0% of the lines (1/1)",
    printed on nearly every Kokoro TTS line (dozens per session).
    kokoro_onnx phonemizes one line at a time with punctuation preserved, and
    espeak's word count routinely differs from the text's; the audio is fine.
  * ``huggingface_hub.utils._http`` - "Warning: You are sending
    unauthenticated requests to the HF Hub. Please set a HF_TOKEN ...", a
    server X-HF-Warning header the long-term-memory embedder load receives on
    every boot (it pulls a PUBLIC model; no token is wanted).

ONLY these exact messages are dropped, and only on the logger that emits
them: every other record of those loggers - and anything above WARNING, even
with the same text - still reaches the log. A logging.Filter on the emitting
logger, not setLevel(): phonemizer.logger.get_logger() resets its logger's
level AND handlers whenever a phonemizer module that calls it is imported,
but never touches its filters. Stdlib only; installing is idempotent and
never raises.
"""
from __future__ import annotations

import logging
import re

__all__ = ["PHONEMIZER_LOGGER", "HF_HTTP_LOGGER",
           "install_phonemizer_filter", "install_hf_unauthenticated_filter"]

PHONEMIZER_LOGGER = "phonemizer"
HF_HTTP_LOGGER = "huggingface_hub.utils._http"

# phonemizer/backend/espeak/words_mismatch.py:
#   'words count mismatch on %s%% of the lines (%s/%s)'
_PHONEMIZER_MISMATCH_RE = re.compile(
    r"^words count mismatch on \d+(?:\.\d+)?% of the lines \(\d+/\d+\)$")
# The X-HF-Warning body huggingface_hub logs verbatim.
_HF_UNAUTHENTICATED_RE = re.compile(
    r"^(?:Warning:\s*)?You are sending unauthenticated requests to the HF "
    r"Hub\b")

# Marks OUR filters on a logger, so a second install (or a reload of this
# module) never stacks a duplicate.
_TAG_ATTR = "_jarvis_log_filter_tag"


class _DropExactMessage(logging.Filter):
    """Drops a record whose rendered message matches ``pattern`` - at
    WARNING or below only, so an error is never hidden. Counts what it
    dropped (``dropped``) so the silence stays measurable."""

    def __init__(self, tag: str, pattern: "re.Pattern[str]"):
        super().__init__()
        setattr(self, _TAG_ATTR, tag)
        self.pattern = pattern
        self.dropped = 0

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            if record.levelno > logging.WARNING:
                return True
            if self.pattern.search(record.getMessage()):
                self.dropped += 1
                return False
        except Exception:   # a filter must never break logging
            pass
        return True


def _install(logger_name: str, tag: str,
             pattern: "re.Pattern[str]") -> "logging.Filter | None":
    try:
        logger = logging.getLogger(logger_name)
        for f in list(logger.filters):
            if getattr(f, _TAG_ATTR, None) == tag:
                return f
        f = _DropExactMessage(tag, pattern)
        logger.addFilter(f)
        return f
    except Exception:
        return None


def install_phonemizer_filter() -> "logging.Filter | None":
    """Drop phonemizer's per-line "words count mismatch" warning. Returns
    the installed filter (None only if logging itself failed)."""
    return _install(PHONEMIZER_LOGGER, "phonemizer-words-mismatch",
                    _PHONEMIZER_MISMATCH_RE)


def install_hf_unauthenticated_filter() -> "logging.Filter | None":
    """Drop huggingface_hub's "unauthenticated requests to the HF Hub"
    server warning. Returns the installed filter (None only if logging
    itself failed)."""
    return _install(HF_HTTP_LOGGER, "hf-unauthenticated",
                    _HF_UNAUTHENTICATED_RE)
