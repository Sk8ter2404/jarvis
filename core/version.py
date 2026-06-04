"""Single source of truth for the JARVIS release version.

The public release/distribution version lives in the top-level ``VERSION``
file and is read once here. This is intentionally SEPARATE from the
self-upgrade pipeline's internal CHANGELOG counter (which bumps a patch
number every pipeline run); ``__version__`` is the version JARVIS reports as
its shareable build (e.g. "1.28.0").

Zero side effects at import beyond a single small file read, so this stays in
the import-light tier and is safe to import anywhere (incl. bare CI).
"""
from __future__ import annotations

import os

_VERSION_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "VERSION"
)

_FALLBACK = "0.0.0-dev"


def _read_version() -> str:
    try:
        with open(_VERSION_FILE, "r", encoding="utf-8") as fh:
            return fh.read().strip() or _FALLBACK
    except OSError:
        return _FALLBACK


__version__ = _read_version()
VERSION = __version__


def version_string() -> str:
    """Human-facing release string, e.g. ``1.28.0``."""
    return __version__
