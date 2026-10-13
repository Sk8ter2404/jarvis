"""Shared plumbing for the parts of the daily briefing skill.

The briefing used to be ten separate skills (morning_arrival, _arrival_v2,
_briefing, _chain, _handoff, evening_briefing, daily_recap, news_briefing and
the daily_briefing itself) that found each other by guessing module names in
sys.modules and, failing that, importing a SECOND copy off the skills path
(whose state was empty). They are now the parts of one package, and a part
finds a sibling by its package-relative name: always the instance the loader
registered, never a second copy.
"""
from __future__ import annotations

import importlib
import sys


def package_name(part_package: str) -> str:
    """``__package__`` of a part ('skill_daily_briefing', or
    'skills.daily_briefing' when imported by dotted path)."""
    return part_package or "skill_daily_briefing"


def part(part_package: str, name: str):
    """The sibling part ``name`` ('news', 'chain', 'arrival_v2', ...) of the
    package ``part_package``. Raises ImportError when it cannot be loaded."""
    return importlib.import_module(f"{package_name(part_package)}.{name}")


def package(part_package: str):
    """The package module itself (the daily briefing proper), or None when it
    is not loaded."""
    return sys.modules.get(package_name(part_package))
