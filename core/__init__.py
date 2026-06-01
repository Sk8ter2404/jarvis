"""JARVIS core package — shared helpers that don't pull in the giant
bobert_companion module. Kept tiny so importing it from skills/ or
hud/ is cheap.

Currently exposes:

    BLUE_GREEN_ROLE — "prod" or "staging" — derived from JARVIS_STAGING
        or sys.argv. Skills that want to disable themselves in staging
        (e.g. the Bambu MQTT skill) can `from core import BLUE_GREEN_ROLE`
        and gate on it without importing the full manager.

This file used to be empty — populating it does NOT change behaviour
for any existing import like `from core import emotion_tracker`, since
those still resolve via the package's submodule machinery.
"""

import os
import sys

BLUE_GREEN_ROLE = (
    "staging"
    if (os.environ.get("JARVIS_STAGING", "").strip() == "1"
        or "--staging" in sys.argv)
    else "prod"
)


def is_staging() -> bool:
    return BLUE_GREEN_ROLE == "staging"
