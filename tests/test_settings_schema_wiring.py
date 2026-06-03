"""Systemic guard: every PERSISTED Settings-GUI key must be wired to a real
core/config.py constant.

The Settings GUI (tools/settings_window.py) writes data/user_settings.json, and
core.config._apply_user_settings() applies a saved value ONLY if a same-named
constant already exists as a core/config.py module global (see the
`key not in g` guard there). So a persisted schema key with no matching config
constant is a DEAD toggle: the GUI saves it, but nothing ever reads it.

This test fails if any persisted key (other than the small, explicit allowlist
of intentionally GUI-only reference hints) lacks a core/config.py constant — so
the dead-toggle class of bug can never ship again.

Status rows (``_status_*``) are not persisted and are excluded by
``persisted_keys()``. The MODEL_ROUTING ``"::"`` sub-keys are GUI-internal
widget variable names folded into the nested MODEL_ROUTING dict before save
(see settings_window.py), never top-level persisted keys, so they're naturally
out of scope here.

stdlib unittest + importlib only; CI-safe (no tkinter, no GUI).
"""
from __future__ import annotations

import importlib.util
import os
import unittest

from core import config as cfg

# Load settings_window by file path so the test works whether or not `tools`
# is an importable package on the runner (mirrors test_settings_window.py).
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
_MODULE_PATH = os.path.join(_PROJECT, "tools", "settings_window.py")

_spec = importlib.util.spec_from_file_location("jarvis_settings_window_wiring",
                                               _MODULE_PATH)
sw = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(sw)

# Intentionally GUI-only keys that persist a non-secret connection HINT to
# user_settings.json but deliberately have NO core/config.py constant — they
# are reference values the user reads off, not config the runtime consumes.
# Confirmed against the settings_window.py module docstring ("non-secret
# connection hints (host/port)"). Adding a key here is a deliberate, reviewed
# exception — it should be rare.
_GUI_ONLY_ALLOWLIST = {
    "OBS_HOST_HINT",
    "OBS_PORT_HINT",
    "HUE_BRIDGE_IP_HINT",
}


class SchemaWiringTests(unittest.TestCase):
    def test_every_persisted_key_maps_to_a_config_constant(self):
        """No persisted schema key may be a dead toggle."""
        unwired = []
        for key in sw.persisted_keys():
            if key in _GUI_ONLY_ALLOWLIST:
                continue
            if not hasattr(cfg, key):
                unwired.append(key)
        self.assertEqual(
            unwired, [],
            msg=(
                "These persisted Settings-GUI keys have no matching "
                "core/config.py constant, so toggling them does nothing "
                "(core.config._apply_user_settings drops unknown keys): "
                f"{unwired}. Either add a constant of the same name to "
                "core/config.py (and make the real consumer read it) or, if "
                "the key is an intentional GUI-only hint, add it to "
                "_GUI_ONLY_ALLOWLIST in this test with justification."
            ),
        )

    def test_allowlist_entries_are_actually_persisted_and_unwired(self):
        """Keep the allowlist honest: every entry must still be a persisted
        key AND still lack a config constant. If one gains a config constant
        (or stops being persisted), it should leave the allowlist so the
        allowlist can't silently mask a future real dead toggle."""
        persisted = set(sw.persisted_keys())
        for key in _GUI_ONLY_ALLOWLIST:
            self.assertIn(
                key, persisted,
                msg=f"Allowlisted key {key!r} is no longer a persisted schema "
                    "key — remove it from _GUI_ONLY_ALLOWLIST.",
            )
            self.assertFalse(
                hasattr(cfg, key),
                msg=f"Allowlisted key {key!r} now HAS a core/config.py "
                    "constant — it's properly wired, so remove it from "
                    "_GUI_ONLY_ALLOWLIST and let the main test cover it.",
            )

    def test_persisted_keys_excludes_status_rows(self):
        """Sanity-check the helper this guard relies on: status rows
        (``_status_*``) must never be treated as persisted settings."""
        for key in sw.persisted_keys():
            self.assertFalse(
                key.startswith("_status_"),
                msg=f"status row {key!r} leaked into persisted_keys()",
            )


if __name__ == "__main__":
    unittest.main()
