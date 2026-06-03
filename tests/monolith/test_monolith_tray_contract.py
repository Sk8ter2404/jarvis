"""Tray <-> bobert CONTRACT tests — the gap class that produced the tray
overhaul: a menu item sends a command bobert never handles, or a checkmark reads
a hud_state field bobert never writes (so it's permanently blank). The existing
tray tests assert the tray EMITS the right strings but never check the backend,
which is exactly why `llm_backend` (AI checkmarks) and `open_hud` slipped through.

This test imports BOTH sides and pins the contract.

Monolith-tier (needs bobert's ACTIONS): runs locally, skips on the light CI tier.
    python -m unittest tests.monolith.test_monolith_tray_contract
"""
from __future__ import annotations

import os
import re
import unittest

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith


@requires_monolith
class TrayCommandContractTests(MonolithGlobalsTestCase):
    def _tray_commands(self) -> set[str]:
        path = os.path.join(os.path.dirname(self.bc.__file__), "tray.py")
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()
        return set(re.findall(r'_send_command\(\s*["\']([a-z_]+)["\']', src))

    def _explicit_handlers(self) -> set[str]:
        # _dispatch_tray_command's elif chain matches `cmd == "name"`. Parse them
        # from source so this set never drifts from the actual dispatcher.
        with open(self.bc.__file__, "r", encoding="utf-8") as f:
            src = f.read()
        return set(re.findall(r'cmd\s*==\s*["\']([a-z_]+)["\']', src))

    def test_every_tray_command_has_a_bobert_handler(self):
        cmds = self._tray_commands()
        # Sanity: we actually parsed the sends (incl. the two new items).
        self.assertIn("mic_mute_toggle", cmds)
        self.assertIn("open_hud", cmds)
        explicit = self._explicit_handlers()
        unhandled = sorted(
            c for c in cmds
            if c not in explicit and c not in self.bc.ACTIONS
        )
        self.assertEqual(
            unhandled, [],
            f"tray.py sends commands with NO bobert handler (dead menu items): "
            f"{unhandled}")


@requires_monolith
class TrayStateFieldContractTests(MonolithGlobalsTestCase):
    def test_read_fields_have_writers(self):
        """Every hud_state field the tray reads for a checkmark/indicator must be
        written somewhere in bobert (else it's stale forever). Covers the two we
        just fixed (mic_muted, llm_backend) plus the previously-working ones."""
        with open(self.bc.__file__, "r", encoding="utf-8") as f:
            src = f.read()
        for field in ("mic_muted", "llm_backend", "tts_muted",
                      "ambient_mode_active", "debug_mode", "daemons_paused",
                      "audio_processing_enabled"):
            # Matches the `field=` kwarg of a _write_hud_state(...) call. The
            # \b won't match inside the underscore-prefixed runtime flag
            # (_mic_muted), only the published field name.
            self.assertRegex(
                src, rf"\b{field}\s*=",
                f"tray reads hud_state['{field}'] but bobert never publishes it")


if __name__ == "__main__":
    unittest.main()
