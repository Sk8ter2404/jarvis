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

    def test_open_hud_clears_x_button_latch_before_relaunch(self):
        """P1-hud reopen bug + 2026-07-21 audit ('Open HUD is a silent no-op
        after a voice hide HUD'): the open_hud branch must clear BOTH persisted
        hide latches — the ✕-button 'hidden' flag in unified_hud_state.json AND
        the voice-hide visible=False in hud_state.json — or the relaunched
        subprocess re-hides itself immediately from whichever latch was stale.
        The regression class is 'un-hide rule fixed in one copy, stale in
        another', so the branch must DELEGATE to core.actions._act_show_hud,
        the single owner of the un-hide rule (its visible=True +
        _set_unified_hud_hidden(False) pairing is pinned by
        tests/test_actions_sec1.py ShowHudTests)."""
        with open(self.bc.__file__, "r", encoding="utf-8") as f:
            src = f.read()
        # Isolate the open_hud elif block (up to the next elif/else/def).
        m = re.search(
            r'elif cmd == ["\']open_hud["\']:(.*?)(?:\n    elif |\n    else:|\ndef )',
            src, re.DOTALL)
        self.assertIsNotNone(m, "could not locate the open_hud dispatch block")
        block = m.group(1)
        self.assertIn(
            "_act_show_hud(", block,
            "open_hud must delegate to core.actions._act_show_hud — the single "
            "owner of the un-hide rule (visible=True + ✕-latch clear); a local "
            "re-implementation is exactly the stale-duplicate class that made "
            "the tray button a silent no-op after a voice 'hide HUD'")
        # Ordering 1: the un-hide must precede the relaunch, else the freshly
        # launched HUD reads the stale latches and hides before the clear lands.
        self.assertLess(
            block.index("_act_show_hud("),
            block.index("_launch_hud()"),
            "open_hud must un-hide BEFORE relaunching the HUD subprocess")
        # Ordering 2: the un-hide must come after the HUD_ENABLED flip —
        # _write_hud_state is a no-op while HUD_ENABLED is False, so calling
        # _act_show_hud first would silently drop the visible=True write.
        self.assertIn("HUD_ENABLED = True", block)
        self.assertLess(
            block.index("HUD_ENABLED = True"),
            block.index("_act_show_hud("),
            "open_hud must flip HUD_ENABLED on BEFORE _act_show_hud, else the "
            "visible=True write is a no-op and the stale latch survives")


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
