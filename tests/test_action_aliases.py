"""core/action_aliases.py - the exact invented-name table (v3 W04).

Pinned, stdlib + AST only (light tier):
  * every alias SOURCE is not a registered action: not in the registration
    scan of the tracked tree (tools/registration_scan, the harness's scanner)
    and not one of the live-only names (the call / VIP / Bonnaroo actions
    exist only in the live install; _runner also only consults the table when
    the live ACTIONS has no such name, so an untracked skill always wins);
  * every alias TARGET is registered;
  * no target is self-terminating (core.action_risk.SELF_TERMINATING_ACTIONS),
    guess-protected (core.action_risk.guess_protected), CONFIRM-tier
    (core.config.CONFIRM_KEYWORDS, the _needs_confirmation word rule) or bound
    to the same handler as a name that is;
  * nothing NEVER-listed (sends, deletes, money, shutdown / exit / restart,
    close-all, file moves, shell) is ever a source;
  * alias_for: a registered name wins, a missing target or an empty argument
    for an argument-needing target is None, faults are None;
  * friendly_label speaks words, never backticks or underscores.

The monolith-side checks (_autocorrect_protected with its same-handler rule,
_DESTRUCTIVE_REPLAY_ACTIONS, the _runner rewrite) are in
tests/monolith/test_monolith_action_aliases.py.

    python -m unittest tests.test_action_aliases
"""
from __future__ import annotations

import glob
import importlib.util
import os
import re
import sys
import unittest

_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from core import action_aliases as aa  # noqa: E402
from core import action_risk  # noqa: E402

# Registered only in the live install (05_action_vocabulary.md section 2:
# seen in the live "added actions" lines, not in the tracked tree). A source
# may not shadow them.
_LIVE_ONLY_PREFIXES = ("vip_", "bonnaroo_")
_LIVE_ONLY_NAMES = frozenset({"answer_call", "decline_call",
                              "send_vip_reply", "scrap_vip_reply"})

_REGISTRY: "dict | None" = None


def _registry() -> dict:
    """name -> (file, handler symbol) for every scanned registration."""
    global _REGISTRY
    if _REGISTRY is not None:
        return _REGISTRY
    spec = importlib.util.spec_from_file_location(
        "registration_scan",
        os.path.join(_PROJECT, "tools", "registration_scan.py"))
    rs = importlib.util.module_from_spec(spec)
    sys.modules["registration_scan"] = rs    # before exec (dataclasses)
    spec.loader.exec_module(rs)
    srcs = ([os.path.join(_PROJECT, "bobert_companion.py")]
            + sorted(glob.glob(os.path.join(_PROJECT, "skills", "*.py")))
            + sorted(glob.glob(os.path.join(_PROJECT, "skills", "*", "*.py")))
            + sorted(glob.glob(os.path.join(_PROJECT, "core", "*.py"))))
    reg: dict = {}
    for path in srcs:
        rel = os.path.relpath(path, _PROJECT).replace("\\", "/")
        targets = (("ACTIONS",) if rel == "bobert_companion.py"
                   else rs.DEFAULT_TARGETS)
        try:
            found = rs.scan_file(path, targets=targets, filename=rel)
        except (SyntaxError, OSError):
            continue
        for name, r in found.items():
            reg.setdefault(name, (rel, getattr(r, "symbol", "")))
    _REGISTRY = reg
    return reg


def _confirm_tier(name: str) -> bool:
    """bobert_companion._needs_confirmation's word rule on the name alone."""
    from core.config import CONFIRM_KEYWORDS
    hay = name.lower()
    return any(re.search(rf"(?<![a-z]){re.escape(k.lower())}", hay)
               for k in CONFIRM_KEYWORDS)


def _name_protected(name: str) -> bool:
    return (name in action_risk.SELF_TERMINATING_ACTIONS
            or action_risk.guess_protected(name)
            or _confirm_tier(name))


class AliasTableShapeTests(unittest.TestCase):
    def test_registry_scan_is_not_blind(self):
        reg = _registry()
        self.assertGreater(len(reg), 500)
        for real in ("see_screen", "launch_app", "volume_mute"):
            self.assertIn(real, reg)

    def test_rows_are_lower_snake_case(self):
        for src, dst in aa.ALIASES.items():
            for n in (src, dst):
                self.assertRegex(n, r"^[a-z][a-z0-9_]*$")
            self.assertNotEqual(src, dst)

    def test_no_source_is_a_registered_action(self):
        reg = _registry()
        for src in aa.ALIASES:
            with self.subTest(src=src):
                self.assertNotIn(src, reg, f"{src} is a real action "
                                 f"({reg.get(src)}); a real name always wins")
                self.assertFalse(src.startswith(_LIVE_ONLY_PREFIXES))
                self.assertNotIn(src, _LIVE_ONLY_NAMES)

    def test_every_target_is_registered(self):
        reg = _registry()
        for src, dst in aa.ALIASES.items():
            with self.subTest(src=src):
                self.assertIn(dst, reg)

    def test_no_target_is_self_terminating_protected_or_confirm(self):
        for src, dst in aa.ALIASES.items():
            with self.subTest(src=src, dst=dst):
                self.assertNotIn(dst, action_risk.SELF_TERMINATING_ACTIONS)
                self.assertFalse(action_risk.guess_protected(dst))
                self.assertFalse(_confirm_tier(dst))
                self.assertFalse(action_risk.confirm_reasons(dst),
                                 f"{dst} has a confirm reason")

    def test_no_target_shares_a_handler_with_a_protected_name(self):
        reg = _registry()
        by_handler: dict = {}
        for name, (rel, sym) in reg.items():
            if sym:
                by_handler.setdefault((rel, sym), set()).add(name)
        for dst in set(aa.ALIASES.values()):
            rel, sym = reg[dst]
            for other in by_handler.get((rel, sym), ()):
                with self.subTest(dst=dst, other=other):
                    self.assertFalse(_name_protected(other))

    def test_no_source_is_never_listed(self):
        for src in aa.ALIASES:
            with self.subTest(src=src):
                self.assertFalse(aa.is_never_alias(src))

    def test_never_names_stay_unaliased(self):
        for n in ("send_email", "send_money", "delete_folder",
                  "delete_folder_contents", "delete", "transfer",
                  "transfer_funds", "shutdown", "shutdown_pc", "exit", "quit",
                  "restart", "reboot", "close_all_windows", "close_everything",
                  "move_file", "copy_file", "run_shell", "run_code", "kill_x",
                  "wipe_drive", "format_disk", "pay_invoice", "buy_item"):
            with self.subTest(n=n):
                self.assertTrue(aa.is_never_alias(n))
                self.assertIsNone(aa.alias_for(n, "x", set(_registry())))

    def test_table_covers_the_planned_rows(self):
        # PLAN.md W04 names these rows explicitly.
        expect = {"read_screen": "see_screen", "browser_open": None,
                  "pause": "media_playpause", "mute": "volume_mute",
                  "toggle_mute": "volume_mute", "volume_set": "set_volume",
                  "open_app": "launch_app", "play_video": "youtube_play"}
        for src, dst in expect.items():
            with self.subTest(src=src):
                self.assertEqual(aa.ALIASES.get(src), dst)


class AliasForTests(unittest.TestCase):
    REG = {"see_screen": 1, "launch_app": 1, "youtube_play": 1,
           "media_playpause": 1, "volume_mute": 1, "set_volume": 1}

    def test_rewrites_an_invented_name(self):
        self.assertEqual(aa.alias_for("read_screen", "", self.REG),
                         "see_screen")
        self.assertEqual(aa.alias_for("pause", "", self.REG),
                         "media_playpause")
        self.assertEqual(aa.alias_for(" Toggle_Mute ", "", self.REG),
                         "volume_mute")

    def test_mute_music_is_mute_not_pause(self):
        # The fuzzy layer used to send mute_music to pause_music; the W04
        # verb-clash guard refuses that guess (mute vs pause), so the exact
        # row is what keeps "mute the music" working: mute is the literal ask.
        self.assertEqual(aa.ALIASES.get("mute_music"), "volume_mute")
        self.assertEqual(aa.alias_for("mute_music", "", self.REG),
                         "volume_mute")
        self.assertIsNone(aa.alias_for("mute_music", "", {"pause_music": 1}))
        import command_autocorrect as ca
        self.assertTrue(ca._verb_clash("mute_music", "pause_music"))
        self.assertFalse(ca._verb_clash("mute_music", "volume_mute"))

    def test_registered_name_wins(self):
        reg = dict(self.REG, read_screen=1)
        self.assertIsNone(aa.alias_for("read_screen", "", reg))

    def test_target_must_be_registered(self):
        self.assertIsNone(aa.alias_for("describe_screen", "", self.REG))

    def test_arg_needing_target_needs_an_arg(self):
        self.assertIsNone(aa.alias_for("play_video", "", self.REG))
        self.assertIsNone(aa.alias_for("play_video", "   ", self.REG))
        self.assertEqual(aa.alias_for("play_video", "lofi beats", self.REG),
                         "youtube_play")
        self.assertIsNone(aa.alias_for("open_app", "", self.REG))
        self.assertEqual(aa.alias_for("open_app", "Spotify", self.REG),
                         "launch_app")
        self.assertIsNone(aa.alias_for("volume_set", "", self.REG))
        self.assertEqual(aa.alias_for("volume_set", "30", self.REG),
                         "set_volume")

    def test_unknown_and_junk_are_none(self):
        self.assertIsNone(aa.alias_for("no_such_thing", "", self.REG))
        self.assertIsNone(aa.alias_for("", "", self.REG))
        self.assertIsNone(aa.alias_for(None, None, self.REG))

    def test_faults_are_none(self):
        class Boom:
            def __contains__(self, item):
                raise RuntimeError("broken registry")
        self.assertIsNone(aa.alias_for("read_screen", "", Boom()))


class FriendlyLabelTests(unittest.TestCase):
    def test_words_not_identifiers(self):
        self.assertEqual(aa.friendly_label("recall_screen"), "recall screen")
        self.assertEqual(aa.friendly_label("read_thread"), "read thread")
        self.assertEqual(aa.friendly_label("toggle_hud"), "toggle HUD")
        self.assertEqual(aa.friendly_label("open_url"), "open URL")

    def test_never_backticks_or_underscores(self):
        for n in list(aa.ALIASES.values()) + list(aa.ALIASES):
            lab = aa.friendly_label(n)
            self.assertNotIn("_", lab)
            self.assertNotIn("`", lab)
            self.assertTrue(lab.strip())

    def test_label_still_matches_the_disambig_reply(self):
        # _resolve_autocorrect_choice matches name.replace("_", " ") in the
        # owner's lower-cased reply: saying the label back must pick it.
        for n in ("recall_screen", "toggle_hud", "list_smart_home_devices"):
            self.assertIn(n.replace("_", " "), aa.friendly_label(n).lower())

    def test_empty(self):
        self.assertEqual(aa.friendly_label(""), "that")
        self.assertEqual(aa.friendly_label(None), "that")


if __name__ == "__main__":
    unittest.main()
