"""Config keys after the 2026-10-02 integration (claude/integrate-1002).

Fifteen branches added or moved settings on the same day, and three files
hold every one of them: core/config.py (the value), tools/settings_window.py
SCHEMA (the GUI row) and tools/user_settings.example.json (the template).
A merge can register one key twice - a second assignment in core/config.py,
a repeated key in the SCHEMA literal or in the template (Python and json
both keep the LAST one silently) - or leave a call site whose fallback
default disagrees with the shipped value (the stale-duplicate class).

Pinned here, stdlib + AST only (light tier):
  * every top-level name in core/config.py is assigned once;
  * the SCHEMA dict literal and the example template repeat no key;
  * every in-code fallback for a key this batch added
    (``globals().get(KEY, d)`` / ``getattr(obj, KEY, d)``) is either the
    core/config.py literal or a blank "unknown" sentinel the reader then
    normalises - never a different live value;
  * the instant-actions module's own default and allowlist agree with the
    shipped config.

    python -m unittest tests.test_batch_1002_config_keys
"""
from __future__ import annotations

import ast
import collections
import glob
import json
import os
import unittest

_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The settings the 2026-10-02 branches added or re-pointed.
BATCH_KEYS = (
    "INSTANT_ACTIONS_MODE", "INSTANT_ACTIONS_ALLOW",          # instant-actions
    "TURN_CHECK_MODE", "TURN_CHECK_ESCALATE_MODEL",           # turncheck-ship
    "BACKGROUND_TAG_STRICT",                                  # r5-tagging
    "PROCESSING_FILLER_LATE_START_S",                         # r3-ship
    "PROCESSING_FILLER_SKIP_PLEASANTRIES", "FILLER_DUCK_HOLD",
    "PROCESSING_FILLER_PRERENDER",
    "NOTIFY_SORTER_BACKEND", "BROWSER_AGENT_BACKEND",         # local-features
    "CREDITS_CHECK_BACKEND", "CLAUDE_FAST_MODEL",
    "CLAUDE_MODEL_SUCCESSORS", "ORCHESTRATOR_BACKEND",
    "ORCHESTRATOR_WORKER_MODEL",
    "MEMORY_EMBED_MODEL",                                     # voyage-embed
)

# A fallback that means "not set - normalise it", never a live value.
_BLANKS = ("", None, (), [], {})


def _read(rel: str) -> str:
    with open(os.path.join(_PROJECT, rel), encoding="utf-8") as f:
        return f.read()


def _config_literals() -> dict:
    lits: dict = {}
    for node in ast.parse(_read(os.path.join("core", "config.py"))).body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            try:
                lits[node.targets[0].id] = ast.literal_eval(node.value)
            except Exception:
                pass
    return lits


def _fallbacks(tree) -> list:
    """(key, default node, line) for every globals().get("KEY", d) and
    getattr(obj, "KEY", d) call in ``tree``."""
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if (isinstance(fn, ast.Attribute) and fn.attr == "get"
                and isinstance(fn.value, ast.Call)
                and isinstance(fn.value.func, ast.Name)
                and fn.value.func.id == "globals"
                and len(node.args) >= 2
                and isinstance(node.args[0], ast.Constant)):
            out.append((node.args[0].value, node.args[1], node.lineno))
        elif (isinstance(fn, ast.Name) and fn.id == "getattr"
                and len(node.args) == 3
                and isinstance(node.args[1], ast.Constant)):
            out.append((node.args[1].value, node.args[2], node.lineno))
    return out


class SingleRegistrationTests(unittest.TestCase):
    def test_every_config_name_is_assigned_once(self):
        seen = collections.Counter()
        for node in ast.parse(_read(os.path.join("core", "config.py"))).body:
            names = []
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        names.append(t.id)
                    elif isinstance(t, (ast.Tuple, ast.List)):
                        names += [e.id for e in t.elts
                                  if isinstance(e, ast.Name)]
            elif (isinstance(node, ast.AnnAssign)
                    and isinstance(node.target, ast.Name)):
                names.append(node.target.id)
            seen.update(names)
        twice = sorted(n for n, c in seen.items() if c > 1)
        self.assertEqual(twice, [], "a second assignment silently wins")
        self.assertGreater(len(seen), 300)   # guard the guard

    def test_the_schema_literal_repeats_no_key(self):
        tree = ast.parse(_read(os.path.join("tools", "settings_window.py")))
        literal = None
        for node in tree.body:
            target = (node.targets[0] if isinstance(node, ast.Assign)
                      else getattr(node, "target", None))
            if isinstance(target, ast.Name) and target.id == "SCHEMA":
                literal = node.value
        self.assertIsInstance(literal, ast.Dict)
        keys = [k.value for k in literal.keys if isinstance(k, ast.Constant)]
        self.assertEqual(len(keys), len(literal.keys),
                         "every SCHEMA key is a plain string literal")
        dup = sorted(k for k, c in collections.Counter(keys).items() if c > 1)
        self.assertEqual(dup, [])
        for key in BATCH_KEYS:
            self.assertLessEqual(keys.count(key), 1, key)

    def test_the_example_template_repeats_no_key(self):
        dups: list = []

        def hook(pairs):
            dups.extend(k for k, c in collections.Counter(
                k for k, _ in pairs).items() if c > 1)
            return dict(pairs)

        json.loads(_read(os.path.join("tools", "user_settings.example.json")),
                   object_pairs_hook=hook)
        self.assertEqual(dups, [])


class FallbackDefaultTests(unittest.TestCase):
    def test_batch_fallbacks_are_the_shipped_value_or_blank(self):
        lits = _config_literals()
        for key in BATCH_KEYS:
            self.assertIn(key, lits, key)
        files = (["bobert_companion.py"]
                 + sorted(glob.glob(os.path.join(_PROJECT, "core", "*.py")))
                 + sorted(glob.glob(os.path.join(_PROJECT, "skills", "*.py")))
                 + sorted(glob.glob(os.path.join(_PROJECT, "tools", "*.py"))))
        checked = 0
        bad = []
        for path in files:
            rel = os.path.relpath(os.path.join(_PROJECT, path), _PROJECT)
            tree = ast.parse(_read(rel))
            for key, default, line in _fallbacks(tree):
                if key not in BATCH_KEYS:
                    continue
                try:
                    value = ast.literal_eval(default)
                except Exception:
                    continue           # a computed default (a module constant)
                checked += 1
                shipped = lits[key]
                if value == shipped:
                    continue
                if isinstance(shipped, bool):
                    bad.append((rel, line, key, value, shipped))
                elif value not in _BLANKS:
                    bad.append((rel, line, key, value, shipped))
        self.assertEqual(bad, [], "a fallback that is a different live value")
        self.assertGreater(checked, 10)    # guard the guard

    def test_instant_actions_defaults_agree_with_config(self):
        from core import instant_actions as ia
        lits = _config_literals()
        self.assertEqual(ia.DEFAULT_MODE, lits["INSTANT_ACTIONS_MODE"])
        self.assertEqual(ia.normalize_mode(""), lits["INSTANT_ACTIONS_MODE"])
        shipped = lits["INSTANT_ACTIONS_ALLOW"]
        # Every shipped allowlist entry is one a rule can produce: no dead
        # entries, and nothing the allowlist could never narrow.
        self.assertEqual(ia.normalize_allow(shipped), frozenset(shipped))
        self.assertEqual(frozenset(shipped), ia.RULE_ACTIONS)


if __name__ == "__main__":   # pragma: no cover
    unittest.main()
