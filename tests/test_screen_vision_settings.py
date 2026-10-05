"""The 2026-10-05 screen-vision settings: four copies agree (core/config.py,
tools/settings_window.py SCHEMA, tools/user_settings.example.json, and the
consuming module's own fallback), each has a real consumer outside the
config / settings / tests files, and the defaults are what was shipped
(AMBIENT_SCREEN_ENABLED stays False in code; the owner turns it on).

    python -m unittest tests.test_screen_vision_settings
"""
from __future__ import annotations

import ast
import glob
import importlib.util
import json
import os
import re
import unittest

from core import config as cfg

_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

KEYS = {
    "VISION_TRACE": "on", "VISION_TRACE_DAYS": 7,
    "VISION_TRACE_MAX_ENTRIES": 300, "VISION_TRACE_MAX_MB": 300,
    "SCREEN_UIA_ENABLED": True, "SCREEN_UIA_NONBROWSER": "on_demand",
    "VLM_MAX_IMAGE_TOKENS": 960, "VISION_GROUNDING_FORMAT": "box2d",
    "CLICK_VERIFY_TIMEOUT_S": 2.5, "CLICK_ROUTE_ENABLED": True,
    "SCREEN_OCR_BACKEND": "auto", "AMBIENT_SCREEN_ENABLED": False,
    "AMBIENT_SCREEN_VLM_ENABLED": False, "SCREEN_MEMORY_INTERVAL_S": 5.0,
    "SCREEN_MEMORY_CPU_PAUSE_PCT": 60.0, "SCREEN_MEMORY_MAX_CORE_PCT": 1.0,
    "SCREEN_TIMELINE_DAYS": 7.0, "SCREEN_TIMELINE_MAX_MB": 200.0,
    "NOTES_FOR_CLAUDE_MIRROR": "",
}
_CONSUMERS = ("core", "skills", "tools", "bobert_companion.py", "tray.py")


def _sw():
    spec = importlib.util.spec_from_file_location(
        "jarvis_settings_window_sv", os.path.join(_PROJECT, "tools",
                                                  "settings_window.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _config_literals() -> dict:
    out = {}
    with open(os.path.join(_PROJECT, "core", "config.py"), encoding="utf-8") as f:
        tree = ast.parse(f.read())
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except Exception:
                pass
    return out


class FourCopiesTests(unittest.TestCase):
    def test_config_values(self):
        lits = _config_literals()
        for k, v in KEYS.items():
            self.assertIn(k, lits, k)
            self.assertEqual(lits[k], v, k)
            self.assertIs(type(lits[k]), type(v), k)

    def test_vision_trace_is_a_string(self):
        # The LOCAL_KEEP_ALIVE lesson: an override is coerced to the
        # DEFAULT's type, so the default must be the string the code reads.
        self.assertIsInstance(cfg.VISION_TRACE, str)

    def test_schema_rows(self):
        schema = _sw().SCHEMA
        for k, v in KEYS.items():
            self.assertIn(k, schema, k)
            self.assertEqual(schema[k]["default"], v, k)

    def test_example_template(self):
        with open(os.path.join(_PROJECT, "tools", "user_settings.example.json"),
                  encoding="utf-8") as f:
            ex = json.load(f)
        for k, v in KEYS.items():
            self.assertIn(k, ex, k)
            self.assertEqual(ex[k], v, k)

    def test_module_fallbacks_match_the_config(self):
        """Every _cfg("KEY", default) / getattr(x, "KEY", default) in the new
        modules uses the shipped value (the stale-duplicate class)."""
        lits = _config_literals()
        files = [os.path.join(_PROJECT, "core", n) for n in (
            "vision_trace.py", "screen_memory.py", "screen_timeline.py",
            "grounded_click.py", "screen_ocr.py", "vision_grounding.py",
            "screen_privacy.py", "dev_notes.py", "screen_digest.py",
            "actions.py")]
        bad = []
        for fp in files:
            with open(fp, encoding="utf-8") as f:
                tree = ast.parse(f.read())
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or len(node.args) < 2:
                    continue
                fn = node.func
                name = (fn.id if isinstance(fn, ast.Name) else
                        fn.attr if isinstance(fn, ast.Attribute) else "")
                if name not in ("_cfg", "getattr"):
                    continue
                key_node = node.args[0] if name == "_cfg" else (
                    node.args[1] if len(node.args) >= 3 else None)
                def_node = node.args[1] if name == "_cfg" else (
                    node.args[2] if len(node.args) >= 3 else None)
                if not (isinstance(key_node, ast.Constant)
                        and key_node.value in KEYS and def_node is not None):
                    continue
                try:
                    default = ast.literal_eval(def_node)
                except Exception:
                    continue
                if default != lits[key_node.value]:
                    bad.append((os.path.basename(fp), key_node.value, default))
        self.assertEqual(bad, [])


class ConsumerTests(unittest.TestCase):
    def test_every_key_has_a_real_consumer(self):
        declare_only = {os.path.normcase(os.path.join(_PROJECT, "core",
                                                      "config.py")),
                        os.path.normcase(os.path.join(_PROJECT, "tools",
                                                      "settings_window.py"))}
        texts = []
        for root in _CONSUMERS:
            path = os.path.join(_PROJECT, root)
            files = ([path] if os.path.isfile(path) else
                     glob.glob(os.path.join(path, "**", "*.py"), recursive=True))
            for fp in files:
                if os.path.normcase(fp) in declare_only or "__pycache__" in fp:
                    continue
                with open(fp, encoding="utf-8", errors="replace") as f:
                    texts.append(f.read())
        blob = "\n".join(texts)
        dead = [k for k in KEYS if not re.search(rf"\b{re.escape(k)}\b", blob)]
        self.assertEqual(dead, [])


class StructuralTests(unittest.TestCase):
    def test_new_modules_are_import_light(self):
        import subprocess
        import sys
        mods = ["core.screen_resolve", "core.onscreen_refs",
                "core.vision_grounding", "core.screen_privacy",
                "core.vision_trace", "core.screen_timeline", "core.dev_notes",
                "core.screen_scope", "core.uia_host", "core.screen_text",
                "core.screen_ocr", "core.grounded_click", "core.screen_digest",
                "core.screen_memory"]
        code = ("import sys; " + "; ".join(f"import {m}" for m in mods)
                + "; heavy=[m for m in ('numpy','cv2','torch','comtypes',"
                "'PIL','mss','requests','bobert_companion') if m in "
                "sys.modules]; print(heavy)")
        r = subprocess.run([sys.executable, "-B", "-c", code], cwd=_PROJECT,
                           capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr[-500:])
        self.assertEqual(r.stdout.strip(), "[]", r.stdout)


if __name__ == "__main__":
    unittest.main()
