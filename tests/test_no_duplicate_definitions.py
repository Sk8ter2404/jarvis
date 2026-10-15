"""Ratchet: no NEW same-module duplicate def/class (the stale-duplicate class).

A module-level ``def``/``class`` defined twice in one module means the later
one silently replaces the earlier (the classic stale duplicate); a name both
defined and ``from X import``-ed in a module has the same problem. 2026-10-15
scan of core/ skills/ tools/ audio/ hud/*.py bobert_companion.py
upgrade_jarvis.py overnight_upgrade.py found zero straight-line cases, only
the deliberate ``try: from X import f / except: def f`` fallbacks below.
Any new hit must be fixed (delete the dead copy) or added here WITH a reason.
"""
import ast
import collections
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent

_FALLBACK = "try/except ImportError-style fallback: the def only runs if the import fails"
# (relative posix path, name) -> reason. All INTENTIONAL.
ALLOWED = {
    ("core/gpu_usage.py", "model_route"): _FALLBACK,
    ("core/scheduler.py", "_atomic_write_json"): _FALLBACK,
    ("skills/email_triage.py", "_atomic_write_json"): _FALLBACK,
    ("skills/network_deco.py", "_atomic_write_json"): _FALLBACK,
    ("skills/notification_triage.py", "_atomic_write_json"): _FALLBACK,
    ("skills/phone_bridge.py", "_atomic_write_json"): _FALLBACK,
    ("skills/self_diagnostic.py", "_atomic_write_json"): _FALLBACK,
    ("hud/jarvis_hud.py", "_atomic_write_json"): _FALLBACK,
    ("hud/arc_reactor_status_hud.py", "_hud_brain"): _FALLBACK,
    ("hud/holo_workshop_canvas.py", "_hud_brain"): _FALLBACK,
    ("hud/jarvis_holo.py", "_hud_brain"): _FALLBACK,
    ("hud/jarvis_unified_hud.py", "_hud_brain"): _FALLBACK,
    ("bobert_companion.py", "detect_tts_emotion"): _FALLBACK + " (core.tts re-export)",
}

_DIRS = ("core", "skills", "tools", "audio")
_EXTRA = ("bobert_companion.py", "upgrade_jarvis.py", "overnight_upgrade.py")


def _files():
    out = []
    for d in _DIRS:
        out += sorted((ROOT / d).rglob("*.py"))
    out += sorted((ROOT / "hud").glob("*.py"))
    out += [ROOT / f for f in _EXTRA if (ROOT / f).exists()]
    return out


def _walk(body):
    for n in body:
        yield n
        if isinstance(n, (ast.If, ast.Try, ast.With, ast.For, ast.While)):
            for attr in ("body", "orelse", "finalbody"):
                yield from _walk(getattr(n, attr, []))
            for h in getattr(n, "handlers", []):
                yield from _walk(h.body)


def _scan():
    hits = {}
    for f in _files():
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        defs = collections.defaultdict(list)
        imported = set()
        for n in _walk(tree.body):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                defs[n.name].append(n.lineno)
            elif isinstance(n, ast.ImportFrom):
                imported.update(a.asname or a.name for a in n.names if a.name != "*")
        rel = f.relative_to(ROOT).as_posix()
        for name, lines in defs.items():
            if len(lines) > 1 or name in imported:
                hits[(rel, name)] = lines
    return hits


class NoDuplicateDefinitionTests(unittest.TestCase):
    def test_no_new_duplicate_definitions(self):
        hits = _scan()
        new = {k: v for k, v in hits.items() if k not in ALLOWED}
        self.assertEqual(new, {}, "same-module duplicate def/class (later silently "
                         "wins): delete the dead copy or allowlist with a reason: "
                         f"{new}")

    def test_allowlist_has_no_stale_entries(self):
        hits = _scan()
        stale = sorted(k for k in ALLOWED if k not in hits)
        self.assertEqual(stale, [], f"allowlist entries no longer needed: {stale}")


if __name__ == "__main__":
    unittest.main()
