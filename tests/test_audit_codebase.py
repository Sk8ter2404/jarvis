"""Unit tests for tools/audit_codebase.py — the in-tree static codebase auditor.

WHAT THIS COVERS
  The auditor is a large (~1.5K-stmt) collection of mostly-pure checker and
  helper functions that walk a Python tree, parse with `ast`, and emit
  `Finding` objects (P0/P1/P2). This suite drives each checker DIRECTLY against
  tiny synthetic source strings + temp files, asserting the clean-vs-dirty
  verdict per branch. main() is exercised against a minimal temp tree with the
  real-repo walk / heavy import / leak loop neutralised.

CI-SAFETY CONTRACT (runs on BOTH the Windows dev box AND a bare Linux 3.14
runner with only light deps):
  * The target top-level-imports ONLY stdlib, so importing it is CI-safe.
  * We NEVER run the auditor against the real C:\\JARVIS tree (slow, host- and
    OS-dependent, and its import-resolution flags every absent heavy dep on a
    light runner). Every filesystem-touching checker is pointed at a per-test
    tempdir by patching the module's PROJECT_DIR / SKILLS_DIR / CORE_DIR / …
    path globals; the originals are restored in tearDown.
  * The action smoke-test's optional `import bobert_companion` is forced to
    fail (mock) so the deterministic static-fallback path is what we assert —
    otherwise the result is host-dependent (the real monolith may or may not
    import / register a given action).
  * No real subprocess / network / threads / sleep.
  * Tests that hinge on Windows backslash-path normalisation are guarded with
    @skipUnless(win) — that code can't be faithfully exercised on posix.

PRIVACY: every secret-shaped / key-shaped fixture string is BUILT AT RUNTIME by
concatenation so no literal credential pattern lives in this file (the CI PII
gate greps tests/). No real names or LAN IPs.

BUGS / QUIRKS found while writing this are annotated `# QUIRK:` / `# NOTE:` —
the target source is NOT modified to "make tests pass".
"""
from __future__ import annotations

import ast
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

# ─────────────────────────────────────────────────────────────────────────
#  Import the target from tools/ by file location.
#
#  tools/ is not on sys.path under the test runner (only the project root is),
#  so load the module from its absolute path. CRITICAL: the module must be
#  registered in sys.modules under its own name BEFORE exec_module — Python
#  3.14's @dataclass introspection does `sys.modules[cls.__module__].__dict__`,
#  which raises AttributeError(NoneType) if the module isn't registered yet.
# ─────────────────────────────────────────────────────────────────────────

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_TESTS_DIR)
_TARGET_PATH = os.path.join(_PROJECT_ROOT, "tools", "audit_codebase.py")


def _load_audit_module():
    spec = importlib.util.spec_from_file_location("audit_codebase", _TARGET_PATH)
    assert spec and spec.loader, "could not build import spec for audit_codebase"
    mod = importlib.util.module_from_spec(spec)
    sys.modules["audit_codebase"] = mod  # MUST precede exec for dataclass (3.14)
    spec.loader.exec_module(mod)
    return mod


audit = _load_audit_module()


# Source-fragment building blocks for PC_CONTROL_PROMPT fixtures. The auditor's
# prompt regexes key off the literal two-character `\n` then `"` that terminate
# each Python source line, plus an em-dash (U+2014) for column-aligned no-arg
# table rows. Build these explicitly to keep the test file ASCII-clean and to
# avoid any editor/encoding ambiguity.
_BSLASH_N = chr(92) + "n"   # the two chars backslash + n, as they sit in source
_QUOTE = '"'
_EMDASH = chr(0x2014)


def _prompt_line(inner: str) -> str:
    """One PC_CONTROL_PROMPT body source line: indent + quote + inner + \\n" ."""
    return "    " + _QUOTE + inner + _BSLASH_N + _QUOTE + "\n"


def _make_prompt(*inner_lines: str) -> str:
    """Wrap inner prompt lines in the `PC_CONTROL_PROMPT = ( ... )` envelope the
    extractor matches (it requires a `\\n)\\n` close)."""
    return ("PC_CONTROL_PROMPT = (\n"
            + "".join(_prompt_line(s) for s in inner_lines)
            + ")\n")


# ─────────────────────────────────────────────────────────────────────────
#  Base: redirect every path global into a fresh tempdir; restore in teardown.
# ─────────────────────────────────────────────────────────────────────────

_PATCHED_PATH_ATTRS = (
    "PROJECT_DIR", "SKILLS_DIR", "HUD_DIR", "CORE_DIR", "TOOLS_DIR",
    "REQUIREMENTS_FILE",
)


class _AuditTestBase(unittest.TestCase):
    def setUp(self):
        # Snapshot every path constant so a test can freely reassign them.
        self._saved = {k: getattr(audit, k) for k in _PATCHED_PATH_ATTRS}
        # Per-test sandbox tree the checkers will see as the whole project.
        self.tmp = tempfile.mkdtemp(prefix="audit_test_")
        self.addCleanup(self._restore_paths)
        self.addCleanup(self._rm_tmp)
        self._point_paths_at(self.tmp)
        # The skill-profile cache is module-global; isolate tests from each
        # other (paths are reused-by-value, file contents differ).
        audit._skill_profile_cache.clear()
        self.addCleanup(audit._skill_profile_cache.clear)

    def _point_paths_at(self, root: str):
        audit.PROJECT_DIR = root
        audit.SKILLS_DIR = os.path.join(root, "skills")
        audit.HUD_DIR = os.path.join(root, "hud")
        audit.CORE_DIR = os.path.join(root, "core")
        audit.TOOLS_DIR = os.path.join(root, "tools")
        audit.REQUIREMENTS_FILE = os.path.join(root, "requirements.txt")

    def _restore_paths(self):
        for k, v in self._saved.items():
            setattr(audit, k, v)

    def _rm_tmp(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    # — helpers —
    def write(self, relpath: str, text: str) -> str:
        """Write a UTF-8 file under the sandbox; return its absolute path."""
        ap = os.path.join(self.tmp, relpath)
        os.makedirs(os.path.dirname(ap), exist_ok=True)
        with open(ap, "w", encoding="utf-8") as fh:
            fh.write(text)
        return ap

    def mkdir(self, relpath: str) -> str:
        ap = os.path.join(self.tmp, relpath)
        os.makedirs(ap, exist_ok=True)
        return ap

    def read_text(self, relpath: str) -> str:
        with open(os.path.join(self.tmp, relpath), encoding="utf-8") as fh:
            return fh.read()

    def read_json(self, relpath: str):
        with open(os.path.join(self.tmp, relpath), encoding="utf-8") as fh:
            return json.load(fh)

    @staticmethod
    def cats(findings):
        return [f.category for f in findings]

    @staticmethod
    def by_cat(findings, category):
        return [f for f in findings if f.category == category]


# ═════════════════════════════════════════════════════════════════════════
#  module import + data model
# ═════════════════════════════════════════════════════════════════════════

class ImportAndModelTests(_AuditTestBase):
    def test_module_imported_with_expected_symbols(self):
        for name in ("Finding", "check_syntax", "check_imports", "main",
                     "walk_py_files", "STATE_FILES", "SECRET_PATTERNS"):
            self.assertTrue(hasattr(audit, name), f"missing symbol {name}")

    def test_finding_as_md_line_with_line_and_fix(self):
        f = audit.Finding(severity="P1", category="imports",
                          file="skills/x.py", line=12, message="boom",
                          fixable=True)
        line = f.as_md_line()
        self.assertIn("[P1]", line)
        self.assertIn("skills/x.py:12", line)
        self.assertIn("boom", line)
        self.assertIn("auto-fixable", line)

    def test_finding_as_md_line_file_wide_no_fix(self):
        # line == 0 → location is the bare file, no ":0" suffix, no fix marker.
        f = audit.Finding(severity="P2", category="cat", file="a.py", line=0,
                          message="msg")
        line = f.as_md_line()
        self.assertIn("`a.py`", line)
        self.assertNotIn("a.py:0", line)
        self.assertNotIn("auto-fixable", line)

    def test_stdlib_modules_populated_includes_ast(self):
        # The auditor relies on sys.stdlib_module_names to classify imports.
        self.assertIn("ast", audit.STDLIB_MODULES)
        self.assertIn("os", audit.STDLIB_MODULES)


# ═════════════════════════════════════════════════════════════════════════
#  helpers: _rel, walk_py_files, _read_source
# ═════════════════════════════════════════════════════════════════════════

class HelperTests(_AuditTestBase):
    @unittest.skipUnless(sys.platform.startswith("win"),
                         "backslash→slash normalisation only meaningful on Windows")
    def test_rel_normalises_backslashes_on_windows(self):
        abs_path = os.path.join(audit.PROJECT_DIR, "core", "tts.py")
        self.assertEqual(audit._rel(abs_path), "core/tts.py")

    def test_rel_returns_forward_slash_relative_path(self):
        # On any OS, a path under PROJECT_DIR comes back relative + slash-joined.
        abs_path = os.path.join(audit.PROJECT_DIR, "skills", "foo.py")
        self.assertEqual(audit._rel(abs_path), "skills/foo.py")

    def test_read_source_ok_and_missing(self):
        p = self.write("a.py", "x = 1\ny = 2\n")
        text, lines = audit._read_source(p)
        self.assertEqual(text, "x = 1\ny = 2\n")
        self.assertEqual(lines, ["x = 1", "y = 2"])
        # Nonexistent → (None, [])
        text2, lines2 = audit._read_source(os.path.join(self.tmp, "nope.py"))
        self.assertIsNone(text2)
        self.assertEqual(lines2, [])

    def test_walk_py_files_prunes_excludes_and_underscore(self):
        self.write("a.py", "")
        self.write("__init__.py", "")        # kept (package entry point)
        self.write("_private.py", "")        # skipped (underscore template)
        self.write("note.txt", "")           # skipped (not .py)
        self.write("skills/s.py", "")        # kept
        self.write("backups/b.py", "")       # pruned dir
        self.write("__pycache__/c.py", "")   # pruned dir
        self.write("tests/t.py", "")         # pruned dir (in EXCLUDE_DIRS)
        got = {os.path.basename(p) for p in audit.walk_py_files()}
        self.assertEqual(got, {"a.py", "__init__.py", "s.py"})
        # No pruned-dir file leaked through.
        self.assertFalse(any("backups" in p or "__pycache__" in p
                             for p in audit.walk_py_files()))


# ═════════════════════════════════════════════════════════════════════════
#  requirements parsing
# ═════════════════════════════════════════════════════════════════════════

class RequirementsTests(_AuditTestBase):
    def _write_reqs(self, body):
        return self.write("requirements.txt", body)

    def test_parse_requirements_basic_and_optional(self):
        self._write_reqs(
            "requests>=2.0\n"
            "opencv-python  # optional: cv2\n"
            "# optional: RealtimeTTS\n"
            "numpy==1.0  # needed\n"
            "\n"
            "# a plain comment\n"
            "paho-mqtt\n"
        )
        pkgs, opt = audit._parse_requirements_full()
        self.assertEqual(pkgs, {"requests", "opencv-python", "numpy", "paho-mqtt"})
        # Both the inline `# optional` marker on opencv-python AND the bare
        # `# optional: RealtimeTTS` directive land in the optional set.
        self.assertIn("opencv-python", opt)
        self.assertIn("RealtimeTTS", opt)

    def test_public_parse_wrappers(self):
        self._write_reqs("alpha  # optional\nbeta\n")
        self.assertEqual(audit.parse_requirements(), {"alpha", "beta"})
        self.assertEqual(audit.parse_optional_requirements(), {"alpha"})

    def test_parse_requirements_missing_file_returns_empty(self):
        audit.REQUIREMENTS_FILE = os.path.join(self.tmp, "does_not_exist.txt")
        self.assertEqual(audit.parse_requirements(), set())
        self.assertEqual(audit.parse_optional_requirements(), set())

    def test_req_to_import_name_mapping(self):
        self.assertEqual(audit.req_to_import_name("opencv-python"), "cv2")
        self.assertEqual(audit.req_to_import_name("paho-mqtt"), "paho.mqtt.client")
        self.assertEqual(audit.req_to_import_name("pillow"), "PIL")
        # Unknown name → hyphens become underscores.
        self.assertEqual(audit.req_to_import_name("some-thing"), "some_thing")


# ═════════════════════════════════════════════════════════════════════════
#  CHECK 2 — syntax
# ═════════════════════════════════════════════════════════════════════════

class SyntaxCheckTests(_AuditTestBase):
    def test_clean_file_no_findings(self):
        good = self.write("good.py", "def f():\n    return 1\n")
        self.assertEqual(audit.check_syntax([good]), [])

    def test_broken_file_yields_p0(self):
        bad = self.write("bad.py", "def (:\n")
        out = audit.check_syntax([bad])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].severity, "P0")
        self.assertEqual(out[0].category, "syntax")
        self.assertEqual(os.path.basename(out[0].file), "bad.py")


# ═════════════════════════════════════════════════════════════════════════
#  CHECK 3 — imports (incl. _iter_imports)
# ═════════════════════════════════════════════════════════════════════════

class IterImportsTests(_AuditTestBase):
    def test_iter_imports_marks_conditional(self):
        tree = ast.parse(
            "import os\n"
            "import numpy as np\n"
            "if True:\n"
            "    import cond_mod\n"
            "def g():\n"
            "    import deferred_mod\n"
            "from . import rel_mod\n"      # relative → skipped (level>0)
        )
        got = {(mod, cond) for mod, _ln, cond in audit._iter_imports(tree)}
        self.assertIn(("os", False), got)
        self.assertIn(("numpy", False), got)
        self.assertIn(("cond_mod", True), got)
        self.assertIn(("deferred_mod", True), got)
        # Relative import contributes no top-module entry.
        self.assertFalse(any(mod == "rel_mod" for mod, _l, _c in
                             audit._iter_imports(tree)))


class ImportsCheckTests(_AuditTestBase):
    def setUp(self):
        super().setUp()
        self.mkdir("skills")  # check_imports lists SKILLS_DIR

    def test_stdlib_and_intra_imports_clean(self):
        self.write("requirements.txt", "")
        self.write("sibling.py", "")  # makes 'sibling' an intra-project root
        src = "import os\nimport sys\nimport sibling\nfrom skills import x\n"
        fp = self.write("mod.py", src)
        self.assertEqual(audit.check_imports([fp]), [])

    def test_undeclared_third_party_is_p1(self):
        self.write("requirements.txt", "")
        # A module name that is definitely not importable anywhere.
        fp = self.write("mod.py", "import zzz_totally_absent_pkg_98765\n")
        out = audit.check_imports([fp])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].severity, "P1")
        self.assertEqual(out[0].category, "imports")
        self.assertIn("undeclared dependency", out[0].message)

    def test_declared_but_missing_is_p1_with_note(self):
        # Declared in requirements.txt but not installed → different message.
        self.write("requirements.txt", "zzz-absent-declared-pkg-4242\n")
        fp = self.write("mod.py", "import zzz_absent_declared_pkg_4242\n")
        out = audit.check_imports([fp])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].severity, "P1")
        self.assertIn("declared in requirements.txt", out[0].message)

    def test_conditional_absent_import_is_p2(self):
        self.write("requirements.txt", "")
        src = ("try:\n"
               "    import zzz_absent_conditional_111\n"
               "except Exception:\n"
               "    pass\n")
        fp = self.write("mod.py", src)
        out = audit.check_imports([fp])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].severity, "P2")
        self.assertIn("conditional import", out[0].message)

    def test_conditional_optional_import_is_suppressed(self):
        # Marked '# optional' in requirements AND imported conditionally →
        # finding suppressed entirely.
        self.write("requirements.txt", "# optional: zzz_optional_mod_222\n")
        src = ("if True:\n"
               "    import zzz_optional_mod_222\n")
        fp = self.write("mod.py", src)
        self.assertEqual(audit.check_imports([fp]), [])


# ═════════════════════════════════════════════════════════════════════════
#  CHECK 4 — cross-references
# ═════════════════════════════════════════════════════════════════════════

class CrossReferenceTests(_AuditTestBase):
    def setUp(self):
        super().setUp()
        self.mkdir("skills")
        self.mkdir("core")

    def _bc(self, body):
        self.write("bobert_companion.py", body)

    def test_no_bc_module_returns_empty(self):
        # bobert_companion absent → check bails (syntax check owns that case).
        fp = self.write("skills/s.py", "import bobert_companion as bc\nx = bc.foo\n")
        self.assertEqual(audit.check_cross_references([fp]), [])

    def test_from_import_unknown_name_flagged(self):
        self._bc("def real_fn():\n    pass\n")
        fp = self.write("skills/s.py",
                        "from bobert_companion import does_not_exist\n")
        out = audit.check_cross_references([fp])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].severity, "P1")
        self.assertEqual(out[0].category, "cross-ref")
        self.assertIn("does_not_exist", out[0].message)

    def test_attribute_access_resolution(self):
        self._bc("def real_fn():\n    pass\nCONST = 1\n")
        fp = self.write("skills/s.py",
                        "import bobert_companion as bc\n"
                        "a = bc.real_fn()\n"      # ok
                        "b = bc.CONST\n"          # ok
                        "c = bc.ghost_attr\n")    # bad
        out = audit.check_cross_references([fp])
        self.assertEqual(len(out), 1)
        self.assertIn("ghost_attr", out[0].message)

    def test_core_module_from_import_resolution(self):
        self._bc("X = 1\n")  # must parse so check proceeds
        self.write("core/tts.py", "def synth():\n    pass\nPRESET = 1\n")
        fp = self.write("skills/s.py",
                        "from core.tts import synth\n"        # ok
                        "from core.tts import missing_sym\n")  # bad
        out = audit.check_cross_references([fp])
        self.assertEqual(len(out), 1)
        self.assertIn("missing_sym", out[0].message)
        self.assertIn("core/tts.py", out[0].message)

    def test_star_import_names_resolve(self):
        # bobert_companion does `from core.config import *`; star-exported names
        # must count as bc attributes (no false cross-ref).
        self.write("core/config.py", "MONITORS = []\nWIDTH = 1\n")
        self._bc("from core.config import *\nBAR = 2\n")
        names = audit.collect_bobert_companion_names()
        self.assertIn("MONITORS", names)
        self.assertIn("WIDTH", names)
        self.assertIn("BAR", names)

    def test_global_declared_names_resolve(self):
        # Names assigned via `global` inside a function are module globals too.
        self._bc("def boot():\n    global LAZY_G\n    LAZY_G = 1\n")
        names = audit.collect_bobert_companion_names()
        self.assertIn("LAZY_G", names)
        self.assertIn("boot", names)

    def test_names_in_module_level_try_if_with_blocks_resolve(self):
        # Module-level conditional/try/with blocks still define module globals;
        # the collector descends into body/orelse/handlers/with (but not into
        # nested funcs/classes).
        self._bc(
            "try:\n    import blue_green_manager\n    ROLE = 1\n"
            "except Exception:\n    ROLE_FALLBACK = 2\n"
            "if True:\n    IF_NAME = 3\n"
            "with open('x') as fh:\n    WITH_NAME = 4\n"
        )
        names = audit.collect_bobert_companion_names()
        for n in ("ROLE", "ROLE_FALLBACK", "IF_NAME", "WITH_NAME"):
            self.assertIn(n, names)


# ═════════════════════════════════════════════════════════════════════════
#  collect_core_module_names / _module_exported_names
# ═════════════════════════════════════════════════════════════════════════

class ModuleNameCollectionTests(_AuditTestBase):
    def test_collect_core_module_names_skips_underscore(self):
        self.mkdir("core")
        self.write("core/tts.py", "def synth():\n    pass\nPRESET = 1\n")
        self.write("core/_private.py", "x = 1\n")
        cm = audit.collect_core_module_names()
        self.assertIn("tts", cm)
        self.assertEqual(cm["tts"], {"synth", "PRESET"})
        self.assertNotIn("_private", cm)

    def test_collect_core_module_names_no_dir(self):
        audit.CORE_DIR = os.path.join(self.tmp, "no_core_here")
        self.assertEqual(audit.collect_core_module_names(), {})

    def test_module_exported_names_with_all(self):
        p = self.write("m.py",
                       "__all__ = ['foo', 'bar']\n"
                       "def foo():\n    pass\n"
                       "def _hidden():\n    pass\n"
                       "baz = 1\n")
        self.assertEqual(audit._module_exported_names(p), {"foo", "bar"})

    def test_module_exported_names_without_all(self):
        p = self.write("m.py",
                       "def foo():\n    pass\n"
                       "def _hidden():\n    pass\n"
                       "baz = 1\n"
                       "import os\n")
        got = audit._module_exported_names(p)
        self.assertIn("foo", got)
        self.assertIn("baz", got)
        self.assertIn("os", got)
        self.assertNotIn("_hidden", got)  # underscore excluded sans __all__


# ═════════════════════════════════════════════════════════════════════════
#  CHECK 5 — ACTIONS registration
# ═════════════════════════════════════════════════════════════════════════

class ActionsCheckTests(_AuditTestBase):
    def setUp(self):
        super().setUp()
        self.mkdir("skills")

    def test_clean_single_arg_handler(self):
        src = ("def _act_ok(payload):\n    return ''\n"
               "def register(actions):\n    actions['ok'] = _act_ok\n")
        fp = self.write("skills/s.py", src)
        out, owners = audit.check_actions([fp], set())
        self.assertEqual(out, [])
        self.assertEqual(owners.get("ok"), "skills/s.py")

    def test_handler_requiring_two_args_flagged(self):
        src = ("def _act_bad(a, b):\n    return ''\n"
               "def register(actions):\n    actions['bad'] = _act_bad\n")
        fp = self.write("skills/s.py", src)
        out, _ = audit.check_actions([fp], set())
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].severity, "P1")
        self.assertIn("requires 2 positional args", out[0].message)

    def test_zero_arg_handler_flagged(self):
        src = ("def _act_zero():\n    return ''\n"
               "def register(actions):\n    actions['z'] = _act_zero\n")
        fp = self.write("skills/s.py", src)
        out, _ = audit.check_actions([fp], set())
        self.assertEqual(len(out), 1)
        self.assertIn("takes zero args", out[0].message)

    def test_collision_with_existing_owner_flagged(self):
        src = ("def _h(p):\n    return ''\n"
               "def register(actions):\n    actions['dup'] = _h\n")
        fp = self.write("skills/s.py", src)
        # 'dup' already owned by bobert_companion.
        out, owners = audit.check_actions([fp], {"dup"})
        collisions = [f for f in out if "collides" in f.message]
        self.assertEqual(len(collisions), 1)
        self.assertEqual(collisions[0].severity, "P1")
        # Last-load-wins: ownership transfers to the skill.
        self.assertEqual(owners["dup"], "skills/s.py")

    def test_intentional_wrap_comment_suppresses_collision(self):
        src = ("def _h(p):\n    return ''\n"
               "def register(actions):\n"
               "    actions['dup'] = _h  # INTENTIONAL_WRAP override\n")
        fp = self.write("skills/s.py", src)
        out, _ = audit.check_actions([fp], {"dup"})
        self.assertEqual([f for f in out if "collides" in f.message], [])

    def test_register_without_args_flagged(self):
        src = "def register():\n    pass\n"
        fp = self.write("skills/s.py", src)
        out, _ = audit.check_actions([fp], set())
        self.assertEqual(len(out), 1)
        self.assertIn("takes no arguments", out[0].message)

    def test_no_register_function_is_skipped(self):
        fp = self.write("skills/s.py", "def helper():\n    pass\n")
        out, owners = audit.check_actions([fp], set())
        self.assertEqual(out, [])


# ═════════════════════════════════════════════════════════════════════════
#  CHECK 6 — bad patterns (regex + ast hybrid)
# ═════════════════════════════════════════════════════════════════════════

class BadPatternTests(_AuditTestBase):
    def test_all_patterns_in_one_file(self):
        src = (
            "import threading, subprocess, os\n"
            "eval('1+1')\n"                       # P0 eval
            "exec('y=2')\n"                       # P1 exec
            "os.system('ls')\n"                   # P1 os-system
            "f = open('a.txt')\n"                 # P2 open-no-encoding
            "t = threading.Thread(target=foo)\n"  # P1 thread-no-daemon
            "subprocess.run(['x'])\n"             # P1 subprocess-no-timeout
            "subprocess.Popen(['y'])\n"           # P2 popen-no-close-fds
            "try:\n    pass\nexcept:\n    pass\n"  # P2 bare-except
        )
        fp = self.write("mod.py", src)
        cats = self.cats(audit.check_bad_patterns([fp]))
        for expected in ("eval", "exec", "os-system", "open-no-encoding",
                         "thread-no-daemon", "subprocess-no-timeout",
                         "popen-no-close-fds", "bare-except"):
            self.assertIn(expected, cats, f"missing {expected}")

    def test_open_with_encoding_is_clean(self):
        fp = self.write("mod.py", "f = open('a.txt', encoding='utf-8')\n")
        self.assertEqual(self.by_cat(audit.check_bad_patterns([fp]),
                                     "open-no-encoding"), [])

    def test_open_binary_mode_is_skipped(self):
        fp = self.write("mod.py", "f = open('a.bin', 'rb')\n")
        self.assertEqual(self.by_cat(audit.check_bad_patterns([fp]),
                                     "open-no-encoding"), [])

    def test_thread_with_daemon_kwarg_clean(self):
        fp = self.write("mod.py",
                        "import threading\n"
                        "t = threading.Thread(target=f, daemon=True)\n")
        self.assertEqual(self.by_cat(audit.check_bad_patterns([fp]),
                                     "thread-no-daemon"), [])

    def test_thread_post_construction_daemon_clean(self):
        # `t = Thread(...); t.daemon = True` idiom must NOT be flagged.
        fp = self.write("mod.py",
                        "import threading\n"
                        "t = threading.Thread(target=f)\n"
                        "t.daemon = True\n"
                        "t.start()\n")
        self.assertEqual(self.by_cat(audit.check_bad_patterns([fp]),
                                     "thread-no-daemon"), [])

    def test_subprocess_with_timeout_clean(self):
        fp = self.write("mod.py",
                        "import subprocess\n"
                        "subprocess.run(['x'], timeout=5)\n")
        self.assertEqual(self.by_cat(audit.check_bad_patterns([fp]),
                                     "subprocess-no-timeout"), [])

    def test_strings_and_comments_do_not_false_positive(self):
        # eval(/except: inside a string literal or comment must be ignored.
        src = (
            "X = 'this mentions eval( and except: but is a string'\n"
            "# eval('nope') and os.system('nope') in a comment\n"
            "DOC = \"\"\"\nexcept:\neval(\n\"\"\"\n"
        )
        fp = self.write("mod.py", src)
        out = audit.check_bad_patterns([fp])
        self.assertEqual(out, [])

    def test_hardcoded_path_in_string_literal_is_not_flagged_quirk(self):
        # QUIRK (reported, not "fixed" in source): _HARDCODED_PATH_RE is applied
        # only to lines that are NOT inside a string literal (the loop skips
        # `i in string_lines`). A hardcoded filesystem path is ALWAYS a string
        # literal, so in practice this regex branch never fires for real code —
        # the hardcoded-path check is effectively dead. We pin the actual
        # (surprising) behaviour: a path-in-a-string yields NO hardcoded-path
        # finding. Build the path at runtime to keep the literal out of source;
        # emit it as a RAW string literal in the fixture so parsing it doesn't
        # raise an invalid-escape SyntaxWarning for the backslashes.
        bad = "PATH = r'" + "D:" + chr(92) + "PC Files" + chr(92) + "thing'\n"
        fp = self.write("mod.py", bad)
        cats = self.cats(audit.check_bad_patterns([fp]))
        self.assertNotIn("hardcoded-path", cats)

    def test_callee_name_resolution(self):
        call = ast.parse("a.b.c(1)").body[0].value
        self.assertEqual(audit._callee_name(call), "a.b.c")
        bare = ast.parse("open(1)").body[0].value
        self.assertEqual(audit._callee_name(bare), "open")
        # Non-Name/Attribute base (a call result) → None.
        weird = ast.parse("f()()").body[0].value
        self.assertIsNone(audit._callee_name(weird))

    def test_kwarg_helpers(self):
        call = ast.parse("open('x', mode='w', encoding='utf-8')").body[0].value
        self.assertTrue(audit._has_kwarg(call, "encoding"))
        self.assertFalse(audit._has_kwarg(call, "buffering"))
        self.assertEqual(audit._const_str(audit._kwarg_value(call, "mode")), "w")
        self.assertIsNone(audit._kwarg_value(call, "nope"))

    def test_unresolvable_callee_in_file_is_skipped(self):
        # A call whose callee is itself a call result (`f()()`) resolves to None
        # in _ast_bad_pattern_checks and is skipped — no crash, no finding.
        fp = self.write("mod.py", "f()()\n")
        out = audit.check_bad_patterns([fp])
        self.assertEqual(out, [])

    def test_json_dump_call_is_noop_branch(self):
        # `json.dump(d, f)` hits the dedicated json.dump branch (a pass) without
        # producing a bad-pattern finding here.
        fp = self.write("mod.py", "import json\njson.dump(d, f)\n")
        out = audit.check_bad_patterns([fp])
        self.assertEqual(self.by_cat(out, "open-no-encoding"), [])

    def test_open_with_non_constant_mode_treated_as_text(self):
        # open(path, modevar) — mode arg is a Name, not a str constant, so
        # _const_str returns None; the call is treated as text (encoding wanted).
        fp = self.write("mod.py", "m = 'r'\nf = open('a.txt', m)\n")
        cats = self.cats(audit.check_bad_patterns([fp]))
        self.assertIn("open-no-encoding", cats)

    def test_hardcoded_path_in_trailing_comment_flagged(self):
        # The hardcoded-path regex fires on a non-string, non-comment-leading
        # line — here a code line with the path in a trailing comment. Build the
        # path at runtime so no literal lives in this test file.
        bad_path = "D:" + chr(92) + "PC Files" + chr(92) + "thing"
        fp = self.write("mod.py", "x = 1  # see " + bad_path + "\n")
        cats = self.cats(audit.check_bad_patterns([fp]))
        self.assertIn("hardcoded-path", cats)

    def test_bare_unassigned_thread_flagged_no_daemon(self):
        # A Thread call that is NOT bound to `<name> = ...` exercises the
        # _has_post_construction_daemon early-out (parent is not an Assign).
        fp = self.write("mod.py",
                        "import threading\nthreading.Thread(target=foo)\n")
        cats = self.cats(audit.check_bad_patterns([fp]))
        self.assertIn("thread-no-daemon", cats)

    def test_thread_assigned_to_attribute_target_flagged(self):
        # `self.t = Thread(...)` — the assign target is an Attribute (not a bare
        # Name), so the post-construction-daemon check bails and the finding fires.
        fp = self.write("mod.py",
                        "import threading\n"
                        "class C:\n"
                        "    def m(self):\n"
                        "        self.t = threading.Thread(target=foo)\n")
        cats = self.cats(audit.check_bad_patterns([fp]))
        self.assertIn("thread-no-daemon", cats)


class SyntaxErrorResilienceTests(_AuditTestBase):
    """Every AST-walking checker must SWALLOW a SyntaxError file (it's already
    reported by check_syntax) and continue, never raising. Feed each a file
    that does not parse and assert no exception + no spurious finding from it."""

    def setUp(self):
        super().setUp()
        self.mkdir("skills")
        # A file that reads fine but does NOT parse.
        self.bad = self.write("skills/broken.py", "def (:\n    pass\n")

    def test_ast_checkers_skip_unparseable_file(self):
        # None of these should raise; broken.py contributes no finding of its
        # own category (only check_syntax would flag it, which we don't call).
        self.assertEqual(audit.check_imports([self.bad]), [])
        self.assertEqual(audit.check_bad_patterns([self.bad]), [])
        self.assertEqual(audit.check_state_file_writes([self.bad]), [])
        self.assertEqual(audit.check_mutation_hygiene([self.bad]), [])
        thread_f, _ = audit.check_thread_audit([self.bad])
        self.assertEqual(thread_f, [])
        _, owners = audit.check_actions([self.bad], set())
        self.assertEqual(owners, {})
        # profile of an unparseable file is the empty default profile.
        prof = audit._profile_skill(self.bad)
        self.assertEqual(prof["actions"], {})

    def test_collectors_return_empty_on_unparseable(self):
        # bobert_companion that doesn't parse → empty name set.
        self.write("bobert_companion.py", "def (:\n")
        self.assertEqual(audit.collect_bobert_companion_names(), set())
        # core module that doesn't parse → skipped (not in result map).
        self.write("core/broke.py", "def (:\n")
        self.assertNotIn("broke", audit.collect_core_module_names())
        # exported-names of an unparseable module → empty set.
        self.assertEqual(audit._module_exported_names(self.bad), set())


# ═════════════════════════════════════════════════════════════════════════
#  CHECK 7 — secrets
# ═════════════════════════════════════════════════════════════════════════

class SecretCheckTests(_AuditTestBase):
    def test_anthropic_key_pattern_p0(self):
        # Build the key shape at runtime so no literal pattern is in this file.
        key = "sk-" + "ant-" + ("A" * 25)
        fp = self.write("mod.py", "KEY = '" + key + "'\n")
        out = audit.check_secrets([fp])
        p0 = [f for f in out if f.severity == "P0"]
        self.assertTrue(p0)
        self.assertIn("Anthropic API key", p0[0].message)

    def test_aws_access_key_pattern_p0(self):
        key = "AKIA" + ("Q" * 16)
        fp = self.write("mod.py", "AWS = '" + key + "'\n")
        out = audit.check_secrets([fp])
        self.assertTrue(any(f.severity == "P0" and "AWS" in f.message
                            for f in out))

    def test_password_literal_p1(self):
        val = "hunter2hunter2"  # 14 chars, not all-caps, not placeholder
        fp = self.write("mod.py", "password = '" + val + "'\n")
        out = audit.check_secrets([fp])
        p1 = [f for f in out if f.severity == "P1"]
        self.assertTrue(p1)
        self.assertIn("password", p1[0].message)

    def test_password_placeholder_whitelisted(self):
        fp = self.write("mod.py", "password = 'your-password-here'\n")
        self.assertEqual(audit.check_secrets([fp]), [])

    def test_password_env_read_whitelisted(self):
        fp = self.write("mod.py",
                        "token = os.environ['SOME_TOKEN_PLACEHOLDER_VALUE']\n")
        # 'environ' / 'os.environ' substrings whitelist the line.
        self.assertEqual([f for f in audit.check_secrets([fp])
                          if f.severity == "P1"], [])

    def test_constant_reference_value_whitelisted(self):
        # All-caps value → looks like a constant name reference, not a literal.
        fp = self.write("mod.py", "secret = 'SOME_CONSTANT_NAME'\n")
        self.assertEqual([f for f in audit.check_secrets([fp])
                          if f.severity == "P1"], [])

    def test_comment_line_ignored(self):
        key = "sk-" + "ant-" + ("B" * 25)
        fp = self.write("mod.py", "# example: KEY = '" + key + "'\n")
        self.assertEqual(audit.check_secrets([fp]), [])


# ═════════════════════════════════════════════════════════════════════════
#  CHECK 10/11 — state-file writes
# ═════════════════════════════════════════════════════════════════════════

class StateFileWriteTests(_AuditTestBase):
    def test_direct_hud_state_write_flagged_p1(self):
        src = ("import json\n"
               "with open('hud_state.json', 'w') as f:\n"
               "    json.dump({}, f)\n")
        fp = self.write("skills/s.py", src)
        self.mkdir("skills")
        out = audit.check_state_file_writes([fp])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].severity, "P1")
        self.assertEqual(out[0].category, "state-write")
        self.assertIn("_write_hud_state", out[0].message)

    def test_pending_speech_write_flagged_p1(self):
        src = ("import json\n"
               "with open('pending_speech.json', 'w') as f:\n"
               "    json.dump([], f)\n")
        fp = self.write("skills/s.py", src)
        out = audit.check_state_file_writes([fp])
        self.assertEqual(len(out), 1)
        self.assertIn("_enqueue_speech", out[0].message)

    def test_other_state_file_write_p2(self):
        src = ("import json\n"
               "with open('credits_state.json', 'w') as f:\n"
               "    json.dump({}, f)\n")
        fp = self.write("skills/s.py", src)
        out = audit.check_state_file_writes([fp])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].severity, "P2")

    def test_bobert_companion_hud_write_exempted(self):
        # bobert_companion.py owns the canonical _write_hud_state helper.
        src = ("import json\n"
               "with open('hud_state.json', 'w') as f:\n"
               "    json.dump({}, f)\n")
        fp = self.write("bobert_companion.py", src)
        self.assertEqual(audit.check_state_file_writes([fp]), [])

    def test_read_mode_open_not_flagged(self):
        src = ("import json\n"
               "with open('hud_state.json', 'r') as f:\n"
               "    json.load(f)\n")
        fp = self.write("skills/s.py", src)
        self.assertEqual(audit.check_state_file_writes([fp]), [])

    def test_non_state_file_write_not_flagged(self):
        src = ("import json\n"
               "with open('whatever.json', 'w') as f:\n"
               "    json.dump({}, f)\n")
        fp = self.write("skills/s.py", src)
        self.assertEqual(audit.check_state_file_writes([fp]), [])

    def test_various_non_matching_open_forms_skipped(self):
        # Each `with open(...)` here fails a different early-continue guard:
        #  - unresolvable Name path  - read mode  - state file but no json.dump.
        src = ("import json\n"
               "with open(somevar, 'w') as f:\n"
               "    json.dump({}, f)\n"
               "with open('hud_state.json', 'r') as f:\n"
               "    pass\n"
               "with open('hud_state.json', 'w') as f:\n"
               "    f.write('x')\n")
        fp = self.write("skills/s.py", src)
        self.assertEqual(audit.check_state_file_writes([fp]), [])

    def test_resolve_open_target_heuristics(self):
        self.assertEqual(
            audit._resolve_open_target(ast.parse("'hud_state.json'").body[0].value),
            "hud_state.json")
        self.assertEqual(
            audit._resolve_open_target(ast.parse("_SPEECH_QUEUE_FILE").body[0].value),
            "pending_speech.json")
        self.assertEqual(
            audit._resolve_open_target(ast.parse("_HUD_STATE_PATH").body[0].value),
            "hud_state.json")
        self.assertEqual(
            audit._resolve_open_target(
                ast.parse("os.path.join(D, 'pending_speech.json')").body[0].value),
            "pending_speech.json")
        self.assertIsNone(
            audit._resolve_open_target(ast.parse("random_var").body[0].value))


# ═════════════════════════════════════════════════════════════════════════
#  CHECK 13 — readback edge cases
# ═════════════════════════════════════════════════════════════════════════

class ReadbackTests(_AuditTestBase):
    def test_corrupt_json_flagged(self):
        self.write("bobert_memory.json", "{ not valid json")
        out = audit.check_readback()
        self.assertTrue(any(f.category == "state-corruption"
                            and "bobert_memory.json" in f.file for f in out))

    def test_valid_json_clean(self):
        self.write("hud_state.json", json.dumps({"ok": True}))
        self.write("bobert_memory.json", json.dumps({"a": 1}))
        out = audit.check_readback()
        self.assertEqual([f for f in out if f.file in
                          ("hud_state.json", "bobert_memory.json")], [])

    def test_todo_without_task_lines_p2(self):
        self.write("jarvis_todo.md", "Just a heading\nno checkbox lines\n")
        out = audit.check_readback()
        todo = [f for f in out if f.file == "jarvis_todo.md"]
        self.assertEqual(len(todo), 1)
        self.assertEqual(todo[0].severity, "P2")

    def test_todo_with_task_lines_clean(self):
        self.write("jarvis_todo.md", "# Tasks\n- [ ] do thing\n- [x] done\n")
        out = audit.check_readback()
        self.assertEqual([f for f in out if f.file == "jarvis_todo.md"], [])

    def test_absent_files_clean(self):
        # No state files written → nothing to read back.
        self.assertEqual(audit.check_readback(), [])


# ═════════════════════════════════════════════════════════════════════════
#  CHECK 14 — mutation hygiene
# ═════════════════════════════════════════════════════════════════════════

class MutationHygieneTests(_AuditTestBase):
    def test_unlocked_global_mutation_in_thread_target_flagged(self):
        src = (
            "import threading\n"
            "CACHE = {}\n"
            "def worker():\n"
            "    CACHE.clear()\n"
            "    CACHE = {}\n"
            "t = threading.Thread(target=worker)\n"
        )
        fp = self.write("mod.py", src)
        out = audit.check_mutation_hygiene([fp])
        self.assertTrue(out)
        self.assertTrue(all(f.category == "mutation-hygiene" for f in out))
        self.assertTrue(all(f.severity == "P2" for f in out))

    def test_locked_mutation_not_flagged(self):
        src = (
            "import threading\n"
            "CACHE = {}\n"
            "_lock = threading.Lock()\n"
            "def worker():\n"
            "    with _lock:\n"
            "        CACHE.clear()\n"
            "        CACHE['k'] = 1\n"
            "t = threading.Thread(target=worker)\n"
        )
        fp = self.write("mod.py", src)
        self.assertEqual(audit.check_mutation_hygiene([fp]), [])

    def test_non_thread_function_not_flagged(self):
        # Same mutation but the function is never a thread target.
        src = (
            "CACHE = {}\n"
            "def helper():\n"
            "    CACHE.clear()\n"
        )
        fp = self.write("mod.py", src)
        self.assertEqual(audit.check_mutation_hygiene([fp]), [])

    def test_find_unlocked_mutations_subscript_is_lockfree(self):
        # `d[k] = v` is the intentional GIL-atomic lock-free pattern → NOT
        # flagged; only whole-name rebind + mutating method calls are.
        fn = ast.parse(
            "def w():\n"
            "    CACHE['k'] = 1\n"   # subscript → ignored
            "    CACHE = {}\n"       # rebind → flagged
            "    BUF.append(2)\n"    # method → flagged
        ).body[0]
        muts = audit._find_unlocked_mutations(fn, {"CACHE", "BUF"})
        flagged_names = {n for _ln, n in muts}
        self.assertEqual(flagged_names, {"CACHE", "BUF"})
        # The subscript line (2) is not among the flagged lines.
        self.assertNotIn(2, {ln for ln, _n in muts})

    def test_find_unlocked_mutations_augassign_flagged(self):
        fn = ast.parse("def w():\n    CNT += 1\n").body[0]
        muts = audit._find_unlocked_mutations(fn, {"CNT"})
        self.assertEqual({n for _l, n in muts}, {"CNT"})

    def test_find_unlocked_mutations_acquire_context_is_guard(self):
        # `with lk.acquire():` counts as a lock guard → AugAssign inside is safe.
        fn = ast.parse(
            "def w():\n"
            "    with lk.acquire():\n"
            "        CNT += 1\n"
        ).body[0]
        self.assertEqual(audit._find_unlocked_mutations(fn, {"CNT"}), [])

    def test_find_unlocked_mutations_attribute_lock_is_guard(self):
        # `with self._state_lock:` (Attribute whose name contains 'lock').
        fn = ast.parse(
            "def w():\n"
            "    with self._state_lock:\n"
            "        BUF.append(1)\n"
        ).body[0]
        self.assertEqual(audit._find_unlocked_mutations(fn, {"BUF"}), [])


# ═════════════════════════════════════════════════════════════════════════
#  --fix transforms
# ═════════════════════════════════════════════════════════════════════════

class FixTransformTests(_AuditTestBase):
    def test_encoding_fix(self):
        self.assertEqual(audit._apply_fix_to_line("f = open(p)", "encoding"),
                         ['f = open(p, encoding="utf-8")'])

    def test_encoding_fix_already_present_returns_none(self):
        self.assertIsNone(
            audit._apply_fix_to_line('f = open(p, encoding="utf-8")', "encoding"))

    def test_encoding_fix_binary_mode_skipped(self):
        self.assertIsNone(audit._apply_fix_to_line('f = open(p, "rb")', "encoding"))

    def test_daemon_fix_thread(self):
        self.assertEqual(
            audit._apply_fix_to_line("t = threading.Thread(target=x)", "daemon"),
            ["t = threading.Thread(target=x, daemon=True)"])

    def test_daemon_fix_timer_uses_post_construction(self):
        # Timer.__init__ rejects daemon= kwarg → must use the `.daemon = True`
        # post-construction idiom, only when bound to `<var> = Timer(...)`.
        out = audit._apply_fix_to_line("    t = threading.Timer(5, x)", "daemon")
        self.assertEqual(out, ["    t = threading.Timer(5, x)",
                               "    t.daemon = True"])

    def test_daemon_fix_timer_without_assignment_returns_none(self):
        self.assertIsNone(
            audit._apply_fix_to_line("threading.Timer(5, x).start()", "daemon"))

    def test_timeout_fix(self):
        self.assertEqual(
            audit._apply_fix_to_line("subprocess.run(['x'])", "timeout"),
            ["subprocess.run(['x'], timeout=60)"])

    def test_daemon_fix_no_thread_on_line_returns_none(self):
        # fix_kind=daemon but the line has no Thread/Timer call → no match → None.
        self.assertIsNone(audit._apply_fix_to_line("x = 1", "daemon"))

    def test_daemon_fix_already_has_daemon_returns_none(self):
        self.assertIsNone(
            audit._apply_fix_to_line("t = threading.Thread(target=x, daemon=True)",
                                     "daemon"))

    def test_timeout_fix_no_subprocess_on_line_returns_none(self):
        self.assertIsNone(audit._apply_fix_to_line("x = 1", "timeout"))

    def test_timeout_fix_already_has_timeout_returns_none(self):
        self.assertIsNone(
            audit._apply_fix_to_line("subprocess.run(['x'], timeout=5)", "timeout"))

    def test_unknown_fix_kind_returns_none(self):
        self.assertIsNone(audit._apply_fix_to_line("x = 1", "bogus"))


class ApplyFixesTests(_AuditTestBase):
    def test_apply_fixes_rewrites_file(self):
        self.write("mod.py", "f = open('a.txt')\n")
        finding = audit.Finding(severity="P2", category="open-no-encoding",
                                file="mod.py", line=1, message="x",
                                fixable=True, fix_kind="encoding")
        applied, residual = audit.apply_fixes([finding])
        self.assertEqual(applied, 1)
        self.assertEqual(residual, [])
        new = self.read_text("mod.py")
        self.assertIn('encoding="utf-8"', new)

    def test_apply_fixes_out_of_range_line_becomes_residual(self):
        self.write("mod.py", "x = 1\n")
        finding = audit.Finding(severity="P2", category="open-no-encoding",
                                file="mod.py", line=999, message="x",
                                fixable=True, fix_kind="encoding")
        applied, residual = audit.apply_fixes([finding])
        self.assertEqual(applied, 0)
        self.assertEqual(len(residual), 1)

    def test_apply_fixes_unfixable_line_becomes_residual(self):
        # Line exists but the transform can't match → residual, not applied.
        self.write("mod.py", "y = 1\n")
        finding = audit.Finding(severity="P2", category="open-no-encoding",
                                file="mod.py", line=1, message="x",
                                fixable=True, fix_kind="encoding")
        applied, residual = audit.apply_fixes([finding])
        self.assertEqual(applied, 0)
        self.assertEqual(len(residual), 1)

    def test_apply_fixes_unreadable_file_skipped(self):
        # The finding points at a file that can't be read → that file is skipped
        # (the `_read_source is None` continue) with nothing applied.
        finding = audit.Finding(severity="P2", category="open-no-encoding",
                                file="ghost_does_not_exist.py", line=1,
                                message="x", fixable=True, fix_kind="encoding")
        applied, residual = audit.apply_fixes([finding])
        self.assertEqual(applied, 0)

    def test_apply_fixes_write_failure_swallowed(self):
        # The fix matches and modifies the lines, but the write-back raises;
        # apply_fixes must swallow it (printing to stderr) and still count it.
        # _read_source reads in mode "r"; only the write-back open(mode "w")
        # raises, so the read succeeds and we reach the write except.
        self.write("mod.py", "f = open('a.txt')\n")
        finding = audit.Finding(severity="P2", category="open-no-encoding",
                                file="mod.py", line=1, message="x",
                                fixable=True, fix_kind="encoding")
        real_open = open

        def open_fail_on_write(path, *a, **k):
            mode = a[0] if a else k.get("mode", "r")
            if "w" in mode:
                raise OSError("read-only")
            return real_open(path, *a, **k)

        with mock.patch("builtins.open", side_effect=open_fail_on_write):
            applied, _residual = audit.apply_fixes([finding])
        # the in-memory transform succeeded (applied counted) even though the
        # write failed; no exception escaped.
        self.assertEqual(applied, 1)


# ═════════════════════════════════════════════════════════════════════════
#  _safe wrapper
# ═════════════════════════════════════════════════════════════════════════

class SafeWrapperTests(_AuditTestBase):
    def test_safe_converts_exception_to_p2(self):
        def boom(_x):
            raise ValueError("kaboom")
        findings, data = audit._safe(boom)(1)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].category, "integration-check-error")
        self.assertEqual(findings[0].severity, "P2")
        self.assertIsNone(data)

    def test_safe_passes_through_tuple_result(self):
        def fine(_x):
            return [audit.Finding(severity="P2", category="c", file="f",
                                  line=0, message="m")], {"k": 1}
        findings, data = audit._safe(fine)(1)
        self.assertEqual(data, {"k": 1})
        self.assertEqual(len(findings), 1)

    def test_safe_wraps_non_tuple_result(self):
        def single(_x):
            return [audit.Finding(severity="P0", category="z", file="f",
                                  line=0, message="m")]
        result = audit._safe(single)(1)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result[0]), 1)
        self.assertIsNone(result[1])


# ═════════════════════════════════════════════════════════════════════════
#  _profile_skill (drives checks B & C)
# ═════════════════════════════════════════════════════════════════════════

class ProfileSkillTests(_AuditTestBase):
    def setUp(self):
        super().setUp()
        self.mkdir("skills")

    def test_direct_action_registration_and_threads_and_locks(self):
        src = (
            "import threading\n"
            "_LOCK = threading.Lock()\n"
            "def h(p):\n    return ''\n"
            "def _loop():\n    while True:\n        x = 1\n"
            "def register(actions):\n"
            "    actions['a'] = h\n"
            "    t = threading.Thread(target=_loop)\n"
        )
        fp = self.write("skills/s.py", src)
        prof = audit._profile_skill(fp)
        self.assertIn("a", prof["actions"])
        self.assertEqual(len(prof["thread_lines"]), 1)
        self.assertIn("_LOCK", prof["lock_names"])
        self.assertIn("_loop", prof["thread_targets"])

    def test_dict_update_registration_form(self):
        # `handlers = {...}; actions.update(handlers)` must be detected.
        # NOTE: the dict literal is discovered by walking the register() body
        # only, so it must be DEFINED INSIDE register() (a module-level
        # HANDLERS dict is invisible to this heuristic).
        src = (
            "def h1(p):\n    return ''\n"
            "def h2(p):\n    return ''\n"
            "def register(actions):\n"
            "    handlers = {'a': h1, 'b': h2}\n"
            "    actions.update(handlers)\n"
        )
        fp = self.write("skills/s.py", src)
        prof = audit._profile_skill(fp)
        self.assertEqual(set(prof["actions"]), {"a", "b"})

    def test_update_inline_dict_literal_form(self):
        src = (
            "def h1(p):\n    return ''\n"
            "def register(actions):\n"
            "    actions.update({'a': h1})\n"
        )
        fp = self.write("skills/s.py", src)
        prof = audit._profile_skill(fp)
        self.assertIn("a", prof["actions"])

    def test_for_loop_items_registration_form(self):
        # As with update(), the source dict must live inside register().
        src = (
            "def h1(p):\n    return ''\n"
            "def register(actions):\n"
            "    mp = {'x': h1}\n"
            "    for k, v in mp.items():\n"
            "        actions[k] = v\n"
        )
        fp = self.write("skills/s.py", src)
        prof = audit._profile_skill(fp)
        self.assertIn("x", prof["actions"])

    def test_state_write_detection(self):
        src = (
            "import json\n"
            "def w():\n"
            "    with open('hud_state.json', 'w') as f:\n"
            "        json.dump({}, f)\n"
            "def register(actions):\n    pass\n"
        )
        fp = self.write("skills/s.py", src)
        prof = audit._profile_skill(fp)
        self.assertIn("hud_state.json", prof["writes_state"])

    def test_atomic_writer_detection(self):
        src = (
            "from core.atomic_io import _atomic_write_json\n"
            "def register(actions):\n    pass\n"
        )
        fp = self.write("skills/s.py", src)
        prof = audit._profile_skill(fp)
        self.assertTrue(prof["uses_atomic_writer"])

    def test_profile_cache_returns_same_object(self):
        fp = self.write("skills/s.py", "def register(actions):\n    pass\n")
        a = audit._profile_skill(fp)
        b = audit._profile_skill(fp)
        self.assertIs(a, b)


# ═════════════════════════════════════════════════════════════════════════
#  CHECK A — action smoke tests (static-fallback path)
# ═════════════════════════════════════════════════════════════════════════

class SmokeTestChecksTests(_AuditTestBase):
    def setUp(self):
        super().setUp()
        self.mkdir("skills")

    def test_static_fallback_when_bc_import_fails(self):
        # Force the optional bobert_companion import to fail so we exercise the
        # deterministic static-only path (this is also the CI reality).
        src = ("def h(p):\n    return ''\n"
               "def register(actions):\n"
               "    actions['myact'] = h\n"
               "    actions['upgrade'] = h\n")  # 'upgrade' ∈ _NO_SMOKE_TEST
        fp = self.write("skills/s.py", src)
        with mock.patch.object(audit.importlib, "import_module",
                               side_effect=ImportError("forced")):
            findings, results = audit.check_action_smoke_tests([fp])
        self.assertEqual(results.get("myact"), "static-only")
        self.assertEqual(results.get("upgrade"), "skipped (no-smoke-test)")
        # The only finding is the advisory that bc couldn't import.
        self.assertTrue(any(f.category == "smoke-test"
                            and "could not import bobert_companion" in f.message
                            for f in findings))

    def test_live_dispatch_ok_badreturn_and_raise(self):
        # Drive the live-dispatch branch by injecting a FAKE bobert_companion
        # whose ACTIONS map is fully controllable — deterministic, no real
        # monolith. Covers: ok / bad-return(P1) / raised-target-exc(P1).
        import types
        skill = (
            "def good(p):\n    return 'ok'\n"
            "def badret(p):\n    return 0\n"
            "def boom(p):\n    raise KeyError('x')\n"
            "def register(actions):\n"
            "    actions['good'] = good\n"
            "    actions['badret'] = badret\n"
            "    actions['boom'] = boom\n")
        fp = self.write("skills/s.py", skill)
        fake = types.ModuleType("bobert_companion")
        fake.ACTIONS = {
            "good": lambda payload: "fine",
            "badret": lambda payload: 999,             # non-str → P1
            "boom": lambda payload: (_ for _ in ()).throw(KeyError("boom")),
        }
        with mock.patch.object(audit.importlib, "import_module",
                               return_value=fake):
            findings, results = audit.check_action_smoke_tests([fp])
        self.assertEqual(results["good"], "ok")
        self.assertEqual(results["badret"], "bad-return: int")
        self.assertTrue(results["boom"].startswith("error: KeyError"))
        cats = {(f.severity, f.category) for f in findings}
        self.assertIn(("P1", "smoke-test"), cats)
        self.assertTrue(any("instead of str" in f.message for f in findings))
        self.assertTrue(any("raised KeyError" in f.message for f in findings))

    def test_live_dispatch_unexpected_exception_is_p2(self):
        # A non-target exception class (e.g. RuntimeError) → P2, not P1.
        import types
        skill = ("def h(p):\n    return ''\n"
                 "def register(actions):\n    actions['weird'] = h\n")
        fp = self.write("skills/s.py", skill)
        fake = types.ModuleType("bobert_companion")
        fake.ACTIONS = {
            "weird": lambda payload: (_ for _ in ()).throw(RuntimeError("boom")),
        }
        with mock.patch.object(audit.importlib, "import_module",
                               return_value=fake):
            findings, results = audit.check_action_smoke_tests([fp])
        self.assertTrue(results["weird"].startswith("error: RuntimeError"))
        self.assertTrue(any(f.severity == "P2" and "unexpected" in f.message
                            for f in findings))

    def test_live_dispatch_not_registered_and_no_actions_dict(self):
        import types
        skill = ("def h(p):\n    return ''\n"
                 "def register(actions):\n    actions['ghost'] = h\n")
        fp = self.write("skills/s.py", skill)
        # (a) ACTIONS present but missing this action → "not registered".
        fake = types.ModuleType("bobert_companion")
        fake.ACTIONS = {}
        with mock.patch.object(audit.importlib, "import_module",
                               return_value=fake):
            _f, results = audit.check_action_smoke_tests([fp])
        self.assertEqual(results["ghost"], "skipped (not registered at runtime)")
        # (b) no ACTIONS dict attribute at all → "no ACTIONS dict".
        audit._skill_profile_cache.clear()
        fake2 = types.ModuleType("bobert_companion")  # no ACTIONS attr
        with mock.patch.object(audit.importlib, "import_module",
                               return_value=fake2):
            _f2, results2 = audit.check_action_smoke_tests([fp])
        self.assertEqual(results2["ghost"], "skipped (no ACTIONS dict)")

    def test_install_and_restore_stubs(self):
        saved = audit._install_smoke_stubs()
        try:
            self.assertIn("cv2", sys.modules)
            self.assertIsInstance(sys.modules["cv2"], audit._StubModule)
            # webbrowser stub is callable & truthy-returning.
            self.assertTrue(sys.modules["webbrowser"].open("http://x"))
        finally:
            audit._restore_smoke_stubs(saved)

    def test_stub_callable_behaviours(self):
        sc = audit._StubCallable("cv2.VideoCapture")
        # chained call + attribute access never explode
        self.assertIsInstance(sc(0), audit._StubCallable)
        self.assertIsInstance(sc.isOpened, audit._StubCallable)
        self.assertFalse(bool(sc))             # short-circuits truthiness
        self.assertEqual(len(sc), 0)
        self.assertEqual(int(sc), 0)
        self.assertEqual(str(sc), "")
        self.assertEqual(sc.read(), (False, None))
        self.assertEqual(list(iter(sc)), [])
        # string concat returns the str operand
        self.assertEqual(sc + "tail", "tail")
        self.assertEqual("head" + sc, "head")
        with sc as ctx:
            self.assertIs(ctx, sc)

    def test_stub_module_dunder_raises(self):
        sm = audit._StubModule("cv2")
        with self.assertRaises(AttributeError):
            sm.__wrapped__  # dunder lookups must not be stubbed

    def test_stub_callable_arithmetic_and_comparison_dunders(self):
        # The stub overloads a wide arithmetic/comparison surface so skill code
        # doing maths on a stubbed return value never explodes. Exercise them.
        sc = audit._StubCallable("m.x")
        for result in (sc - 1, 1 - sc, sc * 2, 2 * sc, sc / 2, 2 / sc,
                       sc // 2, sc % 2):
            self.assertIsInstance(result, audit._StubCallable)
        # numeric add with non-str returns a stub; index/float coercions.
        self.assertIsInstance(sc + 1, audit._StubCallable)
        self.assertIsInstance(1 + sc, audit._StubCallable)
        self.assertEqual(sc.__index__(), 0)
        self.assertEqual(float(sc), 0.0)
        self.assertEqual(repr(sc), "<_StubCallable m.x>")
        # ordering comparisons are all False; equality keys on the name.
        self.assertFalse(sc < sc)
        self.assertFalse(sc <= sc)
        self.assertFalse(sc > sc)
        self.assertFalse(sc >= sc)
        self.assertTrue(sc == audit._StubCallable("m.x"))
        self.assertTrue(sc != audit._StubCallable("other"))
        self.assertEqual(hash(sc), hash("m.x"))
        self.assertFalse("anything" in sc)
        self.assertIsInstance(sc["key"], audit._StubCallable)


# ═════════════════════════════════════════════════════════════════════════
#  CHECK B — skill-pair conflicts
# ═════════════════════════════════════════════════════════════════════════

class SkillPairConflictTests(_AuditTestBase):
    def setUp(self):
        super().setUp()
        self.mkdir("skills")

    def test_shared_state_write_and_action_collision(self):
        body = ("import json\n"
                "def h(p):\n    return ''\n"
                "def w():\n"
                "    with open('hud_state.json', 'w') as f:\n"
                "        json.dump({}, f)\n"
                "def register(actions):\n    actions['shared'] = h\n")
        fa = self.write("skills/aa.py", body)
        fb = self.write("skills/bb.py", body)
        findings, rows = audit.check_skill_pair_conflicts([fa, fb])
        msgs = " ".join(f.message for f in findings)
        self.assertIn("hud_state.json", msgs)
        self.assertIn("shared", msgs)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["shared_state_files"], ["hud_state.json"])
        self.assertEqual(rows[0]["shared_actions"], ["shared"])
        # The conflict-matrix markdown is written to PROJECT_DIR.
        self.assertTrue(os.path.exists(
            os.path.join(self.tmp, "audit_conflict_matrix.md")))

    def test_both_atomic_suppresses_state_finding(self):
        # Both sides import the canonical atomic writer → state co-write is safe;
        # only action collisions remain.
        body = ("import json\n"
                "from core.atomic_io import _atomic_write_json\n"
                "def h(p):\n    return ''\n"
                "def w():\n"
                "    with open('hud_state.json', 'w') as f:\n"
                "        json.dump({}, f)\n"
                "def register(actions):\n    actions['x'] = h\n")
        fa = self.write("skills/aa.py", body)
        fb = self.write("skills/bb.py", body)
        findings, rows = audit.check_skill_pair_conflicts([fa, fb])
        # No state-file conflict finding...
        self.assertFalse(any("hud_state.json" in f.message for f in findings))
        # ...but the action collision still fires.
        self.assertTrue(any("'x'" in f.message for f in findings))
        self.assertTrue(rows[0]["both_canonical"])

    def test_no_overlap_no_findings_but_matrix_written(self):
        fa = self.write("skills/aa.py",
                        "def h(p):\n    return ''\n"
                        "def register(actions):\n    actions['a'] = h\n")
        fb = self.write("skills/bb.py",
                        "def h(p):\n    return ''\n"
                        "def register(actions):\n    actions['b'] = h\n")
        findings, rows = audit.check_skill_pair_conflicts([fa, fb])
        self.assertEqual(findings, [])
        self.assertEqual(rows, [])
        self.assertTrue(os.path.exists(
            os.path.join(self.tmp, "audit_conflict_matrix.md")))


# ═════════════════════════════════════════════════════════════════════════
#  CHECK C — background-thread audit
# ═════════════════════════════════════════════════════════════════════════

class ThreadAuditTests(_AuditTestBase):
    def test_tight_spin_and_no_try_flagged(self):
        src = ("import threading\n"
               "def spin():\n"
               "    while True:\n"
               "        x = 1\n"
               "t = threading.Thread(target=spin)\n")
        fp = self.write("mod.py", src)
        findings, summary = audit.check_thread_audit([fp])
        cats = self.cats(findings)
        self.assertIn("thread-tight-spin", cats)
        self.assertIn("thread-no-try", cats)
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["target"], "spin")

    def test_paced_loop_with_try_is_clean(self):
        src = ("import threading, time\n"
               "def worker():\n"
               "    try:\n"
               "        while True:\n"
               "            time.sleep(1)\n"
               "    except Exception:\n"
               "        pass\n"
               "t = threading.Thread(target=worker)\n")
        fp = self.write("mod.py", src)
        findings, _ = audit.check_thread_audit([fp])
        self.assertEqual(findings, [])

    def test_blocking_wait_counts_as_paced(self):
        # `_stop.wait(0.5)` paces the loop even though it's not time.sleep.
        src = ("import threading\n"
               "def worker():\n"
               "    try:\n"
               "        while True:\n"
               "            _stop.wait(0.5)\n"
               "    except Exception:\n"
               "        pass\n"
               "t = threading.Thread(target=worker)\n")
        fp = self.write("mod.py", src)
        findings, _ = audit.check_thread_audit([fp])
        self.assertEqual(self.by_cat(findings, "thread-tight-spin"), [])

    def test_drain_loop_not_flagged_as_spin(self):
        # `while q: q.popleft()` terminates — not a CPU peg.
        src = ("import threading\n"
               "def worker():\n"
               "    try:\n"
               "        while q:\n"
               "            q.popleft()\n"
               "    except Exception:\n"
               "        pass\n"
               "t = threading.Thread(target=worker)\n")
        fp = self.write("mod.py", src)
        findings, _ = audit.check_thread_audit([fp])
        self.assertEqual(self.by_cat(findings, "thread-tight-spin"), [])

    def test_delegation_only_loop_treated_resilient(self):
        # Loop body is pure delegation (bare calls) → no try required.
        src = ("import threading\n"
               "def worker():\n"
               "    while True:\n"
               "        do_step()\n"
               "        time.sleep(1)\n"
               "t = threading.Thread(target=worker)\n")
        fp = self.write("mod.py", src)
        findings, _ = audit.check_thread_audit([fp])
        self.assertEqual(self.by_cat(findings, "thread-no-try"), [])

    def test_unknown_target_skipped(self):
        # target not defined in this module → only a summary row, no findings.
        src = ("import threading\n"
               "t = threading.Thread(target=external_fn)\n")
        fp = self.write("mod.py", src)
        findings, summary = audit.check_thread_audit([fp])
        self.assertEqual(findings, [])
        self.assertEqual(len(summary), 1)


# ═════════════════════════════════════════════════════════════════════════
#  CHECK D — prompt ↔ action consistency (+ _extract_prompt_actions)
# ═════════════════════════════════════════════════════════════════════════

class PromptExtractionTests(_AuditTestBase):
    def test_extract_table_list_and_citation_forms(self):
        prompt = _make_prompt(
            "  screenshot                   " + _EMDASH + " save it",      # table no-arg
            "  open_url, <url>   " + _EMDASH + " open a website",          # table w/arg
            "    volume_up, volume_down, volume_mute (system-wide)",        # narrative list
            "    Media keys: media_next, media_prev, media_playpause",      # labelled list
            "  do thing then [ACTION: cited_action]",                       # citation
        )
        got = audit._extract_prompt_actions(prompt)
        self.assertEqual(
            got,
            {"screenshot", "open_url", "volume_up", "volume_down",
             "volume_mute", "media_next", "media_prev", "media_playpause",
             "cited_action"})

    def test_extract_returns_empty_without_prompt_block(self):
        self.assertEqual(audit._extract_prompt_actions("x = 1\n"), set())

    def test_narrative_continuation_text_not_misparsed(self):
        # A deeply-indented English description line must NOT yield tokens.
        prompt = _make_prompt(
            "          this is just descriptive prose about volume and stuff")
        self.assertEqual(audit._extract_prompt_actions(prompt), set())


class PromptConsistencyTests(_AuditTestBase):
    def test_missing_and_undocumented(self):
        findings, summary = audit.check_prompt_action_consistency(
            prompt_actions={"documented_only", "shared"},
            registered_actions={"shared", "registered_only"})
        # documented but unregistered → P1
        missing = [f for f in findings if f.severity == "P1"]
        self.assertTrue(any("documented_only" in f.message for f in missing))
        # registered but undocumented → P2
        undoc = [f for f in findings if f.severity == "P2"]
        self.assertTrue(any("registered_only" in f.message for f in undoc))
        self.assertEqual(summary["missing_from_registry"], ["documented_only"])
        self.assertEqual(summary["missing_from_prompt"], ["registered_only"])

    def test_internal_only_actions_suppressed(self):
        # An action in _INTERNAL_ONLY_ACTIONS that's registered-not-documented
        # must NOT be reported as undocumented.
        internal = next(iter(audit._INTERNAL_ONLY_ACTIONS))
        findings, summary = audit.check_prompt_action_consistency(
            prompt_actions=set(),
            registered_actions={internal})
        self.assertEqual(findings, [])
        self.assertEqual(summary["missing_from_prompt"], [])

    def test_fully_consistent_no_findings(self):
        findings, summary = audit.check_prompt_action_consistency(
            prompt_actions={"a", "b"}, registered_actions={"a", "b"})
        self.assertEqual(findings, [])


# ═════════════════════════════════════════════════════════════════════════
#  CHECK E — voice-trigger coverage
# ═════════════════════════════════════════════════════════════════════════

class VoiceCoverageTests(_AuditTestBase):
    def test_present_trigger_phrase_ok_missing_flagged(self):
        prompt = _make_prompt("  hide the HUD overlay now")
        findings, coverage = audit.check_voice_trigger_coverage(
            prompt, registered_actions={"hide_hud", "show_hud"})
        self.assertEqual(coverage["hide_hud"], "ok")
        self.assertEqual(coverage["show_hud"], "MISSING")
        self.assertTrue(any("show_hud" in f.message and f.severity == "P2"
                            for f in findings))

    def test_unregistered_action_skipped(self):
        findings, coverage = audit.check_voice_trigger_coverage(
            _make_prompt("anything"), registered_actions=set())
        # Every example action is reported as skipped (not registered).
        self.assertTrue(all(v.startswith("skipped")
                            for v in coverage.values()))
        self.assertEqual(findings, [])


# ═════════════════════════════════════════════════════════════════════════
#  CHECK F — TTS pipeline
# ═════════════════════════════════════════════════════════════════════════

class TtsPipelineTests(_AuditTestBase):
    def test_clean_presets_with_neutral_reference(self):
        tts = ("_TTS_EMOTION_PRESETS = {\n"
               "    'neutral': {},\n"
               "    'excited': {},\n"
               "}\n")
        bc = "v = _TTS_EMOTION_PRESETS['neutral']\nw = 'excited'\n"
        findings, coverage = audit.check_tts_pipeline(bc, tts)
        self.assertEqual(findings, [])
        self.assertEqual(coverage.get("neutral"), "ok")
        self.assertEqual(coverage.get("excited"), "ok")

    def test_missing_dict_literal_p1(self):
        findings, _ = audit.check_tts_pipeline("", "")
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].severity, "P1")
        self.assertIn("dict literal not found", findings[0].message)

    def test_no_neutral_preset_p0(self):
        tts = "_TTS_EMOTION_PRESETS = {\n    'happy': {},\n}\n"
        findings, _ = audit.check_tts_pipeline("", tts)
        self.assertTrue(any(f.severity == "P0" and "neutral" in f.message
                            for f in findings))

    def test_neutral_present_but_unreferenced_p1(self):
        # 'neutral' defined but resolver doesn't subscript it → P1 fallback note.
        tts = "_TTS_EMOTION_PRESETS = {\n    'neutral': {},\n}\n"
        findings, _ = audit.check_tts_pipeline("", tts)
        self.assertTrue(any(f.severity == "P1"
                            and "fallback chain unclear" in f.message
                            for f in findings))


# ═════════════════════════════════════════════════════════════════════════
#  CHECK G — crash recovery
# ═════════════════════════════════════════════════════════════════════════

class CrashRecoveryTests(_AuditTestBase):
    def test_outer_try_ok_missing_try_flagged(self):
        bc = ("def main():\n"
              "    try:\n        pass\n    except Exception:\n        pass\n"
              "def _speak(x):\n    return x\n")
        findings, status = audit.check_crash_recovery(bc)
        self.assertEqual(status["main"], "ok (outer try)")
        self.assertEqual(status["_speak"], "NO TRY/EXCEPT")
        self.assertTrue(any("_speak" in f.message and f.severity == "P1"
                            for f in findings))

    def test_loop_body_try_accepted(self):
        bc = ("def main():\n"
              "    while True:\n"
              "        try:\n            pass\n        except Exception:\n"
              "            pass\n")
        findings, status = audit.check_crash_recovery(bc)
        self.assertEqual(status["main"], "ok (loop body try)")
        self.assertEqual([f for f in findings if "main" in f.message], [])

    def test_unparseable_source_p0(self):
        findings, _ = audit.check_crash_recovery("def (:\n")
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].severity, "P0")
        self.assertIn("cannot parse", findings[0].message)

    def test_missing_critical_function_status(self):
        # No critical functions defined at all → each is 'missing', no finding.
        findings, status = audit.check_crash_recovery("x = 1\n")
        self.assertEqual(findings, [])
        self.assertTrue(all(v == "missing" for v in status.values()))


# ═════════════════════════════════════════════════════════════════════════
#  CHECK H — leak test
# ═════════════════════════════════════════════════════════════════════════

def _psutil_leak_probe_unavailable() -> str | None:
    """Return a skip reason if THIS host's psutil cannot introspect the current
    process the way ``audit.check_leak`` requires, else ``None``.

    ``check_leak`` is deliberately best-effort: it imports psutil and reads
    ``Process().num_handles()`` / ``open_files()`` / ``memory_info()`` inside a
    broad ``try`` and, on ANY failure, reports ``ran=False`` plus a ``skipped``
    reason — a contract pinned by ``test_inner_exception_marks_skipped``. psutil
    being merely IMPORTABLE is therefore necessary but NOT sufficient: on a
    locked-down Windows host an EDR / security agent (or a sandbox) can block
    handle enumeration, so ``Process().open_files()`` raises ``AccessDenied``
    even though ``import psutil`` succeeds — and ``check_leak`` then correctly
    skips with ``ran=False``. That is an environmental limitation, not a
    regression. This probe mirrors check_leak's own calls so the live-psutil
    smoke test below gates on exactly what the tool needs to actually run."""
    if importlib.util.find_spec("psutil") is None:
        return "psutil not installed"
    try:
        import psutil  # used immediately below, so pyflakes sees no unused import
    except Exception as exc:  # find_spec found a spec but the import still failed
        return f"psutil import failed: {type(exc).__name__}: {exc}"
    try:
        proc = psutil.Process()
        if hasattr(proc, "num_handles"):
            proc.num_handles()
        proc.open_files()
        proc.memory_info()
    except Exception as exc:  # AccessDenied / OSError under EDR/sandbox, etc.
        return f"psutil cannot introspect this process here ({type(exc).__name__}: {exc})"
    return None


class LeakTestTests(_AuditTestBase):
    def test_leak_skips_cleanly_without_psutil(self):
        # Force the optional psutil import to fail → graceful skip, no findings.
        real_import = __import__

        def fake_import(name, *a, **k):
            if name == "psutil":
                raise ImportError("no psutil")
            return real_import(name, *a, **k)

        with mock.patch("builtins.__import__", side_effect=fake_import):
            findings, summary = audit.check_leak()
        self.assertEqual(findings, [])
        self.assertIn("psutil not available", summary.get("skipped", ""))

    def test_leak_runs_when_psutil_present(self):
        # When this host's psutil can introspect the current process (the CI
        # runner + a normal dev box), the 100-iter no-op loop must RUN and report
        # no growth (well under the thresholds) — assert exactly that, so the real
        # leak gate stands. When it can't (an EDR/sandbox host blocks handle
        # enumeration, so check_leak takes its documented best-effort skip and
        # reports ran=False), skip with a tracked reason rather than failing on an
        # environmental limitation. The leak-detection LOGIC itself stays fully
        # exercised by LeakDetectionBranchTests (fake psutil, env-independent), so
        # this conditional never weakens the gate.
        reason = _psutil_leak_probe_unavailable()
        if reason:
            self.skipTest(f"leak smoke needs live psutil introspection: {reason}")
        findings, summary = audit.check_leak()
        self.assertTrue(summary.get("ran"))
        # A trivial mkstemp/close/unlink loop must not leak >20 handles.
        self.assertEqual([f for f in findings if f.category == "leak-test"], [])


# ═════════════════════════════════════════════════════════════════════════
#  CHECK I — import graph + cycle detection
# ═════════════════════════════════════════════════════════════════════════

class ImportGraphTests(_AuditTestBase):
    def setUp(self):
        super().setUp()
        self.mkdir("skills")
        self.mkdir("core")

    def test_cycle_detected(self):
        self.write("skills/aa.py", "import bb\n")
        self.write("skills/bb.py", "import aa\n")
        files = [os.path.join(self.tmp, "skills", "aa.py"),
                 os.path.join(self.tmp, "skills", "bb.py")]
        findings, info = audit.check_import_graph(files)
        self.assertTrue(any(f.category == "import-cycle" for f in findings))
        self.assertTrue(info["cycles"])

    def test_deferred_import_does_not_create_cycle(self):
        # bb imports aa only inside a function → lazy, breaks the import-time
        # cycle, must NOT be reported.
        self.write("skills/aa.py", "import bb\n")
        self.write("skills/bb.py",
                   "def f():\n    import aa\n    return aa\n")
        files = [os.path.join(self.tmp, "skills", "aa.py"),
                 os.path.join(self.tmp, "skills", "bb.py")]
        findings, info = audit.check_import_graph(files)
        self.assertEqual(info["cycles"], [])
        self.assertEqual([f for f in findings if f.category == "import-cycle"], [])

    def test_acyclic_graph_reports_edges_no_cycle(self):
        self.write("skills/aa.py", "import bb\n")
        self.write("skills/bb.py", "x = 1\n")
        files = [os.path.join(self.tmp, "skills", "aa.py"),
                 os.path.join(self.tmp, "skills", "bb.py")]
        findings, info = audit.check_import_graph(files)
        self.assertEqual(info["cycles"], [])
        self.assertGreaterEqual(info["edge_count"], 1)

    def test_from_import_creates_intra_edge(self):
        # `from bobert_companion import X` and `from core.x import Y` are
        # intra-project edges in the graph.
        self.write("core/cfg.py", "Y = 1\n")
        self.write("skills/aa.py",
                   "from bobert_companion import foo\n"
                   "from core.cfg import Y\n")
        files = [os.path.join(self.tmp, "skills", "aa.py")]
        _findings, info = audit.check_import_graph(files)
        self.assertGreaterEqual(info["edge_count"], 2)
        self.assertIn("aa", info["nodes"])


# ═════════════════════════════════════════════════════════════════════════
#  CHECK J — state-file integrity sweep
# ═════════════════════════════════════════════════════════════════════════

class StateFileSweepTests(_AuditTestBase):
    def test_corrupt_json_flagged_valid_clean(self):
        self.write("hud_state.json", "{}")
        self.write("credits_state.json", "{ not json")
        findings, status = audit.check_state_files()
        self.assertEqual(status["hud_state.json"], "ok")
        self.assertTrue(status["credits_state.json"].startswith("corrupt"))
        self.assertTrue(any(f.file == "credits_state.json"
                            and f.severity == "P1" for f in findings))

    def test_absent_files_marked_ok(self):
        findings, status = audit.check_state_files()
        self.assertEqual(findings, [])
        self.assertTrue(all(v == "absent (ok)" for v in status.values()))

    def test_md_schema_drift_flagged(self):
        self.write("jarvis_todo.md", "no checkbox lines here\n")
        findings, status = audit.check_state_files()
        self.assertEqual(status["jarvis_todo.md"], "schema drift")
        self.assertTrue(any(f.file == "jarvis_todo.md" and f.severity == "P2"
                            for f in findings))

    def test_md_with_expected_pattern_ok(self):
        self.write("jarvis_todo.md", "- [ ] a task\n")
        _findings, status = audit.check_state_files()
        self.assertEqual(status["jarvis_todo.md"], "ok")


# ═════════════════════════════════════════════════════════════════════════
#  reporting
# ═════════════════════════════════════════════════════════════════════════

class WriteReportsTests(_AuditTestBase):
    def test_writes_json_and_md_with_severity_grouping(self):
        findings = [
            audit.Finding(severity="P0", category="eval", file="x.py", line=2,
                          message="eval call"),
            audit.Finding(severity="P1", category="imports", file="y.py", line=3,
                          message="undeclared dep"),
            audit.Finding(severity="P2", category="bare-except", file="z.py",
                          line=5, message="bare except"),
        ]
        audit.write_reports(findings, total_files=7)
        rep = self.read_json("audit_report.json")
        self.assertEqual(rep["summary"]["files_audited"], 7)
        self.assertEqual(rep["summary"]["total_findings"], 3)
        self.assertEqual(rep["summary"]["p0"], 1)
        self.assertEqual(rep["summary"]["p1"], 1)
        self.assertEqual(rep["summary"]["p2"], 1)
        self.assertEqual(len(rep["findings"]["P0"]), 1)
        self.assertNotIn("integration", rep)
        md = self.read_text("audit_report.md")
        self.assertIn("# JARVIS Codebase Audit Report", md)
        self.assertIn("## P0", md)
        self.assertIn("eval call", md)

    def test_integration_section_enriches_report(self):
        integration = {
            "smoke_tests": {"a": "ok", "b": "error: X"},
            "conflict_matrix": [{"skill_a": "p", "skill_b": "q",
                                 "shared_state_files": [], "shared_actions": [],
                                 "both_have_threads": False}],
            "thread_audit": [{"file": "f", "line": 1}],
            "prompt_consistency": {"missing_from_registry": ["m"],
                                   "missing_from_prompt": []},
            "voice_coverage": {"hide_hud": "ok", "show_hud": "MISSING"},
            "tts_pipeline": {"neutral": "ok"},
            "crash_recovery": {"main": "ok (outer try)", "x": "NO TRY/EXCEPT"},
            "leak_test": {"ran": True, "handles_before": 1, "handles_after": 1,
                          "open_files_before": 0, "open_files_after": 0},
            "import_graph": {"nodes": ["a"], "edge_count": 2, "cycles": []},
            "state_files": {"hud_state.json": "ok", "x.json": "corrupt: bad"},
        }
        audit.write_reports([], total_files=1, integration=integration)
        rep = self.read_json("audit_report.json")
        self.assertIn("integration", rep)
        self.assertEqual(rep["integration"]["smoke_tests"]["a"], "ok")
        md = self.read_text("audit_report.md")
        self.assertIn("Integration & Conflict Checks", md)
        self.assertIn("Action smoke tests", md)
        self.assertIn("Crash recovery", md)
        self.assertIn("Leak test", md)

    def test_integration_md_leak_skipped_branch(self):
        # When the leak test was skipped, the markdown reports the skip reason.
        audit.write_reports([], total_files=1, integration={
            "leak_test": {"ran": False, "skipped": "psutil not available"},
        })
        md = self.read_text("audit_report.md")
        self.assertIn("Leak test:** skipped (psutil not available)", md)


# ═════════════════════════════════════════════════════════════════════════
#  main() — driven against a tiny temp tree, heavy bits neutralised
# ═════════════════════════════════════════════════════════════════════════

class MainTests(_AuditTestBase):
    def _seed_minimal_tree(self, bc_body="ACTIONS = {}\n"
                                          "def main():\n    try:\n        pass\n"
                                          "    except Exception:\n        pass\n"):
        self.mkdir("skills")
        self.mkdir("core")
        self.write("requirements.txt", "")
        self.write("bobert_companion.py", bc_body)
        self.write("clean.py", "x = 1\n")
        return

    def _run_main(self, argv):
        with mock.patch.object(sys, "argv", argv):
            return audit.main()

    def test_mutually_exclusive_flags_return_2(self):
        self._seed_minimal_tree()
        rc = self._run_main(["audit", "--integration-only", "--no-integration"])
        self.assertEqual(rc, 2)

    def test_no_integration_clean_tree_returns_0(self):
        self._seed_minimal_tree()
        rc = self._run_main(["audit", "--no-integration", "--quiet"])
        self.assertEqual(rc, 0)
        self.assertTrue(os.path.exists(
            os.path.join(self.tmp, "audit_report.json")))

    def test_p0_finding_returns_1(self):
        # A clean-syntax file containing eval() → P0 → exit code 1.
        self._seed_minimal_tree()
        self.write("danger.py", "eval('1+1')\n")
        rc = self._run_main(["audit", "--no-integration", "--quiet"])
        self.assertEqual(rc, 1)

    def test_p1_finding_returns_2(self):
        self._seed_minimal_tree()
        # Undeclared third-party import (no requirements entry) → P1 → exit 2.
        self.write("needs_dep.py", "import zzz_absent_pkg_for_main_31415\n")
        rc = self._run_main(["audit", "--no-integration", "--quiet"])
        self.assertEqual(rc, 2)

    def test_integration_path_runs_with_import_neutralised(self):
        # Exercise the full A–J branch on the tiny tree. Force the optional
        # bobert_companion import (smoke test) to fail so the smoke check takes
        # its deterministic static fallback instead of importing the real
        # monolith (host-dependent). check_leak runs for real — psutil is in the
        # dev AND CI dep set and the no-op loop leaks nothing — and is itself
        # wrapped in _safe(), so it can't crash the run regardless.
        self._seed_minimal_tree()
        with mock.patch.object(audit.importlib, "import_module",
                               side_effect=ImportError("forced")):
            rc = self._run_main(["audit", "--integration-only", "--quiet"])
        # Exit code is data-dependent but must be one of the defined codes.
        self.assertIn(rc, (0, 1, 2, 3))
        rep = self.read_json("audit_report.json")
        self.assertIn("integration", rep)

    def test_non_quiet_prints_summary_and_top_findings(self):
        # Cover the non-quiet print path + the "top critical findings" block,
        # including the ">5 more" overflow line (seed 6 distinct P0 eval files).
        import io
        self._seed_minimal_tree()
        for i in range(6):
            self.write(f"danger{i}.py", "eval('1+1')\n")
        buf = io.StringIO()
        with mock.patch.object(sys, "stdout", buf), \
             mock.patch.object(sys, "argv", ["audit", "--no-integration"]):
            rc = audit.main()
        out = buf.getvalue()
        self.assertEqual(rc, 1)
        self.assertIn("=== AUDIT COMPLETE ===", out)
        self.assertIn("Files audited:", out)
        self.assertIn("Top critical findings:", out)
        self.assertIn("more in audit_report.md", out)

    def test_p2_only_findings_return_3(self):
        # A bare-except is the lone P2 finding → exit code 3.
        self._seed_minimal_tree()
        self.write("smell.py", "try:\n    pass\nexcept:\n    pass\n")
        rc = self._run_main(["audit", "--no-integration", "--quiet"])
        self.assertEqual(rc, 3)

    def test_fix_flag_applies_and_reruns_syntax(self):
        # A fixable open()-without-encoding finding should be auto-fixed and the
        # post-fix syntax re-check should pass (rc 0).
        self._seed_minimal_tree()
        self.write("fixme.py", "f = open('a.txt')\n")
        rc = self._run_main(["audit", "--no-integration", "--quiet", "--fix"])
        new = self.read_text("fixme.py")
        self.assertIn('encoding="utf-8"', new)
        # open-no-encoding is the only finding and it's fixable → clean exit.
        self.assertEqual(rc, 0)


# ═════════════════════════════════════════════════════════════════════════
#  Targeted branch-coverage completion: defensive guards and node-type
#  branches that the per-checker suites above don't exercise. Each test names
#  the exact branch it pins.
# ═════════════════════════════════════════════════════════════════════════

class CollectorBranchTests(_AuditTestBase):
    """AnnAssign / star-package / unreadable-file branches in the name
    collectors and _module_exported_names."""

    def setUp(self):
        super().setUp()
        self.mkdir("skills")
        self.mkdir("core")

    def test_module_exported_names_includes_annassign(self):
        # `x: int = 1` at module top is an AnnAssign with a Name target → x is
        # an exported name (no __all__ present).
        p = self.write("m.py", "y: int = 5\ndef f():\n    pass\n")
        got = audit._module_exported_names(p)
        self.assertIn("y", got)
        self.assertIn("f", got)

    def test_module_exported_names_unreadable_returns_empty(self):
        # _read_source None → empty set (the early return).
        with mock.patch.object(audit, "_read_source", return_value=(None, None)):
            self.assertEqual(audit._module_exported_names("anything.py"), set())

    def test_bc_collector_includes_annassign(self):
        self.write("bobert_companion.py", "CONST: int = 1\ndef boot():\n    pass\n")
        names = audit.collect_bobert_companion_names()
        self.assertIn("CONST", names)

    def test_bc_collector_star_import_from_package(self):
        # `from pkg import *` where pkg is a package dir (pkg/__init__.py) →
        # the collector resolves the package __init__ exports (the not-isfile
        # branch that switches to __init__.py).
        self.mkdir("pkg")
        self.write("pkg/__init__.py", "EXPORTED_FROM_PKG = 1\n")
        self.write("bobert_companion.py", "from pkg import *\nLOCAL = 2\n")
        names = audit.collect_bobert_companion_names()
        self.assertIn("EXPORTED_FROM_PKG", names)
        self.assertIn("LOCAL", names)

    def test_core_collector_skips_unreadable_file(self):
        # A core/*.py whose _read_source yields None is skipped (continue) and
        # absent from the result; a readable sibling still appears.
        self.write("core/good.py", "def g():\n    pass\n")
        self.write("core/bad.py", "def b():\n    pass\n")
        bad_abs = os.path.join(self.tmp, "core", "bad.py")
        real_read = audit._read_source

        def selective_read(path, *a, **k):
            if os.path.normpath(path) == os.path.normpath(bad_abs):
                return (None, None)
            return real_read(path, *a, **k)

        with mock.patch.object(audit, "_read_source", side_effect=selective_read):
            cm = audit.collect_core_module_names()
        self.assertIn("good", cm)
        self.assertNotIn("bad", cm)

    def test_core_collector_includes_annassign(self):
        self.write("core/c.py", "SLOT: int = 0\n")
        cm = audit.collect_core_module_names()
        self.assertIn("SLOT", cm["c"])


class LocalInternalActionsTests(_AuditTestBase):
    """_load_local_internal_actions: absent file → empty; broken file → empty."""

    def test_absent_file_returns_empty(self):
        # No tools/audit_local.py next to the module → empty set.
        with mock.patch.object(audit.os.path, "exists", return_value=False):
            self.assertEqual(audit._load_local_internal_actions(), set())

    def test_unreadable_or_broken_file_returns_empty(self):
        # File "exists" but exec/read raises → the except returns an empty set.
        with mock.patch.object(audit.os.path, "exists", return_value=True), \
             mock.patch("builtins.open", side_effect=OSError("locked")):
            self.assertEqual(audit._load_local_internal_actions(), set())


class RelAndRequirementsBranchTests(_AuditTestBase):
    def test_rel_valueerror_returns_input(self):
        # relpath raising ValueError (e.g. cross-drive on Windows) → return the
        # path unchanged. Patch relpath to raise so this is OS-independent.
        with mock.patch.object(audit.os.path, "relpath",
                               side_effect=ValueError("different drive")):
            self.assertEqual(audit._rel("C:/some/abs/path.py"), "C:/some/abs/path.py")

    def test_requirements_blank_spec_line_skipped(self):
        # A requirements line that strips to nothing after removing the version
        # marker hits the `if not spec: continue` branch and is ignored.
        self.write("requirements.txt", "requests==2.0\n>=1.0\n# comment\nflask\n")
        pkgs, _optional = audit._parse_requirements_full()
        self.assertIn("requests", pkgs)
        self.assertIn("flask", pkgs)

    def test_requirements_read_error_swallowed(self):
        # An unreadable requirements.txt → the outer except returns empty sets.
        self.write("requirements.txt", "requests\n")
        with mock.patch("builtins.open", side_effect=OSError("locked")):
            pkgs, optional = audit._parse_requirements_full()
        self.assertEqual(pkgs, set())
        self.assertEqual(optional, set())


class CheckSyntaxBranchTests(_AuditTestBase):
    def test_non_pycompile_exception_becomes_p0(self):
        # py_compile raising a NON-PyCompileError (e.g. OSError) hits the generic
        # except and still yields a P0 syntax finding.
        fp = self.write("m.py", "x = 1\n")
        with mock.patch.object(audit.py_compile, "compile",
                               side_effect=OSError("disk gone")):
            out = audit.check_syntax([fp])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].severity, "P0")
        self.assertIn("could not compile", out[0].message)


class CheckImportsBranchTests(_AuditTestBase):
    def setUp(self):
        super().setUp()
        self.mkdir("skills")
        self.write("requirements.txt", "")

    def test_unreadable_file_skipped(self):
        # _read_source None for a file in the list → continue (no crash).
        fp = self.write("skills/s.py", "import os\n")
        with mock.patch.object(audit, "_read_source", return_value=(None, None)):
            out = audit.check_imports([fp])
        self.assertEqual(out, [])

    def test_duplicate_import_on_same_line_deduped(self):
        # `import os, os` yields the same (module, line) key twice → the second
        # hits the `if key in seen: continue` branch.
        fp = self.write("skills/s.py", "import os, os\n")
        out = audit.check_imports([fp])   # os is stdlib → no finding either way
        self.assertEqual(out, [])

    def test_successful_third_party_import_is_clean(self):
        # A module that is NOT stdlib and NOT intra but imports successfully
        # (coverage — present in the CI dep set) hits the post-__import__
        # `continue`, producing no finding.
        if importlib.util.find_spec("coverage") is None:
            self.skipTest("coverage not importable")
        fp = self.write("skills/s.py", "import coverage\n")
        out = audit.check_imports([fp])
        self.assertEqual([f for f in out if f.category == "imports"], [])

    def test_import_raising_non_import_error_skipped(self):
        # __import__ raising something other than ImportError (weird import-time
        # side effect) → the `except Exception: continue` branch; no finding.
        fp = self.write("skills/s.py", "import weird_side_effect_mod\n")
        real_import = __import__

        def fake_import(name, *a, **k):
            if name == "weird_side_effect_mod":
                raise RuntimeError("import had a side effect")
            return real_import(name, *a, **k)

        with mock.patch("builtins.__import__", side_effect=fake_import):
            out = audit.check_imports([fp])
        self.assertEqual([f for f in out if f.category == "imports"], [])

    def test_project_dir_listdir_oserror_swallowed(self):
        # The dynamic intra-module discovery wraps os.listdir(PROJECT_DIR) in
        # try/except OSError. Make listdir raise only for PROJECT_DIR; the
        # function must still complete.
        fp = self.write("skills/s.py", "import os\n")
        real_listdir = os.listdir
        proj = audit.PROJECT_DIR

        def fail_proj(path, *a, **k):
            if os.path.normpath(path) == os.path.normpath(proj):
                raise OSError("denied")
            return real_listdir(path, *a, **k)

        with mock.patch.object(audit.os, "listdir", side_effect=fail_proj):
            out = audit.check_imports([fp])   # must not raise
        self.assertIsInstance(out, list)

    def test_package_discovery_inner_listdir_oserror_swallowed(self):
        # The package-discovery loop lists each top-level dir; an inner
        # os.listdir(<subdir>) OSError is swallowed. Raise only for a specific
        # subdir while PROJECT_DIR itself lists fine.
        self.mkdir("somedir")
        fp = self.write("skills/s.py", "import os\n")
        real_listdir = os.listdir
        sub = os.path.join(audit.PROJECT_DIR, "somedir")

        def fail_sub(path, *a, **k):
            if os.path.normpath(path) == os.path.normpath(sub):
                raise OSError("denied")
            return real_listdir(path, *a, **k)

        with mock.patch.object(audit.os, "listdir", side_effect=fail_sub):
            out = audit.check_imports([fp])   # must not raise
        self.assertIsInstance(out, list)


class CrossRefBranchTests(_AuditTestBase):
    def setUp(self):
        super().setUp()
        self.mkdir("skills")
        self.mkdir("core")

    def test_core_dotted_import_alias_attribute_resolution(self):
        # `import core.state as st` records the alias; `st.MISSING` (not defined
        # in core/state.py) is flagged, while `st.KNOWN` is clean.
        self.write("bobert_companion.py", "X = 1\n")
        self.write("core/state.py", "KNOWN = 1\n")
        fp = self.write("skills/s.py",
                        "import core.state as st\n"
                        "a = st.KNOWN\n"
                        "b = st.MISSING\n")
        out = audit.check_cross_references([fp])
        msgs = " ".join(f.message for f in out)
        self.assertIn("MISSING", msgs)
        self.assertNotIn("KNOWN", msgs)

    def test_star_imports_from_bc_and_core_are_ignored(self):
        # `from bobert_companion import *` and `from core.x import *` both hit
        # the `alias.name == "*": continue` guards (no cross-ref findings).
        self.write("bobert_companion.py", "X = 1\n")
        self.write("core/cfg.py", "Y = 1\n")
        fp = self.write("skills/s.py",
                        "from bobert_companion import *\n"
                        "from core.cfg import *\n")
        out = audit.check_cross_references([fp])
        self.assertEqual([f for f in out if f.category == "cross-ref"], [])

    def test_attribute_on_non_name_base_skipped(self):
        # With a bc alias present (so the attribute walk runs), an attribute
        # access whose base is itself an Attribute (`a.b.c`) is skipped at the
        # `not isinstance(node.value, ast.Name)` guard — no spurious finding.
        self.write("bobert_companion.py", "X = 1\n")
        fp = self.write("skills/s.py",
                        "import bobert_companion as bc\n"
                        "import os\n"
                        "v = os.path.join\n"      # os.path.join → nested Attribute
                        "w = bc.X\n")             # keeps the walk meaningful
        out = audit.check_cross_references([fp])
        self.assertEqual([f for f in out if f.category == "cross-ref"], [])

    def test_unreadable_target_file_skipped(self):
        # bc_names is non-empty (so the check proceeds past the early bail), but
        # the file in the list is unreadable → `_read_source is None` continue.
        # _read_source is also used by collect_bobert_companion_names(); make
        # ONLY the skill file unreadable so bc_names stays populated.
        self.write("bobert_companion.py", "X = 1\ndef boot():\n    pass\n")
        skill = self.write("skills/s.py", "x = 1\n")
        real_read = audit._read_source

        def selective(path, *a, **k):
            if os.path.normpath(path) == os.path.normpath(skill):
                return (None, None)
            return real_read(path, *a, **k)

        with mock.patch.object(audit, "_read_source", side_effect=selective):
            out = audit.check_cross_references([skill])
        self.assertEqual(out, [])

    def test_syntaxerror_target_file_skipped(self):
        # bc_names non-empty + a syntactically-broken file → SyntaxError continue
        # inside check_cross_references (not just the syntax checker's own path).
        self.write("bobert_companion.py", "X = 1\n")
        broken = self.write("skills/broken.py", "def (:\n")
        out = audit.check_cross_references([broken])
        self.assertEqual([f for f in out if f.category == "cross-ref"], [])


class ActionsCheckBranchTests(_AuditTestBase):
    def setUp(self):
        super().setUp()
        self.mkdir("skills")

    def test_unreadable_skill_skipped(self):
        fp = self.write("skills/s.py", "def register(actions):\n    pass\n")
        with mock.patch.object(audit, "_read_source", return_value=(None, None)):
            out, owners = audit.check_actions([fp], set())
        self.assertEqual(out, [])

    def test_non_actions_subscript_and_dynamic_key_skipped(self):
        # In register(): a subscript on a DIFFERENT name (not the actions param)
        # and a dynamic (non-constant) key both hit their `continue` branches.
        src = ("def _h(p):\n    return ''\n"
               "def register(actions):\n"
               "    other = {}\n"
               "    other['x'] = _h\n"          # subscript on non-param → skip
               "    k = 'dyn'\n"
               "    actions[k] = _h\n")         # dynamic key → skip
        fp = self.write("skills/s.py", src)
        out, owners = audit.check_actions([fp], set())
        # neither produced a registered action / finding
        self.assertEqual(out, [])
        self.assertNotIn("x", owners)

    def test_rhs_attribute_and_lambda_callables(self):
        # `actions["a"] = mod.fn` (Attribute rhs) and `actions["b"] = lambda ...`
        # (Lambda rhs) resolve callable_name via their dedicated branches.
        src = ("import mod\n"
               "def register(actions):\n"
               "    actions['a'] = mod.fn\n"
               "    actions['b'] = lambda payload: ''\n")
        fp = self.write("skills/s.py", src)
        out, owners = audit.check_actions([fp], set())
        # both register cleanly (lambda takes 1 arg; attribute callable unknown
        # arity → no arity finding)
        self.assertIn("a", owners)
        self.assertIn("b", owners)


class UnreadableFileSkipBranchTests(_AuditTestBase):
    """The `_read_source is None` continue at the top of several per-file
    checkers (an unreadable file is silently skipped)."""

    def setUp(self):
        super().setUp()
        self.mkdir("skills")

    def test_check_bad_patterns_skips_unreadable(self):
        fp = self.write("m.py", "x = 1\n")
        with mock.patch.object(audit, "_read_source", return_value=(None, None)):
            self.assertEqual(audit.check_bad_patterns([fp]), [])

    def test_check_secrets_skips_unreadable(self):
        fp = self.write("m.py", "x = 1\n")
        with mock.patch.object(audit, "_read_source", return_value=(None, None)):
            self.assertEqual(audit.check_secrets([fp]), [])

    def test_check_state_file_writes_skips_unreadable(self):
        fp = self.write("m.py", "x = 1\n")
        with mock.patch.object(audit, "_read_source", return_value=(None, None)):
            self.assertEqual(audit.check_state_file_writes([fp]), [])


class SecretPlaceholderBranchTests(_AuditTestBase):
    def test_all_x_placeholder_value_whitelisted(self):
        # A password value that is all-X (>=8) looks like a placeholder → the
        # `re.fullmatch(r"[xX*]{8,}", val)` continue suppresses the P1.
        fp = self.write("m.py", "password = 'XXXXXXXXXX'\n")
        out = audit.check_secrets([fp])
        self.assertEqual([f for f in out if f.severity == "P1"], [])


class StateFileWriteNonOpenBranchTests(_AuditTestBase):
    def test_with_block_non_open_context_skipped(self):
        # A `with <non-open call>():` inside the file → the context_expr is a
        # Call but not open() → the `continue` guard fires (no state-write check).
        src = ("def f():\n"
               "    with contextlib.suppress(Exception):\n"
               "        pass\n")
        fp = self.write("m.py", src)
        out = audit.check_state_file_writes([fp])
        self.assertEqual(out, [])


class ReadbackBranchTests(_AuditTestBase):
    def test_json_file_unreadable_non_decode_error(self):
        # A tracked json file that exists but raises a non-JSONDecodeError on
        # open → the generic except emits a state-corruption P1.
        self.write("hud_state.json", "{}")
        with mock.patch("builtins.open", side_effect=OSError("locked")):
            out = audit.check_readback()
        self.assertTrue(any(f.category == "state-corruption"
                            and "unreadable" in f.message for f in out))

    def test_todo_unreadable_flagged(self):
        # jarvis_todo.md present but unreadable → its own except emits a P1.
        self.write("jarvis_todo.md", "- [ ] task\n")
        real_open = open

        def fail_todo(path, *a, **k):
            if str(path).endswith("jarvis_todo.md"):
                raise OSError("locked")
            return real_open(path, *a, **k)

        with mock.patch("builtins.open", side_effect=fail_todo):
            out = audit.check_readback()
        self.assertTrue(any(f.file == "jarvis_todo.md"
                            and "unreadable" in f.message for f in out))


class ResolveOpenTargetBranchTests(_AuditTestBase):
    def test_join_without_string_args_returns_none(self):
        # _resolve_open_target on an os.path.join() whose args are all
        # non-constant falls through to the final `return None`.
        node = ast.parse("os.path.join(a, b)").body[0].value
        self.assertIsNone(audit._resolve_open_target(node))


class MutationHygieneBranchTests(_AuditTestBase):
    def test_unreadable_file_skipped(self):
        fp = self.write("m.py", "x = 1\n")
        with mock.patch.object(audit, "_read_source", return_value=(None, None)):
            self.assertEqual(audit.check_mutation_hygiene([fp]), [])

    def test_thread_target_attribute_recorded(self):
        # `threading.Thread(target=self.worker)` → target is an Attribute; the
        # attr name is recorded as a thread target. Pair it with an unlocked
        # global mutation in a method of the same name to surface the finding.
        src = ("import threading\n"
               "CACHE = {}\n"
               "class C:\n"
               "    def worker(self):\n"
               "        CACHE.clear()\n"
               "    def go(self):\n"
               "        threading.Thread(target=self.worker).start()\n")
        fp = self.write("m.py", src)
        out = audit.check_mutation_hygiene([fp])
        self.assertTrue(all(f.category == "mutation-hygiene" for f in out))


class ProfileSkillBranchTests(_AuditTestBase):
    def setUp(self):
        super().setUp()
        self.mkdir("skills")

    def test_unreadable_skill_returns_empty_profile(self):
        fp = self.write("skills/s.py", "def register(actions):\n    pass\n")
        with mock.patch.object(audit, "_read_source", return_value=(None, None)):
            prof = audit._profile_skill(fp)
        self.assertEqual(prof["actions"], {})
        self.assertEqual(prof["writes_state"], set())

    def test_thread_target_attribute_recorded(self):
        src = ("import threading\n"
               "def register(actions):\n"
               "    t = threading.Thread(target=obj.run)\n")
        fp = self.write("skills/s.py", src)
        prof = audit._profile_skill(fp)
        self.assertIn("run", prof["thread_targets"])

    def test_atomic_writer_via_whole_module_import(self):
        # `import core.atomic_io` (the whole module) marks uses_atomic_writer.
        src = "import core.atomic_io\ndef register(actions):\n    pass\n"
        fp = self.write("skills/s.py", src)
        self.assertTrue(audit._profile_skill(fp)["uses_atomic_writer"])

    def test_atomic_writer_via_from_core_import(self):
        # `from core import atomic_io` also marks uses_atomic_writer.
        src = "from core import atomic_io\ndef register(actions):\n    pass\n"
        fp = self.write("skills/s.py", src)
        self.assertTrue(audit._profile_skill(fp)["uses_atomic_writer"])

    def test_register_without_args_skipped_in_profile(self):
        # `def register():` (no args) hits the `if not node.args.args: continue`
        # guard in the action-extraction pass.
        src = "def register():\n    pass\n"
        fp = self.write("skills/s.py", src)
        prof = audit._profile_skill(fp)
        self.assertEqual(prof["actions"], {})

    def test_annassign_dict_update_registration(self):
        # `handlers: dict = {...}` (AnnAssign with a Dict value) inside register
        # is recognised as a registration dict for the update() form.
        src = ("def h1(p):\n    return ''\n"
               "def register(actions):\n"
               "    handlers: dict = {'aa': h1}\n"
               "    actions.update(handlers)\n")
        fp = self.write("skills/s.py", src)
        prof = audit._profile_skill(fp)
        self.assertIn("aa", prof["actions"])


class StubObjectsBranchTests(unittest.TestCase):
    def test_stub_module_returns_callable(self):
        m = audit._StubModule("fakecv2")
        attr = m.SomeFunc
        self.assertIsInstance(attr, audit._StubCallable)
        # calling it returns another stub (chainable)
        self.assertIsInstance(attr(), audit._StubCallable)

    def test_stub_callable_dunder_attr_raises(self):
        sc = audit._StubCallable("x")
        with self.assertRaises(AttributeError):
            getattr(sc, "__wrapped__")
        # normal attribute access still returns a stub
        self.assertIsInstance(sc.normal, audit._StubCallable)

    def test_stub_module_dunder_attr_raises(self):
        m = audit._StubModule("fake")
        with self.assertRaises(AttributeError):
            getattr(m, "__signature__")


class ThreadAuditBranchTests(_AuditTestBase):
    def test_unreadable_file_skipped(self):
        fp = self.write("m.py", "import threading\nt = threading.Thread(target=f)\n")
        with mock.patch.object(audit, "_read_source", return_value=(None, None)):
            findings, summary = audit.check_thread_audit([fp])
        self.assertEqual(findings, [])
        self.assertEqual(summary, [])

    def test_thread_target_attribute_in_summary(self):
        # target=self.run → Attribute; recorded as target name in the summary.
        src = ("import threading\n"
               "t = threading.Thread(target=obj.run)\n")
        fp = self.write("m.py", src)
        _findings, summary = audit.check_thread_audit([fp])
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["target"], "run")

    def test_delegation_loop_with_pass_break_and_assign(self):
        # A thread target whose loop body's top-level statements are pure
        # delegation: a call-assignment (Assign-of-Call branch), a bare call
        # (Expr-Call branch), and a Break (Pass/Break/Continue branch). All hit
        # their respective `continue`s in _delegation_only → resilient, so no
        # thread-no-try finding is emitted.
        src = ("import threading\n"
               "def worker():\n"
               "    while True:\n"
               "        x = step()\n"     # Assign-of-Call branch (2244-2245)
               "        do()\n"           # Expr-Call branch (2240-2241)
               "        break\n"          # Break branch (2242-2243)
               "t = threading.Thread(target=worker)\n")
        fp = self.write("m.py", src)
        findings, _ = audit.check_thread_audit([fp])
        self.assertEqual(self.by_cat(findings, "thread-no-try"), [])


class SkillPairMatrixWriteBranchTests(_AuditTestBase):
    def setUp(self):
        super().setUp()
        self.mkdir("skills")

    def test_matrix_write_failure_swallowed(self):
        # The conflict-matrix markdown write is best-effort; an OSError on the
        # open() must be swallowed and the check must still return.
        self.write("skills/a.py", "def register(actions):\n    pass\n")
        self.write("skills/b.py", "def register(actions):\n    pass\n")
        files = [os.path.join(self.tmp, "skills", "a.py"),
                 os.path.join(self.tmp, "skills", "b.py")]
        real_open = open

        def fail_md(path, *a, **k):
            if str(path).endswith("audit_conflict_matrix.md"):
                raise OSError("read-only")
            return real_open(path, *a, **k)

        with mock.patch("builtins.open", side_effect=fail_md):
            out, rows = audit.check_skill_pair_conflicts(files)
        self.assertIsInstance(out, list)


class CrashRecoveryNestedTryBranchTests(_AuditTestBase):
    def test_nested_try_accepted(self):
        # A critical function with a try that is neither a top-level statement
        # nor the loop body's first level (it's nested inside an `if`) is still
        # accepted via the `any(Try in ast.walk(fn))` branch.
        crit = next(iter(audit._CRITICAL_FUNCTIONS))
        bc = (f"def {crit}():\n"
              "    if cond:\n"
              "        try:\n"
              "            pass\n"
              "        except Exception:\n"
              "            pass\n")
        findings, status = audit.check_crash_recovery(bc)
        self.assertEqual(status[crit], "ok (nested try)")
        self.assertEqual([f for f in findings if crit in f.message], [])


class LeakDetectionBranchTests(_AuditTestBase):
    """Drive check_leak's growth-detection branches with a fake psutil.Process
    so the otherwise leak-only paths are deterministically exercised."""

    def _fake_proc(self, handles_after=0, files_after=0, raise_on=None):
        class _FakeProc:
            def __init__(self):
                self._first_handles = True
                self._first_files = True

            def num_handles(self):
                if self._first_handles:
                    self._first_handles = False
                    return 0
                return handles_after

            def open_files(self):
                if raise_on == "open_files":
                    raise RuntimeError("psutil hiccup")
                if self._first_files:
                    self._first_files = False
                    return []
                return [object()] * files_after

            def memory_info(self):
                class _M:
                    rss = 0
                return _M()

        return _FakeProc

    def test_handle_growth_flags_leak(self):
        if importlib.util.find_spec("psutil") is None:
            self.skipTest("psutil not installed")
        import psutil
        with mock.patch.object(psutil, "Process",
                               self._fake_proc(handles_after=50)):
            findings, summary = audit.check_leak()
        self.assertTrue(summary.get("ran"))
        self.assertTrue(any("file-handle count grew" in f.message
                            for f in findings))

    def test_open_files_growth_flags_leak(self):
        if importlib.util.find_spec("psutil") is None:
            self.skipTest("psutil not installed")
        import psutil
        with mock.patch.object(psutil, "Process",
                               self._fake_proc(files_after=10)):
            findings, _summary = audit.check_leak()
        self.assertTrue(any("open-file count grew" in f.message
                            for f in findings))

    def test_inner_exception_marks_skipped(self):
        if importlib.util.find_spec("psutil") is None:
            self.skipTest("psutil not installed")
        import psutil
        with mock.patch.object(psutil, "Process",
                               self._fake_proc(raise_on="open_files")):
            findings, summary = audit.check_leak()
        self.assertEqual(findings, [])
        self.assertIn("skipped", summary)


class ImportGraphBranchTests(_AuditTestBase):
    def setUp(self):
        super().setUp()
        self.mkdir("skills")
        self.mkdir("core")

    def test_core_file_module_key_and_edges(self):
        # A core/*.py file in the list maps to a `core.<stem>` module key and
        # its intra import becomes an edge.
        self.write("core/aa.py", "import bb\n")
        self.write("skills/bb.py", "x = 1\n")
        files = [os.path.join(self.tmp, "core", "aa.py"),
                 os.path.join(self.tmp, "skills", "bb.py")]
        _findings, info = audit.check_import_graph(files)
        self.assertIn("core.aa", info["nodes"])

    def test_syntaxerror_file_skipped(self):
        # A syntactically-broken file is skipped (SyntaxError continue) without
        # crashing the graph build.
        broken = self.write("skills/broken.py", "def (:\n")
        findings, info = audit.check_import_graph([broken])
        self.assertEqual([f for f in findings if f.category == "import-cycle"], [])

    def test_unreadable_file_skipped(self):
        # _read_source None for a file in the list → the `continue` guard.
        good = self.write("skills/good.py", "import bobert_companion\n")
        with mock.patch.object(audit, "_read_source", return_value=(None, None)):
            findings, info = audit.check_import_graph([good])
        self.assertEqual(info["edge_count"], 0)

    def test_relative_from_import_skipped(self):
        # `from . import x` has node.module is None → the `continue` guard.
        self.write("skills/aa.py", "from . import sibling\n")
        files = [os.path.join(self.tmp, "skills", "aa.py")]
        findings, info = audit.check_import_graph(files)
        self.assertEqual([f for f in findings if f.category == "import-cycle"], [])


class StateFilesSweepBranchTests(_AuditTestBase):
    def test_unreadable_state_file_flagged(self):
        # A tracked state file present but unreadable → P1 + 'unreadable' status.
        self.write("hud_state.json", "{}")
        real_open = open

        def fail_hud(path, *a, **k):
            if str(path).endswith("hud_state.json"):
                raise OSError("locked")
            return real_open(path, *a, **k)

        with mock.patch("builtins.open", side_effect=fail_hud):
            findings, status = audit.check_state_files()
        self.assertTrue(status["hud_state.json"].startswith("unreadable"))
        self.assertTrue(any(f.file == "hud_state.json" and f.severity == "P1"
                            for f in findings))


class MainBranchCoverageTests(_AuditTestBase):
    def _run_main(self, argv):
        with mock.patch.object(sys, "argv", argv):
            return audit.main()

    def _seed(self, bc_body, extra=None):
        self.mkdir("skills")
        self.mkdir("core")
        self.write("requirements.txt", "")
        self.write("bobert_companion.py", bc_body)
        if extra:
            for rel, body in extra.items():
                self.write(rel, body)

    def test_bc_actions_extracted_from_all_three_forms(self):
        # ACTIONS literal keys + ACTIONS["x"]= + ACTIONS.update({...}) are all
        # parsed out of bobert_companion.py during main().
        bc = ("ACTIONS = {\n    'lit_a': h,\n    'lit_b': h,\n}\n"
              "ACTIONS['idx_c'] = h\n"
              "ACTIONS.update({'upd_d': h})\n"
              "def main():\n    try:\n        pass\n    except Exception:\n        pass\n")
        # A skill that documents these actions so the integration prompt check
        # has registered actions to compare (also exercises _profile_skill in
        # the all_registered loop).
        skill = ("def _h(p):\n    return ''\n"
                 "def register(actions):\n"
                 "    actions['skill_e'] = _h\n")
        self._seed(bc, extra={"skills/s.py": skill})
        with mock.patch.object(audit.importlib, "import_module",
                               side_effect=ImportError("forced")):
            rc = self._run_main(["audit", "--quiet"])
        self.assertIn(rc, (0, 1, 2, 3))
        rep = self.read_json("audit_report.json")
        self.assertIn("integration", rep)

    def test_fix_non_quiet_prints_applied_line(self):
        # --fix WITHOUT --quiet prints the "[fix] applied N" summary line.
        import io
        self._seed("ACTIONS = {}\ndef main():\n    try:\n        pass\n"
                   "    except Exception:\n        pass\n",
                   extra={"fixme.py": "f = open('a.txt')\n"})
        buf = io.StringIO()
        with mock.patch.object(sys, "stdout", buf), \
             mock.patch.object(sys, "argv",
                               ["audit", "--no-integration", "--fix"]):
            audit.main()
        self.assertIn("[fix] applied", buf.getvalue())

    def test_fix_introduces_syntax_error_warning_path(self):
        # If the post-fix syntax re-check reports problems, main prints the
        # WARNING block and keeps the (now-real) syntax findings. We simulate a
        # clean pre-fix syntax pass and a dirty post-fix pass.
        import io
        self._seed("ACTIONS = {}\ndef main():\n    try:\n        pass\n"
                   "    except Exception:\n        pass\n",
                   extra={"fixme.py": "f = open('a.txt')\n"})
        fake_post = [audit.Finding(severity="P0", category="syntax",
                                   file="fixme.py", line=1,
                                   message="py_compile failed: simulated")]
        call_state = {"n": 0}
        real_check_syntax = audit.check_syntax

        def syntax_side_effect(files):
            call_state["n"] += 1
            # First call (pre-fix, line 2996) → real/clean; second call
            # (post-fix, line 3083) → simulated syntax error.
            if call_state["n"] >= 2:
                return list(fake_post)
            return real_check_syntax(files)

        buf = io.StringIO()
        err = io.StringIO()
        with mock.patch.object(audit, "check_syntax",
                               side_effect=syntax_side_effect), \
             mock.patch.object(sys, "stdout", buf), \
             mock.patch.object(sys, "stderr", err), \
             mock.patch.object(sys, "argv",
                               ["audit", "--no-integration", "--fix"]):
            rc = audit.main()
        self.assertIn("--fix introduced syntax errors", err.getvalue())
        self.assertEqual(rc, 1)   # a P0 syntax finding now dominates

    def test_duplicate_findings_deduplicated(self):
        # Two identical Thread calls on ONE source line yield two findings with
        # an identical (file, line, category, message) key → the second is
        # dropped by the dedup `continue`.
        self._seed("ACTIONS = {}\ndef main():\n    try:\n        pass\n"
                   "    except Exception:\n        pass\n",
                   extra={"dup.py":
                          "import threading\n"
                          "threading.Thread(target=a); threading.Thread(target=a)\n"})
        rc = self._run_main(["audit", "--no-integration", "--quiet"])
        rep = self.read_json("audit_report.json")
        # report["findings"] is keyed by severity; thread-no-daemon is P1.
        p1 = rep["findings"]["P1"]
        dups = [f for f in p1
                if f.get("category") == "thread-no-daemon"
                and f.get("file") == "dup.py"]
        # exactly one survives despite two call sites on the same line
        self.assertEqual(len(dups), 1)
        self.assertIn(rc, (0, 1, 2, 3))


if __name__ == "__main__":
    unittest.main()
