"""Documentation-truth gate: the numbers and contracts the docs assert must hold.

WHY THIS FILE EXISTS
--------------------
Every hand-transcribed fact in this repo's prose has rotted at least once:

  * README.md / CONTRIBUTING.md claimed "~2,100 tests" while the suite had grown
    past 14,000 -- roughly 7x low, untouched since 2026-06-01.
  * README.md said "~78 skills" in three places against 96 loadable skills.
  * SETUP.md / SETUP_GUIDE.md told a NEW USER to `ollama pull qwen2.5:14b` as
    "the default" local LLM, two promotions after the shipped default became
    gemma4:26b-a4b-it-qat -- so following the guide installed the wrong brain.
  * SETUP.md's first-run smoke test documented the answer to "what version are
    you on" as a hardcoded 2.0.29, 65 VERSION bumps ago.
  * README.md / CONTRIBUTING.md claimed CI runs `tools/audit_codebase.py`, which
    ci.yml explicitly says it does NOT (it is a local/full-deps gate).

The contrast that proves the point: ``docs/ACTION_INDEX.md`` is MACHINE-generated
by ``tools/gen_action_index.py`` and is accurate, while every number a human typed
drifted. This file is the cheap version of that discipline -- it does not generate
the prose, it just refuses to let the prose lie.

DESIGN NOTES
------------
* Tolerance, not equality. These are "~N" approximations; asserting an exact
  number would flap on every added skill and teach people to edit the assertion
  instead of the doc. The band is wide enough to ignore normal growth and narrow
  enough to catch the 7x / 23% drifts that actually happened.
* Nothing is IMPORTED from the tree. Model tags are read as SOURCE literals via
  ``ast``, because the docs document the SHIPPED default and importing
  ``core.config`` would hand back the owner's ``data/user_settings.json``
  override instead. Test counting is a pure AST pass -- no test module is
  imported, so counting cannot execute anything.
* Files that ``.gitignore`` excludes (SETUP_GUIDE.md) are skipped when absent,
  so a fresh clone and the CI runner stay green.

CI-safety: stdlib only. No monolith boot, no hardware, no network, no deps.
"""
from __future__ import annotations

import ast
import os
import re
import unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TESTS = os.path.join(_ROOT, "tests")

# How far a documented "~N" may drift from the measured value before it is a lie.
# 15% absorbs ordinary growth between doc passes; the real regressions this file
# was written for were 690% (tests) and 23% (skills).
_TOLERANCE = 0.15


def _read(*parts):
    path = os.path.join(_ROOT, *parts)
    with open(path, "rb") as fh:
        return fh.read().decode("utf-8-sig")


def _exists(*parts):
    return os.path.exists(os.path.join(_ROOT, *parts))


# ── measurement helpers (the "truth" side of every assertion) ────────────────

def measure_skills():
    """Count what ``bobert_companion.load_skills()`` would actually load.

    Mirrors that function's rule exactly: every ``skills/<name>/__init__.py``
    package plus every ``skills/*.py`` whose name does not start with ``_``,
    with packages winning a name collision.
    """
    skills_dir = os.path.join(_ROOT, "skills")
    names = set()
    for entry in sorted(os.listdir(skills_dir)):
        sub = os.path.join(skills_dir, entry)
        if not os.path.isdir(sub) or entry.startswith("_") or entry == "__pycache__":
            continue
        if os.path.isfile(os.path.join(sub, "__init__.py")):
            names.add(entry)
    for fn in sorted(os.listdir(skills_dir)):
        if not fn.endswith(".py") or fn.startswith("_"):
            continue
        names.add(fn[:-3])
    return len(names)


def measure_monolith_lines():
    return len(_read("bobert_companion.py").splitlines())


def _class_index():
    """Parse every module under tests/ into {(mod, cls): (bases, test_methods)}."""
    modfiles, classes, imports = {}, {}, {}
    for dirpath, dirnames, filenames in os.walk(_TESTS):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(dirpath, fn)
            rel = os.path.relpath(path, _ROOT).replace("\\", "/")
            modfiles[rel[:-3].replace("/", ".")] = path
    for mod, path in modfiles.items():
        imports[mod] = {}
        try:
            with open(path, "rb") as fh:
                tree = ast.parse(fh.read().decode("utf-8-sig"))
        except (SyntaxError, UnicodeDecodeError):  # pragma: no cover - defensive
            continue
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module:
                for alias in node.names:
                    imports[mod][alias.asname or alias.name] = (node.module, alias.name)
            elif isinstance(node, ast.ClassDef):
                bases = []
                for b in node.bases:
                    if isinstance(b, ast.Name):
                        bases.append(b.id)
                    elif isinstance(b, ast.Attribute):
                        bases.append(ast.unparse(b))
                methods = set(
                    n.name for n in node.body
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and n.name.startswith("test"))
                classes[(mod, node.name)] = (bases, methods)
    return modfiles, classes, imports


_TESTCASE_BASES = {
    "unittest.TestCase", "TestCase", "unittest.case.TestCase",
    "unittest.IsolatedAsyncioTestCase", "IsolatedAsyncioTestCase",
}


def measure_tests():
    """(discovered test files, effective test methods) by pure AST.

    Emulates ``unittest`` discovery (``pattern="test_*.py"``, recursive) and
    expands inheritance, so a shared base class's tests are counted once per
    concrete subclass -- which is what the runner reports.
    """
    modfiles, classes, imports = _class_index()

    def resolve(mod, base):
        if base in _TESTCASE_BASES:
            return "TESTCASE"
        if (mod, base) in classes:
            return (mod, base)
        imp = imports.get(mod, {}).get(base.split(".")[0])
        if imp:
            srcmod, srcname = imp
            name = srcname or base.split(".")[-1]
            if (srcmod, name) in classes:
                return (srcmod, name)
        for key in classes:
            if key[1] == base.split(".")[-1]:
                return key
        return None

    def is_testcase(key, seen=None):
        if key == "TESTCASE":
            return True
        if key is None or key not in classes:
            return False
        seen = seen or set()
        if key in seen:
            return False
        seen.add(key)
        return any(is_testcase(resolve(key[0], b), seen) for b in classes[key][0])

    def inherited(key, seen=None):
        if key == "TESTCASE" or key is None or key not in classes:
            return set()
        seen = seen or set()
        if key in seen:
            return set()
        seen.add(key)
        out = set(classes[key][1])
        for b in classes[key][0]:
            out |= inherited(resolve(key[0], b), seen)
        return out

    discovered = [m for m, p in modfiles.items()
                  if os.path.basename(p).startswith("test_")]
    total = 0
    for mod in discovered:
        for key in classes:
            if key[0] == mod and is_testcase(key):
                total += len(inherited(key))
    return len(discovered), total


def config_literal(name):
    """The SHIPPED value of a top-level ``core/config.py`` string constant.

    Read statically on purpose: importing core.config runs
    ``_apply_user_settings()``, which overlays the owner's
    ``data/user_settings.json`` -- the docs document what SHIPS, not what this
    particular box is currently configured for.
    """
    tree = ast.parse(_read("core", "config.py"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (isinstance(target, ast.Name) and target.id == name
                        and isinstance(node.value, ast.Constant)
                        and isinstance(node.value.value, str)):
                    return node.value.value
    raise AssertionError("core/config.py has no top-level string %r" % name)


def local_llm_preference():
    """The ``_LOCAL_LLM_PREFERENCE`` tuple, read from monolith SOURCE."""
    src = _read("bobert_companion.py")
    m = re.search(r"^_LOCAL_LLM_PREFERENCE\s*=\s*\((.*?)\)",
                  src, re.MULTILINE | re.DOTALL)
    assert m, ("could not locate _LOCAL_LLM_PREFERENCE in bobert_companion.py -- "
               "if it was renamed, every doc that defers to it is now wrong too")
    return re.findall(r'"([^"]+)"', m.group(1))


def _docs_numbers(text, pattern):
    return [int(n.replace(",", "")) for n in re.findall(pattern, text)]


def _assert_close(case, documented, measured, where):
    low = measured * (1 - _TOLERANCE)
    high = measured * (1 + _TOLERANCE)
    case.assertTrue(
        low <= documented <= high,
        "%s documents %s but the tree measures %s (allowed %d-%d). Update the "
        "prose -- a stale number in a public doc is a defect, not a rounding "
        "choice." % (where, format(documented, ","), format(measured, ","),
                     int(low), int(high)))


# ── the assertions ───────────────────────────────────────────────────────────

class SkillCountTests(unittest.TestCase):
    def test_readme_skill_counts_track_the_loader(self):
        readme = _read("README.md")
        counts = _docs_numbers(readme, r"~([\d,]+)\s*(?:\w+[- ])?skills")
        self.assertTrue(counts, "README.md no longer states a skill count at all")
        measured = measure_skills()
        for n in counts:
            _assert_close(self, n, measured, "README.md")

    def test_readme_and_n_more_tracks_the_loader(self):
        """The intro's "...and ~N more" is a skill count too, minus the examples
        it just listed. It rotted with the others and must not be forgotten."""
        readme = _read("README.md")
        m = re.search(r"and ~([\d,]+) more\)", readme)
        self.assertIsNotNone(m, "README.md intro no longer says 'and ~N more'")
        # 4 skills are named inline before the "~N more"; allow a wide band.
        _assert_close(self, int(m.group(1).replace(",", "")) + 4,
                      measure_skills(), "README.md intro ('and ~N more')")


class TestSuiteSizeTests(unittest.TestCase):
    def test_readme_test_count_tracks_the_suite(self):
        readme = _read("README.md")
        m = re.search(r"\*\*~([\d,]+) tests\*\*", readme)
        self.assertIsNotNone(
            m, "README.md no longer states a test count in the Testing section")
        files, methods = measure_tests()
        _assert_close(self, int(m.group(1).replace(",", "")), methods,
                      "README.md test count")
        fm = re.search(r"across ([\d,]+) files", readme)
        if fm:
            _assert_close(self, int(fm.group(1).replace(",", "")), files,
                          "README.md test-file count")

    def test_contributing_does_not_pin_an_unmeasured_runtime(self):
        """CONTRIBUTING.md used to advertise "~30s" beside a 7x-wrong test count.

        Nobody ever measured it, and an absent number cannot rot -- run_tests.py
        prints its own real total. Refuse to let a fresh guess creep back in.
        """
        text = _read("CONTRIBUTING.md")
        stray = re.findall(r"~\s*\d+\s*(?:s\b|sec|seconds)", text)
        self.assertEqual(
            stray, [],
            "CONTRIBUTING.md pins a suite runtime (%s). Either quote a figure you "
            "just measured, or say nothing -- run_tests.py already prints the "
            "real 'N run' line." % stray)

    def test_contributing_does_not_restate_a_test_count(self):
        text = _read("CONTRIBUTING.md")
        self.assertEqual(
            re.findall(r"~[\d,]+\s*tests", text), [],
            "CONTRIBUTING.md restates the test count; that is a second copy that "
            "rotted once already. Point at README.md or at run_tests.py output.")


class LocalModelDocsTests(unittest.TestCase):
    """The highest-cost doc bug: a new user follows SETUP and installs the wrong
    brain, then silently runs on a fallback model forever."""

    RETIRED = ("qwen2.5:14b", "qwen2.5vl", "qwen2.5:32b", "llama3.1:8b")

    def test_setup_pulls_the_shipped_default_model(self):
        shipped = config_literal("LOCAL_LLM_MODEL")
        setup = _read("SETUP.md")
        self.assertIn(
            shipped, setup,
            "SETUP.md must tell a new user to pull core/config.py's shipped "
            "LOCAL_LLM_MODEL (%s); anything else installs the wrong brain."
            % shipped)

    def test_setup_does_not_present_a_retired_tag_as_the_default(self):
        setup = _read("SETUP.md")
        for tag in self.RETIRED:
            self.assertNotIn(
                "ollama pull " + tag, setup,
                "SETUP.md still instructs `ollama pull %s`; the shipped default "
                "is %s." % (tag, config_literal("LOCAL_LLM_MODEL")))

    def test_setup_guide_pulls_the_shipped_default_model(self):
        if not _exists("SETUP_GUIDE.md"):
            self.skipTest("SETUP_GUIDE.md is gitignored and absent in this checkout")
        shipped = config_literal("LOCAL_LLM_MODEL")
        guide = _read("SETUP_GUIDE.md")
        self.assertIn("ollama pull " + shipped, guide,
                      "SETUP_GUIDE.md must pull the shipped default (%s)" % shipped)
        for tag in self.RETIRED:
            self.assertNotIn(
                "ollama pull " + tag, guide,
                "SETUP_GUIDE.md still instructs `ollama pull %s`" % tag)

    def test_vision_tag_is_in_lockstep_with_the_chat_tag(self):
        """Both setup docs now say "one multimodal tag serves chat AND vision,
        there is no second VLM to pull". That is only true while the two config
        constants match -- so make the claim machine-checked, not trusted."""
        self.assertEqual(
            config_literal("LOCAL_VISION_MODEL"), config_literal("LOCAL_LLM_MODEL"),
            "LOCAL_VISION_MODEL has forked from LOCAL_LLM_MODEL. SETUP.md and "
            "SETUP_GUIDE.md both promise one shared multimodal tag and no "
            "separate vision pull -- fix the docs or restore the lockstep.")

    def test_low_vram_alternative_is_really_in_the_fallback_chain(self):
        """SETUP.md/SETUP_GUIDE.md tell low-VRAM users to pull gemma4:12b and
        promise the selector picks it up automatically. Prove it is in the chain
        rather than taking the sentence's word for it."""
        chain = local_llm_preference()
        self.assertEqual(chain[0], config_literal("LOCAL_LLM_MODEL"),
                         "the chain no longer leads with the shipped default")
        setup = _read("SETUP.md")
        m = re.search(r"pull `([\w.:\-]+)` instead", setup)
        self.assertIsNotNone(m, "SETUP.md no longer names a low-VRAM alternative")
        self.assertIn(m.group(1), chain,
                      "SETUP.md tells low-VRAM users to pull %r and says the "
                      "selector falls back to it, but it is not in "
                      "_LOCAL_LLM_PREFERENCE %s" % (m.group(1), chain))


class VersionSmokeTestDocTests(unittest.TestCase):
    def test_setup_smoke_test_does_not_pin_a_version_literal(self):
        """SETUP.md's first-run table told users to expect `2.0.29` for 65
        releases after that stopped being true. The action reads the VERSION
        file, so the doc must describe that, never a frozen number."""
        for line in _read("SETUP.md").splitlines():
            if "what version are you on" in line:
                self.assertIsNone(
                    re.search(r"\b\d+\.\d+\.\d+\b", line),
                    "SETUP.md pins a version literal in the first-run smoke "
                    "test: %r. version_info reads the top-level VERSION file, "
                    "so describe that instead." % line.strip())
                return
        self.fail("SETUP.md no longer contains the version smoke-test row")


class CiGateClaimTests(unittest.TestCase):
    """"CI runs exactly these gates" was false: ci.yml deliberately excludes the
    codebase auditor, and a contributor who believes the docs skips the only gate
    that would have caught an auditor-detectable defect before a public push."""

    def _ci_run_blob(self):
        if not _exists(".github", "workflows", "ci.yml"):
            self.skipTest("no CI workflow in this checkout")
        return _read(".github", "workflows", "ci.yml")

    def test_auditor_is_still_not_a_ci_step(self):
        ci = self._ci_run_blob()
        run_lines = [ln for ln in ci.splitlines()
                     if re.match(r"\s*(run:|python )", ln)]
        self.assertNotIn(
            "audit_codebase.py", "\n".join(run_lines),
            "ci.yml now RUNS the auditor -- README.md and CONTRIBUTING.md say it "
            "deliberately does not. Update both docs in the same commit.")

    def test_docs_do_not_claim_ci_runs_the_auditor(self):
        for name in ("README.md", "CONTRIBUTING.md"):
            text = _read(name)
            self.assertIn(
                "audit_codebase.py", text,
                "%s no longer mentions the auditor at all" % name)
            self.assertTrue(
                re.search(r"(?is)audit_codebase\.py[^.]{0,200}?"
                          r"(not a CI step|NOT `?tools/audit_codebase|LOCAL only|"
                          r"local/full-deps|not run)", text)
                or re.search(r"(?is)(not a CI step|\*\*not `tools/audit_codebase)",
                             text),
                "%s must say plainly that tools/audit_codebase.py is a LOCAL "
                "gate CI does not run (see the NOTE in ci.yml)." % name)

    def test_docs_do_not_claim_ci_runs_exactly_the_listed_gates(self):
        self.assertNotIn(
            "runs exactly\nthese gates", _read("CONTRIBUTING.md"),
            "CONTRIBUTING.md is back to claiming CI runs exactly the listed "
            "gates; it does not run the auditor.")

    def test_named_ci_gates_really_exist_in_the_workflow(self):
        ci = self._ci_run_blob()
        for script in ("check_no_pii.py", "run_tests.py", "run_coverage.py",
                       "pyflakes", "compileall"):
            self.assertIn(script, ci,
                          "CONTRIBUTING.md/README.md name %s as a CI gate but "
                          "ci.yml does not run it" % script)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    unittest.main()
