"""Regression tests for the 2026-07-21 audit finding "Action-index generator
misses lambda- and loop-registered actions" (actions-registry dimension).

The registration-scan rule used to live as FOUR diverging copies (two regexes
in tools/gen_action_index.py, an AST pass in audit_codebase._profile_skill,
a regex trio in audit_codebase.main()); every lambda-valued monolith entry and
the whole browser agent were absent from docs/ACTION_INDEX.md — the file
tools/web_interface.py serves as the control panel's complete Actions
inventory. The fix consolidated the rule into tools/registration_scan.py.

These are source-scanning INVARIANT tests: they independently enumerate what
the tree registers and demand the shared scanner sees all of it, so ANY future
registration form the scanner can't parse fails here — not just the lambda
class this audit caught.
"""
from __future__ import annotations

import ast
import glob
import importlib.util
import os
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
_TOOLS = os.path.join(_PROJECT, "tools")
_MONO = os.path.join(_PROJECT, "bobert_companion.py")


def _load_registration_scan():
    spec = importlib.util.spec_from_file_location(
        "registration_scan", os.path.join(_TOOLS, "registration_scan.py"))
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RS = _load_registration_scan()


def _read(path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        return fh.read()


def _skill_sources():
    return (glob.glob(os.path.join(_PROJECT, "skills", "*.py"))
            + glob.glob(os.path.join(_PROJECT, "skills", "*", "__init__.py")))


class ScannerFormsTests(unittest.TestCase):
    """Unit coverage of every registration form on synthetic source — each one
    is a form some real file uses and at least one old copy of the rule missed."""

    def test_all_known_forms_on_synthetic_source(self):
        src = (
            "ACTIONS = {\n"
            "    'plain': _act_plain,\n"
            "    'lam': lambda _='': _act_set(True),\n"
            "    'opaque_lam': lambda _='': 'just a string',\n"
            "}\n"
            "ACTIONS['sub'] = mod.handler\n"
            "ACTIONS.update({'upd': _act_upd})\n"
            "def register(actions):\n"
            "    handlers = {'dict_a': _fn_a, 'dict_b': _fn_b}\n"
            "    for k, v in handlers.items():\n"
            "        if k not in actions:\n"
            "            actions[k] = v\n"
            "    actions.update(handlers)\n"
            "    for alias in ('al_one', 'al_two'):\n"
            "        actions[alias] = _fn_a\n"
            "    actions['fact'] = _make_handler(cfg)\n"
            "    actions['alias'] = actions['fact']\n"
        )
        regs = RS.scan_registrations(src, filename="synth.py")
        self.assertEqual(
            set(regs),
            {"plain", "lam", "opaque_lam", "sub", "upd", "dict_a", "dict_b",
             "al_one", "al_two", "fact", "alias"})
        self.assertEqual(regs["plain"].symbol, "_act_plain")
        self.assertEqual(regs["lam"].symbol, "_act_set")     # lambda → callee
        self.assertEqual(regs["lam"].kind, "lambda")
        # A lambda that resolves to no call still carries its own location.
        self.assertEqual(regs["opaque_lam"].symbol, "lambda@synth.py:4")
        self.assertEqual(regs["sub"].symbol, "mod.handler")
        self.assertEqual(regs["dict_a"].symbol, "_fn_a")
        self.assertEqual(regs["al_two"].symbol, "_fn_a")
        # Factory-call RHS: symbol anchors to the factory but kind says "call"
        # so audit arity checks must not treat the factory as the handler.
        self.assertEqual((regs["fact"].symbol, regs["fact"].kind),
                         ("_make_handler", "call"))
        # actions["a"] = actions["b"] resolves through the fixed point.
        self.assertEqual((regs["alias"].symbol, regs["alias"].kind),
                         ("_make_handler", "alias"))

    def test_unresolvable_alias_keeps_the_name(self):
        src = ("def register(actions):\n"
               "    actions['ghost'] = actions['never_defined']\n")
        regs = RS.scan_registrations(src, filename="synth.py")
        self.assertIn("ghost", regs)   # the NAME must never vanish
        self.assertEqual(regs["ghost"].symbol, "?alias:never_defined")

    def test_unrelated_dicts_and_dynamic_keys_ignored(self):
        src = ("def register(actions):\n"
               "    cfg = {'host': 'x'}\n"          # never feeds actions
               "    for k, v in cfg.items():\n"
               "        print(k, v)\n"              # loop does NOT assign actions
               "    k = 'dyn'\n"
               "    actions[k] = _h\n")             # dynamic key, no literal loop
        regs = RS.scan_registrations(src, filename="synth.py")
        self.assertEqual(set(regs), set())


class MonolithCompletenessInvariant(unittest.TestCase):
    """COMPLETENESS INVARIANT — independently collect every string key the
    monolith registers into ACTIONS (dict-literal keys, ACTIONS.update({...})
    literal keys, ACTIONS["x"] = assigns) WITHOUT caring what the value looks
    like, then demand the shared scanner saw every one. Fails on ANY future
    value form the scanner can't see (lambda, partial, walrus, ...)."""

    @classmethod
    def setUpClass(cls):
        cls.mono = _read(_MONO)
        cls.tree = ast.parse(cls.mono)
        cls.scanned = RS.scan_registrations(
            cls.mono, filename="bobert_companion.py", targets=("ACTIONS",))

    def _independent_names(self):
        exp: set[str] = set()
        for node in ast.walk(self.tree):
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == "ACTIONS"
                    and isinstance(node.value, ast.Dict)):
                exp.update(k.value for k in node.value.keys
                           if isinstance(k, ast.Constant)
                           and isinstance(k.value, str))
            elif (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "update"
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "ACTIONS"
                    and node.args and isinstance(node.args[0], ast.Dict)):
                exp.update(k.value for k in node.args[0].keys
                           if isinstance(k, ast.Constant)
                           and isinstance(k.value, str))
            elif (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Subscript)
                    and isinstance(node.targets[0].value, ast.Name)
                    and node.targets[0].value.id == "ACTIONS"
                    and isinstance(node.targets[0].slice, ast.Constant)
                    and isinstance(node.targets[0].slice.value, str)):
                exp.add(node.targets[0].slice.value)
        return exp

    def test_every_monolith_registration_is_scanned(self):
        exp = self._independent_names()
        self.assertGreater(len(exp), 100,
                           "sanity: the monolith ACTIONS collection collapsed")
        missing = exp - set(self.scanned)
        self.assertFalse(
            missing,
            f"shared scanner is blind to monolith registration(s): "
            f"{sorted(missing)[:20]} — every copy of the rule (gen_action_index"
            f" AND audit_codebase) inherits this hole; fix registration_scan.py")

    def test_lambda_sentinels_present_and_resolved(self):
        # The concrete names the 2026-07-21 audit found missing.
        for name in ("ambient_mode_on", "wake_word_mode_status",
                     "wake_resume_answer_then_quiet"):   # multi-line lambda
            self.assertIn(name, self.scanned, f"{name} missing — lambda "
                          "registrations have vanished from the index again")
        reg = self.scanned["ambient_mode_on"]
        self.assertEqual(reg.symbol, "_act_ambient_mode_set",
                         "lambda body call must resolve to the real handler")
        # ... and that handler must be a resolvable def somewhere the index's
        # def_index looks (monolith, core/, skills/) — no `?` location.
        defs = self.mono
        for p in (glob.glob(os.path.join(_PROJECT, "core", "*.py"))
                  + _skill_sources()):
            defs += _read(p)
        self.assertRegex(defs, r"def _act_ambient_mode_set\(",
                         "handler symbol must resolve to a real def location")


class SkillCompletenessInvariant(unittest.TestCase):
    """Per-skill invariant: every Constant string that feeds the register()
    actions param — direct subscript keys, keys of dict literals piped in via
    .update()/.items() loops, and tuple/list alias-loop literals — must appear
    in the shared scanner's output for that file."""

    @staticmethod
    def _independent_names(tree, param):
        names: set[str] = set()
        targets = {param, "actions"}
        # dict literal vars in the file: var -> keys
        dict_keys: dict[str, set[str]] = {}
        for node in ast.walk(tree):
            tgt = dval = None
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and isinstance(node.value, ast.Dict)):
                tgt, dval = node.targets[0].id, node.value
            elif (isinstance(node, ast.AnnAssign)
                    and isinstance(node.target, ast.Name)
                    and isinstance(node.value, ast.Dict)):
                tgt, dval = node.target.id, node.value
            if tgt:
                dict_keys.setdefault(tgt, set()).update(
                    k.value for k in dval.keys
                    if isinstance(k, ast.Constant) and isinstance(k.value, str))

        def assigns_target(n):
            return any(isinstance(b, ast.Assign) and len(b.targets) == 1
                       and isinstance(b.targets[0], ast.Subscript)
                       and isinstance(b.targets[0].value, ast.Name)
                       and b.targets[0].value.id in targets
                       for b in ast.walk(n))

        for node in ast.walk(tree):
            # direct: actions["name"] = ...
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Subscript)
                    and isinstance(node.targets[0].value, ast.Name)
                    and node.targets[0].value.id in targets
                    and isinstance(node.targets[0].slice, ast.Constant)
                    and isinstance(node.targets[0].slice.value, str)):
                names.add(node.targets[0].slice.value)
            # actions.update(<literal or dict var>)
            elif (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "update"
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id in targets and node.args):
                a0 = node.args[0]
                if isinstance(a0, ast.Dict):
                    names.update(k.value for k in a0.keys
                                 if isinstance(k, ast.Constant)
                                 and isinstance(k.value, str))
                elif isinstance(a0, ast.Name):
                    names.update(dict_keys.get(a0.id, ()))
            elif isinstance(node, ast.For) and assigns_target(node):
                it = node.iter
                # for k, v in handlers.items(): actions[k] = v
                if (isinstance(it, ast.Call)
                        and isinstance(it.func, ast.Attribute)
                        and it.func.attr == "items"
                        and isinstance(it.func.value, ast.Name)):
                    names.update(dict_keys.get(it.func.value.id, ()))
                # for alias in ("a", "b"): actions[alias] = fn
                elif isinstance(it, (ast.Tuple, ast.List)):
                    names.update(e.value for e in it.elts
                                 if isinstance(e, ast.Constant)
                                 and isinstance(e.value, str))
        return names

    def test_every_skill_registration_is_scanned(self):
        checked = 0
        problems = []
        for path in sorted(_skill_sources()):
            src = _read(path)
            try:
                tree = ast.parse(src)
            except SyntaxError:
                continue
            param = None
            for node in tree.body:
                if (isinstance(node, ast.FunctionDef)
                        and node.name == "register" and node.args.args):
                    param = node.args.args[0].arg
                    break
            if param is None:
                continue
            checked += 1
            rel = os.path.relpath(path, _PROJECT).replace("\\", "/")
            expected = self._independent_names(tree, param)
            scanned = set(RS.scan_registrations(src, filename=rel))
            missing = expected - scanned
            if missing:
                problems.append(f"{rel}: {sorted(missing)}")
        self.assertGreater(checked, 20, "sanity: skill sweep found no skills")
        self.assertFalse(
            problems,
            "shared scanner blind to skill registrations (every consumer — "
            "ACTION_INDEX, audit checks — inherits the hole):\n  "
            + "\n  ".join(problems))

    def test_concrete_regression_names(self):
        # The exact names the audit proved absent from docs/ACTION_INDEX.md.
        browser = set(RS.scan_file(
            os.path.join(_PROJECT, "skills", "browser_agent.py")))
        for name in ("browser_task", "browser_reset_profile"):
            self.assertIn(name, browser,
                          "dict-plus-loop registration lost again")
        air = set(RS.scan_file(
            os.path.join(_PROJECT, "skills", "kinect_air_mouse.py")))
        for name in ("mouse_control_on", "hand_mouse_off"):
            self.assertIn(name, air, "tuple-alias-loop registration lost again")


class CommittedIndexEndToEnd(unittest.TestCase):
    """The committed docs/ACTION_INDEX.md (what the web panel serves) must
    carry the rows the old regexes dropped."""

    def _index(self):
        p = os.path.join(_PROJECT, "docs", "ACTION_INDEX.md")
        if not os.path.exists(p):
            self.skipTest("ACTION_INDEX.md not present in this checkout")
        return _read(p)

    def test_regression_rows_present(self):
        idx = self._index()
        for name in ("`browser_task`", "`ambient_mode_on`"):
            self.assertIn(name, idx,
                          f"{name} row missing — regenerate with "
                          "`python tools/gen_action_index.py` (and if it is "
                          "still missing, the scanner regressed)")


class AntiForkGuard(unittest.TestCase):
    """The registration-scan rule has exactly ONE home. Both consumers must
    import tools/registration_scan.py, and neither may re-grow the old regex
    (this codebase's #1 bug class is the stale duplicate)."""

    def _src(self, name):
        return _read(os.path.join(_TOOLS, name))

    def test_both_consumers_import_the_shared_scanner(self):
        for name in ("gen_action_index.py", "audit_codebase.py"):
            self.assertIn("import registration_scan", self._src(name),
                          f"tools/{name} no longer uses the shared scanner — "
                          "the rule is forking again")

    def test_old_regexes_are_dead(self):
        lower = "actions" + "\\[\\s*"    # the old `actions\[\s*['"]...` regex
        upper = "ACTIONS" + "\\[\\s*"    # the old `ACTIONS\[\s*"..."` regex
        for name in ("gen_action_index.py", "audit_codebase.py"):
            src = self._src(name)
            self.assertNotIn(lower, src,
                             f"tools/{name} regrew the old registration regex")
            self.assertNotIn(upper, src,
                             f"tools/{name} regrew the old registration regex")


if __name__ == "__main__":
    unittest.main()
