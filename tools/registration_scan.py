#!/usr/bin/env python3
"""Shared AST scanner for JARVIS action registrations — THE one home for the
"which action names does this file register, and with which handler?" rule.

Consumers:
  * tools/gen_action_index.py — builds docs/ACTION_INDEX.md, which
    tools/web_interface.py serves as the control panel's Actions inventory.
  * tools/audit_codebase.py — check 5 (collisions/arity), the skill profiles
    behind checks B/D/E, and the monolith bc_actions set in main().

History (audit 2026-07-21, "Action-index generator misses lambda- and
loop-registered actions"): this rule used to live as FOUR diverging copies —
two regexes in gen_action_index.py that missed every lambda-valued entry and
every dict-plus-loop registration (the whole browser agent), an AST pass in
audit_codebase._profile_skill that missed tuple-alias loops
(kinect_air_mouse), and a second regex trio in audit_codebase.main(). Fix a
registration form HERE and every consumer sees it; do not fork this logic.

Uses ast.parse only — never imports or executes the scanned file, so it is
safe to run against a live tree (the guarantee gen_action_index always had).

Recognised forms (targets: ``ACTIONS`` / ``actions`` by default, plus the
first parameter of any ``def register(...)`` in the scanned source):

    ACTIONS = {"name": fn, ...}                       # dict literal
    ACTIONS.update({"name": fn, ...})                 # update w/ literal
    ACTIONS["name"] = fn                              # subscript assign
    handlers = {...}; actions.update(handlers)        # update w/ named dict
    handlers = {...}
    for k, v in handlers.items(): actions[k] = v      # dict-plus-loop
    for alias in ("a", "b"): actions[alias] = fn      # tuple/list alias loop
    actions["a"] = actions["b"]                       # alias → fixed-point

Each entry is a ``Registration(lineno, symbol, kind)``:

  * ``kind="direct"`` — Name/Attribute RHS; ``symbol`` is the dotted handler.
  * ``kind="lambda"`` — lambda RHS; if the body is a plain call, ``symbol`` is
    the callee (so ``ambient_mode_on`` maps to ``_act_ambient_mode_set``),
    else ``lambda@<filename>:<lineno>`` (its own site IS the location).
  * ``kind="call"`` — factory call RHS; ``symbol`` is the FACTORY's dotted
    name (a location anchor — the real handler is the returned closure, so
    arity checks must not apply to it).
  * ``kind="alias"`` — ``actions["a"] = actions["b"]``; ``symbol`` is the
    target's symbol after fixed-point resolution, or ``?alias:<target>`` if
    the target is unknown (the action NAME must never vanish).
  * ``kind="expr"`` — anything else; ``symbol`` is ``expr@<filename>:<lineno>``.
"""
import ast
import os
from typing import NamedTuple

DEFAULT_TARGETS = ("ACTIONS", "actions")

# Symbol prefixes that do NOT name a def anywhere — consumers must not try to
# resolve them as function names (they either embed their own file:line or
# mark an unresolvable alias).
OPAQUE_SYMBOL_PREFIXES = ("lambda@", "expr@", "?alias:")


class Registration(NamedTuple):
    lineno: int    # line of the registration site (dict key / assign / loop)
    symbol: str    # handler symbol — see module docstring
    kind: str      # "direct" | "lambda" | "call" | "alias" | "expr"


def _dotted(node):
    """Name/Attribute chain → dotted string (``mod.sub.fn``), else None."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _handler_symbol(value, filename):
    """(symbol, kind) for a registration RHS (see module docstring)."""
    sym = _dotted(value)
    if sym:
        return sym, "direct"
    if isinstance(value, ast.Lambda):
        if isinstance(value.body, ast.Call):
            callee = _dotted(value.body.func)
            if callee:
                return callee, "lambda"
        return f"lambda@{filename}:{value.lineno}", "lambda"
    if isinstance(value, ast.Call):          # actions["x"] = make_handler(...)
        callee = _dotted(value.func)
        if callee:
            return callee, "call"
    return f"expr@{filename}:{value.lineno}", "expr"


def _subscript_assigns_on(node, target_names):
    """Yield every single-target Assign under ``node`` whose target is a
    subscript on one of ``target_names``."""
    for b in ast.walk(node):
        if (isinstance(b, ast.Assign) and len(b.targets) == 1
                and isinstance(b.targets[0], ast.Subscript)
                and isinstance(b.targets[0].value, ast.Name)
                and b.targets[0].value.id in target_names):
            yield b


def scan_registrations(tree_or_source, filename="<src>", targets=DEFAULT_TARGETS):
    """Return ``{action_name: Registration(lineno, symbol, kind)}`` for every
    action registration visible in the given source text (or pre-parsed ast
    tree).

    First registration wins for a repeated name (mirrors the runtime
    ``if name not in actions`` guard most skills use). Raises SyntaxError if
    given unparseable source — callers that scan arbitrary files should guard.
    """
    tree = (tree_or_source if isinstance(tree_or_source, ast.AST)
            else ast.parse(tree_or_source))
    target_names = set(targets)
    for node in ast.walk(tree):
        if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == "register" and node.args.args):
            target_names.add(node.args.args[0].arg)

    # Pass 1 — dict-literal variables that may later feed the registry via
    # `.update(name)` or `for k, v in name.items(): actions[k] = v`.
    dict_vars = {}   # var name → [(key, value_node, key_lineno), ...]
    for node in ast.walk(tree):
        tgt = dval = None
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Dict)):
            tgt, dval = node.targets[0].id, node.value
        elif (isinstance(node, ast.AnnAssign)          # handlers: dict = {...}
                and isinstance(node.target, ast.Name)
                and isinstance(node.value, ast.Dict)):
            tgt, dval = node.target.id, node.value
        if tgt is None or tgt in target_names:
            continue
        entries = [(k.value, v, k.lineno)
                   for k, v in zip(dval.keys, dval.values)
                   if isinstance(k, ast.Constant) and isinstance(k.value, str)]
        if entries:
            dict_vars.setdefault(tgt, []).extend(entries)

    out = {}       # action name → Registration; first wins
    aliases = []   # (alias, target_key, lineno) from actions["a"] = actions["b"]

    def record(name, lineno, value_node):
        if name not in out:
            sym, kind = _handler_symbol(value_node, filename)
            out[name] = Registration(lineno, sym, kind)

    def record_dict(dval):
        for k, v in zip(dval.keys, dval.values):
            if isinstance(k, ast.Constant) and isinstance(k.value, str):
                record(k.value, k.lineno, v)

    # Pass 2 — the registrations themselves.
    for node in ast.walk(tree):
        # ACTIONS = {...}  (registry-target dict literal, incl. AnnAssign)
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in target_names
                and isinstance(node.value, ast.Dict)):
            record_dict(node.value)
        elif (isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id in target_names
                and isinstance(node.value, ast.Dict)):
            record_dict(node.value)
        # ACTIONS["x"] = fn   |   actions["a"] = actions["b"]   (alias)
        elif (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Subscript)
                and isinstance(node.targets[0].value, ast.Name)
                and node.targets[0].value.id in target_names):
            key = node.targets[0].slice
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                v = node.value
                if (isinstance(v, ast.Subscript)
                        and isinstance(v.value, ast.Name)
                        and v.value.id in target_names
                        and isinstance(v.slice, ast.Constant)
                        and isinstance(v.slice.value, str)):
                    aliases.append((key.value, v.slice.value, node.lineno))
                else:
                    record(key.value, node.lineno, v)
            # non-constant keys: loop-registered forms are handled below;
            # truly dynamic keys are unanalysable and skipped.
        # ACTIONS.update({...})  |  actions.update(handlers)
        elif (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "update"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in target_names
                and node.args):
            a0 = node.args[0]
            if isinstance(a0, ast.Dict):
                record_dict(a0)
            elif isinstance(a0, ast.Name):
                for k, v, ln in dict_vars.get(a0.id, ()):
                    record(k, ln, v)
        elif isinstance(node, ast.For):
            _scan_for_loop(node, target_names, dict_vars, record)

    # Aliases → fixed point, so chains (a=b, b=c) land on the real symbol.
    for _ in range(len(aliases) + 1):
        changed = False
        for alias, target, lineno in aliases:
            tgt = out.get(target)
            if tgt is not None and (alias not in out
                                    or out[alias].symbol != tgt.symbol):
                out[alias] = Registration(lineno, tgt.symbol, "alias")
                changed = True
        if not changed:
            break
    # An alias whose target is unknown still gets an entry — the action NAME
    # must never vanish from the index just because its handler is opaque.
    for alias, target, lineno in aliases:
        out.setdefault(alias, Registration(lineno, f"?alias:{target}", "alias"))
    return out


def _scan_for_loop(node, target_names, dict_vars, record):
    """Handle the two loop registration forms:
    ``for k, v in handlers.items(): actions[k] = v`` and
    ``for alias in ("a", "b"): actions[alias] = fn``."""
    assigns = [a for a in _subscript_assigns_on(node, target_names)]
    if not assigns:
        return
    it = node.iter
    # for k, v in <dictname>.items(): actions[k] = v
    if (isinstance(it, ast.Call) and isinstance(it.func, ast.Attribute)
            and it.func.attr == "items"
            and isinstance(it.func.value, ast.Name)):
        for k, v, ln in dict_vars.get(it.func.value.id, ()):
            record(k, ln, v)
        return
    # for alias in ("a", "b", ...): actions[alias] = fn
    if isinstance(it, (ast.Tuple, ast.List)) and isinstance(node.target, ast.Name):
        names = [e.value for e in it.elts
                 if isinstance(e, ast.Constant) and isinstance(e.value, str)]
        if not names:
            return
        loop_var = node.target.id
        for a in assigns:
            key = a.targets[0].slice
            if isinstance(key, ast.Name) and key.id == loop_var:
                for nm in names:
                    record(nm, a.lineno, a.value)


def scan_file(path, targets=DEFAULT_TARGETS, filename=None):
    """Convenience wrapper: read ``path`` (utf-8, errors replaced) and scan it.
    ``filename`` (for lambda@/expr@ symbols) defaults to the file's basename;
    pass a repo-relative path when the consumer renders locations."""
    with open(path, encoding="utf-8", errors="replace") as f:
        src = f.read()
    return scan_registrations(src, filename=filename or os.path.basename(path),
                              targets=targets)
