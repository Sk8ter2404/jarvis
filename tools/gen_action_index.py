#!/usr/bin/env python3
"""Machine-verified action index generator for JARVIS.

Parses the monolith ACTIONS dict + INFORMATIVE_ACTIONS / SPEAK_RESULT_VERBATIM_ACTIONS
sets, plus every action registration across skills/ and core/, and writes
docs/ACTION_INDEX.md. Never imports the monolith (ast.parse only — textual), so
it is safe to run against a live tree. Run: ``python tools/gen_action_index.py``.

Two coverage columns per action (2026-10-02): ``spoken note`` (speak-set
membership, including a skill's module-level SPEAK_VERBATIM_ACTIONS /
INFORMATIVE_ACTIONS / SELF_VOICED_ACTIONS declaration) and ``tested`` (the name
is a string literal in a tracked tests/**/*.py file).
tests/test_action_index_coverage.py fails CI when the committed index's action
names drift from ``collect_rows`` and when the untested count grows.

Registration discovery lives in tools/registration_scan.py — the ONE shared
home for that rule (audit 2026-07-21: the two regexes that used to live here
missed every lambda-valued monolith entry and every dict-plus-loop / tuple-
alias-loop skill registration, so ~39 live actions — the whole browser agent
included — were absent from the web panel's Actions inventory).

PUBLIC-REPO PRIVACY (2026-09-30). docs/ACTION_INDEX.md is a TRACKED file in a
PUBLIC repo, but this generator used to glob every ``skills/*.py`` ON DISK. On
the owner's machine that includes the gitignored personal skills (.gitignore
keeps them out of the repo precisely because their action names embed a
specific person or a one-off personal event), so regenerating the index on the
live tree copied those private action names — and their file:line locations —
straight back into a public document. So every source this generator reads is
now filtered through ``publishable_sources``: only files git TRACKS are
indexed (``git ls-files``), which drops both gitignored files and untracked
work-in-progress. When git cannot answer (no git on PATH, not a checkout) the
fallback is the repo's own ``.gitignore`` patterns, applied conservatively,
with a warning — never "index everything on disk".
"""
import ast
import fnmatch
import glob
import os
import re
import subprocess
import sys

_TOOLS = os.path.dirname(os.path.abspath(__file__))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)
import registration_scan

ROOT = os.path.dirname(_TOOLS)


def read(p):
    with open(p, encoding="utf-8", errors="replace") as f:
        return f.read()


def _rel(p, root):
    return os.path.relpath(p, root).replace("\\", "/")


# ---- which files may appear in a PUBLIC document -----------------------------

def git_tracked_files(root):
    """Repo-relative POSIX paths git tracks under ``root`` (``git ls-files``),
    or None when git cannot answer (git missing, not a checkout, timeout).

    Tracked == in the index: a gitignored file is never in it, and neither is
    an untracked new file, which is exactly the set a public doc may name."""
    try:
        r = subprocess.run(["git", "-C", root, "ls-files", "-z"],
                           capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0:
        return None
    return {p.decode("utf-8", "replace") for p in r.stdout.split(b"\0") if p}


def _gitignore_patterns(root):
    """Plain (non-negated) patterns from ``root``/.gitignore. Negations are
    ignored on purpose: the fallback may only ever EXCLUDE more, never less."""
    pats = []
    try:
        text = read(os.path.join(root, ".gitignore"))
    except OSError:
        return pats
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("!"):
            continue
        pats.append(s)
    return pats


def _ignored_by_patterns(rel, patterns):
    """Conservative .gitignore match for the no-git fallback: a pattern matches
    the full relative path, its basename, or any leading directory."""
    parts = rel.split("/")
    for pat in patterns:
        p = pat.strip("/")
        if not p:
            continue
        if fnmatch.fnmatch(rel, p) or fnmatch.fnmatch(parts[-1], p):
            return True
        for i in range(1, len(parts)):
            if fnmatch.fnmatch("/".join(parts[:i]), p):
                return True
    return False


def publishable_sources(paths, root, tracked=None):
    """Filter ``paths`` (absolute) down to the ones a PUBLIC doc may name.

    ``tracked`` is the ``git_tracked_files`` set (looked up when omitted). With
    git available: only tracked files survive. Without it: every file matching
    a .gitignore pattern is dropped, and a warning says the index could still
    include untracked, non-ignored work in progress."""
    if tracked is None:
        tracked = git_tracked_files(root)
    if tracked is not None:
        return [p for p in paths if _rel(p, root) in tracked]
    print("  [gen_action_index] WARNING: git could not list tracked files; "
          "falling back to .gitignore patterns - review the diff before "
          "committing docs/ACTION_INDEX.md", file=sys.stderr)
    pats = _gitignore_patterns(root)
    return [p for p in paths if not _ignored_by_patterns(_rel(p, root), pats)]


def _skill_and_core_sources(root):
    # PACKAGE SKILLS (2026-07-14 audit). `skills/*.py` misses PACKAGE skills whose
    # registration lives in skills/<name>/__init__.py (e.g. holographic_overlay) — a
    # whole package's actions rendered with `?` locations. Add the package inits.
    return (
        glob.glob(os.path.join(root, "skills", "*.py"))
        + glob.glob(os.path.join(root, "skills", "*", "__init__.py"))
        + glob.glob(os.path.join(root, "core", "*.py"))
    )


# ---- coverage columns: spoken note + test reference --------------------------

# A skill's module-level speak-set declarations. When the skill loads,
# bobert_companion._collect_skill_speak_sets folds them into the monolith's
# sets; this reads the same declarations from SOURCE.
SKILL_SPEAK_DECLARATIONS = ("SPEAK_VERBATIM_ACTIONS", "INFORMATIVE_ACTIONS",
                            "SELF_VOICED_ACTIONS")


def declared_speak_names(tree):
    """``{attr: {names}}`` for a parsed skill's module-level speak-set
    declarations: a tuple / list / set literal of strings, optionally wrapped
    in one ``frozenset(...)`` / ``set(...)`` / ``tuple(...)`` / ``list(...)``."""
    attrs = set(SKILL_SPEAK_DECLARATIONS)
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        hit = {t.id for t in targets if isinstance(t, ast.Name)} & attrs
        if not hit:
            continue
        if (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
                and value.func.id in ("frozenset", "set", "tuple", "list")
                and len(value.args) == 1):
            value = value.args[0]
        if not isinstance(value, (ast.Tuple, ast.List, ast.Set)):
            continue
        names = {e.value.strip() for e in value.elts
                 if isinstance(e, ast.Constant) and isinstance(e.value, str)
                 and e.value.strip()}
        for attr in hit:
            out.setdefault(attr, set()).update(names)
    return out


def spoken_note(name, verbatim, informative, self_voiced):
    """The action's spoken-note label. Same precedence and case rule as the
    runtime lookups and the web panel's live labels (tools/web_interface.py):
    self-voiced names are stored lower-cased."""
    if name in verbatim:
        return "VERBATIM"
    if name in informative:
        return "INFORMATIVE"
    if name.lower() in self_voiced:
        return "SELF-VOICED"
    return "neither"


def string_literals(path):
    """Every str constant in the Python file at ``path``, or None when it does
    not parse. The BYTES are parsed, so a UTF-8 BOM is fine."""
    try:
        with open(path, "rb") as f:
            tree = ast.parse(f.read())
    except (OSError, SyntaxError, ValueError):
        return None
    return {n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)}


def collect_rows(root=ROOT, tracked=None):
    """One dict per registered action, read ONLY from publishable (tracked)
    sources: ``action``, ``handler``, ``loc``, ``origin``, ``speak`` (the
    spoken note), ``example``, ``tests`` (tracked tests/ files naming it as a
    string literal) and ``tested``. Sorted by action name."""
    if tracked is None:
        tracked = git_tracked_files(root)

    def keep(paths):
        # Order preserved (NOT sorted): skill_actions is last-wins and
        # def_index first-wins, so the historical glob order decides ties.
        return publishable_sources(list(paths), root, tracked=tracked)

    mono_path = os.path.join(root, "bobert_companion.py")
    mono = read(mono_path)

    # ---- 1. Monolith ACTIONS dict: name -> handler symbol ----
    # Shared AST scanner: catches the dict literal, every ACTIONS.update({...})
    # block, ACTIONS["x"] = assigns (top-level or inside functions), and resolves
    # lambda-wrapped handlers to their callee (ambient_mode_on → _act_ambient_mode_set).
    actions = {name: reg.symbol for name, reg in registration_scan.scan_registrations(
        mono, filename="bobert_companion.py", targets=("ACTIONS",)).items()}

    def extract_set(name):
        m = re.search(name + r'\s*[:=].*?\{(.*?)\}', mono, re.S)
        return set(re.findall(r'"([a-zA-Z_0-9]+)"', m.group(1))) if m else set()

    informative = extract_set("INFORMATIVE_ACTIONS")
    verbatim = extract_set("SPEAK_RESULT_VERBATIM_ACTIONS")
    self_voiced = set()   # the monolith's own set starts empty; skills fill it

    # ---- 2. skill/core-registered actions (TRACKED sources only) ----
    skill_actions = {}
    for p in keep(_skill_and_core_sources(root)):
        base = _rel(p, root)
        # Shared AST scanner: direct subscript assigns, dict-plus-loop /
        # actions.update(handlers) (browser_agent's 12 actions), tuple-alias loops
        # (kinect_air_mouse's mouse_control_on family), and the #45 alias form
        # `actions["a"] = actions["b"]` (resolved to the target's factory symbol
        # via a fixed point inside the scanner, so chained aliases land right).
        try:
            tree = ast.parse(read(p))
        except SyntaxError:
            continue   # unparseable file — nothing registerable to index
        for name, reg in registration_scan.scan_registrations(
                tree, filename=base).items():
            skill_actions[name] = (base, reg.symbol)
        if base.startswith("skills/"):
            declared = declared_speak_names(tree)
            verbatim |= declared.get("SPEAK_VERBATIM_ACTIONS", set())
            informative |= declared.get("INFORMATIVE_ACTIONS", set())
            self_voiced |= {n.lower() for n in
                            declared.get("SELF_VOICED_ACTIONS", ())}

    # ---- 3. handler def locations (TRACKED sources only) ----
    def_index = {}
    for p in [mono_path] + keep(
            glob.glob(os.path.join(root, "core", "*.py"))
            + glob.glob(os.path.join(root, "skills", "*.py"))
            + glob.glob(os.path.join(root, "skills", "*", "__init__.py"))):
        base = _rel(p, root)
        for ln, line in enumerate(read(p).splitlines(), 1):
            dm = re.match(r'\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(', line)
            if dm and dm.group(1) not in def_index:
                def_index[dm.group(1)] = f"{base}:{ln}"

    def handler_loc(sym):
        # Inline lambda / expression handlers carry their own site as location
        # (registration_scan emits `lambda@<file>:<line>` when the lambda body
        # isn't a plain call it can resolve to a def).
        if sym.startswith(("lambda@", "expr@")):
            return sym.split("@", 1)[1]
        return def_index.get(sym.split(".")[-1], "?")

    prompts = read(os.path.join(root, "core", "prompts.py"))
    # ---- 4. test references: string literals in TRACKED tests/**/*.py ----
    test_literals = {}
    for p in keep(glob.glob(os.path.join(root, "tests", "**", "*.py"),
                            recursive=True)):
        lits = string_literals(p)
        if lits is not None:   # a file that does not parse tests nothing
            test_literals[_rel(p, root)] = lits

    def has_example(name):
        return bool(re.search(r'ACTION:\s*' + re.escape(name) + r'\b', prompts))

    def test_refs(name):
        return sorted(rel for rel, lits in test_literals.items() if name in lits)

    # ---- assemble rows ----
    rows = []
    for name in sorted(set(actions) | set(skill_actions)):
        if name in actions:
            sym, origin = actions[name], "monolith"
        else:
            origin, sym = skill_actions[name]
        refs = test_refs(name)
        rows.append({"action": name, "handler": sym, "loc": handler_loc(sym),
                     "origin": origin,
                     "speak": spoken_note(name, verbatim, informative, self_voiced),
                     "example": has_example(name), "tests": refs,
                     "tested": bool(refs)})
    return rows


def build_index(root=ROOT, tracked=None):
    """Return ``(markdown_text, counts, n_handlers)`` for the tree at ``root``,
    reading ONLY publishable (tracked) sources."""
    rows = collect_rows(root, tracked=tracked)
    n_handlers = len({(r["origin"], r["loc"]) for r in rows})

    c = {"total": len(rows), "monolith": sum(r["origin"] == "monolith" for r in rows),
         "skill": sum(r["origin"] != "monolith" for r in rows),
         "verbatim": sum(r["speak"] == "VERBATIM" for r in rows),
         "informative": sum(r["speak"] == "INFORMATIVE" for r in rows),
         "self_voiced": sum(r["speak"] == "SELF-VOICED" for r in rows),
         "neither": sum(r["speak"] == "neither" for r in rows),
         "no_example": sum(not r["example"] for r in rows),
         "tested": sum(r["tested"] for r in rows),
         "no_tests": sum(not r["tested"] for r in rows)}

    def esc(s):
        return str(s).replace("|", "\\|")

    out = []
    w = out.append
    w("# JARVIS Action Index\n")
    w("> Machine-checked inventory of every dispatchable voice action — its handler, its spoken")
    w("> note, whether it has a `core/prompts.py` routing example, and whether a test names it.")
    w("> Regenerate with `python tools/gen_action_index.py`. CI fails when the action NAMES here")
    w("> differ from what the generator finds (tests/test_action_index_coverage.py), so adding,")
    w("> renaming or deleting an action means regenerating this file in the same change.")
    w("> Only git-TRACKED sources are indexed: locally-installed private skills (gitignored) never")
    w("> appear here. The web dashboard's Actions tab reads the LIVE registry instead.")
    w(">")
    w("> **spoken note** — the action's declared speak routing, the convention the runtime uses:")
    w("> **VERBATIM** = in `SPEAK_RESULT_VERBATIM_ACTIONS` (bobert_companion.py) or a tracked")
    w("> skill's module-level `SPEAK_VERBATIM_ACTIONS`, so the result string is spoken as-is;")
    w("> *INFORMATIVE* = in `INFORMATIVE_ACTIONS` (bobert_companion.py or a skill's module-level")
    w("> declaration), so a follow-up LLM round restates the result; SELF-VOICED = in a skill's")
    w("> module-level `SELF_VOICED_ACTIONS`, so the action does all of its own talking; neither =")
    w("> no spoken note, so only the preamble is heard unless the handler speaks for itself. The")
    w("> skill declarations are folded into the monolith's sets at load time by")
    w("> `_collect_skill_speak_sets`; this file reads them from source. A set patched at run time")
    w("> (inside `register()`, or `register_self_voiced`) is invisible here and shows as neither.")
    w(">")
    w("> **tested** — `yes` when the action name is a Python string literal (exact match) in a")
    w("> git-tracked `tests/**/*.py` file. A ratchet in tests/test_action_index_coverage.py stops")
    w("> the untested count from growing.\n")
    w("## Summary\n")
    w("| metric | count |\n|---|---|")
    w(f"| Total registered actions (incl. aliases) | {c['total']} |")
    w(f"| — monolith `ACTIONS` dict | {c['monolith']} |")
    w(f"| — skill / core registered | {c['skill']} |")
    w(f"| tested | {c['tested']} |")
    w(f"| **untested** (no test names it) | {c['no_tests']} |")
    w(f"| spoken note: VERBATIM | {c['verbatim']} |")
    w(f"| spoken note: INFORMATIVE | {c['informative']} |")
    w(f"| spoken note: SELF-VOICED | {c['self_voiced']} |")
    w(f"| **no spoken note** (neither) | {c['neither']} |")
    w(f"| no `prompts.py` example | {c['no_example']} |\n")
    w("A result with no spoken note is correct for side-effect actions but is the recurring")
    w("\"logged but never voiced\" bug for read-outs — see the audit that seeded the 2026-07")
    w("read-out completeness sweep.\n")
    w("## Full index\n")
    w("One row per action, sorted by name; aliases share their handler's location.")
    w("`ex?` = has a prompts.py `[ACTION: …]` example.\n")
    w("| action | handler | spoken note | ex? | tested |")
    w("|---|---|---|:--:|:--:|")
    badge = {"VERBATIM": "**VERBATIM**", "INFORMATIVE": "*INFORMATIVE*",
             "SELF-VOICED": "SELF-VOICED", "neither": "neither"}
    for r in rows:
        w(f"| `{esc(r['action'])}` | `{esc(r['loc'])}` | {badge[r['speak']]} | "
          f"{'yes' if r['example'] else '—'} | {'yes' if r['tested'] else 'no'} |")
    return "\n".join(out) + "\n", c, n_handlers


def main(root=ROOT):
    text, c, groups = build_index(root)
    docs = os.path.join(root, "docs")
    os.makedirs(docs, exist_ok=True)
    outp = os.path.join(docs, "ACTION_INDEX.md")
    with open(outp, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    print(f"wrote {outp}: {c['total']} actions, {groups} handlers, "
          f"VERBATIM={c['verbatim']} INFORMATIVE={c['informative']} "
          f"SELF-VOICED={c['self_voiced']} neither={c['neither']}, "
          f"untested={c['no_tests']}")
    return outp


if __name__ == "__main__":
    main()
