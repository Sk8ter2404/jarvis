#!/usr/bin/env python3
"""Machine-verified action index generator for JARVIS.

Parses the monolith ACTIONS dict + INFORMATIVE_ACTIONS / SPEAK_RESULT_VERBATIM_ACTIONS
sets, plus every action registration across skills/ and core/, and writes
docs/ACTION_INDEX.md. Never imports the monolith (ast.parse only — textual), so
it is safe to run against a live tree. Run: ``python tools/gen_action_index.py``.

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
import collections
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


def build_index(root=ROOT, tracked=None):
    """Return ``(markdown_text, counts, n_groups)`` for the tree at ``root``,
    reading ONLY publishable (tracked) sources."""
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
            regs = registration_scan.scan_file(p, filename=base)
        except SyntaxError:
            continue   # unparseable file — nothing registerable to index
        for name, reg in regs.items():
            skill_actions[name] = (base, reg.symbol)

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
    tests = {p: read(p) for p in keep(glob.glob(
        os.path.join(root, "tests", "**", "*.py"), recursive=True))}

    def has_example(name):
        return bool(re.search(r'ACTION:\s*' + re.escape(name) + r'\b', prompts))

    def test_refs(name):
        return [_rel(p, root) for p, t in tests.items()
                if ('"' + name + '"') in t or ("'" + name + "'") in t]

    # ---- assemble rows ----
    rows = []
    for name in sorted(set(actions) | set(skill_actions)):
        if name in actions:
            sym, origin = actions[name], "monolith"
        else:
            origin, sym = skill_actions[name]
        speak = "VERBATIM" if name in verbatim else ("INFORMATIVE" if name in informative else "neither")
        rows.append({"action": name, "handler": sym, "loc": handler_loc(sym), "origin": origin,
                     "speak": speak, "example": has_example(name), "tests": test_refs(name)})

    by_handler = collections.OrderedDict()
    for r in sorted(rows, key=lambda r: (r["origin"] != "monolith", r["loc"], r["action"])):
        key = (r["origin"], r["loc"], r["speak"])
        g = by_handler.setdefault(key, {"aliases": [], "example": False, "tests": set()})
        g["aliases"].append(r["action"])
        g["example"] = g["example"] or r["example"]
        g["tests"].update(r["tests"])

    c = {"total": len(rows), "monolith": sum(r["origin"] == "monolith" for r in rows),
         "skill": sum(r["origin"] != "monolith" for r in rows),
         "verbatim": sum(r["speak"] == "VERBATIM" for r in rows),
         "informative": sum(r["speak"] == "INFORMATIVE" for r in rows),
         "neither": sum(r["speak"] == "neither" for r in rows),
         "no_example": sum(not r["example"] for r in rows),
         "no_tests": sum(not r["tests"] for r in rows)}

    def esc(s):
        return str(s).replace("|", "\\|")

    out = []
    w = out.append
    w("# JARVIS Action Index\n")
    w("> Machine-verified inventory of every dispatchable voice action — its handler, whether its")
    w("> result is spoken (INFORMATIVE = LLM restates / VERBATIM = spoken as-is / neither = only the")
    w("> preamble is heard), whether it has a `core/prompts.py` routing example, and whether a test")
    w("> references it. Regenerate with `python tools/gen_action_index.py`.")
    w("> Only git-TRACKED sources are indexed: locally-installed private skills (gitignored) never")
    w("> appear here. The web dashboard's Actions tab reads the LIVE registry instead.\n")
    w("## Summary\n")
    w("| metric | count |\n|---|---|")
    w(f"| Total registered actions (incl. aliases) | {c['total']} |")
    w(f"| — monolith `ACTIONS` dict | {c['monolith']} |")
    w(f"| — skill / core registered | {c['skill']} |")
    w(f"| VERBATIM speak set | {c['verbatim']} |")
    w(f"| INFORMATIVE speak set | {c['informative']} |")
    w(f"| neither set | {c['neither']} |")
    w(f"| no `prompts.py` example | {c['no_example']} |")
    w(f"| no test reference | {c['no_tests']} |\n")
    w("A result in **neither** set is spoken only if the handler self-speaks; otherwise the answer")
    w("is dropped. That is correct for side-effect actions but is the recurring \"logged but never")
    w("voiced\" bug for read-outs — see the audit that seeded the 2026-07 read-out completeness sweep.\n")
    w("## Full index\n")
    w("Aliases sharing a handler are collapsed. `ex?` = has a prompts.py `[ACTION: …]` example.\n")
    w("| action(s) | handler | speak | ex? | tests |")
    w("|---|---|---|:--:|:--:|")
    badge = {"VERBATIM": "**VERBATIM**", "INFORMATIVE": "*INFORMATIVE*", "neither": "neither"}
    for (origin, loc, speak), g in by_handler.items():
        al = ", ".join(f"`{esc(a)}`" for a in sorted(g["aliases"]))
        w(f"| {al} | `{esc(loc)}` | {badge[speak]} | {'yes' if g['example'] else '—'} | {len(g['tests'])} |")
    return "\n".join(out) + "\n", c, len(by_handler)


def main(root=ROOT):
    text, c, groups = build_index(root)
    docs = os.path.join(root, "docs")
    os.makedirs(docs, exist_ok=True)
    outp = os.path.join(docs, "ACTION_INDEX.md")
    with open(outp, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    print(f"wrote {outp}: {c['total']} actions, {groups} handler groups, "
          f"VERBATIM={c['verbatim']} INFORMATIVE={c['informative']} neither={c['neither']}")
    return outp


if __name__ == "__main__":
    main()
