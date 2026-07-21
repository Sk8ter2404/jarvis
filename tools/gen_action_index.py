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
"""
import os, re, sys, glob, collections

_TOOLS = os.path.dirname(os.path.abspath(__file__))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)
import registration_scan

ROOT = os.path.dirname(_TOOLS)
MONO = os.path.join(ROOT, "bobert_companion.py")


def read(p):
    with open(p, encoding="utf-8", errors="replace") as f:
        return f.read()


mono = read(MONO)

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

# ---- 2. skill/core-registered actions ----
# PACKAGE SKILLS (2026-07-14 audit). `skills/*.py` misses PACKAGE skills whose
# registration lives in skills/<name>/__init__.py (e.g. holographic_overlay) — a
# whole package's actions rendered with `?` locations. Add the package inits.
_SKILL_SOURCES = (
    glob.glob(os.path.join(ROOT, "skills", "*.py"))
    + glob.glob(os.path.join(ROOT, "skills", "*", "__init__.py"))
    + glob.glob(os.path.join(ROOT, "core", "*.py"))
)
skill_actions = {}
for p in _SKILL_SOURCES:
    base = os.path.relpath(p, ROOT).replace("\\", "/")
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

# ---- 3. handler def locations ----
def_index = {}
for p in ([MONO]
          + glob.glob(os.path.join(ROOT, "core", "*.py"))
          + glob.glob(os.path.join(ROOT, "skills", "*.py"))
          + glob.glob(os.path.join(ROOT, "skills", "*", "__init__.py"))):
    base = os.path.relpath(p, ROOT).replace("\\", "/")
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


prompts = read(os.path.join(ROOT, "core", "prompts.py"))
tests = {p: read(p) for p in glob.glob(os.path.join(ROOT, "tests", "**", "*.py"), recursive=True)}


def has_example(name):
    return bool(re.search(r'ACTION:\s*' + re.escape(name) + r'\b', prompts))


def test_refs(name):
    return [os.path.relpath(p, ROOT).replace("\\", "/") for p, t in tests.items()
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
w("> references it. Regenerate with `python tools/gen_action_index.py`.\n")
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

docs = os.path.join(ROOT, "docs")
os.makedirs(docs, exist_ok=True)
outp = os.path.join(docs, "ACTION_INDEX.md")
with open(outp, "w", encoding="utf-8", newline="") as f:
    f.write("\n".join(out) + "\n")
print(f"wrote {outp}: {c['total']} actions, {len(by_handler)} handler groups, "
      f"VERBATIM={c['verbatim']} INFORMATIVE={c['informative']} neither={c['neither']}")
