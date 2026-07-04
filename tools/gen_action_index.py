#!/usr/bin/env python3
"""Machine-verified action index generator for JARVIS.

Parses the monolith ACTIONS dict + INFORMATIVE_ACTIONS / SPEAK_RESULT_VERBATIM_ACTIONS
sets, plus every ``actions["…"] =`` registration across skills/ and core/, and
writes docs/ACTION_INDEX.md. Never imports the monolith (textual parse only), so
it is safe to run against a live tree. Run: ``python tools/gen_action_index.py``.
"""
import os, re, glob, collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MONO = os.path.join(ROOT, "bobert_companion.py")


def read(p):
    with open(p, encoding="utf-8", errors="replace") as f:
        return f.read()


mono = read(MONO)
mono_lines = mono.splitlines()

# ---- 1. Monolith ACTIONS dict: name -> handler symbol ----
actions = {}


def scan_region(start_ln, end_ln):
    for ln in range(start_ln, min(end_ln, len(mono_lines))):
        line = mono_lines[ln]
        for m in re.finditer(r'"([a-zA-Z_][a-zA-Z0-9_]*)"\s*:\s*([A-Za-z_][A-Za-z0-9_\.]*)\s*,', line):
            actions[m.group(1)] = m.group(2)
        am = re.match(r'\s*ACTIONS\[\s*"([a-zA-Z_0-9]+)"\s*\]\s*=\s*([A-Za-z_][A-Za-z0-9_\.]*)', line)
        if am:
            actions[am.group(1)] = am.group(2)


def block_end(open_ln):
    depth = 0
    started = False
    for ln in range(open_ln, len(mono_lines)):
        depth += mono_lines[ln].count("{") - mono_lines[ln].count("}")
        if "{" in mono_lines[ln]:
            started = True
        if started and depth <= 0:
            return ln
    return open_ln


# locate "ACTIONS = {" and "ACTIONS.update({" blocks by content (line-drift safe)
for idx, line in enumerate(mono_lines):
    if re.match(r'\s*ACTIONS(\.update\(\{|\s*=\s*\{)', line):
        scan_region(idx, block_end(idx) + 1)
    am = re.match(r'\s*ACTIONS\[\s*"([a-zA-Z_0-9]+)"\s*\]\s*=\s*([A-Za-z_][A-Za-z0-9_\.]*)', line)
    if am:
        actions[am.group(1)] = am.group(2)


def extract_set(name):
    m = re.search(name + r'\s*[:=].*?\{(.*?)\}', mono, re.S)
    return set(re.findall(r'"([a-zA-Z_0-9]+)"', m.group(1))) if m else set()


informative = extract_set("INFORMATIVE_ACTIONS")
verbatim = extract_set("SPEAK_RESULT_VERBATIM_ACTIONS")

# ---- 2. skill/core-registered actions ----
skill_actions = {}
for p in glob.glob(os.path.join(ROOT, "skills", "*.py")) + glob.glob(os.path.join(ROOT, "core", "*.py")):
    base = os.path.relpath(p, ROOT).replace("\\", "/")
    for m in re.finditer(r'actions\[\s*[\'"]([a-zA-Z_0-9]+)[\'"]\s*\]\s*=\s*([A-Za-z_][A-Za-z0-9_\.]*)', read(p)):
        skill_actions[m.group(1)] = (base, m.group(2))

# ---- 3. handler def locations ----
def_index = {}
for p in [MONO] + glob.glob(os.path.join(ROOT, "core", "*.py")) + glob.glob(os.path.join(ROOT, "skills", "*.py")):
    base = os.path.relpath(p, ROOT).replace("\\", "/")
    for ln, line in enumerate(read(p).splitlines(), 1):
        dm = re.match(r'\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(', line)
        if dm and dm.group(1) not in def_index:
            def_index[dm.group(1)] = f"{base}:{ln}"


def handler_loc(sym):
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
