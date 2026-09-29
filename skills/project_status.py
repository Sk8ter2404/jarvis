"""Answer "what am I working on?" from the owner's OWN project list.

WHY THIS SKILL EXISTS (live sweep, 2026-09-29): "what am I working on lately"
was answered from the auto-learned topics and projects in memory — and those
were Whisper mis-hearings of TV / room audio, so JARVIS confidently named a
"mystery" and a "project" that never existed. Learned topics are now framed as
unverified hints (core/topic_hygiene.py, build_system_prompt); this skill is
the GROUNDED source the question deserves: a short list the owner keeps
himself.

DATA — ``data/projects_status.json``. Private: ``data/`` is gitignored, and the
path is resolved through ``core.paths`` so a staging / test process (and
``JARVIS_DATA_DIR``) reads its own copy. Schema::

    {
      "projects": [
        {
          "name":    "Garden shed",           # required; entries without one are skipped
          "status":  "Frame is up; roof next", # free text ("done" / "finished" /
                                               #   "complete" / "archived" ... = finished)
          "updated": "2026-09-20",             # YYYY-MM-DD the status was last true
          "next":    "Order shingles"          # optional next step
        }
      ]
    }

Unknown keys are ignored. A generic example with fake content ships as
``tools/projects_status.example.json`` — copy it to ``data/projects_status.json``
and edit it.

Rules:
  * No file, an empty list, or no named projects -> say there is no project
    list yet. NEVER invent one and never fall back to the learned topics.
  * A file that is not valid JSON / not the schema -> say it could not be read.
  * Finished projects are not read out as current work.
  * ``project_status, <name>`` reports just that project.

Action: ``project_status`` — a finished sentence, spoken verbatim.
"""
from __future__ import annotations

import datetime
import json
import os
import re

# Finished sentence -> spoken as-is (load_skills folds this into the monolith's
# SPEAK_RESULT_VERBATIM_ACTIONS). Without it the answer is computed and dropped.
SPEAK_VERBATIM_ACTIONS = ("project_status",)

_FILE_NAME = "projects_status.json"
_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MAX_READ = 5
_MAX_FIELD = 160
_FINISHED = frozenset({
    "done", "complete", "completed", "finished", "archived", "shelved",
    "cancelled", "canceled", "abandoned", "shipped",
})
# Words an argument may carry that do not name a project ("lately", "all").
_NOT_A_NAME = frozenset({
    "all", "everything", "my", "the", "projects", "project", "status", "list",
    "lately", "recently", "now", "current", "currently", "today", "work",
    "working", "on", "am", "i", "what", "doing", "been", "have", "of", "a",
    "please",
})

NO_LIST = ("I don't have a project list yet, sir, so I won't guess what "
           "you're working on.")
UNREADABLE = ("I couldn't read your project list, sir, so I won't guess what "
              "you're working on.")


def _path() -> str:
    """data/projects_status.json for THIS process (staging/test aware)."""
    try:
        from core.paths import data_file
        return data_file(_FILE_NAME, create_dir=False)
    except Exception:   # pragma: no cover - core.paths is in-tree
        return os.path.join(_PROJECT_DIR, "data", _FILE_NAME)


def _today() -> datetime.date:
    return datetime.date.today()


def _text(v) -> str:
    if not isinstance(v, str):
        return ""
    t = " ".join(v.split())
    return t[:_MAX_FIELD].rstrip()


def _date(v):
    if not isinstance(v, str):
        return None
    try:
        return datetime.date.fromisoformat(v.strip()[:10])
    except ValueError:
        return None


def load_projects(path: str | None = None):
    """``(projects, state)``: state is 'ok', 'absent', 'empty' or
    'unreadable'. ``projects`` is a list of normalised dicts (name, status,
    updated: date|None, next, finished: bool); None when unreadable."""
    path = path or _path()
    if not os.path.isfile(path):
        return [], "absent"
    try:
        with open(path, "r", encoding="utf-8-sig") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None, "unreadable"
    if not isinstance(data, dict):
        return None, "unreadable"
    raw = data.get("projects", [])
    if not isinstance(raw, list):
        return None, "unreadable"
    out = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        name = _text(entry.get("name"))
        if not name:
            continue
        status = _text(entry.get("status"))
        out.append({
            "name": name,
            "status": status,
            "updated": _date(entry.get("updated")),
            "next": _text(entry.get("next")),
            "finished": status.lower().strip(" .!") in _FINISHED,
        })
    return out, ("ok" if out else "empty")


def _when(updated, today) -> str:
    if updated is None:
        return ""
    days = (today - updated).days
    if days == 0:
        return "updated today"
    if days == 1:
        return "updated yesterday"
    if 1 < days < 14:
        return f"updated {days} days ago"
    stamp = f"{updated:%B} {updated.day}"
    if updated.year != today.year:
        stamp += f", {updated.year}"
    return f"last updated {stamp}"


def _lead_lower(text: str) -> str:
    """'Frame is up' -> 'frame is up' mid-sentence; 'PCB ordered' stays."""
    if len(text) > 1 and text[0].isupper() and text[1].islower():
        return text[0].lower() + text[1:]
    return text


def _sentence(p, today, sir: bool = False) -> str:
    status = _lead_lower(p["status"]) or "no status recorded"
    when = _when(p["updated"], today)
    line = f"{p['name']}{', sir' if sir else ''}: {status}"
    if when:
        line += f" ({when})"
    if p["next"] and not p["finished"]:
        line += f"; next, {_lead_lower(p['next'])}"
    return line.rstrip(".") + "."


def _by_recency(projects):
    return sorted(projects,
                  key=lambda p: p["updated"] or datetime.date.min,
                  reverse=True)


def _words(text: str) -> list:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _word_hit(w: str, name_words) -> bool:
    """``w`` is one of the name's words, or the same word with a short
    ending ("shed" ~ "sheds", "feeders" ~ "feeder") — never a mere prefix
    ("car" is not "carburettor")."""
    return any(w == n or (min(len(w), len(n)) >= 3
                          and abs(len(w) - len(n)) <= 2
                          and (n.startswith(w) or w.startswith(n)))
               for n in name_words)


def _find(projects, query: str):
    """Projects whose name matches ``query``: the exact name, else every
    meaningful query word present in the name (whole words, not substrings,
    so "car" never matches "carburettor rebuild")."""
    q = " ".join(_words(query))
    exact = [p for p in projects if " ".join(_words(p["name"])) == q]
    if exact:
        return exact
    words = [w for w in q.split() if w not in _NOT_A_NAME]
    if not words:
        return []
    return [p for p in projects
            if all(_word_hit(w, _words(p["name"])) for w in words)]


def _is_name_query(query: str) -> bool:
    return any(w not in _NOT_A_NAME for w in _words(query))


def _count(n: int) -> str:
    return {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
            7: "seven", 8: "eight", 9: "nine", 10: "ten"}.get(n, str(n))


def project_status(arg: str = "") -> str:
    """Read back the owner's project list (or one project from it)."""
    projects, state = load_projects()
    if state == "unreadable":
        return UNREADABLE
    if state in ("absent", "empty"):
        return NO_LIST
    today = _today()
    query = (arg or "").strip()

    if query and _is_name_query(query):
        hits = _find(projects, query)
        if len(hits) == 1:
            return _sentence(hits[0], today, sir=True)
        if hits:
            return (f"{_count(len(hits)).capitalize()} projects match, sir. "
                    + " ".join(_sentence(p, today) for p in _by_recency(hits)))
        names = [p["name"] for p in _by_recency(projects)][:_MAX_READ]
        return (f"I don't see {query} on your project list, sir. It has "
                + _join(names) + ".")

    active = _by_recency([p for p in projects if not p["finished"]])
    finished = [p for p in projects if p["finished"]]
    if not active:
        return ("Everything on your project list is marked finished, sir, "
                "so nothing is active right now.")
    n = len(active)
    head = ("You're working on one project, sir. " if n == 1
            else f"You have {_count(n)} active projects, sir. ")
    body = " ".join(_sentence(p, today) for p in active[:_MAX_READ])
    tail = ""
    if n > _MAX_READ:
        tail += f" And {_count(n - _MAX_READ)} more on the list."
    if finished:
        k = len(finished)
        tail += (f" {_count(k).capitalize()} more "
                 f"{'is' if k == 1 else 'are'} marked finished.")
    return head + body + tail


def _join(names) -> str:
    names = list(names)
    if len(names) <= 1:
        return "".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


def register(actions):
    actions["project_status"] = project_status
    print("  [project_status] ready — action: project_status "
          "(reads data/projects_status.json).")
