#!/usr/bin/env python3
"""Audit the auto-learned TOPICS and PROJECTS already stored in memory.

WHY THIS EXISTS (2026-09-29)
============================
The write gate in ``core/topic_hygiene.py`` stops NEW mis-heard topics and
projects (TV / room audio that Whisper turned into words) from reaching the
system prompt. It cannot un-learn the ones stored before it existed, and
those are exactly what JARVIS was volunteering ("you mentioned a project
involving ..."). This tool finds them with the SAME heuristics the gate uses —
``topic_hygiene.find_suspects``, one home, so the audit and the gate can never
disagree — and can set them aside, all of them or one at a time.

It NEVER deletes. The owner's standing rule is "clean up = tidy, never
delete". Every mode below is a MOVE inside the store file, done in one atomic
write, and nothing in the ``"quarantined"`` section is rendered into the
prompt:

  (default)             DRY RUN. Lists every suspect as ``[kind #N]`` with its
                        reason. Writes nothing.
  --quarantine          MOVE every suspect into ``"quarantined"``.
  --quarantine --only "<exact label>" / --pick topic:N,project:M
                        MOVE just those items (flagged or not — the owner
                        decides). ``--only`` / ``--pick`` WITHOUT --quarantine
                        is a dry run of that selection.
  --list-quarantined    Show what is in quarantine, as ``[quarantined #N]``.
  --restore             MOVE everything in quarantine back where it came from
                        (topics in time order, projects to their old slot).
  --restore --only "<exact label>" / --pick quarantined:N
                        MOVE just those back.

``--only`` and ``--pick`` may be repeated; ``--pick`` also takes a
comma-separated list. Indexes are the ones the LATEST dry run /
--list-quarantined printed: quarantining shifts them, so list again before a
second per-item run. An unknown label or index refuses the whole run.

What makes an entry suspect (details in core/topic_hygiene.py):
  1. its label is MOSTLY not real words, with at least one nonce-shaped token
     (4+ letters, not an ACRONYM). Known words: the ~23k-word bundled lexicon
     (with inflections), place names, tech acronyms, words the owner used in
     2+ logged turns, and words used in 2+ of the store's own ``facts`` (or
     written there as an acronym);
  2. it is not corroborated: it appears in fewer than two stored topics AND the
     owner's own logged utterances (memory/voice_commands.jsonl) mention it in
     fewer than two separate turns. The reason says whether NO logged turn
     uses its words or only ONE does (the evidence is word-based, so an
     abstract label can be real and still have none). Needs that log;
     without it only rule 1 runs, because "no log" must not read as "never
     said it".

The semantic long-term store (data/long_term_memory/facts.json) also mirrors
every learned project, and keeps it after the MAX_PROJECTS trim drops it from
bobert_memory.json. Its project facts are judged by the same rules and
REPORTED (report only) — that store is edited only through
core.long_term_memory, never by hand.

Usage
-----
    python tools/audit_learned_topics.py --store PATH/bobert_memory.json
    python tools/audit_learned_topics.py --store ... --quarantine --pick topic:3
    python tools/audit_learned_topics.py --store ... --quarantine --only "Some label"
    python tools/audit_learned_topics.py --store ... --list-quarantined
    python tools/audit_learned_topics.py --store ... --restore --pick quarantined:0

Stop JARVIS before any writing mode: the running process loads and re-saves
this file on every learned turn, so a write racing it can be overwritten.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

if __package__ in (None, ""):  # direct `python tools/audit_learned_topics.py`
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import topic_hygiene as th  # noqa: E402
from core.atomic_io import _atomic_write_json  # noqa: E402

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_STORE = os.path.join(_PROJECT_DIR, "bobert_memory.json")
OWNER_SELECTED = "owner-selected (not flagged by the audit)"


class SelectionError(ValueError):
    """A --only / --pick that names nothing (or something malformed)."""


def default_vocab_log(store: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(store)),
                        "memory", "voice_commands.jsonl")


def default_ltm_facts(store: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(store)),
                        "data", "long_term_memory", "facts.json")


def load_store(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a JSON object")
    return data


def _quarantine_list(memory: dict) -> list:
    q = memory.get(th.QUARANTINE_KEY)
    if not isinstance(q, list):
        q = []
        memory[th.QUARANTINE_KEY] = q
    return q


def _item_text(kind: str, item) -> str:
    if kind == "topic":
        return item.get("topic", "") if isinstance(item, dict) else ""
    return item if isinstance(item, str) else ""


def live_items(memory: dict) -> list:
    """Every surfaced item as ``{"kind", "index", "text"}``."""
    out = []
    for kind, key in (("topic", "topics"), ("project", "projects")):
        items = memory.get(key)
        for i, item in enumerate(items if isinstance(items, list) else []):
            text = _item_text(kind, item)
            if text:
                out.append({"kind": kind, "index": i, "text": text})
    return out


def _parse_picks(picks) -> list:
    """['topic:3,project:1', 'quarantined:0'] -> [('topic', 3), ...]."""
    out = []
    for spec in picks or ():
        for part in str(spec).split(","):
            part = part.strip()
            if not part:
                continue
            kind, sep, num = part.partition(":")
            kind = kind.strip().lower()
            if not sep or kind not in ("topic", "project", "quarantined"):
                raise SelectionError(
                    f"bad --pick {part!r}: use topic:N, project:N or "
                    "quarantined:N")
            try:
                out.append((kind, int(num)))
            except ValueError:
                raise SelectionError(f"bad --pick {part!r}: N must be a number")
    return out


def select_live(memory: dict, suspects: list, *, only=(), picks=()) -> list:
    """The live items named by ``--only`` / ``--pick``, each with the audit's
    reason when it was flagged (else OWNER_SELECTED). Raises SelectionError
    if any label or index matches nothing — a partial run is how the wrong
    item gets moved."""
    items = live_items(memory)
    reasons = {(s["kind"], s["index"]): s["reason"] for s in suspects}
    chosen: dict = {}
    for label in only or ():
        want = str(label).strip()
        hits = [it for it in items if it["text"].strip() == want]
        if not hits:
            raise SelectionError(f"--only {label!r}: no topic or project has "
                                 "exactly that label")
        for it in hits:
            chosen[(it["kind"], it["index"])] = it
    by_pos = {(it["kind"], it["index"]): it for it in items}
    for kind, idx in _parse_picks(picks):
        if kind == "quarantined":
            raise SelectionError("quarantined:N only works with --restore")
        if (kind, idx) not in by_pos:
            raise SelectionError(f"--pick {kind}:{idx}: no such {kind}")
        chosen[(kind, idx)] = by_pos[(kind, idx)]
    return [dict(it, reason=reasons.get(key, OWNER_SELECTED))
            for key, it in sorted(chosen.items())]


def select_quarantined(memory: dict, *, only=(), picks=()) -> set:
    """Quarantine indexes named by ``--only`` (item label) / ``--pick
    quarantined:N``. Raises SelectionError when one matches nothing."""
    q = memory.get(th.QUARANTINE_KEY)
    q = q if isinstance(q, list) else []
    chosen = set()
    for label in only or ():
        want = str(label).strip()
        hits = [i for i, rec in enumerate(q) if isinstance(rec, dict)
                and _item_text(rec.get("kind"), rec.get("item")).strip() == want]
        if not hits:
            raise SelectionError(f"--only {label!r}: nothing in quarantine has "
                                 "exactly that label")
        chosen.update(hits)
    for kind, idx in _parse_picks(picks):
        if kind != "quarantined":
            raise SelectionError(f"--restore takes quarantined:N, not {kind}:N")
        if not 0 <= idx < len(q):
            raise SelectionError(f"--pick quarantined:{idx}: no such entry")
        chosen.add(idx)
    return chosen


def quarantine(memory: dict, suspects: list, *, now: str | None = None) -> dict:
    """Return a COPY of ``memory`` with every listed item (``suspects``: dicts
    with kind / index / reason) moved into the quarantine section — item kept
    whole, plus kind / old index / reason / timestamp. Input not modified."""
    out = copy.deepcopy(memory)
    now = now or time.strftime("%Y-%m-%dT%H:%M:%S")
    q = _quarantine_list(out)
    by_kind: dict = {"topic": set(), "project": set()}
    for s in suspects:
        by_kind.setdefault(s["kind"], set()).add(s["index"])
    for kind, key in (("topic", "topics"), ("project", "projects")):
        items = out.get(key)
        if not isinstance(items, list) or not by_kind.get(kind):
            continue
        reasons = {s["index"]: s["reason"] for s in suspects
                   if s["kind"] == kind}
        kept = []
        for i, item in enumerate(items):
            if i in by_kind[kind]:
                q.append({"kind": kind, "item": item, "index": i,
                          "reason": reasons.get(i, ""),
                          "quarantined_at": now})
            else:
                kept.append(item)
        out[key] = kept
    return out


def _topic_ts(entry) -> float | None:
    try:
        return float(entry.get("ts"))
    except (AttributeError, TypeError, ValueError):
        return None


def restore(memory: dict, which=None) -> tuple:
    """Return ``(new_memory, restored_count)``: a COPY with the quarantined
    entries (all of them, or the indexes in ``which``) moved back into their
    lists. Topics go back in time order, projects to their old slot; an item
    already live again is not duplicated (its record is retired, the item
    itself is live). Malformed records are kept, never dropped."""
    out = copy.deepcopy(memory)
    q = out.get(th.QUARANTINE_KEY)
    if not isinstance(q, list) or not q:
        return out, 0
    topics = out.get("topics") if isinstance(out.get("topics"), list) else []
    projects = (out.get("projects")
                if isinstance(out.get("projects"), list) else [])
    left = []
    restored = 0
    for i, rec in enumerate(q):
        if which is not None and i not in which:
            left.append(rec)
            continue
        if not isinstance(rec, dict) or rec.get("kind") not in ("topic",
                                                                "project"):
            left.append(rec)        # unknown shape: keep it, never drop it
            continue
        item = rec.get("item")
        if rec["kind"] == "project":
            if not isinstance(item, str):
                left.append(rec)    # malformed record: keep, never drop
                continue
            if item not in projects:
                # Back to its old slot (records are in list order, so a full
                # restore reproduces the original order exactly).
                try:
                    pos = min(int(rec.get("index")), len(projects))
                except (TypeError, ValueError):
                    pos = len(projects)
                projects.insert(max(pos, 0), item)
            restored += 1
            continue
        if not isinstance(item, dict):
            left.append(rec)
            continue
        if item not in topics:
            ts = _topic_ts(item)
            pos = len(topics)
            if ts is not None:
                for j, t in enumerate(topics):
                    other = _topic_ts(t)
                    if other is not None and other > ts:
                        pos = j
                        break
            topics.insert(pos, item)
        restored += 1
    out["topics"] = topics
    out["projects"] = projects
    out[th.QUARANTINE_KEY] = left
    return out, restored


def ltm_suspects(path: str, memory: dict, owner_texts) -> list:
    """Project facts in the semantic long-term store that the SAME rules flag.

    merge_memory mirrors every learned project there, and that store keeps
    them after bobert_memory.json's MAX_PROJECTS trim drops them — so it is
    judged on its own, against the stored topics, facts and the owner-turn
    log. Read-only: returns ``[{"id", "text", "reason"}, ...]``."""
    if not path or not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return []
    rows = [e for e in (data if isinstance(data, list) else [])
            if isinstance(e, dict) and "project" in (e.get("tags") or [])
            and isinstance(e.get("text"), str) and e["text"].strip()]
    if not rows:
        return []
    probe = {"topics": memory.get("topics") or [],
             "facts": memory.get("facts") or [],
             "projects": [e["text"] for e in rows]}
    return [{"id": rows[s["index"]].get("id", "?"), "text": s["text"],
             "reason": s["reason"]}
            for s in th.find_suspects(probe, owner_texts)
            if s["kind"] == "project"]


def _print_items(items: list) -> None:
    for s in items:
        print(f"  [{s['kind']:<7} #{s['index']:>3}] {s['text']!r}")
        print(f"      reason: {s['reason']}")


def _print_quarantined(memory: dict) -> int:
    q = memory.get(th.QUARANTINE_KEY)
    q = q if isinstance(q, list) else []
    if not q:
        print("quarantine is empty")
        return 0
    print(f"{len(q)} quarantined entr{'y' if len(q) == 1 else 'ies'}:")
    for i, rec in enumerate(q):
        if not isinstance(rec, dict):
            print(f"  [quarantined #{i:>3}] (malformed record, kept as-is)")
            continue
        kind = rec.get("kind", "?")
        text = _item_text(kind, rec.get("item"))
        print(f"  [quarantined #{i:>3}] {kind}: {text!r} "
              f"(since {rec.get('quarantined_at', '?')})")
        print(f"      reason: {rec.get('reason', '')}")
    return len(q)


def _write(store: str, memory: dict) -> None:
    _atomic_write_json(store, memory, indent=2)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Audit learned topics/projects (dry run by default; "
                    "never deletes).")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--quarantine", action="store_true",
                      help="move suspects (or just the --only/--pick "
                           "selection) into the store's quarantine section")
    mode.add_argument("--restore", action="store_true",
                      help="move quarantined entries back (all, or the "
                           "--only/--pick selection)")
    mode.add_argument("--list-quarantined", action="store_true",
                      help="show what is in quarantine (read-only)")
    ap.add_argument("--only", action="append", default=[], metavar="LABEL",
                    help="select the item(s) with exactly this label "
                         "(repeatable)")
    ap.add_argument("--pick", action="append", default=[], metavar="SPEC",
                    help="select by index: topic:N,project:M (or "
                         "quarantined:N with --restore); repeatable")
    ap.add_argument("--store", default=DEFAULT_STORE,
                    help="path to bobert_memory.json (default: %(default)s)")
    ap.add_argument("--vocab-log", default=None,
                    help="owner voice-command log (default: "
                         "<store dir>/memory/voice_commands.jsonl)")
    ap.add_argument("--ltm-facts", default=None,
                    help="semantic-store facts.json, report only (default: "
                         "<store dir>/data/long_term_memory/facts.json)")
    args = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass

    store = os.path.abspath(args.store)
    if not os.path.isfile(store):
        print(f"no memory store at {store} - nothing to audit")
        return 1
    try:
        memory = load_store(store)
    except (OSError, ValueError) as e:
        print(f"could not read {store}: {e}")
        return 1
    selecting = bool(args.only or args.pick)

    if args.list_quarantined:
        _print_quarantined(memory)
        return 0

    if args.restore:
        try:
            which = (select_quarantined(memory, only=args.only,
                                        picks=args.pick)
                     if selecting else None)
        except SelectionError as e:
            print(f"refused, nothing written: {e}")
            return 2
        new, n = restore(memory, which)
        if not n:
            print(f"nothing quarantined in {store} - nothing to restore")
            return 0
        _write(store, new)
        print(f"restored {n} entr{'y' if n == 1 else 'ies'} from the "
              f"quarantine section of {store}")
        return 0

    vocab_log = args.vocab_log or default_vocab_log(store)
    owner_texts = th.read_owner_texts(vocab_log)
    suspects = th.find_suspects(memory, owner_texts)
    try:
        selection = (select_live(memory, suspects, only=args.only,
                                 picks=args.pick) if selecting else None)
    except SelectionError as e:
        print(f"refused, nothing written: {e}")
        return 2

    n_topics = len(memory.get("topics") or [])
    n_projects = len(memory.get("projects") or [])
    already = len(memory.get(th.QUARANTINE_KEY) or [])
    print(f"learned-topic audit - store: {store}")
    if owner_texts is None:
        print(f"owner-turn log: NOT FOUND ({vocab_log}) - only the non-word "
              "rule runs; the corroboration rule is skipped")
    else:
        print(f"owner-turn log: {len(owner_texts)} logged turns ({vocab_log})")
    print(f"suspects: {sum(1 for s in suspects if s['kind'] == 'topic')} of "
          f"{n_topics} topics, "
          f"{sum(1 for s in suspects if s['kind'] == 'project')} of "
          f"{n_projects} projects"
          + (f"; {already} already quarantined" if already else ""))
    _print_items(suspects)

    ltm_path = args.ltm_facts or default_ltm_facts(store)
    ltm_hits = ltm_suspects(ltm_path, memory, owner_texts)
    if ltm_hits:
        print(f"semantic store (report only, NOT modified - edit it only "
              f"through core.long_term_memory): {len(ltm_hits)} suspect "
              f"project fact(s) in {ltm_path}:")
        for m in ltm_hits:
            print(f"  - {m['text']!r} ({m['id']})")
            print(f"      reason: {m['reason']}")

    targets = selection if selecting else suspects
    if selecting:
        print(f"selected {len(selection)} item(s):")
        _print_items(selection)

    if not args.quarantine:
        if targets:
            what = "the selection" if selecting else "these"
            print(f"DRY RUN - nothing written. To move {what} into the "
                  "store's quarantine section (reversible with --restore):")
            extra = "".join(f' --only "{o}"' for o in args.only) + \
                "".join(f" --pick {p}" for p in args.pick)
            print(f"  python tools/audit_learned_topics.py --store \"{store}\" "
                  f"--quarantine{extra}")
            if not selecting:
                print("  (or just some of them: add --pick topic:N,project:M "
                      "or --only \"<exact label>\")")
        else:
            print("DRY RUN - nothing written. No suspects.")
        return 0

    if not targets:
        print("no suspects - nothing to quarantine")
        return 0
    new = quarantine(memory, targets)
    _write(store, new)
    print(f"quarantined {len(targets)} entr"
          f"{'y' if len(targets) == 1 else 'ies'} into the "
          f"'{th.QUARANTINE_KEY}' section of {store} (nothing deleted; "
          "undo with --restore)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
