"""core/legacy_memory.py — the flat bobert_memory.json store (load / save / empty).

This is the long-term "what JARVIS knows about its owner" file that gets dumped
into the system prompt every turn. Three functions extracted from the monolith:

  _empty_memory()        — the schema with sensible empty defaults.
  load_memory()          — read + forward-migrate missing keys; empty when the
                           file is absent, an empty MARKED unreadable when it
                           exists but cannot be parsed (2026-10-01).
  save_memory(memory)    — ATOMIC write (tempfile + fsync + os.replace) so a
                           crash/power-loss mid-write can't truncate the live
                           store to empty/half-written JSON (2026-05-30 audit);
                           refuses a dict that came from an unreadable store.

The target path and the write lock are INJECTED via configure() rather than
imported, because bobert_companion picks the path at boot (prod
bobert_memory.json vs the isolated data_staging copy for blue/green) and owns
the shared _memory_lock that other writers (learn_from_turn, the ambient
extractor) also hold. A prod default is set so load/save still work if a caller
hits them before configure() runs. merge_memory() stays in bobert_companion —
it orchestrates load→dedupe→save and depends on monolith config (LOCATION,
MAX_FACTS/PROJECTS/TOPICS) and the memory_guards filters.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import time

# Prod default (project root / bobert_memory.json) so load/save work even
# before configure() runs. configure() overrides this for the staging role.
_MEMORY_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bobert_memory.json"
)
_LOCK: "threading.RLock" = threading.RLock()

# Marker on the dict load_memory() returns for a store that EXISTS but cannot
# be read. save_memory() refuses such a dict, so no writer can persist it.
_LOAD_FAILED_KEY = "_load_failed"


class MemoryStoreUnreadable(RuntimeError):
    """save_memory() was handed the empty stand-in for an unreadable store."""


# (mtime_ns, size) of the last unreadable file already copied aside and
# reported, so a store that stays broken is backed up and logged once, not on
# every one of the dozens of load_memory() calls a session makes.
_unreadable_noted: list = [None]


def configure(memory_file: str, lock=None) -> None:
    """Point the store at THIS role's memory file (prod vs staging) and, when
    given, share the caller's lock so writes here serialise with the monolith's
    other _memory_lock holders. Called once at boot."""
    global _MEMORY_FILE, _LOCK
    if memory_file:
        _MEMORY_FILE = memory_file
    if lock is not None:
        _LOCK = lock


def _empty_memory() -> dict:
    return {
        "first_meeting":      time.strftime("%Y-%m-%d"),
        "conversation_count": 0,
        "facts":              [],   # ["User's name is Alex", ...]
        "projects":           [],   # ["Building REPO animatronic robot"]
        "topics":             [],   # [{"date": "...", "topic": "..."}]
        "sessions":           [],   # [{"date": "...", "summary": "..."}]
        # Rotation tracker for MCU_PHRASES: maps intent bucket → last phrase
        # used on the prior turn, so build_system_prompt can ask the LLM to
        # avoid repeating it. Populated by record_phrase_use() after each reply.
        "last_used_phrase_by_intent": {},
    }


def _note_unreadable(err: Exception) -> None:
    """Copy an unreadable store aside (backups/memory_unreadable_<ts>.json)
    and say so loudly, once per distinct file state. Never raises."""
    try:
        st = os.stat(_MEMORY_FILE)
        key = (st.st_mtime_ns, st.st_size)
    except Exception:
        key = None
    if key is not None and _unreadable_noted[0] == key:
        return
    _unreadable_noted[0] = key
    where = "NOT backed up"
    try:
        _dir = os.path.join(
            os.path.dirname(os.path.abspath(_MEMORY_FILE)) or ".", "backups")
        os.makedirs(_dir, exist_ok=True)
        dest = os.path.join(
            _dir, f"memory_unreadable_{time.strftime('%Y%m%d_%H%M%S')}.json")
        shutil.copy2(_MEMORY_FILE, dest)
        where = f"copied to backups/{os.path.basename(dest)}"
    except Exception as e:
        where = f"NOT backed up ({type(e).__name__})"
    print(f"  [legacy_memory] UNREADABLE memory store "
          f"({type(err).__name__}: {err}) -- {where}; refusing to save over "
          f"it until it is fixed")


def load_memory() -> dict:
    if not os.path.exists(_MEMORY_FILE):
        return _empty_memory()
    # An EXISTING file that cannot be parsed is not "no memory" (2026-10-01).
    # Every writer does a whole-file load -> mutate -> save, so returning a
    # plain empty schema here let the next reply's phrase-rotation write (or
    # the boot conversation_count bump) silently replace all 120 facts,
    # projects, topics, sessions and quarantine after one hand-edit typo. The
    # stand-in is marked so save_memory refuses it, and the bad file is copied
    # aside. utf-8-sig: a valid file re-saved with a BOM (Windows PowerShell
    # 5.1 Out-File, Notepad "UTF-8 with BOM") is not a broken one.
    try:
        with open(_MEMORY_FILE, encoding="utf-8-sig") as f:
            mem = json.load(f)
        if not isinstance(mem, dict):
            raise ValueError(f"top level is {type(mem).__name__}, not an object")
    except Exception as e:
        _note_unreadable(e)
        mem = _empty_memory()
        mem[_LOAD_FAILED_KEY] = True
        return mem

    # Migrate old schema (only had facts + sessions): backfill any missing keys.
    base = _empty_memory()
    for k, v in base.items():
        if k not in mem:
            mem[k] = v

    # Type-coerce known keys to the schema's expected types. A corrupted store
    # (hand-edit, half-merged write, bad sync) can leave e.g. facts as a STRING
    # instead of a list — build_system_prompt then does `for f in mem["facts"]`,
    # iterating it CHARACTER BY CHARACTER into the cloud system prompt (token
    # bloat / garbage) or KeyErrors on the dict-valued topics/sessions. Reset
    # any list field that isn't a list to [] so the prompt builder gets sane
    # input; log once so the corruption isn't silently swallowed.
    for k in ("facts", "projects", "topics", "sessions"):
        if not isinstance(mem.get(k), list):
            print(f"  [legacy_memory] '{k}' was {type(mem.get(k)).__name__}, "
                  f"not list — resetting to [] (corrupt memory store)")
            mem[k] = []
    if not isinstance(mem.get("last_used_phrase_by_intent"), dict):
        print("  [legacy_memory] 'last_used_phrase_by_intent' was "
              f"{type(mem.get('last_used_phrase_by_intent')).__name__}, "
              "not dict — resetting to {} (corrupt memory store)")
        mem["last_used_phrase_by_intent"] = {}
    return mem


def save_memory(memory: dict) -> None:
    # ATOMIC write: serialise to a temp file in the same directory, fsync, then
    # os.replace() over the target. A plain open(...,"w") TRUNCATES the live
    # file first, so a crash / exception / power-loss mid-write left
    # bobert_memory.json EMPTY or half-written — a wiped/corrupt memory store
    # (and it's dumped verbatim into the system prompt every turn). os.replace
    # is atomic on both Windows and POSIX. 2026-05-30 file audit.
    #
    # A dict load_memory() marked unreadable is refused (2026-10-01): writing
    # it would replace the owner's whole store with an empty schema. Every
    # read-modify-write caller already catches a failed save and logs it.
    if isinstance(memory, dict) and memory.get(_LOAD_FAILED_KEY):
        raise MemoryStoreUnreadable(
            "the memory store on disk could not be read; not overwriting it")
    with _LOCK:
        _dir = os.path.dirname(os.path.abspath(_MEMORY_FILE)) or "."
        _fd, _tmp = tempfile.mkstemp(dir=_dir, prefix=".mem_", suffix=".tmp")
        try:
            with os.fdopen(_fd, "w", encoding="utf-8") as f:
                json.dump(memory, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(_tmp, _MEMORY_FILE)
        except Exception:
            try:
                if os.path.exists(_tmp):
                    os.remove(_tmp)
            except Exception:
                pass
            raise
