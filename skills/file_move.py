"""move_file / undo_last_move - move ONE of the owner's files between his
folders, with a dry run and a spoken yes (v3 W15, 2026-10-27).

Live 10-16/17 11:12:05: "take my resume to documents" fell back to a run_shell
move and failed. This is the real action.

  move_file       arg "<what> to <where>" (also "<what> from <folder> to
                  <where>", or explicit paths). The SOURCE is an explicit path,
                  else a file-name search of the owner's folders (and, when
                  that finds nothing, the personal file index search_my_files
                  uses), restricted to the allowed roots. More than one
                  plausible match -> "which one?" (up to three, by name and
                  folder); never a guess. The DESTINATION is a folder word
                  (documents, desktop, downloads, pictures, music, videos,
                  nextcloud [<folder>]), a folder name under those, or an
                  explicit path - an EXISTING directory under an allowed root.
  undo_last_move  moves the last moved file back, if it is still where it
                  was put.

SAFETY (each one has a test):
  * ALLOWED ROOTS: the user profile folder and D:\\Nextcloud. REFUSED: the
    Windows folder, Program Files, ProgramData, any AppData, C:\\JARVIS and
    JARVIS's own code / data folders, any iCloud / Apple folder, any junction
    or symlink on the way, hidden or system files and folders, drive roots.
  * Files only (a folder -> "I can only move single files for now, sir").
  * NEVER overwrites: a name already taken at the destination becomes
    "name (1).ext", and the question says so.
  * NEVER deletes: a same-drive move is one rename; a cross-drive move copies
    to a hidden temp name in the destination, checks the size, removes the
    original and renames the copy into place. Success is spoken only after
    the destination holds the same size and the source is gone.
  * A SPOKEN YES, through JARVIS's own confirm queue: bobert_companion.
    _jarvis_pushback asks confirm_question() below for the question (the file
    and both folders) and queues the action on _pending_confirmation; the yes
    runs it. confirm_question() leaves a PLAN; the action only ever executes a
    fresh plan, so a path that skips the question moves nothing.
  * Every move is journalled in data/file_moves.json (core.atomic_io).
"""
from __future__ import annotations

import os
import re
import shutil
import sys
import threading
import time

skill_utils = globals().get("skill_utils", {})

SPEAK_VERBATIM_ACTIONS = ("move_file", "undo_last_move")

PROMPT_EXAMPLES = (
    "FILE MOVES (skills/file_move.py - JARVIS asks before every move):\n"
    "  move_file      - move ONE file between the owner's folders. Arg is\n"
    "                   '<file> to <folder>'. Fire on 'move my resume to\n"
    "                   documents', 'take the invoice from downloads to the\n"
    "                   desktop'. -> [ACTION: move_file, resume to documents]\n"
    "                   Never use run_shell to move a file.\n"
    "  undo_last_move - put the last moved file back. Fire on 'undo that\n"
    "                   move', 'put it back'. -> [ACTION: undo_last_move]\n"
)

NEXTCLOUD_ROOT = r"D:\Nextcloud"
JOURNAL_NAME = "file_moves.json"
JOURNAL_KEEP = 50
PLAN_TTL_S = 120.0
SEARCH_MAX_DEPTH = 4
SEARCH_MAX_ENTRIES = 60000
SEARCH_MAX_S = 4.0
MAX_LISTED = 3

TERMINAL = "failed (final): "   # core.failure_markers.TERMINAL_FAILURE_PREFIX

# Folder words -> (label, Windows known-folder GUID, fallback name under home)
_KNOWN = {
    "downloads": ("Downloads", "{374DE290-123F-4565-9164-39C4925E467B}", "Downloads"),
    "desktop": ("Desktop", "{B4BFCC3A-DB2C-424C-B029-7FE99A87C641}", "Desktop"),
    "documents": ("Documents", "{FDD39AD0-238F-46AF-ADB4-6C85480369C7}", "Documents"),
    "pictures": ("Pictures", "{33E28130-4E1E-4676-835A-98395C3BC3BB}", "Pictures"),
    "music": ("Music", "{4BD8D571-6D19-48D3-BE97-422220080E43}", "Music"),
    "videos": ("Videos", "{18989B1D-99B5-455B-841C-AB7C74E4DDFC}", "Videos"),
}
_FOLDER_WORDS = {
    "downloads": "downloads", "download": "downloads", "download folder": "downloads",
    "desktop": "desktop",
    "documents": "documents", "document": "documents", "docs": "documents",
    "my documents": "documents",
    "pictures": "pictures", "picture": "pictures", "photos": "pictures",
    "music": "music",
    "videos": "videos", "video": "videos", "movies": "videos",
}
_DIR_WORDS_REFUSED = ("icloud", "apple", "mobilesync")
_STOP = {"my", "the", "a", "an", "file", "called", "named", "that", "this"}

FILE_ATTRIBUTE_HIDDEN = 0x2
FILE_ATTRIBUTE_SYSTEM = 0x4

_lock = threading.Lock()
_plans: dict = {}
_failures: dict = {}   # key -> (owner line, monotonic): a dry run that found nothing
FAILURE_MEMO_S = 15.0


# ── environment (each one a seam the tests replace) ──────────────────────
def _cfg(name, default):
    try:
        from core import config as cfg
        return getattr(cfg, name, default)
    except Exception:
        return default


def home_dir() -> str:
    return os.path.expanduser("~")


def _shell_folder(guid: str) -> str:
    """Windows' own path for a known folder (redirected Documents etc.), or ''."""
    if os.name != "nt":
        return ""
    try:
        import ctypes
        from ctypes import wintypes
        import uuid

        class _GUID(ctypes.Structure):
            _fields_ = [("Data1", wintypes.DWORD), ("Data2", wintypes.WORD),
                        ("Data3", wintypes.WORD), ("Data4", ctypes.c_ubyte * 8)]

        u = uuid.UUID(guid)
        g = _GUID(u.fields[0], u.fields[1], u.fields[2],
                  (ctypes.c_ubyte * 8)(*u.bytes[8:]))
        out = ctypes.c_wchar_p()
        if ctypes.windll.shell32.SHGetKnownFolderPath(
                ctypes.byref(g), 0, None, ctypes.byref(out)) != 0:
            return ""
        path = out.value or ""
        ctypes.windll.ole32.CoTaskMemFree(out)
        return path
    except Exception:
        return ""


def known_folders() -> dict:
    """{key: path} for the six folder words, existing folders only."""
    home = home_dir()
    out = {}
    for key, (_label, guid, fallback) in _KNOWN.items():
        p = _shell_folder(guid) or os.path.join(home, fallback)
        if p and os.path.isdir(p):
            out[key] = os.path.abspath(p)
    return out


def nextcloud_root() -> str:
    return NEXTCLOUD_ROOT if os.path.isdir(NEXTCLOUD_ROOT) else ""


def allowed_roots() -> list:
    roots = [os.path.abspath(home_dir())]
    nc = nextcloud_root()
    if nc:
        roots.append(os.path.abspath(nc))
    return roots


def refused_prefixes() -> list:
    """Absolute folders nothing may be moved out of or into."""
    env = os.environ
    out = [env.get("SystemRoot") or r"C:\Windows", r"C:\Windows",
           env.get("ProgramFiles") or r"C:\Program Files",
           env.get("ProgramFiles(x86)") or r"C:\Program Files (x86)",
           env.get("ProgramW6432") or "", env.get("ProgramData") or r"C:\ProgramData",
           r"C:\JARVIS"]
    try:
        from core import paths as _paths
        out.append(_paths.PROJECT_DIR)
        out.append(_paths.data_dir(create=False))
    except Exception:
        pass
    out.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return [os.path.abspath(p) for p in out if p]


def journal_path() -> str:
    try:
        from core import paths as _paths
        return _paths.data_file(JOURNAL_NAME)
    except Exception:
        return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "data", JOURNAL_NAME)


# ── path rules ───────────────────────────────────────────────────────────
def _nc(p: str) -> str:
    return os.path.normcase(os.path.abspath(p))


def _under(path: str, root: str) -> bool:
    p, r = _nc(path), _nc(root)
    return p == r or p.startswith(r.rstrip("\\/") + os.sep)


def _is_link(p: str) -> bool:
    try:
        if os.path.islink(p):
            return True
        isj = getattr(os.path, "isjunction", None)
        return bool(isj and isj(p))
    except Exception:
        return True


def _hidden_or_system(p: str) -> bool:
    if os.path.basename(p.rstrip("\\/")).startswith("."):
        return True
    try:
        attrs = getattr(os.stat(p, follow_symlinks=False), "st_file_attributes", 0)
    except OSError:
        return False
    return bool(attrs & (FILE_ATTRIBUTE_HIDDEN | FILE_ATTRIBUTE_SYSTEM))


def _is_drive_root(p: str) -> bool:
    a = os.path.abspath(p)
    return os.path.dirname(a) == a


def refusal(path: str) -> str:
    """Why ``path`` (a file to move or a folder to move into) is off limits,
    in the owner's words; '' when it is allowed. Never raises."""
    try:
        a = os.path.abspath(path)
        if _is_drive_root(a):
            return "I won't move anything into or out of a drive's top folder, sir."
        for pre in refused_prefixes():
            if _under(a, pre):
                return ("That's in a system or JARVIS folder, sir; I only move files "
                        "between your own folders.")
        roots = [r for r in allowed_roots() if _under(a, r)]
        if not roots:
            return ("That's outside your own folders, sir; I only move files in your "
                    "user folders and Nextcloud.")
        root = max(roots, key=len)
        rel = os.path.relpath(a, root)
        parts = [] if rel == "." else rel.split(os.sep)
        for i, part in enumerate(parts):
            low = part.lower()
            is_last = i == len(parts) - 1
            if low == "appdata":
                return "That's in AppData, sir; program files there are not mine to move."
            if not is_last or os.path.isdir(a):
                if any(w in low for w in _DIR_WORDS_REFUSED):
                    return "That's in an iCloud or Apple folder, sir; I never move Apple data."
        cur = root
        for part in parts:
            cur = os.path.join(cur, part)
            if _is_link(cur):
                return ("That path goes through a junction or shortcut folder, sir; "
                        "I won't move anything there.")
            if _hidden_or_system(cur):
                return "That's a hidden or system item, sir; I leave those alone."
        real = os.path.realpath(a)
        if _nc(real) != _nc(a):
            if not any(_under(real, r) for r in allowed_roots()):
                return "That really lives outside your own folders, sir."
        return ""
    except Exception as e:   # pragma: no cover - defensive
        return f"I couldn't check that path safely, sir ({e})."


# ── speech helpers ───────────────────────────────────────────────────────
def folder_label(folder: str) -> str:
    f = _nc(folder)
    for key, p in known_folders().items():
        if _nc(p) == f:
            return _KNOWN[key][0]
    nc = nextcloud_root()
    if nc and _nc(nc) == f:
        return "Nextcloud"
    if _nc(home_dir()) == f:
        return "your user folder"
    return os.path.basename(os.path.abspath(folder).rstrip("\\/")) or folder


def free_name(folder: str, name: str) -> str:
    """``name`` if nothing in ``folder`` has it, else 'stem (1).ext', (2)..."""
    if not os.path.lexists(os.path.join(folder, name)):
        return name
    stem, ext = os.path.splitext(name)
    for i in range(1, 1000):
        cand = f"{stem} ({i}){ext}"
        if not os.path.lexists(os.path.join(folder, cand)):
            return cand
    return ""


# ── argument parsing ─────────────────────────────────────────────────────
def _clean(s: str) -> str:
    s = (s or "").strip().strip(".!?,;").strip().strip("'\"").strip()
    return s


def parse_arg(arg: str):
    """'<what> [from <folder>] to <where>' -> (what, from_hint, where,
    whole_what); ('', '', '', '') when there is no ' to ' / ' into '.
    whole_what keeps any 'from / in' words, for a file whose NAME holds one
    ("notes in progress to documents")."""
    text = " ".join((arg or "").replace("\u2019", "'").split())
    text = re.sub(r"^(?:please\s+)?(?:move|take|put|send)\s+", "", text, flags=re.I)
    seps = list(re.finditer(r"\s(?:to|into)\s", text, flags=re.I))
    if not seps:
        return "", "", "", ""
    m = seps[-1]
    what, where = text[:m.start()], text[m.end():]
    whole = what
    hint = ""
    fm = re.search(r"\s(?:from|in|out of)\s+(?!.*\s(?:from|in|out of)\s)(.+)$", what, flags=re.I)
    if fm and not _looks_like_path(what):
        hint = fm.group(1)
        what = what[:fm.start()]
    return _clean(what), _clean(hint), _clean(where), _clean(whole)


def _looks_like_path(s: str) -> bool:
    s = (s or "").strip()
    return bool(re.match(r"^[A-Za-z]:[\\/]", s) or s.startswith(("\\\\", "~", "/")))


def _norm_words(s: str) -> str:
    return " ".join(re.sub(r"[_\-.()\[\],']+", " ", (s or "").lower()).split())


def _tokens(q: str) -> list:
    words = _norm_words(q).split()
    toks = [w for w in words if w not in _STOP]
    return toks or words


def name_matches(basename: str, toks: list) -> bool:
    if not toks:
        return False
    hay = " " + _norm_words(basename)
    return all(re.search(r"(?<![a-z0-9])" + re.escape(t), hay) for t in toks)


# ── resolution ───────────────────────────────────────────────────────────
def _dir_word(phrase: str) -> str:
    p = _norm_words(phrase)
    p = re.sub(r"^(?:the|my)\s+", "", p)
    p = re.sub(r"\s+(?:folder|directory)$", "", p)
    return p.strip()


def resolve_folder(phrase: str):
    """(path, '') for an existing allowed folder, else (None, owner line).
    A which-one question comes back as (None, question)."""
    raw = _clean(phrase)
    if not raw:
        return None, "Move it where, sir?"
    if _looks_like_path(raw):
        p = os.path.abspath(os.path.expanduser(raw))
        if not os.path.isdir(p):
            return None, f"There's no folder at {raw}, sir."
        why = refusal(p)
        return (None, why) if why else (p, "")
    word = _dir_word(raw)
    kf = known_folders()
    key = _FOLDER_WORDS.get(word)
    if key and key in kf:
        why = refusal(kf[key])
        return (None, why) if why else (kf[key], "")
    nc = nextcloud_root()
    m = re.match(r"^(?:nextcloud\s*(.*)|(.+?)\s+(?:in|on)\s+nextcloud)$", word)
    if m and nc:
        sub = (m.group(1) or m.group(2) or "").strip()
        if not sub:
            return nc, ""
        hits = _child_dirs([nc], sub, depth=1)
        return _one_folder(hits, raw)
    bases = list(dict.fromkeys(list(kf.values()) + ([nc] if nc else [])))
    hits = _child_dirs(bases, word, depth=2)
    return _one_folder(hits, raw)


def _one_folder(hits: list, raw: str):
    hits = [h for h in hits if not refusal(h)]
    if not hits:
        return None, f"I can't find a folder called '{raw}' in your folders, sir."
    if len(hits) == 1:
        return hits[0], ""
    listed = [f"{os.path.basename(h)} in {folder_label(os.path.dirname(h))}"
              for h in hits[:MAX_LISTED]]
    return None, (f"I found {len(hits)} folders called '{raw}': "
                  f"{_join(listed)}. Which one, sir?")


def _child_dirs(bases: list, word: str, depth: int) -> list:
    want = _norm_words(word)
    out, seen = [], set()
    for base in bases:
        stack = [(base, 0)]
        while stack:
            cur, d = stack.pop()
            if d >= depth:
                continue
            try:
                entries = list(os.scandir(cur))
            except OSError:
                continue
            for e in entries:
                try:
                    if not e.is_dir(follow_symlinks=False) or _entry_skipped(e):
                        continue
                except OSError:
                    continue
                if _norm_words(e.name) == want and _nc(e.path) not in seen:
                    seen.add(_nc(e.path))
                    out.append(os.path.abspath(e.path))
                stack.append((e.path, d + 1))
    return out


def _entry_skipped(e) -> bool:
    """A search never enters or returns: dot-names, AppData, iCloud / Apple
    folders, junctions / symlinks, hidden or system items."""
    name = e.name
    low = name.lower()
    if name.startswith(".") or low == "appdata":
        return True
    try:
        if e.is_symlink() or (getattr(e, "is_junction", None) and e.is_junction()):
            return True
        if (e.is_dir(follow_symlinks=False)
                and any(w in low for w in _DIR_WORDS_REFUSED)):
            return True
        attrs = getattr(e.stat(follow_symlinks=False), "st_file_attributes", 0)
        return bool(attrs & (FILE_ATTRIBUTE_HIDDEN | FILE_ATTRIBUTE_SYSTEM))
    except OSError:
        return True


def _search_roots(hint_dir: str) -> list:
    if hint_dir:
        return [hint_dir]
    kf = known_folders()
    order = [kf[k] for k in ("downloads", "desktop", "documents", "pictures",
                             "music", "videos") if k in kf]
    nc = nextcloud_root()
    if nc:
        order.append(nc)
    out, seen = [], set()
    for p in order:
        if _nc(p) in seen or any(_under(p, s) for s in out):
            continue
        seen.add(_nc(p))
        out.append(p)
    return out


def find_files(query: str, roots: list) -> list:
    """Allowed files under ``roots`` whose name holds every query word,
    depth- and time-bounded. Exact-name matches first."""
    toks = _tokens(query)
    found, seen = [], set()
    deadline = time.monotonic() + SEARCH_MAX_S
    budget = [SEARCH_MAX_ENTRIES]
    root_set = {_nc(r) for r in roots}
    for root in roots:
        stack = [(root, 0)]
        while stack and budget[0] > 0 and time.monotonic() < deadline:
            cur, d = stack.pop()
            try:
                entries = list(os.scandir(cur))
            except OSError:
                continue
            for e in entries:
                budget[0] -= 1
                try:
                    if _entry_skipped(e):
                        continue
                    if e.is_dir(follow_symlinks=False):
                        # Pictures may sit inside Nextcloud: walk it once.
                        if d + 1 < SEARCH_MAX_DEPTH and _nc(e.path) not in root_set:
                            stack.append((e.path, d + 1))
                        continue
                    if not e.is_file(follow_symlinks=False):
                        continue
                except OSError:
                    continue
                if name_matches(e.name, toks) and _nc(e.path) not in seen:
                    seen.add(_nc(e.path))
                    found.append(os.path.abspath(e.path))
    if not found:
        found = _index_hits(query, toks)
    want = _norm_words(query)
    found.sort(key=lambda p: (_norm_words(os.path.splitext(os.path.basename(p))[0]) != want
                              and _norm_words(os.path.basename(p)) != want, p.lower()))
    return [p for p in found if not refusal(p)]


def _index_hits(query: str, toks: list) -> list:
    """The personal file index (skills/personal_rag.py, the search behind
    search_my_files): its hits whose NAME matches, under the allowed roots."""
    try:
        mod = sys.modules.get("skill_personal_rag")
        rag = mod._rag() if mod is not None and hasattr(mod, "_rag") else None
        if rag is None or not rag.is_available():
            return []
        out = []
        for h in rag.search(query, k=10) or []:
            p = os.path.abspath(str(h.get("path") or ""))
            if (p and os.path.isfile(p) and name_matches(os.path.basename(p), toks)
                    and p not in out):
                out.append(p)
        return out
    except Exception:
        return []


def resolve_source(what: str, hint: str = ""):
    """(path, '') for one allowed file, else (None, owner line) - a not-found,
    a refusal, a folder, or a which-one question."""
    raw = _clean(what)
    if not raw:
        return None, "Which file should I move, sir?"
    if _looks_like_path(raw):
        p = os.path.abspath(os.path.expanduser(raw))
        if os.path.isdir(p):
            return None, "I can only move single files for now, sir."
        if not os.path.isfile(p):
            return None, f"I can't find {raw}, sir."
        why = refusal(p)
        return (None, why) if why else (p, "")
    hint_dir = ""
    if hint:
        hint_dir, why = resolve_folder(hint)
        if not hint_dir:
            return None, why
    hits = find_files(raw, _search_roots(hint_dir))
    if not hits:
        where = f" in {folder_label(hint_dir)}" if hint_dir else " in your folders"
        dirs = _child_dirs(_search_roots(hint_dir), raw, depth=2)
        if dirs:
            return None, "I can only move single files for now, sir."
        return None, f"I couldn't find a file called '{raw}'{where}, sir."
    if len(hits) == 1:
        return hits[0], ""
    listed = [f"'{os.path.basename(h)}' in {folder_label(os.path.dirname(h))}"
              for h in hits[:MAX_LISTED]]
    more = f"{len(hits)} files" if len(hits) <= MAX_LISTED else f"{len(hits)} files; the first three are"
    return None, f"I found {more} {_join(listed)}. Which one, sir?"


def _join(items: list) -> str:
    if len(items) <= 1:
        return "".join(items)
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


# ── plans (the dry run the yes executes) ─────────────────────────────────
def _key(name: str, arg: str) -> tuple:
    return (name, " ".join((arg or "").lower().split()))


def _fingerprint(p: str):
    st = os.stat(p)
    return st.st_size, st.st_mtime_ns


def plan_move(arg: str):
    """(plan, '') or (None, owner line). Touches nothing."""
    what, hint, where, whole = parse_arg(arg)
    if not where:
        return None, "Move which file where, sir? Say it as 'the file to the folder'."
    src, why = resolve_source(what, hint)
    if not src and hint and not resolve_folder(hint)[0]:
        # 'from / in <x>' named no folder: it is part of the file's name.
        src, why = resolve_source(whole)
    if not src:
        return None, why
    dest_dir, why = resolve_folder(where)
    if not dest_dir:
        return None, why
    if _nc(os.path.dirname(src)) == _nc(dest_dir):
        return None, (f"'{os.path.basename(src)}' is already in "
                      f"{folder_label(dest_dir)}, sir.")
    name = os.path.basename(src)
    final = free_name(dest_dir, name)
    if not final:
        return None, f"Too many files called '{name}' in {folder_label(dest_dir)}, sir."
    size, mtime = _fingerprint(src)
    return {"kind": "move", "src": src, "dest_dir": dest_dir, "final": final,
            "size": size, "mtime": mtime, "at": time.monotonic()}, ""


def plan_undo():
    entry = _last_entry()
    if entry is None:
        return None, "I haven't moved any files to undo, sir."
    cur = entry.get("dest", "")
    back_dir = os.path.dirname(entry.get("src", ""))
    name = os.path.basename(cur)
    if not os.path.isfile(cur) or os.path.getsize(cur) != entry.get("size"):
        return None, (f"'{name}' isn't where I put it in {folder_label(os.path.dirname(cur))} "
                      "any more, so I've left things as they are, sir.")
    if not os.path.isdir(back_dir):
        return None, f"The folder it came from, {back_dir}, is gone, sir."
    for p in (cur, back_dir):
        why = refusal(p)
        if why:
            return None, why
    final = free_name(back_dir, os.path.basename(entry.get("src", "")) or name)
    if not final:
        return None, f"Too many files called '{name}' in {folder_label(back_dir)}, sir."
    size, mtime = _fingerprint(cur)
    return {"kind": "undo", "src": cur, "dest_dir": back_dir, "final": final,
            "size": size, "mtime": mtime, "at": time.monotonic(),
            "entry_at": entry.get("at")}, ""


def question_for(plan: dict) -> str:
    name = os.path.basename(plan["src"])
    frm = folder_label(os.path.dirname(plan["src"]))
    to = folder_label(plan["dest_dir"])
    back = " back" if plan["kind"] == "undo" else ""
    q = f"Move '{name}'{back} from {frm} to {to}, sir?"
    if plan["final"] != name:
        q += f" There's already one there, so I'll call it '{plan['final']}'."
    return q


def confirm_question(name: str, arg: str = ""):
    """The dry run bobert_companion._jarvis_pushback asks for: (question,
    reason) and a stored plan the confirmed action will execute, or None when
    there is nothing to ask (the action then says what went wrong)."""
    nm = (name or "").strip().lower()
    try:
        if nm == "move_file":
            plan, _why = plan_move(arg)
        elif nm == "undo_last_move":
            plan, _why = plan_undo()
        else:
            return None
    except Exception as e:
        print(f"  [file-move] dry run failed: {e}")
        return None
    if plan is None:
        with _lock:
            _failures[_key(nm, arg)] = (_why, time.monotonic())
        return None
    with _lock:
        _plans[_key(nm, arg)] = plan
    return question_for(plan), f"{nm}: {plan['src']} -> {plan['dest_dir']}"


def _recent_failure(name: str, arg: str):
    """The dry run's own answer when it just failed for this request (so the
    action that runs next says it without searching the disk again)."""
    with _lock:
        got = _failures.pop(_key(name, arg), None)
    if got and time.monotonic() - got[1] <= FAILURE_MEMO_S:
        return got[0]
    return None


def _take_plan(name: str, arg: str):
    with _lock:
        plan = _plans.pop(_key(name, arg), None)
    if plan and time.monotonic() - plan["at"] <= PLAN_TTL_S:
        return plan
    return None


# ── the move itself ──────────────────────────────────────────────────────
def _same_drive(a: str, b: str) -> bool:
    da = os.path.splitdrive(os.path.realpath(a))[0].lower()
    db = os.path.splitdrive(os.path.realpath(b))[0].lower()
    return da == db


def _locked_line(name: str) -> str:
    return (f"'{name}' is open or locked by another program, so I couldn't move it, "
            "sir. Close it and ask me again.")


def safe_move(src: str, dest: str) -> str:
    """Move src to dest (dest must not exist). '' on success, else the
    owner line. Never overwrites, never deletes anything of his."""
    name = os.path.basename(src)
    if os.path.lexists(dest):
        return (f"Something called '{os.path.basename(dest)}' is already in "
                f"{folder_label(os.path.dirname(dest))}, so I left everything where it was, sir.")
    if _same_drive(src, os.path.dirname(dest)):
        try:
            os.rename(src, dest)   # fails, never replaces, when dest exists
            return ""
        except PermissionError:
            return _locked_line(name)
        except OSError as e:
            if getattr(e, "winerror", None) != 17:   # 17 = not the same device
                return f"Windows wouldn't move '{name}', sir: {e.strerror or e}."
    tmp = os.path.join(os.path.dirname(dest), f".{name}.jarvis-moving")
    if os.path.lexists(tmp):
        return f"A leftover temp file is in the way in {folder_label(os.path.dirname(dest))}, sir."
    try:
        shutil.copy2(src, tmp)
    except PermissionError:
        _drop_own_temp(tmp)
        return _locked_line(name)
    except OSError as e:
        _drop_own_temp(tmp)
        return f"Copying '{name}' failed, sir: {e.strerror or e}. Nothing was moved."
    try:
        same = os.path.getsize(tmp) == os.path.getsize(src)
    except OSError:
        same = False
    if not same:
        _drop_own_temp(tmp)
        return f"The copy of '{name}' came out the wrong size, sir, so I kept the original where it was."
    try:
        os.remove(src)
    except OSError:
        _drop_own_temp(tmp)
        return _locked_line(name)
    try:
        os.rename(tmp, dest)
    except OSError as e:
        return (f"'{name}' was copied but I couldn't give it its name, sir: it's in "
                f"{folder_label(os.path.dirname(dest))} as '{os.path.basename(tmp)}' ({e}).")
    return ""


def _drop_own_temp(tmp: str) -> None:
    """Remove the partial copy THIS move made (never one of his files)."""
    try:
        if os.path.isfile(tmp) and os.path.basename(tmp).endswith(".jarvis-moving"):
            os.remove(tmp)
    except OSError:
        pass


def verify(src: str, dest: str, size: int) -> str:
    """'' when dest holds ``size`` bytes and src is gone, else what's wrong."""
    if not os.path.isfile(dest):
        return f"I can't see '{os.path.basename(dest)}' in {folder_label(os.path.dirname(dest))}"
    if os.path.getsize(dest) != size:
        return f"'{os.path.basename(dest)}' in {folder_label(os.path.dirname(dest))} is the wrong size"
    if os.path.lexists(src):
        return f"'{os.path.basename(src)}' is still in {folder_label(os.path.dirname(src))}"
    return ""


def _execute(plan: dict) -> str:
    src, dest_dir = plan["src"], plan["dest_dir"]
    name = os.path.basename(src)
    if not os.path.isfile(src):
        return TERMINAL + f"'{name}' isn't where it was a moment ago, sir, so I moved nothing."
    try:
        if _fingerprint(src) != (plan["size"], plan["mtime"]):
            return TERMINAL + f"'{name}' changed since I asked, sir, so I moved nothing. Ask me again."
    except OSError:
        return TERMINAL + f"I couldn't read '{name}' just now, sir, so I moved nothing."
    if not os.path.isdir(dest_dir):
        return TERMINAL + f"{folder_label(dest_dir)} isn't there any more, sir, so I moved nothing."
    for p in (src, dest_dir):
        why = refusal(p)
        if why:
            return TERMINAL + why
    dest = os.path.join(dest_dir, plan["final"])
    err = safe_move(src, dest)
    if err:
        return TERMINAL + err
    bad = verify(src, dest, plan["size"])
    if bad:
        return TERMINAL + f"The move didn't check out, sir: {bad}."
    note = _journal_record(plan, dest)
    to = folder_label(dest_dir)
    if plan["kind"] == "undo":
        line = f"Done, sir - '{plan['final']}' is back in {to}."
    elif plan["final"] != name:
        line = f"Done, sir - '{name}' is now in {to}, as '{plan['final']}'."
    else:
        line = f"Done, sir - '{name}' is now in {to}."
    return line + note


# ── journal ──────────────────────────────────────────────────────────────
def _journal_load() -> list:
    try:
        import json
        with open(journal_path(), "r", encoding="utf-8") as f:
            data = json.load(f)
        moves = data.get("moves") if isinstance(data, dict) else None
        return [m for m in moves if isinstance(m, dict)] if isinstance(moves, list) else []
    except Exception:
        return []


def _journal_save(moves: list) -> None:
    from core.atomic_io import _atomic_write_json
    _atomic_write_json(journal_path(), {"moves": moves[-JOURNAL_KEEP:]})


def _last_entry():
    with _lock:
        for m in reversed(_journal_load()):
            if not m.get("undone") and m.get("kind", "move") == "move":
                return m
    return None


def _journal_record(plan: dict, dest: str) -> str:
    try:
        with _lock:
            moves = _journal_load()
            now = time.time()
            if plan["kind"] == "undo":
                for m in reversed(moves):
                    if m.get("at") == plan.get("entry_at") and not m.get("undone"):
                        m["undone"] = True
                        m["undone_at"] = now
                        break
                moves.append({"kind": "undo", "at": now, "src": plan["src"],
                              "dest": dest, "size": plan["size"]})
            else:
                moves.append({"kind": "move", "at": now, "src": plan["src"],
                              "dest": dest, "size": plan["size"], "undone": False})
            _journal_save(moves)
        return ""
    except Exception as e:
        print(f"  [file-move] journal write failed: {e}")
        return " I wasn't able to note it down, though, so undo won't know about it."


# ── actions ──────────────────────────────────────────────────────────────
_NO_YES = ("I only move files after you've said yes to my question, sir. "
           "Ask me again and I'll check with you first.")


def move_file(arg: str = "") -> str:
    plan = _take_plan("move_file", arg)
    if plan is None:
        why = _recent_failure("move_file", arg)
        fresh = None
        if why is None:
            fresh, why = plan_move(arg)
        if fresh is None:
            return why if why.endswith("Which one, sir?") else TERMINAL + why
        return TERMINAL + _NO_YES
    return _execute(plan)


def undo_last_move(arg: str = "") -> str:
    plan = _take_plan("undo_last_move", arg)
    if plan is None:
        why = _recent_failure("undo_last_move", arg)
        fresh = None
        if why is None:
            fresh, why = plan_undo()
        if fresh is None:
            return TERMINAL + why
        return TERMINAL + _NO_YES
    return _execute(plan)


def register(actions):
    if not _cfg("FILE_MOVE_ENABLED", True):
        print("  [skill] file_move: FILE_MOVE_ENABLED is off - not registered")
        return
    actions["move_file"] = move_file
    actions["undo_last_move"] = undo_last_move
