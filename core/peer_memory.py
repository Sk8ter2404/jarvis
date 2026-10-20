"""One memory across the owner's machines (2026-10-19, his ask: "if I ask the
desktop what we did on the Dell today he'll say 'while I was over on the
laptop, this is what happened'").

The thinking already runs on the desktop (the laptop's LOCAL_LLM_BASE_URL is
the desktop's GPU); the MEMORY did not - each machine kept its own session
summaries and facts. Each copy now writes a small bundle of its own memory
(recent session summaries + recent facts) and trades it with the other
machine over the same SSH link the handoff uses (core/peer_handoff.py):

  every 10 min, and when it stands down   push own bundle to the peer
                                          (data/peer_memory_from_<host>.json)
  at start                                pull the peer's own bundle
  every minute                            import any bundle that changed

The laptop's copy only runs while the desktop is on (it is the brain), so
pushing as it goes lands everything on the desktop before the laptop is shut.

Imported sessions keep "from_peer" and read "On the <peer>: ...", so the
recall prompt says where they happened; imported facts carry source
"peer:<name>". Neither is ever sent back (no echo between the two).
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import threading
import time

from core import peer_handoff as _handoff

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA = os.path.join(_ROOT, "data")
SELF_BUNDLE = os.path.join(_DATA, "peer_memory_self.json")
IMPORT_MARKS = os.path.join(_DATA, "peer_memory_imported.json")
PUSH_EVERY_S = 600.0
IMPORT_EVERY_S = 60.0
SESSION_DAYS = 14
FACT_DAYS = 30
_FACT_SOURCE = "peer:"
# Only facts the owner himself said cross between machines (2026-10-19). The
# Dell had AMBIENT_LEARN_FROM_EVERYONE on: 43 facts overheard in a classroom on
# 10-06 carry provenance speaker "unknown", and its first push would have
# landed them on the desktop. The bundle names the speaker; the importer
# takes only "owner", so an older copy's bundle (no speaker) shares sessions
# but no facts.
OWNER = "owner"


def _owner_said(fact: dict) -> bool:
    return (fact.get("provenance") or {}).get("speaker") == OWNER


def _host() -> str:
    return socket.gethostname()


def _safe_host(h: str) -> str:
    return "".join(c for c in (h or "") if c.isalnum() or c in "-_")[:40] or "peer"


def peer_bundle_path(host: str) -> str:
    return os.path.join(_DATA, f"peer_memory_from_{_safe_host(host)}.json")


def build_bundle(sessions, facts, now=None, host=None, settings=None,
                 version=None, conversation=None) -> dict:
    """This machine's OWN memory only: anything imported from the peer is
    left out, so nothing echoes back. Since 2026-10-20 ("JARVIS jumping from
    one machine to another like an identical copy") it also carries this
    machine's shared-settings ledger (core/peer_settings.py), its VERSION,
    and - only in the hand-over push - the conversation in progress."""
    now = time.time() if now is None else now
    s_cut, f_cut = now - SESSION_DAYS * 86400, now - FACT_DAYS * 86400
    own_sessions = [s for s in sessions or []
                    if isinstance(s, dict) and not s.get("from_peer")
                    and s.get("summary") and _num(s.get("ts")) >= s_cut]
    own_facts = [{"text": f.get("text"), "tags": list(f.get("tags") or []),
                  "created_at": f.get("created_at"), "speaker": OWNER}
                 for f in facts or []
                 if isinstance(f, dict) and f.get("text") and _owner_said(f)
                 and not str(f.get("source") or "").startswith(_FACT_SOURCE)
                 and _num(f.get("created_at")) >= f_cut]
    bundle = {"host": host or _host(), "written_at": now,
              "sessions": own_sessions, "facts": own_facts}
    if settings:
        bundle["settings"] = settings
    if version:
        bundle["version"] = str(version)
    turns = conversation_turns(conversation)
    if turns:
        bundle["conversation"] = {"at": now, "turns": turns}
    return bundle


# The conversation in progress travels with a handoff: the last turns, as
# plain text (images and tool blocks dropped), each capped.
CONVERSATION_TURNS = 12
CONVERSATION_MAX_CHARS = 1200
# The arriving copy takes the conversation only when the hand-over is fresh.
CONVERSATION_FRESH_S = 900.0


def _turn_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(str(b.get("text") or "") for b in content
                        if isinstance(b, dict) and b.get("type") == "text")
    return ""


def conversation_turns(history) -> list:
    """The last CONVERSATION_TURNS user / assistant messages of ``history``
    as {"role", "content"} with text content only."""
    out = []
    for m in list(history or [])[-CONVERSATION_TURNS:]:
        if not isinstance(m, dict) or m.get("role") not in ("user", "assistant"):
            continue
        text = _turn_text(m.get("content")).strip()
        if text:
            out.append({"role": m["role"],
                        "content": text[:CONVERSATION_MAX_CHARS]})
    return out


def handed_over_conversation(bundle, now=None) -> list:
    """The turns a FRESH hand-over bundle carries ([] when there are none or
    the bundle is older than CONVERSATION_FRESH_S)."""
    now = time.time() if now is None else now
    conv = (bundle or {}).get("conversation") if isinstance(bundle, dict) else None
    if not isinstance(conv, dict):
        return []
    if now - _num(conv.get("at")) > CONVERSATION_FRESH_S:
        return []
    return conversation_turns(conv.get("turns"))


def _num(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def write_self_bundle(bundle: dict, path: str = SELF_BUNDLE) -> bool:
    try:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            # ASCII-only: pull() reads it back through a remote console,
            # whose code page would mangle a curly quote or a dash.
            json.dump(bundle, f, ensure_ascii=True)
        os.replace(tmp, path)
        return True
    except Exception:
        return False


def _remote_data_path(name: str) -> str:
    # Both machines keep the repo at the same path.
    return os.path.join(_DATA, name).replace("\\", "/")


def push(target: str, run=subprocess.run, path: str = SELF_BUNDLE) -> bool:
    """Copy this machine's bundle to the peer. False when the peer is off."""
    if not target or not os.path.exists(path):
        return False
    dest = f"{target}:{_remote_data_path(os.path.basename(peer_bundle_path(_host())))}"
    try:
        out = run(["scp", "-B", "-q", "-o", "ConnectTimeout=6",
                   "-o", "StrictHostKeyChecking=accept-new", path, dest],
                  capture_output=True, text=True, timeout=60,
                  creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return out.returncode == 0
    except Exception:
        return False


def pull(target: str, run=subprocess.run):
    """Fetch the peer's own bundle; the local path it was saved to, or None."""
    if not target:
        return None
    remote = (f'powershell -NoProfile -Command "Get-Content -Raw -Encoding UTF8 '
              f'\'{os.path.join(_DATA, os.path.basename(SELF_BUNDLE))}\'"')
    try:
        out = run(_handoff.ssh_command(target, remote), capture_output=True,
                  text=True, encoding="utf-8", timeout=60,
                  creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if out.returncode != 0:
            return None
        bundle = json.loads(out.stdout or "")
        if not isinstance(bundle, dict) or not bundle.get("host"):
            return None
        dest = peer_bundle_path(bundle["host"])
        with open(dest, "w", encoding="utf-8") as f:
            json.dump(bundle, f, ensure_ascii=False)
        return dest
    except Exception:
        return None


def peer_session_entries(bundle: dict, name: str) -> list:
    """The peer's sessions as entries for this machine's index: marked
    from_peer, and the summary says where it happened."""
    out = []
    for s in bundle.get("sessions") or []:
        if not isinstance(s, dict) or not s.get("summary"):
            continue
        e = dict(s)
        e["from_peer"] = name
        e["location"] = name
        text = str(s["summary"]).strip()
        prefix = f"On the {name}: "
        e["summary"] = text if text.startswith(prefix) else prefix + text
        out.append(e)
    return out


def _load_marks(path: str = IMPORT_MARKS) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def _save_marks(marks: dict, path: str = IMPORT_MARKS) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(marks, f)
    except Exception:
        pass


def import_bundle(path: str, name: str, merge_sessions, add_fact,
                  marks: dict, apply_settings=None) -> tuple:
    """Fold one peer bundle into this machine's memory once per version of
    it. (sessions merged, facts added) - (0, 0) when already imported.
    ``apply_settings(ledger)`` (optional) merges the bundle's shared
    settings into this machine's settings file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            bundle = json.load(f)
    except Exception:
        return (0, 0)
    if not isinstance(bundle, dict):
        return (0, 0)
    key = _safe_host(bundle.get("host") or "")
    written = _num(bundle.get("written_at"))
    if written <= _num(marks.get(key)):
        return (0, 0)
    n_sessions = merge_sessions(peer_session_entries(bundle, name))
    n_facts = 0
    for fact in bundle.get("facts") or []:
        text = str((fact or {}).get("text") or "").strip()
        if not text or (fact or {}).get("speaker") != OWNER:
            continue
        try:
            add_fact(text, source=_FACT_SOURCE + name, tags=fact.get("tags") or [])
            n_facts += 1
        except Exception:
            pass
    if apply_settings is not None and isinstance(bundle.get("settings"), dict):
        try:
            apply_settings(bundle["settings"])
        except Exception:
            pass
    marks[key] = written
    return (n_sessions, n_facts)


def arrival_handover(data_dir: str = _DATA, now=None) -> dict:
    """What the copy that just stood down handed this one: its pushed bundle
    (data/peer_memory_from_<peer>.json, written before it exited). {"turns":
    [...], "age_s": seconds since the hand-over, "version": its VERSION,
    "host": its name}, or {} when there is no fresh hand-over. Never raises."""
    import glob
    try:
        now = time.time() if now is None else now
        best = None
        for path in glob.glob(os.path.join(data_dir, "peer_memory_from_*.json")):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    b = json.load(f)
            except Exception:
                continue
            if not isinstance(b, dict) or _safe_host(b.get("host") or "") == _safe_host(_host()):
                continue
            if best is None or _num(b.get("written_at")) > _num(best.get("written_at")):
                best = b
        if not best:
            return {}
        turns = handed_over_conversation(best, now=now)
        if not turns:
            return {}
        return {"turns": turns,
                "age_s": max(0.0, now - _num((best.get("conversation") or {}).get("at"))),
                "version": str(best.get("version") or ""),
                "host": str(best.get("host") or "")}
    except Exception:
        return {}


def own_version() -> str:
    try:
        with open(os.path.join(_ROOT, "VERSION"), "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return ""


def _own_bundle(conversation=None) -> dict:
    import memory as _pm
    from core import long_term_memory as _ltm
    from core import peer_settings as _peer_settings
    with _pm._session_lock:
        sessions = list(_pm._load_sessions_file())
    try:
        facts = _ltm.list_facts()
    except Exception:
        facts = []
    return build_bundle(sessions, facts, settings=_peer_settings.load_ledger(),
                        version=own_version(), conversation=conversation)


def sync_out(log=print, conversation=None) -> bool:
    """Write this machine's bundle and push it to the peer (if it is on).
    ``conversation`` (the hand-over push only): the chat in progress."""
    if not write_self_bundle(_own_bundle(conversation)):
        return False
    ok = push(_handoff.peer_target())
    if ok:
        log(f"  [peer-memory] sent this machine's memory to the {_handoff.peer_name()}")
    return ok


def import_all(marks: dict, log=print) -> int:
    import glob
    import memory as _pm
    from core import long_term_memory as _ltm
    from core import peer_settings as _peer_settings
    name = _handoff.peer_name()
    own = os.path.basename(peer_bundle_path(_host()))
    total = 0

    def _settings(ledger):
        # Applies at this machine's next start, like any settings change.
        _peer_settings.apply_peer_ledger(ledger, log=log)

    for path in glob.glob(os.path.join(_DATA, "peer_memory_from_*.json")):
        if os.path.basename(path) == own:
            continue
        s, f = import_bundle(path, name, _pm.merge_peer_sessions, _ltm.add_fact,
                             marks, apply_settings=_settings)
        if s or f:
            log(f"  [peer-memory] from the {name}: {s} session(s), {f} fact(s)")
            total += s + f
    _save_marks(marks)
    return total


_stop = threading.Event()


def _loop(log=print) -> None:
    marks = _load_marks()
    target = _handoff.peer_target()
    if pull(target):
        log(f"  [peer-memory] fetched the {_handoff.peer_name()}'s memory")
    import_all(marks, log=log)
    sync_out(log=log)
    last_push = time.monotonic()
    while not _stop.wait(IMPORT_EVERY_S):
        try:
            import_all(marks, log=log)
            if time.monotonic() - last_push >= PUSH_EVERY_S:
                sync_out(log=log)
                last_push = time.monotonic()
        except Exception as e:
            log(f"  [peer-memory] sync error: {e}")


def start(log=print) -> bool:
    """Boot hook, beside peer_handoff.start(): same switch."""
    if not _handoff.enabled():
        return False
    threading.Thread(target=_loop, kwargs={"log": log}, name="peer-memory",
                     daemon=True).start()
    return True
