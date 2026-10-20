"""One set of settings across the owner's machines (2026-10-20, his ask: "the
hand over should be like JARVIS jumping from one machine to another like an
identical copy, except for obvious hardware differences").

Each machine keeps its own data/user_settings.json, and the Dell had drifted
from the desktop (no sarcastic humor, no Claude safety net, the older memory
search). Every setting that is not tied to this machine's hardware is now
SHARED: a change on either machine reaches the other.

How:
  * data/settings_ledger.json records, per shared key, its value and WHEN it
    last changed here (refresh_ledger: a value that differs from the ledger
    changed just now; a key seen for the first time gets t=0, "age unknown",
    which never overrides a real change on the other machine).
  * core/peer_memory.py ships the ledger inside the bundle it already trades
    over SSH every 10 minutes and at a handoff.
  * merge_peer(): per key, the newer change wins (last writer wins).
  * core/config._apply_user_settings() calls apply_pending() at start, before
    it reads user_settings.json, so a machine BOOTS with the other machine's
    newer settings (local files only, no network, no delay). A newer bundle
    arriving while JARVIS runs is merged into the file too and applies at the
    next start, like any other settings change.

Machine-local (never shared): devices, cameras, Kinect, hands, monitors, mic
levels, speech-to-text hardware, folders, endpoints and tunnels, the handoff
pair itself, the server watch (the laptop is often off the home network),
always-on screen memory (a heavy, private feature per machine) and anything
secret-shaped.

Pure stdlib, never raises; core.config imports it at import time.
"""
from __future__ import annotations

import json
import os
import re
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA = os.path.join(_ROOT, "data")
SETTINGS_PATH = os.path.join(_DATA, "user_settings.json")
LEDGER_PATH = os.path.join(_DATA, "settings_ledger.json")
BACKUP_SUFFIX = ".bak-peer"
UNSET = "__unset__"

LOCAL_KEYS = frozenset({
    "CAMERAS", "MONITORS", "HUD_MONITOR", "CONSOLE_MONITOR",
    "MICROPHONE_INDEX", "SPEAKER_INDEX", "VAD_THRESHOLD",
    "SPEECH_FILTER_OVERRIDES", "STT_SHADOW", "LOCATION",
    "LOCAL_LLM_BASE_URL", "RAG_OLLAMA_ENDPOINT", "RAG_INDEX_PATHS",
    "VOICE_CLONE_SERVER_CMD", "VOICE_CLONE_SERVER_URL", "VOICE_CLONE_TIMEOUT_S",
    "AMBIENT_SCREEN_ENABLED", "LISTEN_CARD",
})
LOCAL_PREFIXES = (
    "KINECT_", "HAND_", "AIR_MOUSE_", "HOLO_", "AUDIO_AUTOSWITCH_",
    "WHISPER_", "SERVER_WATCH_", "PEER_", "HANDOFF_", "CAMERA_",
)
LOCAL_SUFFIXES = (
    "_DEVICE", "_DEVICES", "_INDEX", "_CUDA", "_PATH", "_PATHS", "_DIR",
    "_URL", "_ENDPOINT", "_HOST", "_SSH", "_CMD", "_PORT", "_MONITOR",
    "_MONITORS", "_CAMERA", "_CAMERAS", "_RECT", "_BOX",
)
_SECRET_RE = re.compile(r"TOKEN|SECRET|PASSWORD|PASSWD|API_?KEY|COOKIE|CREDENTIAL")


def is_shared(key: str) -> bool:
    """True for a setting both machines keep the same."""
    k = str(key or "")
    if not k or k.startswith("_") or k != k.upper():
        return False
    if k in LOCAL_KEYS or _SECRET_RE.search(k):
        return False
    if k.startswith(LOCAL_PREFIXES) or k.endswith(LOCAL_SUFFIXES):
        return False
    return True


def _same(a, b) -> bool:
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def refresh_ledger(settings: dict, ledger: dict, now: float | None = None) -> list:
    """Stamp this machine's own changes into ``ledger`` (in place); the keys
    that changed here since the last refresh. A key never seen before gets
    t=0 (age unknown); a removed key is recorded as UNSET."""
    now = time.time() if now is None else now
    changed = []
    for key, val in (settings or {}).items():
        if not is_shared(key):
            continue
        cur = ledger.get(key)
        if not isinstance(cur, dict):
            ledger[key] = {"v": val, "t": 0.0}
        elif not _same(cur.get("v"), val):
            ledger[key] = {"v": val, "t": now}
            changed.append(key)
    for key, cur in list(ledger.items()):
        if (is_shared(key) and key not in (settings or {})
                and isinstance(cur, dict) and cur.get("v") != UNSET):
            ledger[key] = {"v": UNSET, "t": now}
            changed.append(key)
    return changed


def merge_peer(settings: dict, ledger: dict, peer_ledger) -> list:
    """Take every shared key whose change on the peer is NEWER than ours
    (``settings`` and ``ledger`` updated in place); the keys taken."""
    taken = []
    if not isinstance(peer_ledger, dict):
        return taken
    for key, entry in peer_ledger.items():
        if not is_shared(key) or not isinstance(entry, dict) or "v" not in entry:
            continue
        try:
            t_peer = float(entry.get("t") or 0.0)
        except (TypeError, ValueError):
            continue
        mine = ledger.get(key) if isinstance(ledger.get(key), dict) else None
        t_mine = float(mine.get("t") or 0.0) if mine else -1.0
        if t_peer <= t_mine:
            continue
        val = entry["v"]
        here = settings.get(key, UNSET)
        ledger[key] = {"v": val, "t": t_peer}
        if _same(here, val):
            continue
        if val == UNSET:
            settings.pop(key, None)
        else:
            settings[key] = val
        taken.append(key)
    return taken


def _read_json(path: str, default):
    """The JSON object in ``path``; ``default`` when it is missing, broken or
    not an object."""
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else default
    except Exception:
        return default


def _write_json(path: str, data, indent=2) -> bool:
    try:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        os.replace(tmp, path)
        return True
    except Exception:
        return False


def load_ledger(settings_path: str = SETTINGS_PATH,
                ledger_path: str = LEDGER_PATH, now: float | None = None) -> dict:
    """This machine's ledger, refreshed against its settings file (and saved
    when anything changed). {} when there is no settings file."""
    settings = _read_json(settings_path, {})
    ledger = _read_json(ledger_path, {})
    if not settings and not ledger:
        return {}
    before = json.dumps(ledger, sort_keys=True)
    refresh_ledger(settings, ledger, now=now)
    if json.dumps(ledger, sort_keys=True) != before:
        _write_json(ledger_path, ledger, indent=None)
    return ledger


def apply_peer_ledger(peer_ledger, settings_path: str = SETTINGS_PATH,
                      ledger_path: str = LEDGER_PATH, log=None) -> list:
    """Merge one peer ledger into this machine's settings file; the keys
    taken. Backs the file up (one rolling copy) before it writes. Never
    raises."""
    try:
        if not isinstance(peer_ledger, dict) or not peer_ledger:
            return []
        if not os.path.exists(settings_path):
            return []
        settings = _read_json(settings_path, None)
        if not isinstance(settings, dict):
            return []
        ledger = _read_json(ledger_path, {})
        refresh_ledger(settings, ledger)
        taken = merge_peer(settings, ledger, peer_ledger)
        if taken:
            try:
                import shutil
                shutil.copy2(settings_path, settings_path + BACKUP_SUFFIX)
            except Exception:
                pass
            if not _write_json(settings_path, settings):
                return []
            if log:
                log(f"  [peer-settings] took {len(taken)} setting(s) from the "
                    f"other machine: {', '.join(sorted(taken))}")
        _write_json(ledger_path, ledger, indent=None)
        return taken
    except Exception:
        return []


def apply_pending(data_dir: str = _DATA, settings_path: str = SETTINGS_PATH,
                  ledger_path: str = LEDGER_PATH, log=None) -> list:
    """Boot hook (core.config, before user_settings.json is read): merge the
    settings from every peer bundle already on this disk. Local files only.
    Off unless this machine has a handoff partner (PEER_HANDOFF_SSH set in
    its settings file). Never raises."""
    try:
        settings = _read_json(settings_path, {})
        if not str(settings.get("PEER_HANDOFF_SSH") or "").strip():
            return []
        import glob
        import socket
        me = socket.gethostname().lower()
        taken = []
        for path in sorted(glob.glob(os.path.join(data_dir, "peer_memory_from_*.json"))):
            bundle = _read_json(path, {})
            if str(bundle.get("host") or "").lower() == me:
                continue
            taken += apply_peer_ledger(bundle.get("settings"), settings_path,
                                       ledger_path, log=log)
        return taken
    except Exception:
        return []
