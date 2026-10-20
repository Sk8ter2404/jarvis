"""Is the home server up? (2026-10-19, owner: "I want him to be able to monitor
the server as well").

READ-ONLY by design: JARVIS only looks. It never writes to the server, never
restarts anything there and never touches the array - the server's care stays
the owner's (array changes, "New Config", parity calls are his alone).

  * Reachability: TCP connects to the server's own services (SSH 22, web 443,
    Plex 32400, DNS 53), in parallel, 1.5 s each. Up = any of them answers.
  * Array health, when it is up and SERVER_WATCH_SSH is set: ONE read-only
    `mdcmd status` over SSH (BatchMode, 10 s) - mdState, the disabled /
    invalid / missing disk counts and a running parity check. Nothing else
    is ever run there.
  * A background poll every SERVER_WATCH_POLL_S believes a change after two
    polls agree, and announces it once (proactive_announce): down, back up,
    or more disabled disks than before. The believed state and when it last
    changed live in data/server_watch.json, so "offline since ..." survives a
    restart and a restart never re-announces an old state.
  * "how's the server" / "is the server up" answer from a FRESH check.

Off unless SERVER_WATCH_HOST is set (data/user_settings.json).
"""
from __future__ import annotations

import importlib
import json
import os
import re
import socket
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor

SPEAK_VERBATIM_ACTIONS = ("server_status",)

PORTS = ((22, "SSH"), (443, "the web front end"), (32400, "Plex"), (53, "DNS"))
CONNECT_TIMEOUT_S = 1.5
SSH_TIMEOUT_S = 10.0
AGREE_POLLS = 2
FIRST_POLL_DELAY_S = 60.0

PROMPT_EXAMPLES = (
    "HOME SERVER (skills/server_watch.py - READ-ONLY, never changes the server):\n"
    "  server_status - is the home server up, which of its services answer, and\n"
    "                  (when it can read it) the array state and disabled disks.\n"
    # No apostrophe inside a quoted trigger: core.prompt_router pairs the
    # single quotes to find them, and "how's" shifted every pair after it.
    "                  Fire on 'how is the server', 'is the server up', 'is unraid\n"
    "                  online', 'check the server', 'any disks down',\n"
    "                  'how is the rebuild going', 'parity check', 'is the array\n"
    "                  ok' (it reads the rebuild's percent and time left).\n"
    "  'is the server up' -> [ACTION: server_status]\n"
)

try:
    from core.paths import data_dir as _data_dir
    _STATE_PATH = os.path.join(_data_dir(), "server_watch.json")
except Exception:   # pragma: no cover - core.paths is in-tree
    _STATE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "data", "server_watch.json")

_lock = threading.Lock()
_stop = threading.Event()


def _cfg(name, default):
    try:
        from core import config as cfg
        return getattr(cfg, name, default)
    except Exception:
        return default


def host() -> str:
    h = str(_cfg("SERVER_WATCH_HOST", "") or "").strip()
    return h if re.match(r"^[A-Za-z0-9._:-]+$", h) else ""


def name() -> str:
    return str(_cfg("SERVER_WATCH_NAME", "server") or "server").strip() or "server"


def ssh_target() -> str:
    t = str(_cfg("SERVER_WATCH_SSH", "") or "").strip()
    return t if re.match(r"^[A-Za-z0-9._-]+@[A-Za-z0-9._:-]+$", t) else ""


# ── checks ─────────────────────────────────────────────────────────────
def _port_open(h: str, port: int, timeout: float = CONNECT_TIMEOUT_S) -> bool:
    try:
        with socket.create_connection((h, port), timeout=timeout):
            return True
    except Exception:
        return False


def probe_ports(h: str, opener=_port_open) -> list:
    """Names of the services that answered, in PORTS order."""
    with ThreadPoolExecutor(max_workers=len(PORTS)) as ex:
        results = list(ex.map(lambda p: opener(h, p[0]), PORTS))
    return [label for (port, label), ok in zip(PORTS, results) if ok]


def parse_mdcmd(text: str) -> dict:
    """The fields of `mdcmd status` this skill reports (strings -> ints where
    numeric). Unknown output -> {}."""
    out = {}
    for line in str(text or "").splitlines():
        k, sep, v = line.strip().partition("=")
        if not sep:
            continue
        if k in ("mdState", "mdNumDisabled", "mdNumInvalid", "mdNumMissing",
                 "mdResync", "mdResyncPos", "mdResyncSize", "mdResyncAction",
                 "mdResyncDt", "mdResyncDb"):
            v = v.strip()
            out[k] = int(v) if v.lstrip("-").isdigit() else v
    return out


def read_array(target: str, run=subprocess.run):
    """One read-only `mdcmd status` over SSH; {} when unreadable."""
    if not target:
        return {}
    try:
        out = run(["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=6",
                   "-o", "StrictHostKeyChecking=accept-new", target, "mdcmd status"],
                  capture_output=True, text=True, timeout=SSH_TIMEOUT_S,
                  creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return parse_mdcmd(out.stdout) if out.returncode == 0 else {}
    except Exception:
        return {}


def check(h=None, target=None, opener=_port_open, run=subprocess.run) -> dict:
    h = host() if h is None else h
    target = ssh_target() if target is None else target
    up_services = probe_ports(h, opener) if h else []
    array = read_array(target, run) if (up_services and target) else {}
    return {"up": bool(up_services), "services": up_services, "array": array,
            "at": time.time()}


# ── state ──────────────────────────────────────────────────────────────
def load_state(path: str = _STATE_PATH) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def save_state(state: dict, path: str = _STATE_PATH) -> None:
    try:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f)
        os.replace(tmp, path)
    except Exception:
        pass


def _when(ts, now=None) -> str:
    try:
        ts = float(ts)
    except (TypeError, ValueError):
        return ""
    now = time.time() if now is None else now
    lt, nt = time.localtime(ts), time.localtime(now)
    clock = time.strftime("%I:%M %p", lt).lstrip("0")
    if lt.tm_yday == nt.tm_yday and lt.tm_year == nt.tm_year:
        return f"{clock} today"
    if now - ts < 6 * 86400:
        return f"{time.strftime('%A', lt)} at {clock}"
    return f"{time.strftime('%B', lt)} {lt.tm_mday} at {clock}"


def _disabled(array: dict) -> int:
    try:
        return int(array.get("mdNumDisabled") or 0)
    except (TypeError, ValueError):
        return 0


def step(state: dict, result: dict, agree: int = AGREE_POLLS):
    """Fold one poll into the believed state. Returns (state, announcement or
    ""). A change is believed after `agree` polls in a row; the first belief
    after a fresh start is never announced (nothing changed that he saw)."""
    seen = "up" if result.get("up") else "down"
    st = dict(state or {})
    if seen == "up":
        st["last_up"] = result.get("at")
    if seen == st.get("believed"):
        st["pending"], st["pending_n"] = None, 0
    else:
        st["pending_n"] = (st.get("pending_n", 0) + 1) if st.get("pending") == seen else 1
        st["pending"] = seen
    msg = ""
    if st.get("pending") and st["pending_n"] >= agree:
        before = st.get("believed")
        st["believed"], st["since"] = seen, result.get("at")
        st["pending"], st["pending_n"] = None, 0
        if before == "up" and seen == "down":
            msg = f"Sir, the {name()} just went offline."
        elif before == "down" and seen == "up":
            msg = f"The {name()} is back online, sir."
    if seen == "up" and result.get("array"):
        a = result["array"]
        n = _disabled(a)
        if n > int(st.get("disabled", 0) or 0) and not msg:
            msg = (f"Sir, the {name()} reports {n} disabled disk"
                   f"{'' if n == 1 else 's'}.")
        st["disabled"] = n
        try:
            active = int(a.get("mdResync") or 0) > 0
        except (TypeError, ValueError):
            active = False
        if st.get("resync") and not active and not msg:
            msg = f"The {name()} finished its {st['resync']}, sir."
        st["resync"] = ((str(a.get("mdResyncAction") or "").split() or ["resync"])[0]
                        .replace("recon", "rebuild").replace("check", "parity check")
                        if active else "")
    return st, msg


def resync_line(a: dict) -> str:
    """'Rebuilding disks 10 and 14, 75 percent done, about 1.5 hours left.'
    from mdResyncAction / Pos / Size / Dt / Db; "" when nothing is running.
    Measured 2026-10-19: "recon D10 D14" while two disks were rebuilt."""
    try:
        size, pos = int(a.get("mdResyncSize") or 0), int(a.get("mdResyncPos") or 0)
        if int(a.get("mdResync") or 0) <= 0 or size <= 0:
            return ""
        action = str(a.get("mdResyncAction") or "").split()
        verb, targets = (action[0].lower(), action[1:]) if action else ("", [])
        def disk(tok):
            return {"P": "parity", "Q": "the second parity"}.get(tok, "disk " + tok.lstrip("D"))
        names = [disk(x) for x in targets]
        what = (" and ".join(names) if len(names) <= 2 else ", ".join(names[:-1]) + " and " + names[-1])
        if verb == "recon":
            head = f"It's rebuilding {what}" if names else "It's rebuilding a disk"
        elif verb == "check":
            head = "A parity check is running"
        elif verb == "clear":
            head = f"It's clearing {what}" if names else "It's clearing a disk"
        else:
            head = "A resync is running"
        line = f"{head}, {100.0 * pos / size:.0f} percent done"
        dt, db = int(a.get("mdResyncDt") or 0), int(a.get("mdResyncDb") or 0)
        if dt > 0 and db > 0:
            left_s = (size - pos) / (db / dt)       # both in KiB
            hours = left_s / 3600.0
            line += (f", about {hours:.1f} hours left" if hours >= 1
                     else f", about {max(1, round(left_s / 60))} minutes left")
        return line + "."
    except Exception:
        return ""


def describe(result: dict, state: dict, now=None) -> str:
    """The spoken answer, built only from what was read."""
    nm = name()
    if not result.get("up"):
        since = state.get("since") if state.get("believed") == "down" else None
        last_up = state.get("last_up")
        bits = [f"The {nm}'s offline, sir: none of its services answered"]
        if since:
            bits.append(f"since {_when(since, now)}")
        line = " ".join(bits)
        if last_up:
            line += f", and it was last up {_when(last_up, now)}"
        return line + "."
    svcs = list(result.get("services") or [])
    listing = (", ".join(svcs[:-1]) + " and " + svcs[-1]) if len(svcs) > 1 else svcs[0]
    line = f"The {nm}'s up, sir: {listing} answering."
    a = result.get("array") or {}
    if a:
        state_word = str(a.get("mdState", "")).lower() or "in an unknown state"
        n = _disabled(a)
        line += f" The array is {state_word}"
        line += (f" with {n} disabled disk{'' if n == 1 else 's'}." if n
                 else " with no disabled disks.")
        rs = resync_line(a)
        if rs:
            line += " " + rs
    elif ssh_target():
        line += " I couldn't read the array's state just now."
    return line


# ── voice + background poll ────────────────────────────────────────────
def server_status(_: str = "") -> str:
    if not host():
        return ("I'm not watching a server yet, sir - its address isn't in my "
                "settings (SERVER_WATCH_HOST).")
    with _lock:
        result = check()
        state, _msg = step(load_state(), result, agree=1)
        save_state(state)
    return describe(result, state)


def _announce(msg: str) -> None:
    try:
        bc = importlib.import_module("bobert_companion")
        fn = getattr(bc, "proactive_announce", None)
        if callable(fn):
            fn(msg, source="server_watch", dedupe_key="server_watch:" + msg)
            return
    except Exception:
        pass
    print(f"  [server-watch] (no announcer) {msg}")


def _poll_loop() -> None:
    if _stop.wait(FIRST_POLL_DELAY_S):
        return
    while True:
        try:
            with _lock:
                result = check()
                state, msg = step(load_state(), result)
                save_state(state)
            if msg:
                print(f"  [server-watch] {msg}")
                _announce(msg)
        except Exception as e:
            print(f"  [server-watch] poll error: {e}")
        try:
            gap = max(60.0, float(_cfg("SERVER_WATCH_POLL_S", 300.0)))
        except (TypeError, ValueError):
            gap = 300.0
        if _stop.wait(gap):
            return


_ROUTE_RE = re.compile(
    r"^\s*(hey\s+)?(jarvis[,.!\s]+)?(how'?s|how is|is|are|check( on)?|what'?s|"
    r"whats|status of|any news on|how are things (with|on))\b.{0,30}?"
    r"\b(server|unraid|nas)\b", re.I)
_NOT_RE = re.compile(r"\b(minecraft|discord|game|voice|clone|web ?server|mod|"
                     r"modshit|jarvis'?s? server|test server)\b", re.I)


# "how long until that data rebuild's done?" (live 19:38) names no server.
_REBUILD_RE = re.compile(
    r"\b(how long|how far|how much longer|when|is|has|what'?s|whats|progress|status)\b.{0,40}?"
    r"\b(rebuild|parity( check| sync)?|data rebuild|array)\b", re.I)
_REBUILD_NOT_RE = re.compile(r"\b(website|site|app|project|index|cache|model|plan)\b", re.I)


def _route(text):
    t = str(text or "")
    if len(t) > 120:
        return None
    if _ROUTE_RE.search(t) and not _NOT_RE.search(t):
        return "[ACTION: server_status]"
    if host() and _REBUILD_RE.search(t) and not (_NOT_RE.search(t) or _REBUILD_NOT_RE.search(t)):
        return "[ACTION: server_status]"
    return None


def register(actions):
    actions["server_status"] = server_status
    try:
        su = globals().get("skill_utils") or {}
        reg = su.get("register_utterance_route") if isinstance(su, dict) else None
        if callable(reg):
            reg(_route, "server watch")
    except Exception:
        pass
    if host() and os.environ.get("JARVIS_NO_SCREEN_READ") != "1":
        threading.Thread(target=_poll_loop, name="server-watch", daemon=True).start()
        print(f"  [server-watch] watching the {name()} every "
              f"{int(float(_cfg('SERVER_WATCH_POLL_S', 300.0)))} s (read-only)")
