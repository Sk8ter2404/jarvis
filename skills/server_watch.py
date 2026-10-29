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
  * Health reads (3.0.6, Home Lab hub OK 2026-10-28, the rules are theirs):
    three more read-only commands, from the fixed READ_COMMANDS tuple only -
    the argv is never composed from model or spoken text:
      - `cat /var/local/emhttp/disks.ini`: speak ONLY when a disk's numErrors
        rises or its temp reaches 50 C.
      - `apcaccess -u`: STATUS only - speak on ONBATT and on the way back to
        ONLINE. NEVER load, overload, low battery or runtime ("record, do not
        nag").
      - `docker ps -a --format '{{.Names}} {{.Status}}'`: an Exited container
        is announced once; silent while an Agent Office board claim on
        server:unraid is held (someone is working on the server).
    SERVER_WATCH_HEALTH_READS turns them off.
  * SERVER_WATCH_SSH_KEY (3.0.6): the restricted key (forced command
    /boot/config/jarvis-ro.sh allows exactly READ_COMMANDS). When set, the
    argv carries `-i <key> -o IdentitiesOnly=yes`: without IdentitiesOnly ssh
    offers the default ~/.ssh and agent keys first, the server accepts the
    full-root key and the restriction silently does nothing (Home Lab hub,
    2026-10-28). Empty = today's behaviour (ssh picks its default key).

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


# The ONLY commands ever run on the server, byte for byte what the restricted
# key's forced command (/boot/config/jarvis-ro.sh, Home Lab hub 2026-10-28)
# allows; anything else is refused + logged there, and refused here first.
# Each goes as ONE argv element. Never built from model or spoken text.
CMD_MDCMD = "mdcmd status"
CMD_DISKS = "cat /var/local/emhttp/disks.ini"
CMD_UPS = "apcaccess -u"
CMD_DOCKER = "docker ps -a --format '{{.Names}} {{.Status}}'"
READ_COMMANDS = (CMD_MDCMD, CMD_DISKS, CMD_UPS, CMD_DOCKER)

_INVALID_KEY = object()


def ssh_key():
    """SERVER_WATCH_SSH_KEY: "" = today's behaviour (ssh's default key).
    A value that could be read as an ssh option or carries control
    characters is _INVALID_KEY - the caller then runs NOTHING rather than
    fall back to the default (full-root) key."""
    k = str(_cfg("SERVER_WATCH_SSH_KEY", "") or "").strip()
    if not k:
        return ""
    if k.startswith("-") or any(ord(c) < 32 for c in k):
        return _INVALID_KEY
    # Review 2026-10-28: ssh only WARNS when an `-i` file cannot be read and
    # then falls back to the default identities (~/.ssh/id_ed25519 = the
    # full-root key) - IdentitiesOnly=yes limits agent keys, not those. So a
    # typo, a deleted key or one not yet deployed would silently run every
    # read as root. Checked on EVERY call (the key can vanish while JARVIS
    # runs). ssh expands a leading "~" itself, so the check does too.
    try:
        present = os.path.isfile(os.path.expanduser(k))
    except Exception:
        present = False
    if not present:
        _warn_missing_key_once(k)
        return _INVALID_KEY
    return k


_warned_key = [None]


def _warn_missing_key_once(k: str) -> None:
    """Log the missing key once per path (a poll runs 4 reads a time)."""
    try:
        if _warned_key[0] != k:
            _warned_key[0] = k
            print("  [server-watch] key file missing - not reading")
    except Exception:
        pass


def ssh_argv(target: str, cmd: str, key=None):
    """The exact ssh argv for one allow-listed read, or None (never run).
    With a key: ["ssh", "-n", "-i", KEY, "-o", "IdentitiesOnly=yes", then
    today's options] - the order the Home Lab hub sent (scope D2)."""
    if cmd not in READ_COMMANDS or not target:
        return None
    key = ssh_key() if key is None else key
    if key is _INVALID_KEY:
        return None
    argv = ["ssh", "-n"]
    if key:
        argv += ["-i", str(key), "-o", "IdentitiesOnly=yes"]
    return argv + ["-o", "BatchMode=yes", "-o", "ConnectTimeout=6",
                   "-o", "StrictHostKeyChecking=accept-new", target, cmd]


def _run_read(target: str, cmd: str, run=subprocess.run):
    """stdout of one allow-listed read, or None when it could not be read."""
    argv = ssh_argv(target, cmd)
    if argv is None:
        return None
    try:
        out = run(argv, capture_output=True, text=True, timeout=SSH_TIMEOUT_S,
                  creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return out.stdout if out.returncode == 0 else None
    except Exception:
        return None


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
    text = _run_read(target, CMD_MDCMD, run)
    return parse_mdcmd(text) if text is not None else {}


def _int_or_none(v):
    v = str(v if v is not None else "").strip().strip('"')
    return int(v) if v.lstrip("-").isdigit() else None


def parse_disks_ini(text: str) -> dict:
    """{disk name: {"temp": int|None, "errors": int|None}} from
    /var/local/emhttp/disks.ini (["name"] sections, key="value" lines). A
    spun-down disk reports temp="*" -> None. Unknown output -> {}."""
    out, cur = {}, None
    try:
        for line in str(text or "").splitlines():
            line = line.strip()
            if line.startswith("[") and line.endswith("]"):
                cur = line[1:-1].strip().strip('"') or None
                if cur:
                    out.setdefault(cur, {"temp": None, "errors": None})
                continue
            k, sep, v = line.partition("=")
            if not sep or cur is None:
                continue
            k = k.strip()
            if k == "temp":
                out[cur]["temp"] = _int_or_none(v)
            elif k == "numErrors":
                out[cur]["errors"] = _int_or_none(v)
    except Exception:
        return {}
    return out


def parse_ups_status(text: str) -> str:
    """The STATUS field of `apcaccess -u` and NOTHING else: "ONBATT",
    "ONLINE" or "" (unknown). LOADPCT, BCHARGE, TIMELEFT and the LOWBATT /
    OVERLOAD flags are deliberately never read (Home Lab hub: record, do not
    nag)."""
    try:
        for line in str(text or "").splitlines():
            k, sep, v = line.partition(":")
            if sep and k.strip().upper() == "STATUS":
                words = v.upper().split()
                if "ONBATT" in words:
                    return "ONBATT"
                if "ONLINE" in words:
                    return "ONLINE"
                return ""
    except Exception:
        pass
    return ""


def parse_docker(text: str) -> dict:
    """{container name: status text} from `docker ps -a --format
    '{{.Names}} {{.Status}}'` ("plex Up 3 days", "foo Exited (0) 2 hours
    ago")."""
    out = {}
    try:
        for line in str(text or "").splitlines():
            nm, _sp, status = line.strip().partition(" ")
            # Docker's own name rule; anything else is not a name to speak.
            if re.match(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$", nm):
                out[nm] = status.strip()
    except Exception:
        return {}
    return out


def _exited(status: str) -> bool:
    return str(status or "").startswith("Exited")


def health_reads_on() -> bool:
    return bool(_cfg("SERVER_WATCH_HEALTH_READS", True))


def read_health(target: str, run=subprocess.run) -> dict:
    """The three health reads, in parallel. A read that fails is LEFT OUT
    (never an empty guess), so a failed docker read never looks like "no
    containers" and a failed UPS read never looks like a power change."""
    if not target or not health_reads_on():
        return {}
    cmds = ((CMD_DISKS, "disks", parse_disks_ini), (CMD_UPS, "ups", parse_ups_status),
            (CMD_DOCKER, "docker", parse_docker))
    try:
        with ThreadPoolExecutor(max_workers=len(cmds)) as ex:
            texts = list(ex.map(lambda c: _run_read(target, c[0], run), cmds))
    except Exception:
        return {}
    out = {}
    for (_cmd, key, parse), text in zip(cmds, texts):
        if text is None:
            continue
        val = parse(text)
        if val:
            out[key] = val
    return out


def check(h=None, target=None, opener=_port_open, run=subprocess.run) -> dict:
    h = host() if h is None else h
    target = ssh_target() if target is None else target
    up_services = probe_ports(h, opener) if h else []
    array, health = {}, {}
    if up_services and target:
        # In parallel, so "how's the server" waits no longer than one read.
        try:
            with ThreadPoolExecutor(max_workers=2) as ex:
                fa = ex.submit(read_array, target, run)
                fh = ex.submit(read_health, target, run)
                array, health = fa.result(), fh.result()
        except Exception:
            array, health = {}, {}
    return {"up": bool(up_services), "services": up_services, "array": array,
            "health": health, "at": time.time()}


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


HOT_C = 50          # Home Lab hub rule: speak at temp >= 50 C
HOT_REARM_C = 45    # a disk hovering at 49/50 is said once, not every poll
BOARD_PATH = "/api/board"
BOARD_TIMEOUT_S = 2.0
BOARD_AREA = "server:unraid"
_OFFICE_DEFAULT = "http://127.0.0.1:8795"


def _disk_word(d: str) -> str:
    d = str(d)
    m = re.match(r"^disk(\d+)$", d)
    if m:
        return "disk " + m.group(1)
    if d == "parity2":
        return "the second parity"
    return d.replace("_", " ")


def _office_url() -> str:
    u = str(_cfg("AGENT_OFFICE_URL", _OFFICE_DEFAULT) or "").strip().rstrip("/")
    return u if re.match(r"^https?://[A-Za-z0-9._:\[\]-]+$", u) else _OFFICE_DEFAULT


def _fetch_board(url: str, timeout: float = BOARD_TIMEOUT_S) -> dict:
    import urllib.request
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with opener.open(req, timeout=timeout) as resp:
        raw = resp.read(1_000_001)
    data = json.loads(raw.decode("utf-8", "replace"))
    return data if isinstance(data, dict) else {}


def board_claim_held(fetch=None, now=None) -> bool:
    """True while an Agent Office board claim on server:unraid is held (the
    owner or a chat is working on the server, e.g. the 10-28 parity swap).
    One GET of AGENT_OFFICE_URL/api/board, 2 s; the Office not running or any
    fault = False, quietly. Board text is data: only "what" and "until" are
    looked at, and nothing from it is ever spoken."""
    fetch = fetch or _fetch_board
    now = time.time() if now is None else now
    try:
        board = fetch(_office_url() + BOARD_PATH, BOARD_TIMEOUT_S)
        for c in (board or {}).get("claims") or []:
            if not isinstance(c, dict):
                continue
            if str(c.get("what") or "").strip().lower() != BOARD_AREA:
                continue
            until = c.get("until")
            try:
                if until is not None and float(until) <= now:
                    continue
            except (TypeError, ValueError):
                pass
            return True
    except Exception:
        return False
    return False


def step_health(state: dict, health: dict, claim_held=None):
    """Fold the health reads into the believed state. Returns (state, [lines])
    - only the Home Lab hub's speak rules ever produce a line:
      * a disk's numErrors rising, or its temp reaching HOT_C (said once,
        again only after it cooled below HOT_REARM_C);
      * UPS STATUS ONBATT, and ONLINE again after ONBATT;
      * a container newly Exited, once - unless a server:unraid board claim
        is held (then it is noted silently and never said later: the work
        that holds the claim stopped it).
    The first read of each kind is a silent baseline (a restart never
    re-announces). Never raises."""
    claim_held = claim_held or board_claim_held
    st = dict(state or {})
    lines = []
    try:
        nm = name()
        disks = (health or {}).get("disks")
        if isinstance(disks, dict) and disks:
            known_err = dict(st.get("disk_errors") or {})
            hot = set(st.get("disk_hot") or [])
            first = "disk_errors" not in st
            for d, info in sorted(disks.items()):
                err, temp = info.get("errors"), info.get("temp")
                if err is not None:
                    before = known_err.get(d)
                    if not first and before is not None and err > int(before):
                        n = err - int(before)
                        lines.append(f"Sir, {_disk_word(d)} on the {nm} logged "
                                     f"{n} new error{'' if n == 1 else 's'}.")
                    known_err[d] = err      # a reset to 0 (array restart) is silent
                if temp is not None:
                    if temp >= HOT_C and d not in hot:
                        hot.add(d)
                        if not first:
                            lines.append(f"Sir, {_disk_word(d)} on the {nm} is at "
                                         f"{temp} degrees Celsius.")
                    elif temp < HOT_REARM_C:
                        hot.discard(d)
            st["disk_errors"], st["disk_hot"] = known_err, sorted(hot)
        ups = (health or {}).get("ups")
        if ups in ("ONBATT", "ONLINE"):
            before = st.get("ups")
            if before == "ONLINE" and ups == "ONBATT":
                lines.append(f"Sir, the {nm} is running on battery.")
            elif before == "ONBATT" and ups == "ONLINE":
                lines.append(f"The {nm} is back on mains power, sir.")
            st["ups"] = ups
        dock = (health or {}).get("docker")
        if isinstance(dock, dict) and dock:
            first = "exited" not in st
            said = set(st.get("exited") or [])
            now_exited = {c for c, status in dock.items() if _exited(status)}
            new = sorted(now_exited - said)
            if new and not first and not claim_held():
                names = (", ".join(new[:-1]) + " and " + new[-1]) if len(new) > 1 else new[0]
                lines.append(f"Sir, the {names} container{'s' if len(new) > 1 else ''} "
                             f"on the {nm} stopped.")
            # A container that is up again is forgotten, so a later exit is
            # said again; one still Exited stays said.
            st["exited"] = sorted(now_exited)
    except Exception:
        return dict(state or {}), []
    return st, lines


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
    # Health reads: only the conditions the poll itself would speak about.
    h = result.get("health") or {}
    if h.get("ups") == "ONBATT":
        line += " It's running on battery."
    hot = [(d, i.get("temp")) for d, i in sorted((h.get("disks") or {}).items())
           if isinstance(i, dict) and i.get("temp") is not None and i["temp"] >= HOT_C]
    for d, t in hot[:3]:
        w = _disk_word(d)
        line += f" {w[:1].upper() + w[1:]} is at {t} degrees Celsius."
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
                lines = []
                if result.get("up") and result.get("health"):
                    state, lines = step_health(state, result["health"])
                save_state(state)
            for m in ([msg] if msg else []) + lines:
                print(f"  [server-watch] {m}")
                _announce(m)
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
