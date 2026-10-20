"""One JARVIS at a time across the owner's machines (2026-10-19).

The laptop is only a body: its brain is the desktop's GPU over Tailscale
(LOCAL_LLM_BASE_URL), so two running copies fight over one card and both
answer "Jarvis". JARVIS starting on one machine asks the other to stand down;
with HANDOFF_RESUME_WHEN_PEER_STOPS (the desktop) the machine that stood down
boots again by itself once the other copy stops.

  JARVIS starts here  -> ssh <peer> tools/stand_down.ps1 (writes the request
                         there and waits for that JARVIS to exit)
  JARVIS there        -> sees data/stand_down.json newer than its own start;
                         with resume on it launches tools/handoff_resume.ps1,
                         then shuts down ("Handing over to the laptop, sir.")
  handoff_resume.ps1  -> every 30 s: JARVIS running here again? exit. The
                         peer's JARVIS gone twice running, or the peer
                         unreachable for about 3 minutes? boot JARVIS here.

Off unless PEER_HANDOFF_SSH is set (data/user_settings.json). Staging and
headless instances never take part. stand_down.ps1 runs over SSH, where there
is no desktop, so it never STARTS anything; the waiter is launched from the
JARVIS process itself, in the owner's session, through WMI (not a child: the
shutdown's subprocess sweep must not take it down).
"""
from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import threading
import time

from core import config as cfg

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REQUEST_PATH = os.path.join(_ROOT, "data", "stand_down.json")
STAND_DOWN_SCRIPT = os.path.join(_ROOT, "tools", "stand_down.ps1")
RESUME_SCRIPT = os.path.join(_ROOT, "tools", "handoff_resume.ps1")
POLL_S = 2.0
SSH_TIMEOUT_S = 75.0

# "user@host" only: the target reaches a command line (ssh, WMI).
_TARGET_RE = re.compile(r"^[A-Za-z0-9._-]+@[A-Za-z0-9._:-]+$")
_HANDOFF_PREFIX = "handoff:"


def peer_target() -> str:
    t = str(getattr(cfg, "PEER_HANDOFF_SSH", "") or "").strip()
    return t if _TARGET_RE.match(t) else ""


def peer_name() -> str:
    return str(getattr(cfg, "PEER_HANDOFF_NAME", "") or "").strip() or "other machine"


def enabled() -> bool:
    if os.environ.get("JARVIS_STAGING") == "1" or os.environ.get("JARVIS_HEADLESS") == "1":
        return False
    return bool(peer_target())


def ssh_command(target: str, remote: str) -> list:
    return ["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=6",
            "-o", "StrictHostKeyChecking=accept-new", target, remote]


def stand_down_remote(from_name: str) -> str:
    """The command the PEER runs. Both machines keep the repo at the same
    path, so this machine's script path is the peer's too."""
    safe = re.sub(r"[^A-Za-z0-9 _-]", "", from_name or "")[:40] or "peer"
    return (f'powershell -NoProfile -ExecutionPolicy Bypass -File '
            f'"{STAND_DOWN_SCRIPT}" -From "{safe}"')


def ask_peer_to_stand_down(run=subprocess.run, log=print) -> str:
    """Ask the other machine's JARVIS to stand down; the one-line outcome.
    Never raises: an unreachable peer (asleep, off Tailscale) is normal."""
    target = peer_target()
    if not target:
        return "off"
    try:
        out = run(ssh_command(target, stand_down_remote(socket.gethostname())),
                  capture_output=True, text=True, timeout=SSH_TIMEOUT_S,
                  creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        lines = [ln.strip() for ln in (out.stdout or "").splitlines() if ln.strip()]
        result = lines[-1] if (out.returncode == 0 and lines) else f"unreachable (exit {out.returncode})"
    except Exception as e:
        result = f"unreachable ({type(e).__name__})"
    log(f"  [handoff] asked the {peer_name()} to stand down: {result}")
    return result


def read_request(path: str = REQUEST_PATH):
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            req = json.load(f)
        return req if isinstance(req, dict) else None
    except Exception:
        return None


def request_is_for_me(req, started_at: float) -> bool:
    """A request written AFTER this process started: one left over from an
    earlier handoff (before this boot) never stops the new instance."""
    try:
        return float(req.get("at")) > float(started_at)
    except Exception:
        return False


def process_started_at() -> float:
    """This process's creation time: a request that lands while JARVIS is
    still booting (before the watcher runs) still counts."""
    try:
        import psutil
        return float(psutil.Process(os.getpid()).create_time())
    except Exception:
        return time.time()


def handoff_reason(name: str) -> str:
    return _HANDOFF_PREFIX + (name or "")


def goodbye_for(reason) -> str:
    """The spoken line for a handoff shutdown; "" for any other shutdown."""
    if not isinstance(reason, str) or not reason.startswith(_HANDOFF_PREFIX):
        return ""
    name = reason[len(_HANDOFF_PREFIX):].strip() or "other machine"
    return f"Handing over to the {name}, sir."


def launch_resume_waiter(run=subprocess.run, log=print) -> bool:
    """Start tools/handoff_resume.ps1 through WMI so it outlives this
    process and runs in the owner's session (it may boot JARVIS later)."""
    target = peer_target()
    if not target or not os.path.exists(RESUME_SCRIPT):
        return False
    cmdline = (f"powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden "
               f"-File \"{RESUME_SCRIPT}\" -Peer {target} -StoodDownPid {os.getpid()}")
    ps = ("Invoke-CimMethod -ClassName Win32_Process -MethodName Create "
          "-Arguments @{CommandLine='" + cmdline.replace("'", "''") + "'} "
          "| Select-Object -ExpandProperty ReturnValue")
    try:
        out = run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
                  capture_output=True, text=True, timeout=30,
                  creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        ok = (out.stdout or "").strip().endswith("0")
    except Exception:
        ok = False
    log(f"  [handoff] resume waiter {'started' if ok else 'FAILED to start'}: "
        f"JARVIS comes back here when the {peer_name()}'s copy stops")
    return ok


def watch(started_at: float, on_stand_down, stop: threading.Event,
          path: str = REQUEST_PATH, poll_s: float = POLL_S,
          resume=launch_resume_waiter, log=print) -> bool:
    """Poll for a stand-down request meant for this instance; act on the
    first one, then stop watching. True when it stood down."""
    while not stop.is_set():
        req = read_request(path)
        if req and request_is_for_me(req, started_at):
            who = str(req.get("from") or "")
            log(f"  [handoff] standing down: JARVIS started on {who or 'the other machine'}")
            if getattr(cfg, "HANDOFF_RESUME_WHEN_PEER_STOPS", False):
                try:
                    resume()
                except Exception as e:
                    log(f"  [handoff] resume waiter error: {e}")
            on_stand_down(handoff_reason(peer_name()))
            return True
        stop.wait(poll_s)
    return False


_stop = threading.Event()


def _ask_then_arrive(on_arrival, log=print) -> str:
    """Ask the peer to stand down; when it did ("stood down"), it has pushed
    its memory and the conversation in progress before exiting, so
    ``on_arrival()`` can pick them up (2026-10-20, "JARVIS jumping from one
    machine to another like an identical copy")."""
    result = ask_peer_to_stand_down(log=log)
    if result == "stood down" and on_arrival is not None:
        try:
            on_arrival()
        except Exception as e:
            log(f"  [handoff] arrival hand-over failed: {e}")
    return result


def start(on_stand_down, log=print, on_arrival=None) -> bool:
    """Boot hook: ask the peer to stand down (background) and watch for the
    peer asking the same of this instance. ``on_arrival`` (optional) runs
    after the peer stood down. False when the feature is off."""
    if not enabled():
        return False
    started_at = process_started_at()
    threading.Thread(target=_ask_then_arrive, args=(on_arrival,),
                     kwargs={"log": log},
                     name="peer-handoff-ask", daemon=True).start()
    threading.Thread(target=watch, args=(started_at, on_stand_down, _stop),
                     kwargs={"log": log}, name="peer-handoff-watch",
                     daemon=True).start()
    log(f"  [handoff] on: the {peer_name()} stands down while JARVIS runs here")
    return True
