"""The clone voice server's memory, read from JARVIS's side (v3.0.27,
overhaul finding reliability-08, the part v3.0.26 left open).

WHY
===
The 11-16/17 RAM alerts said "Python at 22-28 GB" and nobody could tell which
python.exe it was. One suspect was the Chatterbox clone voice server: it ran
from 11-13 to 11-17 without a restart. v3.0.26 names it in the RAM alert's
"[mem-users]" line, but only when RAM is already over 90 % and only when it is
in the top three. Its own size was never tracked, so a slow growth could not
be seen until the whole PC was out of memory.

WHAT THIS MODULE DOES (psutil injected; REPORT ONLY)
====================================================
``VoiceServerMem.sample(pid)`` reads the resident size (RSS) of the voice
server's process - the pid the server reports in its /health, which the
client keeps (core.clone_voice_client.CLIENT.server_pid()). The server
script lives outside this repo and is never touched. A reading is logged:

* the first one for a server pid, and again whenever it moved by
  LOG_STEP_MB or more since the last logged one:
  ``[voice-mem] voice server pid P rss 3.0 GB``;
* ONE warning per server pid when it passes VOICE_SERVER_RSS_WARN_MB
  (core/config.py, default 6000; 0 = no warning):
  ``[voice-mem] WARNING voice server pid P rss 6.2 GB is over the 6000 MB
  limit (VOICE_SERVER_RSS_WARN_MB) - report only, it is not restarted``,
  and the caller's ``on_warn`` hook runs once (the sysmon monitor stamps its
  HUD / tray alert from it).

The server is NEVER restarted, stopped or killed here: the owner forbids
watchdogs and auto-kill on this PC. Restarting it stays his call.
"""
from __future__ import annotations

import sys
import threading
import time

DEFAULT_WARN_MB = 6000.0
LOG_STEP_MB = 1024.0        # log again after this much growth / shrink
SAMPLE_EVERY_S = 60.0       # the sysmon loop samples at most this often
_MB = 1024.0 * 1024.0


def _psutil():
    try:
        import psutil
        return psutil
    except Exception:       # pragma: no cover - psutil is a hard dependency
        return None


def rss_mb(pid, psutil_mod=None):
    """Resident size of ``pid`` in MB, or None (no pid, gone, unreadable).
    Never raises."""
    try:
        if pid is None:
            return None
        pid = int(pid)
        if pid <= 0:
            return None
        ps = psutil_mod if psutil_mod is not None else _psutil()
        if ps is None:
            return None
        return float(ps.Process(pid).memory_info().rss) / _MB
    except Exception:
        return None


def clone_server_pid():
    """The pid the clone voice server reported in /health, when the client
    module is loaded (never imported here) and knows one; else None. Never
    raises."""
    try:
        cvc = sys.modules.get("core.clone_voice_client")
        client = getattr(cvc, "CLIENT", None) if cvc is not None else None
        if client is None:
            return None
        pid = client.server_pid()
        return int(pid) if pid else None
    except Exception:
        return None


def warn_mb_setting(default: float = DEFAULT_WARN_MB) -> float:
    """VOICE_SERVER_RSS_WARN_MB from core.config, read live (0 = off).
    Never raises."""
    try:
        import core.config as _cfg
        v = float(getattr(_cfg, "VOICE_SERVER_RSS_WARN_MB", default))
        return max(0.0, v)
    except Exception:
        return float(default)


def _gb(mb: float) -> str:
    return f"{float(mb) / 1024.0:.1f} GB"


class VoiceServerMem:
    """See the module docstring. Collaborators injected for tests:
    ``pid_fn() -> pid|None``, ``warn_mb() -> float``, ``log(str)``,
    ``clock()``, ``psutil_mod``."""

    def __init__(self, *, pid_fn=None, warn_mb=None, log=None, clock=None,
                 psutil_mod=None, every_s: float = SAMPLE_EVERY_S):
        self._pid_fn = pid_fn or clone_server_pid
        self._warn_mb = warn_mb or warn_mb_setting
        self._log = log or print
        self._clock = clock or time.monotonic
        self._ps = psutil_mod
        self._every_s = float(every_s)
        self._lock = threading.Lock()
        self._last_at = None
        self._logged = {}           # pid -> rss MB last logged
        self._warned = set()        # pids already warned about
        self.last = None            # (pid, rss_mb) of the last reading

    def _say(self, msg: str) -> None:
        try:
            self._log(msg)
        except Exception:
            pass

    def due(self) -> bool:
        with self._lock:
            return (self._last_at is None
                    or (float(self._clock()) - self._last_at) >= self._every_s)

    def tick(self, on_warn=None):
        """Sample when due (every ``every_s``); see sample(). Never raises."""
        try:
            if not self.due():
                return None
            return self.sample(on_warn=on_warn)
        except Exception:
            return None

    def sample(self, pid=None, on_warn=None):
        """Read the server's RSS now. Returns (pid, rss_mb) or None (no
        server / unreadable). Logs and warns as the module docstring says;
        ``on_warn(line)`` runs once per pid with the warning. Never raises,
        never touches the process beyond reading it."""
        try:
            with self._lock:
                self._last_at = float(self._clock())
            if pid is None:
                pid = self._pid_fn()
            mb = rss_mb(pid, self._ps)
            if mb is None:
                return None
            pid = int(pid)
            with self._lock:
                self.last = (pid, mb)
                prev = self._logged.get(pid)
                log_it = prev is None or abs(mb - prev) >= LOG_STEP_MB
                if log_it:
                    self._logged[pid] = mb
                try:
                    limit = float(self._warn_mb())
                except Exception:
                    limit = DEFAULT_WARN_MB
                warn = limit > 0 and mb > limit and pid not in self._warned
                if warn:
                    self._warned.add(pid)
            if log_it:
                self._say(f"  [voice-mem] voice server pid {pid} rss "
                          f"{_gb(mb)}")
            if warn:
                line = (f"  [voice-mem] WARNING voice server pid {pid} rss "
                        f"{_gb(mb)} is over the {limit:.0f} MB limit "
                        "(VOICE_SERVER_RSS_WARN_MB) - report only, it is "
                        "not restarted")
                self._say(line)
                if on_warn is not None:
                    try:
                        on_warn(line)
                    except Exception:
                        pass
            return (pid, mb)
        except Exception:
            return None

    def describe(self) -> str:
        """'voice server pid P rss X GB' for the last reading, or ''."""
        try:
            with self._lock:
                last = self.last
            if not last:
                return ""
            return f"voice server pid {last[0]} rss {_gb(last[1])}"
        except Exception:
            return ""


MONITOR = VoiceServerMem()
