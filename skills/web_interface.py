"""web_interface skill — start/stop/status the live LAN web dashboard.

THE FEATURE
===========
A local-LAN web page (served by tools/web_interface.py) to SEE what JARVIS is
doing — a live tail of the session log plus a status strip (version / awake
state / model routing / VRAM) — and to TALK TO HIM BY TEXT. A typed command is
fed through the EXACT SAME file-based inject channel (injected_commands.json)
that the voice loop drains, so a typed turn behaves identically to a spoken one.

THIS MODULE is only the thin lifecycle wiring: it registers the three voice
actions (web_interface_on / _off / _status), auto-starts the server at boot iff
core.config.WEB_INTERFACE_ENABLED is True, and owns the single daemon-thread
server so on/off/status all see the same instance. ALL of the HTTP + inject +
status logic lives in tools/web_interface.py (stdlib-only, unit-tested without a
GUI, a browser, or a real JARVIS).

SAFETY MODEL (mirrors skills/air_control.py)
============================================
  • WEB_INTERFACE_ENABLED defaults to False. The skill ALWAYS loads (so the
    voice actions exist), but the server only AUTO-STARTS at boot when the knob
    is True. Saying "start the web interface" starts it regardless (the explicit
    command is the owner's consent) — the knob only guards unattended auto-start.
  • LAN EXPOSURE: create_server() REFUSES to bind a non-loopback address with an
    empty token (raises InsecureBindError). web_interface_on surfaces that refusal
    as a spoken sentence so the owner knows to set a token — the server is NOT
    started wide-open, ever.
  • A staging / test instance never starts the server (JARVIS_STAGING gate) so a
    green blue/green instance can't fight prod for the port.
  • Daemon thread + clean shutdown on "off": httpd.shutdown() unblocks
    serve_forever, then server_close() releases the socket.

Voice actions (each returns ONE finished sentence — listed in
bobert_companion.SPEAK_RESULT_VERBATIM_ACTIONS so it's spoken verbatim):
  web_interface_on / web_interface_off / web_interface_status
"""
from __future__ import annotations

import os
import sys
import threading

# The server engine. Import defensively so a (should-never-happen) load failure
# disables the feature without taking down the skill loader — the same pattern
# gpu_usage/air_control use for their core engines.
try:
    from tools import web_interface as _engine
    _HAS_ENGINE = True
except Exception as _exc:   # pragma: no cover - tools.web_interface is in-tree
    _engine = None          # type: ignore[assignment]
    _HAS_ENGINE = False
    print(f"  [web-interface] tools.web_interface unavailable ({_exc}); "
          f"actions reply gracefully, server disabled")


# ─── loop/server state (module-level so on/off/status share one instance) ────
_server_lock = threading.Lock()
_httpd = None                 # the live ThreadingHTTPServer, or None
_serve_thread = None          # its daemon serve_forever() thread
_bound = ("", 0)              # (bind, port) the live server is on, for status


# ─── config helpers (read FRESH each call so a Settings toggle takes effect
#     without a restart — matches skills/air_control.py:_cfg_flag) ────────────
def _cfg(name: str, default):
    try:
        from core import config as _c
        return getattr(_c, name, default)
    except Exception:
        return default


def _bc():
    """The live monolith module if it's importable (for _is_staging). None under
    the isolated skill test harness, which is fine — those tests don't stage."""
    return sys.modules.get("bobert_companion")


def _is_staging() -> bool:
    """True on the staging/test instance — the server must NOT start there (a
    green instance would fight prod for the port). Env var first so it holds even
    before the monolith is importable; then the monolith's own gate if present."""
    if os.environ.get("JARVIS_STAGING", "").strip() == "1":
        return True
    bc = _bc()
    fn = getattr(bc, "_is_staging", None) if bc is not None else None
    if callable(fn):
        try:
            return bool(fn())
        except Exception:
            return False
    return False


# ─── start / stop ────────────────────────────────────────────────────────────
def _start() -> tuple[bool, str]:
    """Start the server from the live config. Returns (ok, message) where message
    is a finished spoken sentence. Idempotent: a second start reports it's already
    running. NEVER raises — an InsecureBindError (non-local bind, empty token) or
    a port-in-use OSError becomes an honest spoken reason, and nothing is left
    half-started."""
    if not _HAS_ENGINE:
        return False, ("The web interface engine didn't load, sir — I can't "
                       "start it.")
    with _server_lock:
        global _httpd, _serve_thread, _bound
        if _httpd is not None:
            b, p = _bound
            return True, f"The web interface is already running, sir, on {b} port {p}."
        if _is_staging():
            return False, "Not while I'm in staging, sir."
        bind = str(_cfg("WEB_INTERFACE_BIND", "127.0.0.1"))
        port = int(_cfg("WEB_INTERFACE_PORT", 8766))
        token = str(_cfg("WEB_INTERFACE_TOKEN", ""))
        try:
            httpd = _engine.create_server(bind=bind, port=port, token=token)
        except _engine.InsecureBindError:
            # The security refusal — bind is non-local and no token is set.
            return False, (
                f"I won't expose the web interface on {bind} without a token, sir "
                f"— it can inject commands I execute. Set a web interface token in "
                f"Settings, or bind it to localhost.")
        except OSError as e:
            return False, (f"I couldn't start the web interface on {bind} port "
                           f"{port}, sir — {e.strerror or e}.")
        except Exception as e:  # pragma: no cover - defensive
            return False, f"The web interface failed to start, sir — {e}."
        _serve_thread = _engine.serve_in_thread(httpd)
        _httpd = httpd
        _bound = (bind, port)
        where = ("localhost" if _engine.is_local_bind(bind) else bind)
        secured = " (token required)" if token else ""
        return True, (f"Web interface online, sir — open http://{where}:{port} "
                      f"in your browser{secured}.")


def _stop() -> tuple[bool, str]:
    """Stop the server and release its socket. Returns (ok, message). Idempotent:
    stopping an already-stopped server reports that plainly. serve_forever is
    unblocked by shutdown(); server_close() frees the port. Never raises."""
    with _server_lock:
        global _httpd, _serve_thread, _bound
        if _httpd is None:
            return True, "The web interface is already off, sir."
        httpd = _httpd
        _httpd = None
        thread = _serve_thread
        _serve_thread = None
        _bound = ("", 0)
    # Do the (potentially blocking) shutdown OUTSIDE the lock so a status call
    # can't deadlock behind it.
    try:
        httpd.shutdown()
    except Exception:
        pass
    try:
        httpd.server_close()
    except Exception:
        pass
    if thread is not None:
        try:
            thread.join(timeout=3.0)
        except Exception:
            pass
    return True, "Web interface off, sir."


def _status() -> str:
    """One finished spoken sentence describing whether the server is running and
    where. Read-only; safe to call anytime."""
    if not _HAS_ENGINE:
        return "The web interface engine isn't loaded, sir."
    with _server_lock:
        running = _httpd is not None
        b, p = _bound
    if running:
        token = str(_cfg("WEB_INTERFACE_TOKEN", ""))
        where = ("localhost" if _engine.is_local_bind(b) else b)
        secured = " with a token required" if token else ""
        return (f"The web interface is running, sir, on {where} port {p}"
                f"{secured}.")
    enabled = bool(_cfg("WEB_INTERFACE_ENABLED", False))
    hint = ("" if enabled
            else " It's set to stay off at boot; say 'start the web interface' "
                 "to bring it up.")
    return f"The web interface is off, sir.{hint}"


# ─── action registration ─────────────────────────────────────────────────────
def register(actions):
    def web_interface_on(_: str = "") -> str:
        ok, msg = _start()
        return msg

    def web_interface_off(_: str = "") -> str:
        ok, msg = _stop()
        return msg

    def web_interface_status(_: str = "") -> str:
        return _status()

    actions["web_interface_on"] = web_interface_on
    actions["web_interface_off"] = web_interface_off
    actions["web_interface_status"] = web_interface_status

    if not _HAS_ENGINE:
        print("  [web-interface] engine missing — actions reply gracefully; "
              "server disabled.")
        return

    # AUTO-START only when the owner opted in via WEB_INTERFACE_ENABLED (default
    # False — a fresh install must never open a LAN socket uninvited). The voice
    # action can still start it later. Never in staging.
    if bool(_cfg("WEB_INTERFACE_ENABLED", False)) and not _is_staging():
        ok, msg = _start()
        if ok:
            b, p = _bound
            print(f"  [web-interface] auto-started (WEB_INTERFACE_ENABLED) on "
                  f"{b}:{p}"
                  f"{' [flask present, using stdlib]' if _engine.FLASK_AVAILABLE else ''}")
        else:
            # The security refusal / port-clash reason — log it clearly so the
            # owner understands why the dashboard isn't up.
            print(f"  [web-interface] enabled but NOT started: {msg}")
    else:
        print("  [web-interface] loaded (server off — say 'start the web "
              "interface' to bring it up; WEB_INTERFACE_ENABLED auto-start is "
              "off by default)")
