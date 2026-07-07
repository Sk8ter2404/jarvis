"""Live web interface for JARVIS — a local-LAN dashboard + text command channel.

WHAT THIS IS
============
A tiny, dependency-light HTTP server that lets the owner (1) SEE what JARVIS is
doing from a browser — a live tail of the session log, plus a status strip
(version / awake state / model routing / VRAM) — and (2) TALK TO HIM BY TEXT: a
typed command is fed through the *exact same* file-based inject channel a spoken
command uses, so a typed "what time is it" behaves identically to the spoken one.

WHY STDLIB http.server (not Flask)
==================================
Flask already lives in this environment (the AirTag tracker runs on it at :8443),
but hard-depending on it here would (a) risk clashing with that server's import
state and (b) break bare-CI / cloud-only installs that don't ship Flask. So this
module is PURE STDLIB (http.server + socketserver + threading + json). We *probe*
for Flask lazily only to note its availability in logs — we never import-fail on
its absence and never actually route through it. One code path, everywhere.

HOW THE INJECT CHANNEL WORKS (reused verbatim from the voice loop)
==================================================================
JARVIS's main loop calls ``_drain_injected_command()`` at the top of every
iteration (bobert_companion.py). That function atomically renames
``injected_commands.json`` to ``.consuming``, pops the FIRST list item, and
requeues the tail. Each item is either a bare string or ``{"text": "...", ...}``.
We APPEND to that same file with the same atomic write-temp-then-os.replace
pattern the run-jarvis driver uses, so a typed command enters the loop exactly
as a mic turn would. The REPLY is read back by tailing the session log
(``logs/session_*.log``) from the byte offset captured at inject time, watching
for the ``JARVIS:`` / ``[action]`` lines the loop prints for that turn — mirroring
driver.py's ``wait_for_reply``. If no reply lands inside the timeout we return
``accepted: true`` (the command still ran; we just didn't capture spoken text).

SECURITY MODEL
==============
The endpoint can INJECT COMMANDS JARVIS EXECUTES, so binding it off-box is a real
exposure. The server therefore:
  • binds 127.0.0.1 by default (loopback — unreachable off the machine),
  • REFUSES TO START on a non-local bind (0.0.0.0 / a LAN IP) when the token is
    empty — ``create_server`` raises ``InsecureBindError`` with a clear reason,
  • when a token IS set, requires it on EVERY request via an Authorization: Bearer
    header, an X-Auth-Token header, or a ?token=… query param; a mismatch is 401.
The GET / dashboard page is served WITHOUT a token (it's just static HTML/JS that
then supplies the token on its API calls) ONLY on a local bind; on a non-local
bind even the page requires the token, so a bare browser hit can't fingerprint us.

TESTABILITY / HEADLESS-CI CONTRACT
==================================
Everything here is stdlib and OS-neutral (no win32, no real JARVIS needed):
  • ``create_server`` takes explicit ``inject_path`` / ``log_dir`` / ``hud_state_path``
    so a test can point them at a temp dir and bind 127.0.0.1:0 (ephemeral port).
  • The status/log/gpu sources all DEGRADE GRACEFULLY when the file/JARVIS is
    absent (missing log → empty tail; missing hud_state → unknown state; gpu
    import failure → omitted). Nothing here raises into a request handler.
  • The reply-wait is injected as ``reply_reader`` so a test can stub it (no live
    log to tail) — the default reader tails the newest session log.
"""
from __future__ import annotations

import glob
import html
import json
import os
import tempfile
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# ── Optional Flask probe (informational only — we NEVER route through it) ───
# Documented in the module header: Flask is present in this env for the AirTag
# tracker, but hard-depending on it would break bare CI. We note availability so
# a boot log can say "(flask present, using stdlib anyway)" but the server is
# always the stdlib ThreadingHTTPServer below.
try:  # pragma: no cover - trivial import probe; result only affects a log line
    import flask as _flask  # noqa: F401
    FLASK_AVAILABLE = True
except Exception:  # pragma: no cover
    FLASK_AVAILABLE = False


# ── project paths (defaults; overridable per-instance for tests / blue-green) ─
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(_THIS_DIR)          # tools/ -> project root
DEFAULT_INJECT_PATH = os.path.join(PROJECT_DIR, "injected_commands.json")
DEFAULT_LOG_DIR = os.path.join(PROJECT_DIR, "logs")
DEFAULT_HUD_STATE_PATH = os.path.join(PROJECT_DIR, "hud_state.json")

# Reply-wait bounds. The main loop can take a while on a cloud LLM turn, so we
# allow a generous ceiling but poll cheaply. A caller (the POST handler) passes a
# per-request timeout; these just bound it so a hostile ?timeout can't hang a
# worker thread forever.
_REPLY_TIMEOUT_DEFAULT = 30.0
_REPLY_TIMEOUT_MAX = 120.0
_LOG_TAIL_MAX_LINES = 2000        # hard cap on ?lines= so a request can't slurp a huge log


class InsecureBindError(RuntimeError):
    """Raised by ``create_server`` when asked to bind a non-local address with an
    empty token. Refusing to start (rather than binding wide-open) is the whole
    point of the security contract, so this is a hard error the caller logs."""


# ── local-bind detection ────────────────────────────────────────────────────
_LOCAL_BINDS = {"127.0.0.1", "localhost", "::1", "127.0.0.1/32"}


def is_local_bind(bind: str) -> bool:
    """True when ``bind`` is a loopback address that nothing off-box can reach.
    Everything else (0.0.0.0, a LAN IP, a hostname) is treated as EXPOSED and so
    requires a token. Kept deliberately strict — an unknown value is 'exposed'."""
    return (bind or "").strip().lower() in _LOCAL_BINDS


# ── data sources (all graceful — a missing JARVIS/file degrades, never raises) ─

def _newest_log(log_dir: str) -> str | None:
    """Path to the most-recently-modified session_*.log, or None."""
    try:
        files = glob.glob(os.path.join(log_dir, "session_*.log"))
        if not files:
            return None
        return max(files, key=os.path.getmtime)
    except Exception:
        return None


def tail_log(log_dir: str, lines: int) -> dict:
    """Return the last ``lines`` lines of the newest session log as a dict::

        {"log": "<basename or ''>", "lines": [...], "running": bool}

    ``running`` is a best-effort liveness flag: the newest log was written to
    within the last 20 s (the loop logs whisper/vad activity constantly). Never
    raises — a missing logs dir yields an empty tail with running=False."""
    lines = max(1, min(int(lines or 50), _LOG_TAIL_MAX_LINES))
    lg = _newest_log(log_dir)
    if not lg:
        return {"log": "", "lines": [], "running": False}
    try:
        with open(lg, encoding="utf-8", errors="replace") as f:
            tail = f.readlines()[-lines:]
    except Exception:
        return {"log": os.path.basename(lg), "lines": [], "running": False}
    try:
        running = (time.time() - os.path.getmtime(lg)) < 20.0
    except Exception:
        running = False
    return {
        "log": os.path.basename(lg),
        "lines": [ln.rstrip("\n") for ln in tail],
        "running": running,
    }


def _read_hud_state(hud_state_path: str) -> dict:
    """Best-effort read of hud_state.json (empty dict on any error). Mirrors
    tray._read_hud_state — the same file the tray + HUD consume."""
    try:
        with open(hud_state_path, encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def _read_version() -> str:
    """The shareable release string (VERSION file via core.version). Falls back
    to reading the VERSION file directly, then to 'unknown' — so a partial tree
    (or a test that imported us bare) still answers."""
    try:
        from core.version import version_string
        return version_string()
    except Exception:
        try:
            with open(os.path.join(PROJECT_DIR, "VERSION"), encoding="utf-8") as f:
                return f.read().strip() or "unknown"
        except Exception:
            return "unknown"


def _awake_state(hud: dict) -> str:
    """Map hud_state.json's sleep/standby flags into a single word for the strip.
    ``state`` is the canonical label the main loop writes ('Idle'/'Standby'/…);
    fall back to the boolean flags if it's absent."""
    state = hud.get("state")
    if isinstance(state, str) and state.strip():
        return state.strip()
    if hud.get("sleep_mode") or hud.get("standby_mode"):
        return "Asleep"
    return "Unknown"


def _gpu_summary() -> dict:
    """One-shot VRAM/model summary via core.gpu_usage, or a graceful stub.
    Returns ``{"lines": [...], "bar": "..."}``; on any failure (module missing on
    a cloud-only box, no GPU) returns empty lines so the strip shows 'GPU: n/a'."""
    try:
        from core import gpu_usage
        snap = gpu_usage.gpu_snapshot()
        return {"lines": gpu_usage.usage_lines(snap), "bar": gpu_usage.usage_bar(14, snap)}
    except Exception:
        return {"lines": [], "bar": ""}


def build_status(hud_state_path: str, log_dir: str) -> dict:
    """Assemble the /api/status payload from every (graceful) source."""
    hud = _read_hud_state(hud_state_path)
    lg = _newest_log(log_dir)
    running = False
    if lg:
        try:
            running = (time.time() - os.path.getmtime(lg)) < 20.0
        except Exception:
            running = False
    gpu = _gpu_summary()
    return {
        "version": _read_version(),
        "state": _awake_state(hud),
        "running": running,
        "model": hud.get("last_intent_tag", ""),   # best-effort; routing is the real model view
        "routing": (_gpu_summary_routing(gpu)),
        "now_playing": hud.get("now_playing", ""),
        "last_spoken": hud.get("last_spoken", ""),
        "last_transcript": hud.get("last_transcript", ""),
        "gpu_lines": gpu["lines"],
        "gpu_bar": gpu["bar"],
        "ts": time.time(),
    }


def _gpu_summary_routing(gpu: dict) -> str:
    """The routing line is the last entry of usage_lines() (chat→…  vision→…).
    Pulled out so the status payload carries a compact 'which brain' string."""
    lines = gpu.get("lines") or []
    for ln in reversed(lines):
        if "→" in ln:
            return ln
    return ""


# ── inject channel (append with the SAME atomic pattern the loop drains) ─────

def inject_command(text: str, inject_path: str) -> None:
    """Append ``{"text": text, "ts": ...}`` to the inject queue atomically.

    Read-modify-write under a fresh temp + os.replace so a concurrent
    ``_drain_injected_command`` (which claims the file by renaming it) never sees
    a half-written array. If the queue was mid-consume (renamed away) we simply
    start a fresh list — the loop will drain ours next pass. Matches driver.py's
    ``inject`` and staging_instance's writer."""
    items: list = []
    try:
        if os.path.exists(inject_path):
            with open(inject_path, encoding="utf-8") as f:
                raw = f.read().strip()
            if raw:
                decoded = json.loads(raw)
                if isinstance(decoded, list):
                    items = decoded
    except Exception:
        items = []
    items.append({"text": text, "ts": time.time()})
    _dir = os.path.dirname(os.path.abspath(inject_path)) or "."
    fd, tmp = tempfile.mkstemp(dir=_dir, suffix=".tmp", prefix=".webinject_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(items, f, indent=2)
        os.replace(tmp, inject_path)
    except Exception:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass
        raise


def wait_for_reply(text: str, log_dir: str, timeout: float) -> dict:
    """Tail the newest session log from its current end and return JARVIS's reply
    lines for this utterance. Mirrors run-jarvis/driver.py's wait_for_reply, but
    trimmed to what the web UI needs.

    Returns ``{"status": "ok"|"accepted"|"no_log", "lines": [...]}``:
      • ok       — we captured JARVIS:/[action] line(s) for this turn.
      • accepted — injected, but no reply text landed within the timeout (the
                   command still ran; we just didn't see spoken output — e.g. a
                   pure side-effect action, or JARVIS is asleep and dropped it).
      • no_log   — no session log exists (JARVIS isn't running); the command was
                   still queued and will fire when it next boots."""
    lg = _newest_log(log_dir)
    if not lg:
        return {"status": "no_log", "lines": []}
    try:
        pos = os.path.getsize(lg)
    except Exception:
        return {"status": "no_log", "lines": []}
    snippet = text[:30].lower()
    saw_inject = False
    lines: list[str] = []
    deadline = time.time() + max(1.0, min(float(timeout), _REPLY_TIMEOUT_MAX))
    while time.time() < deadline:
        time.sleep(0.5)
        try:
            with open(lg, encoding="utf-8", errors="replace") as f:
                f.seek(pos)
                chunk = f.read()
                pos = f.tell()
        except Exception:
            continue
        for line in chunk.splitlines():
            low = line.lower()
            # The loop prints "  [inject] <text>" (see _capture_utterance) when it
            # picks up our command — anchor on it so we don't scrape an unrelated
            # concurrent turn's output. If we never see the anchor we still return
            # whatever JARVIS:/[action] lines appeared (best-effort).
            if "[inject]" in low and snippet and snippet in low:
                saw_inject = True
                continue
            if "jarvis:" in low or "[action]" in low:
                lines.append(line.rstrip())
        if saw_inject and lines:
            # Give a beat for a trailing spoken follow-up line to land, then stop.
            time.sleep(1.0)
            try:
                with open(lg, encoding="utf-8", errors="replace") as f:
                    f.seek(pos)
                    for line in f.read().splitlines():
                        if "jarvis:" in line.lower() or "[action]" in line.lower():
                            lines.append(line.rstrip())
            except Exception:
                pass
            return {"status": "ok", "lines": lines}
    return {"status": "ok" if lines else "accepted", "lines": lines}


# ── the request handler ──────────────────────────────────────────────────────

class _Handler(BaseHTTPRequestHandler):
    """Routes: GET / (dashboard), GET /api/status, GET /api/log/tail, POST
    /api/say. The owning server pins config onto the class instance via the
    ``config`` attribute set in ``create_server`` (a small dict) so handlers are
    stateless beyond it."""

    # Silence the default per-request stderr logging — it would spam the session
    # log we're tailing. (The base class calls this for every request.)
    def log_message(self, fmt, *args):  # noqa: A003 - matches base signature
        return

    # ── auth ────────────────────────────────────────────────────────────────
    def _token(self) -> str:
        return self.server.config.get("token", "")  # type: ignore[attr-defined]

    def _request_token(self, query: dict) -> str:
        """Pull a caller-supplied token from (in priority) the Authorization
        Bearer header, an X-Auth-Token header, or a ?token= query param."""
        auth = self.headers.get("Authorization", "")
        if auth.lower().startswith("bearer "):
            return auth[7:].strip()
        xat = self.headers.get("X-Auth-Token")
        if xat:
            return xat.strip()
        q = query.get("token")
        if q:
            return q[0]
        return ""

    def _authorized(self, query: dict, *, is_page: bool) -> bool:
        """Auth gate. With no token configured (only reachable on a LOCAL bind —
        create_server enforces that) everything is allowed. With a token set, an
        API request must present it. The dashboard PAGE is allowed token-free on a
        local bind (convenience) but requires the token on an exposed bind."""
        token = self._token()
        if not token:
            return True
        if is_page and self.server.config.get("local_bind", True):  # type: ignore[attr-defined]
            return True
        return self._request_token(query) == token

    # ── tiny response helpers ────────────────────────────────────────────────
    def _send_json(self, obj: dict, code: int = 200) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        # No caching of live status/log/reply payloads.
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except Exception:
            pass

    def _send_html(self, text: str, code: int = 200) -> None:
        body = text.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except Exception:
            pass

    def _unauthorized(self) -> None:
        self._send_json({"error": "unauthorized"}, code=401)

    # ── GET ──────────────────────────────────────────────────────────────────
    def do_GET(self):  # noqa: N802 - http.server API
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        query = urllib.parse.parse_qs(parsed.query)
        cfg = self.server.config  # type: ignore[attr-defined]

        if path == "/":
            if not self._authorized(query, is_page=True):
                return self._unauthorized()
            return self._send_html(_dashboard_html(cfg.get("token", "")))

        if path == "/api/status":
            if not self._authorized(query, is_page=False):
                return self._unauthorized()
            return self._send_json(build_status(cfg["hud_state_path"], cfg["log_dir"]))

        if path == "/api/log/tail":
            if not self._authorized(query, is_page=False):
                return self._unauthorized()
            try:
                n = int(query.get("lines", ["50"])[0])
            except (TypeError, ValueError):
                n = 50
            return self._send_json(tail_log(cfg["log_dir"], n))

        return self._send_json({"error": "not found"}, code=404)

    # ── POST ─────────────────────────────────────────────────────────────────
    def do_POST(self):  # noqa: N802 - http.server API
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        query = urllib.parse.parse_qs(parsed.query)
        cfg = self.server.config  # type: ignore[attr-defined]

        if path != "/api/say":
            return self._send_json({"error": "not found"}, code=404)
        if not self._authorized(query, is_page=False):
            return self._unauthorized()

        # Parse the JSON body {"text": "...", "timeout": <optional seconds>}.
        try:
            length = int(self.headers.get("Content-Length", "0") or "0")
        except (TypeError, ValueError):
            length = 0
        raw = b""
        if length > 0:
            try:
                raw = self.rfile.read(min(length, 64 * 1024))  # cap body size
            except Exception:
                raw = b""
        text = ""
        req_timeout = _REPLY_TIMEOUT_DEFAULT
        try:
            data = json.loads(raw.decode("utf-8")) if raw else {}
            if isinstance(data, dict):
                text = str(data.get("text", "")).strip()
                if "timeout" in data:
                    req_timeout = float(data["timeout"])
        except Exception:
            text = ""
        if not text:
            return self._send_json({"error": "empty text"}, code=400)

        # Inject via the SAME channel the voice loop drains, then wait for a reply.
        try:
            inject_command(text, cfg["inject_path"])
        except Exception as e:
            return self._send_json({"error": f"inject failed: {e}"}, code=500)

        reader = cfg.get("reply_reader") or wait_for_reply
        try:
            res = reader(text, cfg["log_dir"], req_timeout)
        except Exception as e:
            # The command was injected and will run; we just couldn't tail a reply.
            return self._send_json({"accepted": True, "reply": "", "status": f"reply_error: {e}"})

        status = res.get("status", "accepted")
        lines = res.get("lines", []) or []
        return self._send_json({
            "accepted": True,
            "status": status,
            "reply": "\n".join(lines),
            "reply_lines": lines,
        })


# ── the dashboard page (single inline dark, arc-reactor-cyan HTML/JS) ────────

def _dashboard_html(token: str) -> str:
    """Return the full dashboard page. Inline CSS/JS only (no external fetches),
    dark theme with arc-reactor cyan accents. Polls /api/status + /api/log/tail
    once a second and POSTs typed commands to /api/say. The token (if any) is
    baked into the JS so the page's own API calls carry it."""
    # html.escape the token so a token with quotes can't break out of the JS
    # string literal (it's a shared secret, not attacker-controlled, but cheap
    # to be correct).
    tok = html.escape(token or "", quote=True)
    # NOTE: literal braces in the CSS/JS are doubled because this is an f-string.
    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>JARVIS — Live</title>
<style>
  :root {{ --cyan:#22d3ee; --cyan-dim:#0e7490; --bg:#05080d; --panel:#0b1420;
           --edge:#123; --text:#cfe9f2; --muted:#5b7a86; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:radial-gradient(1200px 600px at 50% -10%, #0a1a26 0%, var(--bg) 60%);
          color:var(--text); font:14px/1.5 ui-monospace,Menlo,Consolas,monospace; }}
  header {{ display:flex; align-items:center; gap:14px; padding:14px 18px;
            border-bottom:1px solid var(--edge); position:sticky; top:0;
            background:rgba(5,8,13,.9); backdrop-filter:blur(6px); }}
  .reactor {{ width:22px; height:22px; border-radius:50%;
              background:radial-gradient(circle at 50% 50%, #eafcff, var(--cyan) 45%, var(--cyan-dim) 70%, #04222b 100%);
              box-shadow:0 0 12px var(--cyan), 0 0 28px var(--cyan-dim); }}
  h1 {{ font-size:15px; margin:0; letter-spacing:.28em; color:var(--cyan); }}
  .wrap {{ max-width:1000px; margin:0 auto; padding:16px 18px; }}
  .strip {{ display:flex; flex-wrap:wrap; gap:10px; margin-bottom:14px; }}
  .chip {{ background:var(--panel); border:1px solid var(--edge); border-radius:8px;
           padding:8px 12px; min-width:120px; }}
  .chip .k {{ color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.12em; }}
  .chip .v {{ color:var(--cyan); font-size:15px; margin-top:2px; word-break:break-word; }}
  .dot {{ display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:6px;
          background:#666; vertical-align:middle; }}
  .dot.on {{ background:#2ee6a6; box-shadow:0 0 8px #2ee6a6; }}
  #log {{ background:#04070c; border:1px solid var(--edge); border-radius:8px;
          height:52vh; overflow:auto; padding:10px 12px; white-space:pre-wrap;
          font-size:12.5px; color:#a9c7d1; }}
  #log .a {{ color:var(--cyan); }}
  #log .j {{ color:#eafcff; }}
  form {{ display:flex; gap:8px; margin-top:14px; }}
  input[type=text] {{ flex:1; background:#04070c; border:1px solid var(--edge);
           color:var(--text); border-radius:8px; padding:11px 12px; font:inherit; }}
  input[type=text]:focus {{ outline:none; border-color:var(--cyan); box-shadow:0 0 0 1px var(--cyan-dim); }}
  button {{ background:var(--cyan-dim); color:#eafcff; border:1px solid var(--cyan);
            border-radius:8px; padding:0 18px; font:inherit; cursor:pointer; }}
  button:hover {{ background:var(--cyan); color:#04222b; }}
  button:disabled {{ opacity:.5; cursor:default; }}
  #reply {{ margin-top:10px; color:#eafcff; min-height:1.4em; }}
  .muted {{ color:var(--muted); }}
</style></head><body>
<header><div class="reactor"></div><h1>J.A.R.V.I.S.</h1>
  <span id="conn" class="muted" style="margin-left:auto">connecting…</span>
</header>
<div class="wrap">
  <div class="strip" id="strip"></div>
  <div id="log" class="muted">loading log…</div>
  <form id="say">
    <input id="text" type="text" autocomplete="off" placeholder="Type a command for JARVIS…" autofocus>
    <button id="send" type="submit">Send</button>
  </form>
  <div id="reply"></div>
</div>
<script>
const TOKEN = "{tok}";
function hdr() {{ const h = {{'Content-Type':'application/json'}}; if (TOKEN) h['X-Auth-Token']=TOKEN; return h; }}
function q(u) {{ return TOKEN ? (u + (u.includes('?')?'&':'?') + 'token=' + encodeURIComponent(TOKEN)) : u; }}
const strip = document.getElementById('strip');
const logEl = document.getElementById('log');
const conn  = document.getElementById('conn');

function chip(k, v) {{ const d=document.createElement('div'); d.className='chip';
  d.innerHTML = '<div class="k">'+k+'</div><div class="v"></div>';
  d.querySelector('.v').textContent = (v===''||v==null) ? '—' : v; return d; }}

async function refreshStatus() {{
  try {{
    const r = await fetch(q('/api/status'), {{headers:hdr()}});
    if (r.status===401) {{ conn.textContent='unauthorized — token required'; return; }}
    const s = await r.json();
    conn.textContent = ''; conn.innerHTML =
      '<span class="dot '+(s.running?'on':'')+'"></span>'+(s.running?'live':'offline');
    strip.innerHTML='';
    strip.appendChild(chip('version', s.version));
    strip.appendChild(chip('state', s.state));
    strip.appendChild(chip('model / routing', s.routing || s.model));
    const g = (s.gpu_lines&&s.gpu_lines.length) ? s.gpu_lines[s.gpu_lines.length- (s.routing?2:1)] : '';
    strip.appendChild(chip('vram', (s.gpu_bar||'') + (g? '  '+g : '')));
    if (s.now_playing) strip.appendChild(chip('now playing', s.now_playing));
    if (s.last_spoken) strip.appendChild(chip('last said', s.last_spoken));
  }} catch(e) {{ conn.textContent = 'connection lost'; }}
}}

let pinned = true;
logEl.addEventListener('scroll', () => {{
  pinned = (logEl.scrollTop + logEl.clientHeight) >= (logEl.scrollHeight - 24);
}});
async function refreshLog() {{
  try {{
    const r = await fetch(q('/api/log/tail?lines=200'), {{headers:hdr()}});
    if (r.status===401) return;
    const d = await r.json();
    const frag = (d.lines||[]).map(l => {{
      const cls = /\\[action\\]/i.test(l) ? 'a' : (/jarvis:/i.test(l) ? 'j' : '');
      const esc = l.replace(/&/g,'&amp;').replace(/</g,'&lt;');
      return cls ? '<span class="'+cls+'">'+esc+'</span>' : esc;
    }}).join('\\n');
    logEl.innerHTML = frag || '<span class="muted">(no log yet)</span>';
    if (pinned) logEl.scrollTop = logEl.scrollHeight;
  }} catch(e) {{}}
}}

const form = document.getElementById('say');
const textIn = document.getElementById('text');
const sendBtn = document.getElementById('send');
const replyEl = document.getElementById('reply');
form.addEventListener('submit', async (ev) => {{
  ev.preventDefault();
  const text = textIn.value.trim(); if (!text) return;
  sendBtn.disabled = true; replyEl.textContent = '…';
  try {{
    const r = await fetch(q('/api/say'), {{method:'POST', headers:hdr(),
      body: JSON.stringify({{text}})}});
    const d = await r.json();
    if (r.status===401) replyEl.textContent = 'unauthorized';
    else if (d.reply) replyEl.textContent = d.reply;
    else if (d.status==='no_log') replyEl.innerHTML = '<span class="muted">queued — JARVIS is not running; it will run on next boot.</span>';
    else replyEl.innerHTML = '<span class="muted">accepted (no spoken reply captured).</span>';
    textIn.value='';
  }} catch(e) {{ replyEl.textContent = 'send failed'; }}
  finally {{ sendBtn.disabled=false; textIn.focus(); refreshLog(); }}
}});

refreshStatus(); refreshLog();
setInterval(refreshStatus, 1500);
setInterval(refreshLog, 1000);
</script></body></html>"""


# ── server factory + lifecycle ───────────────────────────────────────────────

def create_server(*, bind: str, port: int, token: str = "",
                  inject_path: str = DEFAULT_INJECT_PATH,
                  log_dir: str = DEFAULT_LOG_DIR,
                  hud_state_path: str = DEFAULT_HUD_STATE_PATH,
                  reply_reader=None) -> ThreadingHTTPServer:
    """Build (but do not serve) a ThreadingHTTPServer for the web interface.

    SECURITY GATE: refuses to construct a server on a NON-LOCAL bind when the
    token is empty (raises InsecureBindError) — the caller must supply a token to
    expose it on the LAN. A local (loopback) bind needs no token.

    ``reply_reader`` lets a test stub the log-tail reply wait (default:
    ``wait_for_reply``). All paths default to the live project files but are
    injectable so a test can point them at a temp dir and bind 127.0.0.1:0."""
    bind = (bind or "127.0.0.1").strip()
    local = is_local_bind(bind)
    if not local and not (token or "").strip():
        raise InsecureBindError(
            f"refusing to start the web interface: bind={bind!r} is not loopback "
            f"but WEB_INTERFACE_TOKEN is empty. Set a token to expose it on the "
            f"LAN, or bind 127.0.0.1 for localhost-only (no token needed)."
        )
    httpd = ThreadingHTTPServer((bind, int(port)), _Handler)
    # Pin per-server config onto the instance so the stateless handler reads it.
    httpd.config = {  # type: ignore[attr-defined]
        "token": (token or "").strip(),
        "local_bind": local,
        "inject_path": inject_path,
        "log_dir": log_dir,
        "hud_state_path": hud_state_path,
        "reply_reader": reply_reader,
    }
    return httpd


def serve_in_thread(httpd: ThreadingHTTPServer) -> threading.Thread:
    """Run ``httpd.serve_forever()`` on a daemon thread and return it. The caller
    stops it with ``httpd.shutdown()`` (which unblocks serve_forever) then
    ``httpd.server_close()``. Daemon so a hung request never blocks JARVIS exit."""
    t = threading.Thread(target=httpd.serve_forever, name="web-interface",
                         daemon=True)
    t.start()
    return t
