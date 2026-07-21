"""
skills/browser_agent.py — deep browser automation via `browser-use`.

What it does
------------
Spawns a headed Chromium controlled by Claude that sees the DOM, clicks,
types, recovers from page changes, and reports back. This is the layer
beyond `webbrowser.open` — voice triggers like:

    'JARVIS, book me a haircut Friday afternoon'
    'JARVIS, fill out this form on the open tab'
    'JARVIS, find the cheapest flight to Tokyo and screenshot the result'

are handed verbatim to the agent loop and the LLM drives the browser to
completion, recovering from page changes / popups / re-prompts.

Architecture
------------
`browser-use` is async-only (Playwright + asyncio). JARVIS skill actions
are synchronous. So this module owns a daemon thread running its own
asyncio event loop and the skill actions submit coroutines to it via
`asyncio.run_coroutine_threadsafe`. Exactly one browser-use Agent runs at
a time (a single Chromium instance can't safely host two agents stepping
on each other); `browser_status` / `browser_stop` surface the current run.

Sandbox
-------
The skill NEVER touches the user's regular Chrome / Edge profile. A
dedicated Chromium profile lives at `data/browser_agent_profile/`. The
agent has to log in fresh for each site by design, but cookies persist
across tasks in the same profile so multi-step workflows (book a haircut
→ pay deposit) work. `browser_reset_profile` wipes the dir clean.

Optional dependencies
---------------------
    browser-use      pip install browser-use
    playwright       pip install playwright && python -m playwright install chromium

Both are intentionally NOT in requirements.txt — matches the optional-dep
pattern set by research-2/3/4a/4c/5/6/7/8 (mcp, tplinkrouterc6u, alexapy,
resemblyzer, rank_bm25). Until the user opts in, every action returns a
friendly install hint and the skill loads silently.

Registered actions
------------------
    browser_task <task>            main entry: hand the agent a natural-language goal
    browser_do <task>              alias
    browser_run <task>             alias
    book_appointment <task>        prefixes 'Book me ' onto <task>
    fill_form <details>            uses the current tab as the starting context
    browse_for <query>             search + summarise shortcut
    find_cheapest <item>           bargain-hunt shortcut
    browser_open <url>             just open a URL in the sandboxed browser
    browser_screenshot             screenshot the live tab; returns the saved path
    browser_status                 current task / step count / last action
    browser_stop                   cancel the active task
    browser_reset_profile          wipe the sandboxed Chromium profile dir
"""
from __future__ import annotations

import asyncio
import os
import shutil
import threading
import time
import traceback
from typing import Any, Callable


_PROJECT_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# STAGING ISOLATION (2026-07-21): resolve through core.paths so a
# JARVIS_STAGING process writes data_staging/ instead of the live data/.
# A private join here is how a staging-isolated action sweep overwrote the
# LIVE smart-home catalog while the settings md5 tripwire stayed green.
try:
    from core.paths import data_dir as _jarvis_data_dir
    _DATA_DIR = _jarvis_data_dir()
except Exception:   # pragma: no cover - core.paths is in-tree
    _DATA_DIR = os.path.join(_PROJECT_DIR, "data")
_PROFILE_DIR     = os.path.join(_DATA_DIR, "browser_agent_profile")
_SCREENSHOT_DIR  = os.path.join(_DATA_DIR, "browser_agent_shots")

# Per-task budget. browser-use steps can legitimately take 30s+ when the
# LLM is reasoning about a complex page; this caps how long a single
# task may run before we forcibly cancel it.
_DEFAULT_TASK_TIMEOUT = 5 * 60
_DEFAULT_MAX_STEPS    = 30

# Wait this long when a synchronous caller asks `browser_task` to block
# on the result. Beyond this we return a "still running" message and let
# them poll with `browser_status`.
_SYNC_WAIT_SECONDS = 90


# ── shared state ────────────────────────────────────────────────────
_lock = threading.Lock()
_state: dict[str, Any] = {
    "loop":          None,    # asyncio loop running on bg thread
    "thread":        None,    # the daemon thread
    "current_task":  None,    # str | None — natural-language task in flight
    "current_fut":   None,    # concurrent.futures.Future for the run
    "agent":         None,    # the live browser_use.Agent (for cancel)
    "browser":       None,    # the live Browser / BrowserSession handle
    "step_count":    0,
    "last_step":     None,
    "started_at":    None,
    "finished_at":   None,
    "last_result":   None,
    "last_error":    None,
}


# ── lazy SDK import ─────────────────────────────────────────────────
def _bu_imports() -> dict[str, Any]:
    """Try several browser-use layouts (the package has shifted between
    0.x and 1.x). Returns the pieces we found, or {} if browser-use is
    not installed at all."""
    try:
        import browser_use  # type: ignore
    except Exception:
        return {}

    Agent = getattr(browser_use, "Agent", None)
    Browser = (getattr(browser_use, "Browser", None)
               or getattr(browser_use, "BrowserSession", None))
    BrowserConfig = (getattr(browser_use, "BrowserConfig", None)
                     or getattr(browser_use, "BrowserSessionConfig", None)
                     or getattr(browser_use, "BrowserProfile", None))
    if Agent is None:
        # Some versions hide Agent under .agent.service
        try:
            from browser_use.agent.service import Agent  # type: ignore
        except Exception:
            return {}

    # LLM — newer browser-use bundles its own chat wrappers; older
    # versions expect langchain_anthropic.
    ChatAnthropic = None
    try:
        from browser_use.llm import ChatAnthropic  # type: ignore
    except Exception:
        try:
            from langchain_anthropic import ChatAnthropic  # type: ignore
        except Exception:
            ChatAnthropic = None

    return {
        "Agent":         Agent,
        "Browser":       Browser,
        "BrowserConfig": BrowserConfig,
        "ChatAnthropic": ChatAnthropic,
    }


def is_available() -> bool:
    """True iff browser-use AND a usable Anthropic chat wrapper AND an
    API key are all in place. Playwright presence is checked lazily on
    first run — installing browser-use without Playwright is uncommon
    but possible, and the user-facing error from Playwright is already
    informative, so we don't bother sniffing here."""
    imports = _bu_imports()
    if not imports.get("Agent") or not imports.get("ChatAnthropic"):
        return False
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def _install_hint() -> str:
    return (
        "Browser agent is offline, sir. Install with: "
        "`pip install browser-use` then "
        "`python -m playwright install chromium`. "
        "Also set ANTHROPIC_API_KEY."
    )


# ── SSRF guard ──────────────────────────────────────────────────────
# The browser agent will happily fetch whatever URL the task names, so a
# prompt-injection ('browse to http://169.254.169.254/latest/meta-data/…')
# could pull cloud-metadata creds or hit services bound to localhost / the
# LAN. Before any user-facing navigation we resolve the target host and
# refuse private / loopback / link-local / reserved addresses. Resolution
# failures are allowed through so ordinary public hostnames still work even
# when DNS is briefly unavailable.
_SSRF_REFUSAL = "I won't browse to internal/private addresses, sir."


def _ip_is_blocked(ip_str: str) -> bool:
    """True if `ip_str` is a loopback / private / link-local / reserved
    address (covers 127/8, ::1, RFC-1918 10/8·172.16/12·192.168/16,
    169.254/16 incl. the 169.254.169.254 metadata IP, and friends)."""
    try:
        import ipaddress
        ip = ipaddress.ip_address(ip_str)
    except Exception:
        return False
    return bool(
        ip.is_private or ip.is_loopback or ip.is_link_local
        or ip.is_reserved or ip.is_multicast or ip.is_unspecified
    )


def _host_is_blocked(host: str) -> bool:
    """Resolve `host` and decide whether navigation to it must be refused.

    A literal IP is checked directly. A hostname is resolved via
    gethostbyname; if it resolves to a blocked address we refuse (this is
    what catches `localhost` and DNS-rebinding to 127.0.0.1 / LAN). If
    resolution fails we allow it — a public-looking name shouldn't be
    blocked just because DNS hiccuped."""
    host = (host or "").strip().strip(".").lower()
    if not host:
        return False
    # Bracketed IPv6 literal, e.g. [::1]
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    # Direct IP literal?
    try:
        import ipaddress
        ipaddress.ip_address(host)
        return _ip_is_blocked(host)
    except Exception:
        pass
    # Obvious loopback alias that may not resolve in odd environments.
    if host == "localhost" or host.endswith(".localhost"):
        return True
    # Resolve the hostname and inspect the address it points at.
    try:
        import socket
        resolved = socket.gethostbyname(host)
    except Exception:
        return False  # resolution failed → allow public-looking hostnames
    return _ip_is_blocked(resolved)


def _extract_hosts(text: str) -> list[str]:
    """Pull candidate hosts out of an arbitrary task/URL string: every
    http(s):// URL's hostname, plus a leading bare host token when the
    whole argument looks like a host/URL the user handed `browser_open`."""
    hosts: list[str] = []
    if not text:
        return hosts
    try:
        import re as _re
        from urllib.parse import urlparse
        for m in _re.findall(r"https?://[^\s'\"<>]+", text, flags=_re.IGNORECASE):
            try:
                h = urlparse(m).hostname
                if h:
                    hosts.append(h)
            except Exception:
                continue
        # Bare 'host[:port]/path' with no scheme — treat the first token as a
        # host when it has no spaces (covers `browser_open localhost:8188`).
        first = text.strip().split()[0] if text.strip().split() else ""
        if first and "://" not in first and not first.lower().startswith("http"):
            tok = first.split("/")[0]
            # Bracketed IPv6 literal, e.g. [::1]:8188 → ::1
            if tok.startswith("[") and "]" in tok:
                hosts.append(tok[1:tok.index("]")])
            else:
                cand = tok.split(":")[0]
                if cand and ("." in cand or cand.lower() == "localhost"):
                    hosts.append(cand)
                # Bare IPv6 with no brackets/port (contains '::' or multiple ':')
                elif tok.count(":") >= 2:
                    hosts.append(tok)
    except Exception:
        pass
    return hosts


def _ssrf_guard(arg: str) -> str | None:
    """Return the refusal string if `arg` names any blocked host, else None.
    Best-effort + exception-safe: a guard that itself errors must not block
    a legitimate public navigation, so we fail open on internal errors."""
    try:
        for host in _extract_hosts(arg):
            if _host_is_blocked(host):
                return _SSRF_REFUSAL
    except Exception:
        return None
    return None


# ── model selection ────────────────────────────────────────────────
def _model_name() -> str:
    """Mirror whatever JARVIS itself is using so the user only has to
    rotate models in one place."""
    env = os.environ.get("BROWSER_AGENT_MODEL")
    if env:
        return env
    try:
        import bobert_companion  # type: ignore
        m = getattr(bobert_companion, "CLAUDE_MODEL", None)
        if m:
            return str(m)
    except Exception:
        pass
    return "claude-sonnet-5"


# ── bg loop ────────────────────────────────────────────────────────
def _start_loop_thread() -> asyncio.AbstractEventLoop:
    """Start (idempotently) the daemon thread that owns our asyncio loop."""
    with _lock:
        loop = _state["loop"]
        if loop is not None and not loop.is_closed():
            return loop

    ready = threading.Event()
    new_loop = asyncio.new_event_loop()

    def _run() -> None:
        asyncio.set_event_loop(new_loop)
        ready.set()
        try:
            new_loop.run_forever()
        finally:
            try:
                new_loop.close()
            except Exception:
                pass

    t = threading.Thread(target=_run, name="browser-agent-asyncio",
                         daemon=True)
    t.start()
    if not ready.wait(timeout=5.0):
        raise RuntimeError("browser_agent asyncio loop failed to start within 5s")
    with _lock:
        _state["loop"]   = new_loop
        _state["thread"] = t
    return new_loop


# ── browser + agent factory ────────────────────────────────────────
async def _make_browser(imports: dict, headless: bool) -> Any:
    """Construct a sandboxed Browser handle. Tolerant of the API shifts
    between 0.x and 1.x: tries the new BrowserSession kwargs, falls back
    to BrowserConfig(user_data_dir=...)."""
    os.makedirs(_PROFILE_DIR, exist_ok=True)
    Browser       = imports.get("Browser")
    BrowserConfig = imports.get("BrowserConfig")
    if Browser is None:
        return None

    # Newer browser-use: BrowserSession(user_data_dir=..., headless=...)
    for kwargs in (
        {"user_data_dir": _PROFILE_DIR, "headless": headless},
        {"user_data_dir": _PROFILE_DIR},
        {"headless": headless},
        {},
    ):
        try:
            handle = Browser(**kwargs)
            # Newer API exposes .start() — older API auto-starts on use.
            starter = getattr(handle, "start", None)
            if callable(starter):
                res = starter()
                if asyncio.iscoroutine(res):
                    await res
            return handle
        except TypeError:
            continue
        except Exception:
            continue

    # Older path: Browser(config=BrowserConfig(...))
    if BrowserConfig is None:
        return None
    for cfg_kwargs in (
        {"user_data_dir": _PROFILE_DIR, "headless": headless},
        {"chrome_instance_path": None, "headless": headless,
         "user_data_dir": _PROFILE_DIR},
        {"headless": headless},
    ):
        try:
            cfg = BrowserConfig(**cfg_kwargs)
            return Browser(config=cfg)
        except TypeError:
            continue
        except Exception:
            continue
    return None


def _make_llm(imports: dict) -> Any:
    ChatAnthropic = imports.get("ChatAnthropic")
    if ChatAnthropic is None:
        return None
    model = _model_name()
    # langchain_anthropic uses `model=`, some browser-use ChatAnthropic
    # variants use `model_name=` — try both.
    for kwargs in (
        {"model": model, "temperature": 0.0},
        {"model_name": model, "temperature": 0.0},
        {"model": model},
    ):
        try:
            return ChatAnthropic(**kwargs)
        except TypeError:
            continue
        except Exception:
            continue
    return None


def _step_hook(*args, **_kwargs) -> None:
    """browser-use calls this after every agent step. The library has
    waved its arguments around across versions (sometimes (browser, state),
    sometimes (agent_step), sometimes nothing) so we ignore them and just
    bump the counter + cache the latest action for `browser_status`."""
    with _lock:
        _state["step_count"] = (_state.get("step_count") or 0) + 1
        # Best-effort scrape of the last action description out of whichever
        # object the lib handed us.
        for obj in args:
            for attr in ("action", "last_action", "current_action",
                         "next_action", "intent"):
                v = getattr(obj, attr, None)
                if v:
                    _state["last_step"] = str(v)[:200]
                    return


async def _run_task_inner(task: str, max_steps: int, headless: bool) -> str:
    imports = _bu_imports()
    Agent = imports.get("Agent")
    if Agent is None:
        return _install_hint()
    llm = _make_llm(imports)
    if llm is None:
        return _install_hint()

    browser = await _make_browser(imports, headless=headless)
    with _lock:
        _state["browser"] = browser

    # Agent construction signature also varies — try the common ones.
    agent = None
    attempts = [
        {"task": task, "llm": llm, "browser": browser,
         "register_new_step_callback": _step_hook},
        {"task": task, "llm": llm, "browser_session": browser,
         "register_new_step_callback": _step_hook},
        {"task": task, "llm": llm, "browser": browser},
        {"task": task, "llm": llm},
    ]
    for kwargs in attempts:
        try:
            agent = Agent(**kwargs)
            break
        except TypeError:
            continue
        except Exception:
            continue
    if agent is None:
        return "Browser agent could not construct an Agent — likely a browser-use version mismatch, sir."

    with _lock:
        _state["agent"] = agent

    # Run signatures also vary: some versions take max_steps positional,
    # some keyword. Try both.
    run_kwargs_options = (
        {"max_steps": max_steps},
        {"steps": max_steps},
        {},
    )
    history: Any = None
    last_err: str | None = None
    for run_kwargs in run_kwargs_options:
        try:
            history = await agent.run(**run_kwargs)
            last_err = None
            break
        except TypeError as e:
            last_err = f"{type(e).__name__}: {e}"
            continue
    if history is None and last_err:
        return f"Browser agent .run() failed, sir — {last_err}"

    # Extract a readable result from the history object. browser-use 1.x
    # returns AgentHistoryList with a .final_result() method; older
    # versions returned a list. Cover both.
    final = None
    for getter in ("final_result", "result", "output"):
        fn = getattr(history, getter, None)
        if callable(fn):
            try:
                final = fn()
                if final is not None:
                    break
            except Exception:
                continue
        elif fn is not None:
            final = fn
            break
    if final is None:
        # Last-ditch: stringify the history.
        try:
            final = str(history)[:1200]
        except Exception:
            final = "task completed."
    return str(final)


async def _close_browser_async() -> None:
    """Best-effort cleanup. We're tolerant of half-built handles because
    a failed _make_browser path can leave us with a partly-initialised
    object."""
    with _lock:
        b = _state.get("browser")
        a = _state.get("agent")
        _state["browser"] = None
        _state["agent"]   = None
    if a is not None:
        for m in ("stop", "close", "shutdown"):
            fn = getattr(a, m, None)
            if callable(fn):
                try:
                    res = fn()
                    if asyncio.iscoroutine(res):
                        await res
                    break
                except Exception:
                    continue
    if b is not None:
        for m in ("close", "stop", "shutdown"):
            fn = getattr(b, m, None)
            if callable(fn):
                try:
                    res = fn()
                    if asyncio.iscoroutine(res):
                        await res
                    break
                except Exception:
                    continue


async def _orchestrate(task: str, max_steps: int, headless: bool) -> str:
    """Top-level coroutine: marks state, runs the agent, always cleans up."""
    try:
        return await _run_task_inner(task, max_steps, headless)
    except Exception as e:
        with _lock:
            _state["last_error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=4)}"
        # The browser-use LLM is cloud-only (ChatAnthropic, see _make_llm) with
        # no local fallback. When the Anthropic API is capped/over-quota/rate-
        # limited the underlying call raises with one of these markers — surface
        # a clear message instead of a raw stack so the user knows it's the cap,
        # not a broken task.
        _err = f"{type(e).__name__}: {e}".lower()
        if any(m in _err for m in ("usage limit", "credit", "quota",
                                   "rate limit", "rate_limit", "400")):
            return ("Browser automation needs the cloud API, sir, which is "
                    "capped until it resets — I can't drive the browser until "
                    "then.")
        return f"Browser agent failed, sir — {type(e).__name__}: {e}"
    finally:
        await _close_browser_async()
        with _lock:
            _state["finished_at"] = time.time()


# ── sync facade ────────────────────────────────────────────────────
def _submit(task: str, max_steps: int, headless: bool) -> Any:
    """Submit `task` to the bg loop. Returns the concurrent.futures.Future."""
    loop = _start_loop_thread()
    with _lock:
        if _state.get("current_fut") is not None and not _state["current_fut"].done():
            raise RuntimeError("a browser task is already running — `browser_stop` to cancel")
        _state.update({
            "current_task": task,
            "step_count":   0,
            "last_step":    None,
            "started_at":   time.time(),
            "finished_at":  None,
            "last_result":  None,
            "last_error":   None,
        })
    fut = asyncio.run_coroutine_threadsafe(
        _orchestrate(task, max_steps, headless),
        loop,
    )
    with _lock:
        _state["current_fut"] = fut

    def _on_done(f) -> None:
        with _lock:
            try:
                _state["last_result"] = f.result()
            except Exception as e:
                _state["last_error"] = f"{type(e).__name__}: {e}"
    fut.add_done_callback(_on_done)
    return fut


def _run_task_sync(
    task: str,
    *,
    max_steps: int = _DEFAULT_MAX_STEPS,
    timeout: float = _SYNC_WAIT_SECONDS,
    headless: bool = False,
) -> str:
    """Submit `task` and block until it finishes OR `timeout` elapses.
    Past the timeout we return a "still running, poll browser_status"
    message rather than killing the agent — long browser tasks are
    legitimate (booking flow w/ 2FA may take minutes)."""
    try:
        fut = _submit(task, max_steps=max_steps, headless=headless)
    except RuntimeError as e:
        return f"{e}, sir."
    try:
        return fut.result(timeout=timeout)
    except Exception as e:
        if "timeout" in type(e).__name__.lower() or "TimeoutError" in type(e).__name__:
            return (
                f"Browser agent is still working on '{task[:60]}', sir. "
                f"Poll `browser_status` for progress or `browser_stop` to cancel."
            )
        return f"Browser agent failed, sir — {type(e).__name__}: {e}"


# ── public helpers exposed as actions ──────────────────────────────
def _truncate(s: str, n: int = 1200) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[: n - 12] + "... (truncated)"


def _action_browser_task(arg: str = "") -> str:
    task = (arg or "").strip()
    if not task:
        return "Give me a task, sir — e.g. 'book me a haircut Friday afternoon'."
    if not is_available():
        return _install_hint()
    blocked = _ssrf_guard(task)
    if blocked:
        return blocked
    return _truncate(_run_task_sync(task))


def _action_book_appointment(arg: str = "") -> str:
    task = (arg or "").strip()
    if not task:
        return "What kind of appointment, sir?"
    if not task.lower().startswith(("book", "schedule", "reserve")):
        task = f"Book me {task}"
    if not is_available():
        return _install_hint()
    blocked = _ssrf_guard(task)
    if blocked:
        return blocked
    return _truncate(_run_task_sync(task))


def _action_fill_form(arg: str = "") -> str:
    details = (arg or "").strip()
    if not details:
        return "Tell me what to fill in, sir."
    task = (
        "Find the open or most-relevant form on the current page and "
        f"fill it in with these details: {details}. "
        "Stop short of final submission — read the filled values back so I can confirm."
    )
    if not is_available():
        return _install_hint()
    blocked = _ssrf_guard(details)
    if blocked:
        return blocked
    return _truncate(_run_task_sync(task))


def _action_browse_for(arg: str = "") -> str:
    query = (arg or "").strip()
    if not query:
        return "What should I look up, sir?"
    task = (
        f"Search the web for '{query}'. Summarise the top three results "
        "with their URLs and one-line takeaways."
    )
    if not is_available():
        return _install_hint()
    blocked = _ssrf_guard(query)
    if blocked:
        return blocked
    return _truncate(_run_task_sync(task))


def _action_find_cheapest(arg: str = "") -> str:
    item = (arg or "").strip()
    if not item:
        return "What should I bargain-hunt for, sir?"
    task = (
        f"Find the cheapest available option for: {item}. Compare at least "
        "three sources, note any obvious caveats (shipping, condition, dates), "
        "and report the price + URL of the best one."
    )
    if not is_available():
        return _install_hint()
    blocked = _ssrf_guard(item)
    if blocked:
        return blocked
    return _truncate(_run_task_sync(task))


def _action_browser_open(arg: str = "") -> str:
    """Just open a URL in the sandboxed browser — no LLM control. Useful
    when the user wants to pre-load a page before handing the agent a
    task ('open this then fill it out')."""
    url = (arg or "").strip()
    if not url:
        return "Give me a URL, sir."
    if not is_available():
        return _install_hint()
    blocked = _ssrf_guard(url)
    if blocked:
        return blocked
    task = f"Navigate to {url} and wait for the page to load. Report the page title."
    return _truncate(_run_task_sync(task, max_steps=3))


def _action_browser_screenshot(_: str = "") -> str:
    """Save a screenshot of the live browser-agent tab. Returns the path."""
    if not is_available():
        return _install_hint()
    with _lock:
        browser = _state.get("browser")
    if browser is None:
        return "Browser is not running, sir — start a task first."
    os.makedirs(_SCREENSHOT_DIR, exist_ok=True)
    path = os.path.join(
        _SCREENSHOT_DIR,
        f"shot_{time.strftime('%Y%m%d_%H%M%S')}.png",
    )
    loop = _state.get("loop")
    if loop is None:
        return "Browser loop is not running, sir."

    async def _shoot() -> str:
        # Try the common shapes the library has exposed.
        page = None
        for attr in ("page", "current_page", "active_page"):
            v = getattr(browser, attr, None)
            if v is not None:
                page = v
                break
        if page is None:
            getter = getattr(browser, "get_current_page", None)
            if callable(getter):
                res = getter()
                if asyncio.iscoroutine(res):
                    res = await res
                page = res
        if page is None:
            raise RuntimeError("no active page on browser handle")
        await page.screenshot(path=path, full_page=True)
        return path

    fut = asyncio.run_coroutine_threadsafe(_shoot(), loop)
    try:
        out = fut.result(timeout=15)
    except Exception as e:
        return f"Screenshot failed, sir — {type(e).__name__}: {e}"
    return f"Saved screenshot to {out}, sir."


def _action_browser_status(_: str = "") -> str:
    with _lock:
        task    = _state.get("current_task")
        fut     = _state.get("current_fut")
        step    = _state.get("step_count") or 0
        last    = _state.get("last_step")
        started = _state.get("started_at")
        result  = _state.get("last_result")
        error   = _state.get("last_error")
    if task is None:
        return "No browser task has run yet, sir."
    running = fut is not None and not fut.done()
    elapsed = int(time.time() - started) if started else 0
    head = f"Browser task: '{task[:80]}'"
    body = [
        f"  state:   {'running' if running else 'finished'}",
        f"  steps:   {step}",
        f"  elapsed: {elapsed}s",
    ]
    if last:
        body.append(f"  last:    {last[:120]}")
    if not running:
        if result:
            body.append(f"  result:  {str(result)[:200]}")
        if error:
            body.append(f"  error:   {error[:200]}")
    return head + ", sir.\n" + "\n".join(body)


def _action_browser_stop(_: str = "") -> str:
    with _lock:
        fut  = _state.get("current_fut")
        loop = _state.get("loop")
        agent = _state.get("agent")
    if fut is None or fut.done():
        return "No browser task is running, sir."

    # First ask the agent to stop politely so the LLM loop exits cleanly.
    if agent is not None:
        for m in ("stop", "request_stop", "cancel"):
            fn = getattr(agent, m, None)
            if callable(fn):
                try:
                    res = fn()
                    if asyncio.iscoroutine(res) and loop is not None:
                        asyncio.run_coroutine_threadsafe(res, loop)
                    break
                except Exception:
                    continue
    # Hard cancel + cleanup.
    fut.cancel()
    if loop is not None:
        try:
            asyncio.run_coroutine_threadsafe(_close_browser_async(), loop)
        except Exception:
            pass
    return "Stopping the browser task, sir."


def _action_browser_reset_profile(_: str = "") -> str:
    with _lock:
        fut = _state.get("current_fut")
    if fut is not None and not fut.done():
        return "Stop the running browser task first, sir — `browser_stop`."
    if not os.path.exists(_PROFILE_DIR):
        return "Sandboxed Chromium profile is already clean, sir."
    try:
        shutil.rmtree(_PROFILE_DIR, ignore_errors=False)
    except Exception as e:
        return f"Could not wipe the profile, sir — {type(e).__name__}: {e}"
    os.makedirs(_PROFILE_DIR, exist_ok=True)
    return "Sandboxed Chromium profile wiped, sir."


# ── skill entry point ──────────────────────────────────────────────
def register(actions: dict) -> None:
    handlers: dict[str, Callable[[str], str]] = {
        "browser_task":          _action_browser_task,
        "browser_do":            _action_browser_task,
        "browser_run":           _action_browser_task,
        "book_appointment":      _action_book_appointment,
        "fill_form":             _action_fill_form,
        "browse_for":            _action_browse_for,
        "find_cheapest":         _action_find_cheapest,
        "browser_open":          _action_browser_open,
        "browser_screenshot":    _action_browser_screenshot,
        "browser_status":        _action_browser_status,
        "browser_stop":          _action_browser_stop,
        "browser_reset_profile": _action_browser_reset_profile,
    }
    for name, fn in handlers.items():
        if name not in actions:
            actions[name] = fn

    if not _bu_imports().get("Agent"):
        # browser-use not installed — stay quiet on the boot log.
        # Actions are still registered and will return the install hint.
        return

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("  [browser_agent] browser-use installed but ANTHROPIC_API_KEY "
              "is not set; actions will surface a hint.")
        return

    os.makedirs(_PROFILE_DIR, exist_ok=True)
    print("  [browser_agent] ready — sandboxed profile at "
          f"{os.path.relpath(_PROFILE_DIR, _PROJECT_DIR)}, "
          f"model={_model_name()}.")
