#!/usr/bin/env python3
"""Process-wide block on REAL browser launches during a JARVIS test run.

THE INCIDENT (2026-08-20, 11:38-11:41)
======================================
Two full-CI runs spawned 12 then 20 new Chrome renderer processes per minute in
the owner's DEFAULT Chrome profile (no ``--user-data-dir``) and spammed dozens
of real tabs into his live window. There was no chromedriver, no selenium, and
Chrome's browser process was legitimately user-launched — the classic signature
of ``webbrowser.open()``, which hands the URL to an ALREADY-RUNNING browser over
its IPC channel and returns immediately, leaving no process lineage back to the
test that did it.

Grepping ``tests/`` for ``webbrowser.open`` finds almost nothing, because the
tests do not call it directly — they call PRODUCTION code that calls it
(``core/actions.py:73`` ``webbrowser.open(url)``, the Apple Music launch path in
``bobert_companion._open_url_in_browser``, ``skills/youtube_search.py``,
``skills/morning_handoff.py``, ``tray.py``'s ``os.startfile`` menu items, …).
Per-call-site test patches are therefore NOT the fix. The fix is one
process-wide interception, installed before a single test is imported.

This module is that interception, and it is the browser sibling of
``tools/mem_guard.py`` (RAM), ``_redirect_settings_to_throwaway()`` and
``_redirect_data_dir_to_throwaway()`` (the owner's live ``data/``): a test run
must not be able to damage — or, here, take over — the box it runs on.

WHAT IS INTERCEPTED
-------------------
* ``webbrowser.open`` / ``open_new`` / ``open_new_tab`` — the leak itself.
* ``webbrowser.get`` — returns a NEUTERED controller whose ``open*`` methods
  record instead of launching, so ``webbrowser.get("chrome").open(url)`` (the
  monolith's preferred path) is covered too.
* ``webbrowser.register`` — a recorded no-op. It is the one API whose entire
  purpose is to make a launch REACHABLE; a run that registers a browser is
  trying to reach one. (``webbrowser._tryorder`` / ``_browsers`` are private and
  deliberately left alone — once ``open`` and ``get`` are stubbed, nothing can
  reach them anyway.)
* ``os.startfile`` on Windows — the shell-launch primitive. ``os.startfile`` of
  an ``https://`` target opens the default browser, and of a folder opens an
  Explorer window on the owner's desktop. Both are unwanted in a test run, so
  every call is blocked. Guarded with ``hasattr`` — the attribute does not exist
  on POSIX, so this whole branch is skipped on the CI runner.
* ``subprocess.Popen`` / ``run`` / ``call`` — but ONLY when the command line
  names a known browser binary. Everything else passes straight through to the
  real function, untouched, so the hundreds of legitimate subprocess calls (and
  the tests that mock them) keep working. ``check_call`` / ``check_output`` are
  covered transitively: they resolve ``call`` / ``run`` as module globals at
  call time, so they land on these wrappers.

WHAT IS DELIBERATELY *NOT* INTERCEPTED
--------------------------------------
* Non-browser subprocess commands — see ``_browser_in_command``.
* READ-ONLY process tools that merely NAME a browser (``tasklist``, ``wmic``,
  ``pgrep``, ``where``, ``reg query``, …). Those count or filter; they cannot
  open a window and cannot close one. See ``_PROCESS_TOOLS``.

  The KILL verbs are NOT in that exemption (they were until 2026-08-20, on a
  justification — "``taskkill /IM chrome.exe`` is how the monolith reuses a
  single media window" — that does not exist anywhere in the tree). A kill verb
  aimed at a browser image is blocked and recorded; aimed at anything else
  (``taskkill /IM ollama.exe``, ``taskkill /F /T /PID n``) it passes straight
  through.
* ``explorer <folder>`` / ``start <folder>``. A shell launcher only counts as a
  browser launch when a URL-shaped argument is on the same command line —
  ``start <url>`` / ``explorer <url>`` / ``rundll32 url.dll,FileProtocolHandler
  <url>`` all open the DEFAULT browser without naming one, and those ARE
  blocked.

Note a command line like ``cmd /c start chrome <url>`` is caught in every
spelling: the argument after a ``-c`` / ``-Command`` / ``/c`` switch is
re-tokenised, so the browser name is found whether the caller passed it as one
string, as a fully split list, or as the ``powershell -Command "…"`` form
everything real actually uses.

COMPATIBILITY WITH ``mock.patch`` (load-bearing)
-----------------------------------------------
Dozens of existing tests patch these very names — ``mock.patch("webbrowser.open")``,
``mock.patch.object(bc.webbrowser, "open")``, ``mock.patch.object(tray.os,
"startfile", create=True)``, ``mock.patch.object(bc.subprocess, "Popen")``. This
guard replaces the MODULE ATTRIBUTE, exactly like mock does; mock then replaces
it a second time on top and, on exit, restores it to THIS module's stub rather
than to the real function. The guard therefore survives every patch/unpatch
cycle, and a test's own mock always wins while it is active (so an assertion
like ``wbopen.assert_called_once_with(url)`` still sees its own mock, not ours).
Tests that instead inject a whole fake ``webbrowser`` MODULE into ``sys.modules``
(tests/skills/test_morning_handoff.py, tools/audit_codebase.py) bypass the stub
entirely — which is fine: a fake module cannot open a real window either.

RETURN VALUES
-------------
A blocked call returns what a SUCCESSFUL one would (``webbrowser.open`` -> True,
``os.startfile`` -> None, ``Popen`` -> a finished-looking stand-in, ``run`` ->
``CompletedProcess(rc=0)``, ``call`` -> 0). That is deliberate: before this
guard existed the leaking tests saw a successful launch, so mimicking success
keeps them green while the launch is neutered. The atexit summary — not a test
failure — is how the offenders get identified.

CONTRACT
--------
* NEVER raises. If an attribute it wants is missing, or a module cannot be
  imported, it skips that target and keeps going.
* Emits exactly ONE stdout banner at install, so any log makes it self-evident
  whether the run was guarded.
* Idempotent: the second and later ``install()`` calls are no-ops returning the
  first result.
* Prints an atexit summary ONLY when something was actually blocked, naming the
  test and the production call site for each distinct offender.

ESCAPE HATCH
------------
``JARVIS_ALLOW_REAL_BROWSER=1`` disables the guard entirely — for a human
debugging a real browser flow by hand. It is deliberately loud in the log.
``install(record_only=True)`` is the middle setting: real launches are ALLOWED
THROUGH and merely recorded, for confirming a suspected offender.

WIRING
------
The chokepoint is ``tests/__init__.py`` — importing ANY test module imports the
``tests`` package first, so ``python -m unittest tests.foo``, the capped
scratchpad harness, an IDE's test runner and all three ``tools/`` runners are
covered by that one line. The three runners ALSO call ``install()`` explicitly
next to ``apply_memory_ceiling()`` (belt and braces; it is idempotent).
"""
from __future__ import annotations

import os
import re
import sys
import typing

_TAG = "[browser-guard]"

# Documented escape hatch — see the module docstring.
ALLOW_ENV_VAR = "JARVIS_ALLOW_REAL_BROWSER"
_ALLOW_WORDS = frozenset({"1", "true", "yes", "on", "allow",
                          "enable", "enabled"})

# Executable names that OPEN a browser window. Windows and POSIX spellings, so
# the same set works here and on the ubuntu-latest CI runner.
_BROWSER_BINARIES = frozenset({
    "chrome.exe", "chrome", "google-chrome", "google-chrome-stable",
    "chromium.exe", "chromium", "chromium-browser",
    "msedge.exe", "msedge", "microsoft-edge", "microsoft-edge-stable",
    "firefox.exe", "firefox", "firefox-bin",
    "brave.exe", "brave", "brave-browser",
    "opera.exe", "opera",
    "iexplore.exe", "safari",
})

# READ-ONLY process tools. They enumerate, filter or query; they cannot open a
# window and they cannot close one. If one of these is argv[0] the whole command
# line is exempt, which costs nothing.
#
# THE KILL VERBS USED TO BE IN HERE (removed 2026-08-20). The comment justifying
# that said `taskkill /IM chrome.exe` is "how the monolith reuses a single media
# window instead of stacking tabs" — and no such call site exists. Every
# `taskkill /IM` in core/, skills/, tools/, bobert_companion.py, tray.py and
# hud/ is `_reap_wedged_ollama`, whose images are ollama.exe / llama-server.exe
# / "ollama app.exe"; window reuse is done with pygetwindow's w.close() in
# `_close_browser_windows_matching`. So the exemption bought nothing and cost an
# unbounded hole: one `taskkill /IM chrome.exe /F` from a test closes every
# Chrome window the owner has open — losing his tabs, which is strictly worse
# than the tab-spam incident this guard was written for.
_PROCESS_TOOLS = frozenset({
    "tasklist.exe", "tasklist", "ps", "pgrep",
    "wmic.exe", "wmic", "sc.exe", "sc", "qprocess",
    "findstr.exe", "findstr", "grep", "where.exe", "where", "which",
    "reg.exe", "reg", "query",
})

# Kill verbs: deliberately NOT exempt. They stay inside the browser scan, so
# `taskkill /F /IM chrome.exe` / `pkill firefox` are blocked and recorded while
# `taskkill /IM ollama.exe` and `taskkill /F /T /PID <n>` (the repo's only real
# users) pass straight through, untouched.
_KILL_TOOLS = frozenset({
    "taskkill.exe", "taskkill", "pkill", "killall", "kill",
})

# Commands that hand a URL to the DEFAULT browser without ever naming one, so
# the binary scan cannot see them. Only a launch: a URL-shaped argument has to
# be present too, which is what keeps `explorer <folder>` and `curl <url>` out.
_URL_LAUNCHERS = frozenset({
    "start", "start.exe", "explorer", "explorer.exe",
    "xdg-open", "gio", "gnome-open", "kde-open", "open",
    "rundll32", "rundll32.exe",
})

_URL_RE = re.compile(r'^["\']?(?:https?|ftp)://', re.IGNORECASE)

# Tokeniser for a shell=True string command line: a double-quoted run, a
# single-quoted run, or a bare whitespace-free run.
_TOKEN_RE = re.compile(r'"[^"]*"|\'[^\']*\'|\S+')

# Switches whose NEXT argv element is a whole command LINE, not a path. ONLY an
# element that follows one of these gets re-tokenised — that discriminator is
# what keeps a claude PROMPT that happens to say "chrome" from being blocked
# while `powershell -Command "Start-Process chrome <url>"` is caught.
_CMDLINE_SWITCHES = frozenset({
    "-c", "--command", "-command", "/c", "/k", "-e", "-ec", "-encodedcommand",
})

_THIS_FILE = os.path.normcase(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class BlockedAttempt(typing.NamedTuple):
    """One refused launch. ``test_id`` is the unittest test that caused it (best
    effort, may be ""), ``call_site`` the PRODUCTION frame that actually made
    the call — together they are enough to find and fix the offender."""
    api: str
    target: str
    test_id: str
    call_site: str


# Result of the first install(): (installed, message). Cached so repeated calls
# (tests/__init__ plus all three runners, or a runner invoked twice in-process)
# neither re-wrap nor re-print. See _reset_for_tests().
_state: tuple[bool, str] | None = None

# Every refused launch, in order. Module-level on purpose: the atexit summary
# has to survive whatever imported us.
_blocked: list[BlockedAttempt] = []

# True => record the attempt and then LET IT THROUGH (install(record_only=True)).
_record_only = False

# Undo callables from the live install, newest first. Only _reset_for_tests()
# uses them; a real run never un-guards itself.
_restore: list[typing.Callable[[], None]] = []

_atexit_registered = False

# Stamped on every stub this module installs (see ``_swap``). It is how
# ``unarmed_targets()`` tells "our wrapper is still there" from "something
# replaced it", and how ``_add`` stays repair-only.
_GUARD_MARK = "_jarvis_browser_guard"

# Every attribute the guard neuters, by module. Kept next to the marker so
# ``unarmed_targets()`` cannot drift out of step with ``_install_into``.
_GUARDED_ATTRS = (
    ("webbrowser", ("open", "open_new", "open_new_tab", "get", "register")),
    ("subprocess", ("Popen", "run", "call")),
    ("os", ("startfile",)),
)


# ─── recording ────────────────────────────────────────────────────────────

def _short_path(path: str) -> str:
    """Repo-relative when it is ours, basename otherwise — the summary has to
    stay readable in a terminal."""
    try:
        full = os.path.abspath(path)
        if os.path.normcase(full).startswith(os.path.normcase(_PROJECT_ROOT) + os.sep):
            return os.path.relpath(full, _PROJECT_ROOT).replace(os.sep, "/")
        return os.path.basename(full) or path
    except Exception:  # noqa: BLE001 - a guard must never fail a run
        return str(path)


def _describe(target: typing.Any, limit: int = 120) -> str:
    """One short, printable string for whatever was about to be launched."""
    try:
        if isinstance(target, (list, tuple)):
            text = " ".join(_describe(item, limit) for item in target)
        else:
            try:
                target = os.fspath(target)
            except TypeError:
                pass
            if isinstance(target, bytes):
                target = target.decode("utf-8", "replace")
            text = str(target)
        text = " ".join(text.split())
        return text if len(text) <= limit else text[:limit - 1] + "…"
    except Exception:  # noqa: BLE001
        return "<unprintable>"


def _caller() -> tuple[str, str]:
    """Return ``(test_id, call_site)`` for the code that tried to launch.

    ``test_id`` is the RUNNING unittest method, identified exactly: a frame
    whose ``self`` carries ``_testMethodName`` equal to that frame's own
    function name. (A bare ``hasattr`` check would be fooled by a MagicMock,
    which answers True to every attribute.) ``call_site`` is the nearest frame
    outside this module — normally the production line that made the call.
    """
    test_id = ""
    call_site = ""
    try:
        frame = sys._getframe(1)
    except Exception:  # noqa: BLE001
        return ("", "")
    fallback = ""
    depth = 0
    while frame is not None and depth < 80:
        depth += 1
        code = frame.f_code
        filename = code.co_filename or ""
        if os.path.normcase(os.path.abspath(filename)) != _THIS_FILE:
            if not call_site:
                call_site = f"{_short_path(filename)}:{frame.f_lineno} in {code.co_name}()"
            if not test_id:
                obj = frame.f_locals.get("self")
                try:
                    method = object.__getattribute__(obj, "_testMethodName")
                except Exception:  # noqa: BLE001 - not a TestCase (or a mock)
                    method = None
                if isinstance(method, str) and method == code.co_name:
                    cls = type(obj)
                    test_id = f"{cls.__module__}.{cls.__qualname__}.{method}"
                    break
                if not fallback and os.path.basename(filename).startswith("test_"):
                    fallback = f"{_short_path(filename)}:{frame.f_lineno} in {code.co_name}()"
        frame = frame.f_back
    return (test_id or fallback, call_site)


def _record(api: str, target: typing.Any) -> None:
    """Log one refused launch. Never raises — recording must not be able to
    fail the very call it is neutering."""
    try:
        test_id, call_site = _caller()
        _blocked.append(BlockedAttempt(api, _describe(target), test_id, call_site))
    except Exception:  # noqa: BLE001
        pass


def blocked_attempts() -> tuple[BlockedAttempt, ...]:
    """Every launch refused so far, oldest first."""
    return tuple(_blocked)


def reset() -> None:
    """Forget the recorded attempts (the guard itself stays installed)."""
    _blocked.clear()


# ─── browser detection ────────────────────────────────────────────────────

def _binary_name(token: typing.Any) -> str:
    """'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe' -> 'chrome.exe'."""
    try:
        try:
            token = os.fspath(token)
        except TypeError:
            pass
        if isinstance(token, bytes):
            token = token.decode("utf-8", "replace")
        if not isinstance(token, str):
            return ""
        token = token.strip().strip('"').strip("'")
        # Normalise BOTH separators: a Windows path reaching the POSIX CI
        # runner still has to split on the backslash.
        return token.replace("\\", "/").rsplit("/", 1)[-1].strip().lower()
    except Exception:  # noqa: BLE001
        return ""


def _command_tokens(args: typing.Any) -> list[str]:
    """Flatten a Popen ``args`` into the tokens to inspect.

    A bare string is a shell command line and gets the quote-aware split. A
    sequence is already tokenised — each element is one token — EXCEPT that one
    element can itself be a whole command LINE, which is the shape every real
    shell-mediated launch takes: ``powershell -Command "Start-Process chrome
    <url>"``, ``cmd /c "start chrome <url>"``, ``bash -c "google-chrome <url>"``.
    Those were invisible to the guard (2026-08-20 review), while the docstring
    asserted ``cmd /c start chrome <url>`` was caught — true only of the fully
    split list form nobody actually writes.

    Such an element is split too, but ONLY when it follows a command-line
    switch. That is the discriminator: it re-tokenises the argument that IS a
    command line without shredding a plain positional argument, so
    tools/multi_agent_pipeline.py's multi-line claude prompts (which mention
    browsers and boot scripts in prose) stay out of the scan.
    """
    try:
        if isinstance(args, (list, tuple)):
            elements = [str(item) for item in args if item is not None]
            tokens = list(elements)
            prev = ""
            for item in elements:
                if prev in _CMDLINE_SWITCHES and item:
                    tokens.extend(m.group(0) for m in _TOKEN_RE.finditer(item))
                prev = item.strip().strip('"').strip("'").lower()
            return tokens
        try:
            args = os.fspath(args)
        except TypeError:
            pass
        if isinstance(args, bytes):
            args = args.decode("utf-8", "replace")
        if not isinstance(args, str):
            return []
        return [m.group(0) for m in _TOKEN_RE.finditer(args)]
    except Exception:  # noqa: BLE001
        return []


def _looks_like_url(token: typing.Any) -> bool:
    try:
        return bool(_URL_RE.match(str(token).strip()))
    except Exception:  # noqa: BLE001
        return False


def _browser_in_command(args: typing.Any) -> str | None:
    """Return the offending token if this command would open a browser.

    Surgical by construction: it returns None for everything that is not a
    browser launch, so a non-browser subprocess call is delegated to the real
    function with its arguments untouched.
    """
    tokens = _command_tokens(args)
    if not tokens:
        return None
    if _binary_name(tokens[0]) in _PROCESS_TOOLS:
        return None  # tasklist/wmic/pgrep name a browser, never touch a window
    for token in tokens:
        name = _binary_name(token)
        if name in _BROWSER_BINARIES:
            return token
        if name in _KILL_TOOLS:
            continue     # the kill verb itself is fine; its TARGET is scanned
    # ``start <url>`` / ``explorer <url>`` / ``rundll32 url.dll,… <url>`` open
    # the DEFAULT browser without naming one. A launcher alone is not enough
    # (``explorer C:\JARVIS\logs`` is a folder), and a URL alone is not either
    # (``curl <url>``) — it takes both.
    if any(_binary_name(t) in _URL_LAUNCHERS for t in tokens):
        for token in tokens:
            if _looks_like_url(token):
                return token
    return None


# ─── the neutered stand-ins ───────────────────────────────────────────────

class _BlockedBrowser:
    """What ``webbrowser.get()`` hands back: a controller that records instead
    of launching. Mirrors the ``webbrowser.BaseBrowser`` surface real code
    touches (``name`` / ``open`` / ``open_new`` / ``open_new_tab``)."""

    def __init__(self, using: typing.Any = None) -> None:
        self.name = "jarvis-browser-guard"
        self.basename = "jarvis-browser-guard"
        self.using = using

    def open(self, url, new=0, autoraise=True):  # noqa: A003 - webbrowser's name
        _record("webbrowser.get().open", url)
        return True

    def open_new(self, url):
        _record("webbrowser.get().open_new", url)
        return True

    def open_new_tab(self, url):
        _record("webbrowser.get().open_new_tab", url)
        return True

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} using={self.using!r}>"


class _BlockedPopen:
    """Stand-in for a browser ``subprocess.Popen`` that was never spawned.

    Looks like a process that started and exited cleanly, because that is what
    the caller saw before this guard existed. Supports the context-manager
    protocol so ``with subprocess.Popen(...) as p:`` still works.
    """

    def __init__(self, args: typing.Any) -> None:
        self.args = args
        self.pid = -1
        self.returncode = 0
        self.stdin = None
        self.stdout = None
        self.stderr = None

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode

    def communicate(self, input=None, timeout=None):  # noqa: A002 - Popen's name
        return (None, None)

    def send_signal(self, sig):
        return None

    def terminate(self):
        return None

    def kill(self):
        return None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} args={self.args!r}>"


def _blocked_completed_process(subprocess_mod, args, kwargs):
    """A ``CompletedProcess`` shaped like the one the real call would return —
    empty captured output only when the caller actually asked to capture, and
    in the right str/bytes flavour."""
    completed = getattr(subprocess_mod, "CompletedProcess", None)
    if completed is None:  # a fake subprocess module in a test
        import subprocess as _real_subprocess
        completed = _real_subprocess.CompletedProcess
    captures = bool(kwargs.get("capture_output")) or (
        kwargs.get("stdout") is not None or kwargs.get("stderr") is not None)
    if not captures:
        return completed(args, 0, None, None)
    textual = bool(kwargs.get("text") or kwargs.get("universal_newlines")
                   or kwargs.get("encoding") or kwargs.get("errors"))
    empty = "" if textual else b""
    return completed(args, 0, empty, empty)


# ─── installation ─────────────────────────────────────────────────────────

def _allowed(env: dict | None = None) -> bool:
    """True iff the human explicitly opted out via JARVIS_ALLOW_REAL_BROWSER."""
    raw = (env if env is not None else os.environ).get(ALLOW_ENV_VAR)
    if raw is None:
        return False
    return str(raw).strip().lower() in _ALLOW_WORDS


def _marked(obj: typing.Any) -> bool:
    """True iff OUR stub is ``obj`` or sits somewhere in its wrapper chain.

    Two hardenings over a bare ``getattr``, both of which bit for real:

    * ``object.__getattribute__`` — a ``MagicMock`` fabricates an attribute for
      every name, so ``getattr(mock, _GUARD_MARK, False)`` is truthy and a stub
      replaced by a mock would report itself still in place.
    * a ``__wrapped__`` walk — ``tests/live_data_guard.py`` wraps ``os.startfile``
      too, and a sibling guard sitting on top of our stub is cooperation, not
      displacement. Without this, two correct guards read as one broken one and
      ``_repair()`` would re-wrap on every call.

    Classes get an MRO scan, because ``object.__getattribute__`` on a class does
    not search its bases (``subprocess.Popen`` is a class).
    """
    if isinstance(obj, type):
        try:
            return any(k.__dict__.get(_GUARD_MARK) is True for k in obj.__mro__)
        except Exception:  # noqa: BLE001
            return False
    depth = 0
    while obj is not None and depth < 12:
        depth += 1
        try:
            if object.__getattribute__(obj, _GUARD_MARK) is True:
                return True
        except Exception:  # noqa: BLE001 - not our stub (or a mock)
            pass
        try:
            nxt = object.__getattribute__(obj, "__wrapped__")
        except Exception:  # noqa: BLE001 - the chain ends here
            return False
        if nxt is obj:
            return False
        obj = nxt
    return False


def _swap(module, name: str, replacement, original) -> typing.Callable[[], None] | None:
    """Replace ``module.name`` with ``replacement``, returning an undo callable
    (or None if the assignment did not take). Never raises."""
    try:
        setattr(replacement, _GUARD_MARK, True)
        # Record what we wrapped so a SIBLING guard's marker check can look
        # through us, and ours through it. See _marked().
        if not isinstance(replacement, type):
            replacement.__wrapped__ = original
    except Exception:  # noqa: BLE001 - an immutable replacement; only a marker
        pass
    try:
        setattr(module, name, replacement)
    except Exception:  # noqa: BLE001 - a read-only module attribute, somehow
        return None

    def _undo() -> None:
        """Put ``original`` back — but ONLY while WE still own the attribute.

        MEASURED 2026-08-20. ``tests/live_data_guard.py`` also wraps
        ``os.startfile``, and which of the two ends up on top is decided by
        install order. ``tests/__init__.py`` arms the live-data guard FIRST so
        our stub sits UNDER it and this snapshot restores the sibling's hook —
        but the three tools/ runners called ``browser_guard.install()`` with no
        live-data guard in the process yet, so under them ``original`` is the
        RAW ``os.startfile`` and the sibling wrapped US. A blind ``setattr``
        then threw the live-data hook away: after the first
        ``_reset_for_tests()`` in tests/test_browser_guard.py the whole rest of
        the CI run had NO interception on ``os.startfile``, while
        ``live_data_guard.is_armed()`` went on answering True. With
        ``JARVIS_ALLOW_REAL_BROWSER=1`` that is a live
        ``os.startfile(r"C:\\JARVIS\\_boot_jarvis.ps1")`` — the shell route
        that boots a real JARVIS and deletes the owner's clean_shutdown.flag.

        So: identity-check first. If the attribute is no longer our
        replacement, someone else owns it now (a sibling guard, or a live
        ``mock.patch``) and restoring our snapshot would destroy their work.
        Leave it alone — an extra layer of blocking is safe, a missing one is
        not. The runner wiring was fixed too; this makes the property hold
        whatever the install order turns out to be."""
        try:
            if getattr(module, name, None) is not replacement:
                return
        except Exception:  # noqa: BLE001 - a diagnostic must never raise
            return
        try:
            setattr(module, name, original)
        except Exception:  # noqa: BLE001
            pass

    return _undo


def _install_into(webbrowser_mod, subprocess_mod, os_mod) -> list[typing.Callable[[], None]]:
    """Do the attribute surgery on the three given modules.

    THE MOCK SEAM: the tests hand this fake module objects, so they exercise the
    real wrappers without touching the live interpreter's ``webbrowser`` /
    ``subprocess`` / ``os`` — and therefore without being able to un-guard the
    suite they are running inside. Returns the undo callables, newest first.
    """
    undos: list[typing.Callable[[], None]] = []

    def _add(module, name, factory):
        """Neuter one attribute. A MISSING attribute is skipped silently — that
        is ``os.startfile`` on POSIX, and the "never raises when the target
        attribute is missing" half of the contract.

        An attribute that ALREADY carries our marker is skipped too, which makes
        this whole function repair-only: re-running it re-wraps exactly the
        targets something else displaced and leaves the intact ones alone, so no
        target is ever double-wrapped. See ``_repair()``."""
        if module is None:
            return
        try:
            if not hasattr(module, name):
                return
            original = getattr(module, name)
            if _marked(original):
                return
        except Exception:  # noqa: BLE001
            return
        try:
            replacement = factory(original)
        except Exception:  # noqa: BLE001
            return
        undo = _swap(module, name, replacement, original)
        if undo is not None:
            undos.insert(0, undo)

    # ── webbrowser: the leak itself ──
    def _make_open(real):
        def open(url, new=0, autoraise=True):  # noqa: A001 - webbrowser's name
            _record("webbrowser.open", url)
            if _record_only and callable(real):
                return real(url, new, autoraise)
            return True
        return open

    def _make_open_new(real):
        def open_new(url):
            _record("webbrowser.open_new", url)
            if _record_only and callable(real):
                return real(url)
            return True
        return open_new

    def _make_open_new_tab(real):
        def open_new_tab(url):
            _record("webbrowser.open_new_tab", url)
            if _record_only and callable(real):
                return real(url)
            return True
        return open_new_tab

    def _make_get(real):
        def get(using=None):
            if _record_only and callable(real):
                return real(using)
            # No webbrowser.Error for an unknown name: always hand back the
            # neutered controller. It cannot launch anything, and raising would
            # push callers down their chrome.exe/msedge.exe fallback chain for
            # no benefit.
            return _BlockedBrowser(using)
        return get

    def _make_register(real):
        def register(name, klass, instance=None, *, preferred=False):
            _record("webbrowser.register", name)
            if _record_only and callable(real):
                return real(name, klass, instance, preferred=preferred)
            return None  # recorded no-op: never make a launcher reachable
        return register

    _add(webbrowser_mod, "open", _make_open)
    _add(webbrowser_mod, "open_new", _make_open_new)
    _add(webbrowser_mod, "open_new_tab", _make_open_new_tab)
    _add(webbrowser_mod, "get", _make_get)
    _add(webbrowser_mod, "register", _make_register)

    # ── os.startfile: Windows-only shell launch (browser, Explorer, anything) ──
    def _make_startfile(real):
        def startfile(path, *args, **kwargs):
            _record("os.startfile", path)
            if _record_only and callable(real):
                return real(path, *args, **kwargs)
            return None
        return startfile

    # On POSIX the attribute simply does not exist; _add skips it silently.
    _add(os_mod, "startfile", _make_startfile)

    # ── subprocess: ONLY when the command line names a browser ──
    # These three are SIGNATURE-TRANSPARENT on purpose (*args/**kwargs, with the
    # command pulled from position 0 or the ``args=`` keyword): a non-browser
    # call must reach the real function with byte-identical arguments, so the
    # ~200 legitimate subprocess call sites in this repo cannot notice the
    # wrapper at all. check_call/check_output need no wrapper — they resolve
    # ``call``/``run`` as subprocess module globals at call time, so they land
    # on these.
    def _command_of(args, kwargs):
        return args[0] if args else kwargs.get("args")

    def _make_popen(real):
        """``subprocess.Popen`` MUST stay a CLASS, and a subclassable one.

        MEASURED 2026-08-20: replacing it with a plain function wrapper made
        ``import unittest.mock`` explode — mock imports asyncio, and
        ``asyncio/windows_utils.py`` line 132 is ``class Popen(subprocess.Popen)``.
        A function base class raises ``TypeError: function() argument 'code' must
        be code, not str`` and takes every test module down with it. Same reason
        an ``isinstance(p, subprocess.Popen)`` check anywhere would have silently
        started returning False.

        So: subclass the real Popen and intercept in ``__new__``. Returning an
        object that is not an instance of ``cls`` makes Python SKIP ``__init__``
        — and ``Popen.__init__`` is where the process is actually spawned — so
        the browser never launches. No ``__init__`` is defined here, so a
        pass-through construction runs the real one, untouched.
        """
        if not isinstance(real, type):  # already wrapped by something else
            def Popen(*args, **kwargs):  # noqa: N802 - subprocess's name
                cmd = _command_of(args, kwargs)
                if _browser_in_command(cmd) is None:
                    return real(*args, **kwargs)
                _record("subprocess.Popen", cmd)
                if _record_only:
                    return real(*args, **kwargs)
                return _BlockedPopen(cmd)
            return Popen

        class GuardedPopen(real):
            def __new__(cls, *args, **kwargs):
                cmd = _command_of(args, kwargs)
                if _browser_in_command(cmd) is None:
                    return super().__new__(cls)   # untouched pass-through
                _record("subprocess.Popen", cmd)
                if _record_only:
                    return super().__new__(cls)
                return _BlockedPopen(cmd)         # not a cls => __init__ skipped

        # Wear the real name: reprs, error messages and anything that formats
        # the type should be indistinguishable from stock subprocess.Popen.
        GuardedPopen.__name__ = getattr(real, "__name__", "Popen")
        GuardedPopen.__qualname__ = getattr(real, "__qualname__", "Popen")
        GuardedPopen.__module__ = getattr(real, "__module__", "subprocess")
        GuardedPopen.__doc__ = getattr(real, "__doc__", None)
        return GuardedPopen

    def _make_run(real):
        def run(*args, **kwargs):
            cmd = _command_of(args, kwargs)
            if _browser_in_command(cmd) is None:
                return real(*args, **kwargs)      # untouched pass-through
            _record("subprocess.run", cmd)
            if _record_only:
                return real(*args, **kwargs)
            return _blocked_completed_process(subprocess_mod, cmd, kwargs)
        return run

    def _make_call(real):
        def call(*args, **kwargs):
            cmd = _command_of(args, kwargs)
            if _browser_in_command(cmd) is None:
                return real(*args, **kwargs)      # untouched pass-through
            _record("subprocess.call", cmd)
            if _record_only:
                return real(*args, **kwargs)
            return 0
        return call

    _add(subprocess_mod, "Popen", _make_popen)
    _add(subprocess_mod, "run", _make_run)
    _add(subprocess_mod, "call", _make_call)

    return undos


def summary() -> str:
    """The atexit report: how many launches were refused, by which tests, from
    which production call sites. This is how the NEXT agent finds the offenders,
    so it names the test method, not just a file."""
    if not _blocked:
        return ""
    # Identity of an "offender" = the test when we could name it, else the
    # frame that called. Distinct call sites are grouped and counted.
    who: dict[str, None] = {}
    sites: dict[tuple[str, str, str], int] = {}
    for att in _blocked:
        who[att.test_id or att.call_site or "<unknown>"] = None
        key = (att.test_id, att.api, att.call_site)
        sites[key] = sites.get(key, 0) + 1
    lines = [f"{_TAG} blocked {len(_blocked)} real browser launch(es) "
             f"from {len(who)} test(s)"]
    shown = sorted(sites.items(), key=lambda kv: (-kv[1], kv[0]))
    for (test_id, api, call_site), count in shown[:40]:
        lines.append(f"{_TAG}   x{count} {test_id or '<unknown test>'}"
                     f" -> {api}() at {call_site or '<unknown site>'}")
    if len(shown) > 40:
        lines.append(f"{_TAG}   … and {len(shown) - 40} more distinct call site(s)")
    lines.append(f"{_TAG}   (these tests reach a REAL browser — patch the "
                 f"boundary in the test, or route it through a fake)")
    return "\n".join(lines)


def _print_summary_at_exit() -> None:
    """Silent on a clean run; loud when something tried to open a browser."""
    try:
        text = summary()
        if text:
            print(text, flush=True)
    except Exception:  # noqa: BLE001 - an atexit hook must never explode
        pass


def install(record_only: bool = False, *, env: dict | None = None,
            quiet: bool = False) -> bool:
    """Arm the guard process-wide. Returns True iff real launches are blocked.

    Never raises. Prints exactly one line describing what happened (armed,
    record-only, or explicitly disabled). Idempotent — later calls return the
    first result without re-wrapping or re-printing, so it is safe to call from
    ``tests/__init__.py`` AND from every runner.
    """
    global _state, _record_only, _atexit_registered
    if _state is not None:
        # Latched — but "latched" is not "armed". Repair anything that displaced
        # one of our stubs since the first call before answering. A guard that
        # reports itself installed while a stub has been swapped out from under
        # it is exactly this repo's #1 bug class (see _repair).
        if _state[0]:
            _repair()
        return _state[0]

    if _allowed(env):
        msg = (f"{_TAG} DISABLED via {ALLOW_ENV_VAR} — this run CAN open REAL "
               f"browser windows in the owner's live profile")
        _state = (False, msg)
    else:
        try:
            _record_only = bool(record_only)
            import subprocess
            import webbrowser
            _restore.extend(_install_into(webbrowser, subprocess, os))
            if not _atexit_registered:
                import atexit
                atexit.register(_print_summary_at_exit)
                _atexit_registered = True
        except Exception as exc:  # noqa: BLE001 - a guard must never fail a run
            msg = (f"{_TAG} WARNING: could NOT arm ({type(exc).__name__}: {exc})"
                   f" — this run CAN open REAL browser windows")
            _state = (False, msg)
        else:
            if _record_only:
                msg = (f"{_TAG} RECORD-ONLY — real browser launches are "
                       f"ALLOWED THROUGH and merely logged")
                _state = (False, msg)
            else:
                msg = (f"{_TAG} armed — webbrowser / os.startfile / subprocess "
                       f"browser launches are blocked and recorded "
                       f"({ALLOW_ENV_VAR}=1 to disable)")
                _state = (True, msg)

    if not quiet:
        print(_state[1], flush=True)
    return _state[0]


def unarmed_targets() -> list[str]:
    """Which of the guard's stubs are NOT currently in place, by name.

    ``is_installed()`` only says install() ran and meant to arm; this says
    whether the interception is still THERE. The two came apart for real on
    2026-08-20: ``tests/live_data_guard.py`` parked its Popen interception on
    whatever ``subprocess.Popen`` named at install time, ``_reset_for_tests()``
    rebound that attribute, and the live-data guard spent the rest of the CI run
    reporting itself installed while a live JARVIS spawn went straight through.
    Same shape of hole, so the same check exists here."""
    bad: list[str] = []
    for modname, names in _GUARDED_ATTRS:
        module = sys.modules.get(modname)
        if module is None:
            continue
        for name in names:
            try:
                if not hasattr(module, name):
                    continue          # os.startfile on POSIX
                if not _marked(getattr(module, name)):
                    bad.append(f"{modname}.{name}")
            except Exception:  # noqa: BLE001 - a diagnostic must never raise
                bad.append(f"{modname}.{name}")
    return bad


def _repair() -> None:
    """Re-wrap any displaced stub. Silent, never raises, never double-wraps.

    ``_install_into`` skips targets that still carry ``_GUARD_MARK``, so this
    touches only what was actually lost."""
    try:
        if not unarmed_targets():
            return
        import subprocess
        import webbrowser
        _restore.extend(_install_into(webbrowser, subprocess, os))
    except Exception:  # noqa: BLE001 - a guard must never fail a run
        pass


def is_installed() -> bool:
    """True iff install() armed the guard in THIS process."""
    return bool(_state and _state[0])


def is_armed() -> bool:
    """True iff install() armed the guard AND every stub is still in place."""
    return is_installed() and not unarmed_targets()


def _reset_for_tests() -> None:
    """Un-wrap and drop the idempotence latch. Tests only — never call this
    from a runner or a test module's body; it would leave the rest of the run
    able to open real windows."""
    global _state, _record_only
    while _restore:
        try:
            _restore.pop(0)()
        except Exception:  # noqa: BLE001
            pass
    _state = None
    _record_only = False
    _blocked.clear()


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    sys.exit(0 if install() else 1)
