#!/usr/bin/env python3
"""Process-wide block on a TEST RUN reaching the owner's LIVE ``data/``.

THE INCIDENT (2026-08-20)
=========================
The owner deliberately shut JARVIS down and parked the resurrection watchdog by
hand-writing ``data/clean_shutdown.flag`` (its body is prose — ``owner asked
JARVIS stay down 2026-08-20`` — not the ``str(time.time())`` production writes,
so it is a MANUAL sentinel that nothing recreates once destroyed). Test runs
destroyed it repeatedly and JARVIS resurrected against his explicit
instruction — at 03:29, 10:24, 17:44 and 18:34 the same day.

Root cause, proven by interception (stack captured live):

    tests/test_audit_2026_07_14.py  LlmIndependentControlPlaneTests
      -> bobert_companion._dispatch_tray_command("restart")   :3341
      -> core.actions._act_restart                            :515
      -> threading.Thread(target=_do_restart).start()
         ... 1.5 s later, on a DAEMON THREAD, after the test returned ...
      -> subprocess.Popen([sys.executable,
                           r"C:\\JARVIS\\bobert_companion.py"],
                          creationflags=DETACHED_PROCESS)     :491
      -> the child runs main(), whose `_clean_flag` unlink (bobert_companion.py,
         in main(); ~:23238 as of 2026-08-20 — the monolith moves, so trust the
         symbol, not the line) os.unlink()s the LIVE data/clean_shutdown.flag.

WHY NO ENV REDIRECT COULD EVER HAVE STOPPED IT
----------------------------------------------
Two independent reasons, and both matter:

1. The child is a SEPARATE, DETACHED process running the real
   ``bobert_companion.py``. It outlives the test runner (``_do_restart`` then
   hard-exits the runner via TerminateProcess), so a runner-scoped redirect is
   irrelevant to it.
2. The flag path is not resolved through the staging-aware helper at all.
   ``bobert_companion.py``'s ``main()._clean_flag`` and its module-level
   ``_CLEAN_SHUTDOWN_FLAG`` both build it as
   ``os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
   "clean_shutdown.flag")`` — bound to ``__file__``, so ``JARVIS_DATA_DIR``
   and ``JARVIS_STAGING`` are BOTH powerless over it. ``core/paths.py`` exists
   precisely to prevent this and is bypassed here.

So ``tools/run_tests.py`` / ``run_tests_ci_sim.py``'s
``_redirect_data_dir_to_throwaway()`` was never the missing piece, and neither
was the scratchpad harness's copy of it: the escape route goes around every one
of them. Only a process-wide interception installed before the first test is
imported can close it — which is what this module is, and it is the live-data
sibling of ``tools/browser_guard.py`` (real browsers) and ``tools/mem_guard.py``
(RAM). A test run must not be able to damage the box it runs on.

WHAT IS INTERCEPTED
-------------------
* **Spawning a real JARVIS.** ``subprocess.Popen`` / ``run`` / ``call`` whose
  command names one of ``_BOOT_SCRIPTS`` — the scripts that ARE a live JARVIS
  (``bobert_companion.py``, ``_boot_jarvis.ps1``, ``upgrade_jarvis.py``,
  ``overnight_upgrade.py``, ``jarvis_watchdog.py``, ``staging_instance.py``)
  AND the second-order LAUNCHERS whose whole job is to start one
  (``stability_smoke_test.py``, ``multi_agent_pipeline.py``, ``bounce_jarvis.py``,
  ``staging_integration.py``, ``staging_inject_smoke.py``,
  ``blue_green_manager.py``). The launchers matter because the guard cannot
  reach into a child process: ``multi_agent_pipeline._run_tester`` spawns
  ``python tools/stability_smoke_test.py``, which spawns
  ``powershell -File _boot_jarvis.ps1``, and only the FIRST hop is ours.

  Matched on each argument's BASENAME, plus a quote-aware tokenisation of any
  argument that FOLLOWS a shell command-line switch (``-Command`` / ``-c`` /
  ``/c`` / ``/k``): ``core/actions._act_upgrade`` spawns
  ``["powershell", "-Command", "…; python 'C:\\JARVIS\\upgrade_jarvis.py'
  --relaunch"]``, where basename-of-the-whole-element is
  ``upgrade_jarvis.py' --relaunch`` and matches nothing. Never on a substring
  of the whole command line, and never on a plain positional argument, so a
  ``claude`` prompt that merely mentions the upgrade pipeline (see
  ``tests/test_multi_agent_pipeline.py``) still passes straight through.

  This is the one that caused the incident, and it is also the only rule that
  can stop it: the delete happens in another process, minutes later.
* **Shell-launching a boot script** — ``os.startfile`` of one. It is the
  Windows shell-launch primitive, it needs no ``subprocess`` anywhere in the
  path, and ``core/actions.py`` and ``tray.py`` both use it. Only
  ``tools/browser_guard.py`` used to hook it, as a side effect of being the
  BROWSER guard, so ``JARVIS_ALLOW_REAL_BROWSER=1`` removed the sole
  interception on that route while this guard still answered ``is_armed()``
  True. The two are now independent.
* **Deleting anything under the live ``data/``** — ``os.remove`` / ``unlink`` /
  ``rmdir`` / ``removedirs``, ``shutil.rmtree``, ``pathlib.Path.unlink`` /
  ``rmdir``. A tripwire sweep of all 230 test modules on 2026-08-20 recorded
  ZERO legitimate deletes under live ``data/``, so this blocks nothing that
  passes today.
* **Clobbering the flag itself** — ``os.rename`` / ``os.replace`` onto it, or
  opening it in a writing mode through ``builtins.open``, ``io.open`` (which is
  where ``pathlib.Path.open`` / ``write_text`` / ``write_bytes`` land),
  ``os.open`` with a write flag, or ``os.truncate``. ``builtins.open`` alone
  used to be the whole of it while ``unarmed_hooks()`` reported the write path
  covered — one refactor to the more idiomatic ``pathlib`` spelling would have
  reopened the exact loss this guard exists to prevent. The owner's sentinel is
  prose; a production ``_write_clean_shutdown_flag`` landing on it would
  overwrite that with a timestamp and lose the note.

WHAT IS DELIBERATELY *NOT* INTERCEPTED
--------------------------------------
``os.replace`` and ``open(..., "w")`` onto OTHER live ``data/`` files. Three
suites do this today through production code that hardcodes the live path —
``test_monolith_kinect_overlay`` (``.hud_camera_preview_kinect.jpg``),
``test_monolith_update_check`` (``update_check.json``) and
``test_monolith_sec6`` (``bug_reports.jsonl``). That is real debt (see
``tests/test_staging_data_isolation.py``'s N-arg ratchet), but it is a
DIFFERENT defect from the one that killed the flag, and blocking it here would
turn three green suites red without protecting the flag any further. They are
RECORDED in ``violations()`` instead, so the debt stays visible.

Escape hatch for a human deliberately driving live state:
``JARVIS_ALLOW_LIVE_DATA=1``. It is announced by ``banner()``, which
``tests/__init__.py`` prints on every entry path — a run that CAN destroy the
owner's state must never be byte-identical in the log to one that cannot.
"""
from __future__ import annotations

import builtins
import io
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import traceback

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIVE_DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CLEAN_SHUTDOWN_FLAG = os.path.join(LIVE_DATA_DIR, "clean_shutdown.flag")

# Scripts whose execution IS a live JARVIS, plus the LAUNCHERS that start one.
#
# The launchers are not optional. This guard lives in ONE process and cannot
# reach into a child (the module docstring's own argument for why a
# runner-scoped redirect could never have worked), so blocking
# ``_boot_jarvis.ps1`` here is irrelevant once a child is already running
# ``tools/stability_smoke_test.py``. Add every launcher, not just every
# launchee: tests/test_live_data_guard.py::SecondOrderLauncherTests pins the
# list and checks each name still exists in the tree.
_BOOT_SCRIPTS = frozenset({
    # ── these ARE a live JARVIS ──
    "bobert_companion.py",
    "_boot_jarvis.ps1",
    "upgrade_jarvis.py",
    "overnight_upgrade.py",
    "jarvis_watchdog.py",
    "staging_instance.py",
    # ── these LAUNCH one, one process further out ──
    "stability_smoke_test.py",   # _launch_jarvis() -> powershell -File _boot_jarvis.ps1
    "multi_agent_pipeline.py",   # _run_tester() -> python stability_smoke_test.py
    "bounce_jarvis.py",
    "staging_integration.py",
    "staging_inject_smoke.py",
    "blue_green_manager.py",
})

# Tokeniser for a shell command LINE: a double-quoted run, a single-quoted run,
# or a bare whitespace-free run. Same shape as tools/browser_guard.py's.
_TOKEN_RE = re.compile(r'"[^"]*"|\'[^\']*\'|\S+')

# Switches whose NEXT argv element is a whole command LINE rather than a path.
# ONLY an element that follows one of these gets re-tokenised — that is what
# keeps false positives out. tests/test_multi_agent_pipeline.py hands the claude
# CLI multi-line PROMPTS that name the boot scripts, and those arrive as plain
# positional arguments, never after -c / -Command.
_CMDLINE_SWITCHES = frozenset({
    "-c", "--command", "-command", "/c", "/k", "-e", "-ec", "-encodedcommand",
})

# Characters a shell leaves clinging to a token (`…bobert_companion.py;`).
_TOKEN_TRIM = "\"';&|,()"

_ENV_ESCAPE = "JARVIS_ALLOW_LIVE_DATA"

_INSTALLED = False
_VIOLATIONS: list[dict] = []

# True while unarmed_hooks() is running its behavioural spawn probe. Mutated,
# never rebound, per house style.
_PROBING = [False]

# The probe's argv. Both elements live in a directory that DOES NOT EXIST, so if
# the interception is gone the real Popen fails at CreateProcess without ever
# starting a process — the probe can only ever be a question, never an action.
_PROBE_DIR = os.path.join(
    tempfile.gettempdir(), f"__jarvis_live_data_guard_probe_{os.getpid()}__")
_PROBE_ARGV = (os.path.join(_PROBE_DIR, "python.exe"),
               os.path.join(_PROBE_DIR, "bobert_companion.py"))

# Every wrapper this module installs carries this marker. Two jobs:
#   * ``is_armed()`` can tell "our hook is still in place" from "something
#     replaced it" — a guard that quietly stops being applied is this repo's #1
#     bug class, and one that is TRUSTED while disarmed is worse than none;
#   * ``_arm()`` stays repair-only, so calling ``install()`` again from a second
#     entry point re-installs what was lost without ever double-wrapping what
#     was not.
_GUARD_MARK = "_jarvis_live_data_guard"


class LiveDataGuardError(RuntimeError):
    """A test tried to touch the owner's live runtime state."""


def violations() -> list[dict]:
    """Every intercepted attempt, blocked or merely recorded."""
    return list(_VIOLATIONS)


def reset() -> None:
    _VIOLATIONS.clear()


def _allowed() -> bool:
    return os.environ.get(_ENV_ESCAPE, "").strip() == "1"


def _mark(fn, real=None):
    """Stamp our marker and record what ``fn`` wraps.

    ``__wrapped__`` is not cosmetic: SIBLING guards wrap these same targets
    (``tools/browser_guard.py`` wraps ``os.startfile``; ``core/no_window_
    subprocess.py`` wraps ``Popen.__init__``), and ``_marked()`` walks the chain
    so two guards cooperating correctly do not read as one broken one."""
    try:
        setattr(fn, _GUARD_MARK, True)
        if real is not None:
            fn.__wrapped__ = real
    except Exception:  # noqa: BLE001 - an immutable wrapper; only a marker
        pass
    return fn


def _marked(obj) -> bool:
    """True iff OUR hook is ``obj`` or sits somewhere in its wrapper chain.

    ``object.__getattribute__``, not ``getattr``: a ``MagicMock`` fabricates an
    attribute for EVERY name, so a bare ``getattr(fn, _GUARD_MARK, False)``
    returns a truthy child mock and reports the guard armed after a leaked
    ``mock.patch("os.remove")`` has displaced it. ``tools/browser_guard.py``
    documents and defends against this exact trap in ``_caller()``; the
    live-data guard's equivalent check did not, and its ratchet — whose whole
    job is to detect displacement — was blind to the most common way a hook
    gets displaced in this suite.

    Classes get an MRO scan instead, because ``object.__getattribute__`` on a
    class does not search its bases.
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
        except Exception:  # noqa: BLE001 - not our wrapper (or a mock)
            pass
        try:
            nxt = object.__getattribute__(obj, "__wrapped__")
        except Exception:  # noqa: BLE001 - chain ends here
            return False
        if nxt is obj:
            return False
        obj = nxt
    return False


def _norm(path) -> str:
    try:
        if hasattr(path, "__fspath__"):
            path = os.fspath(path)
        if isinstance(path, bytes):
            path = path.decode("utf-8", "replace")
        if not isinstance(path, str):
            return ""
        return os.path.normcase(os.path.abspath(path))
    except Exception:
        return ""


_LIVE_N = os.path.normcase(os.path.abspath(LIVE_DATA_DIR))
_FLAG_N = os.path.normcase(os.path.abspath(CLEAN_SHUTDOWN_FLAG))


def _under_live_data(path) -> bool:
    p = _norm(path)
    return bool(p) and (p == _LIVE_N or p.startswith(_LIVE_N + os.sep))


def _is_clean_flag(path) -> bool:
    return _norm(path) == _FLAG_N


def _record(kind: str, detail: str, blocked: bool) -> None:
    if _PROBING[0]:
        # unarmed_hooks()'s behavioural probe deliberately trips the guard. It
        # is a diagnostic, not an offence, so it must not show up in the ledger
        # the next agent reads to find real offenders.
        return
    _VIOLATIONS.append({
        "kind": kind,
        "detail": detail,
        "blocked": blocked,
        "stack": "".join(traceback.format_stack()[:-2]),
    })


def _refuse(kind: str, detail: str) -> None:
    _record(kind, detail, True)
    raise LiveDataGuardError(
        f"[live-data-guard] BLOCKED {kind}: {detail}\n"
        f"A test must never touch the owner's live runtime state. Point the "
        f"code under test at a tempdir (see core.paths.data_file), or patch "
        f"the module constant. If you are a human deliberately driving live "
        f"state, set {_ENV_ESCAPE}=1."
    )


def _as_text(item) -> str:
    """One argv element as a plain string, or "" if it is not text-like."""
    try:
        if isinstance(item, os.PathLike):
            item = os.fspath(item)
        if isinstance(item, bytes):
            item = item.decode("utf-8", "replace")
        return item if isinstance(item, str) else ""
    except Exception:  # noqa: BLE001
        return ""


def _script_basename(token: str) -> str:
    """``"C:\\JARVIS\\upgrade_jarvis.py'"`` -> ``upgrade_jarvis.py``.

    Both separators are normalised, not just ``os.sep``, so a Windows path
    reaching the POSIX CI runner still splits."""
    try:
        text = token.strip().strip(_TOKEN_TRIM).strip()
        return text.replace("\\", "/").rsplit("/", 1)[-1].strip().lower()
    except Exception:  # noqa: BLE001
        return ""


def _command_tokens(args) -> list[str]:
    """The tokens to inspect for a boot-script name.

    A bare string is a shell command line and gets the quote-aware split. A
    sequence is already tokenised — EXCEPT that one element can itself be a
    whole command line, which is the shape ``powershell -Command "…"`` /
    ``cmd /c "…"`` / ``bash -c "…"`` takes and the shape
    ``core/actions._act_upgrade`` actually spawns. Those elements are split too,
    but ONLY when they follow a command-line switch: that is the discriminator
    that keeps a claude PROMPT mentioning ``upgrade_jarvis.py`` from tripping
    the guard, which several tests depend on.
    """
    try:
        if isinstance(args, (str, bytes, os.PathLike)):
            raw = _as_text(args)
            return [m.group(0) for m in _TOKEN_RE.finditer(raw)]
        tokens: list[str] = []
        prev = ""
        for item in args:
            text = _as_text(item)
            tokens.append(text)
            if text and prev in _CMDLINE_SWITCHES:
                tokens.extend(m.group(0) for m in _TOKEN_RE.finditer(text))
            prev = text.strip().strip(_TOKEN_TRIM).lower()
        return tokens
    except Exception:  # noqa: BLE001
        return []


def _boot_script_in(args) -> str | None:
    """The live-JARVIS script named by ``args``, or None.

    Matches a token's BASENAME, never a substring of the whole command line, so
    a long prompt that merely mentions a script name cannot trip the guard."""
    for token in _command_tokens(args):
        try:
            base = _script_basename(token)
            if base in _BOOT_SCRIPTS:
                return base
        except Exception:  # noqa: BLE001
            continue
    return None


# ── the wrappers ────────────────────────────────────────────────────────────
# Module level, not closures inside install(), so _arm() can rebuild exactly the
# one that was displaced without disturbing the others.

def _guard_delete(op: str, real):
    def wrapper(path, *a, **k):
        if _under_live_data(path):
            _refuse(op, str(path))
        return real(path, *a, **k)
    return _mark(wrapper, real)


def _guard_path_method(op: str, real):
    def wrapper(self, *a, **k):
        if _under_live_data(self):
            _refuse(op, str(self))
        return real(self, *a, **k)
    return _mark(wrapper, real)


def _guard_move(op: str, real):
    def wrapper(src, dst, *a, **k):
        if _is_clean_flag(dst) or _is_clean_flag(src):
            _refuse(op, f"{src} -> {dst}")
        if _under_live_data(dst):
            _record(op, f"{src} -> {dst}", False)
        return real(src, dst, *a, **k)
    return _mark(wrapper, real)


def _check_write(op: str, file) -> None:
    """The shared decision for every write-ish entry point. Cheap basename test
    first — some of these wrap EVERY open() in the run."""
    try:
        base = "" if isinstance(file, int) else os.path.basename(os.fspath(file))
    except Exception:  # noqa: BLE001
        base = ""
    if base == "clean_shutdown.flag" and _is_clean_flag(file):
        _refuse(op, str(file))
    elif base and _under_live_data(file):
        _record(op, str(file), False)


def _guard_open(real):
    def wrapper(file, mode="r", *a, **k):
        try:
            writing = isinstance(mode, str) and any(c in mode for c in "wxa+")
        except Exception:
            writing = False
        if writing:
            _check_write("open(%s)" % mode, file)
        return real(file, mode, *a, **k)
    return _mark(wrapper, real)


# Any of these in os.open()'s flags means the caller intends to write. Built by
# getattr so the module still imports on a platform missing one of them.
_O_WRITE = 0
for _name in ("O_WRONLY", "O_RDWR", "O_CREAT", "O_TRUNC", "O_APPEND"):
    _O_WRITE |= getattr(os, _name, 0)
del _name


def _guard_os_open(real):
    """``os.open`` bypasses ``builtins.open`` AND ``io.open`` entirely."""
    def wrapper(path, flags, *a, **k):
        try:
            writing = bool(int(flags) & _O_WRITE)
        except Exception:  # noqa: BLE001
            writing = False
        if writing:
            _check_write("os.open", path)
        return real(path, flags, *a, **k)
    return _mark(wrapper, real)


def _guard_truncate(real):
    def wrapper(path, length, *a, **k):
        _check_write("os.truncate", path)
        return real(path, length, *a, **k)
    return _mark(wrapper, real)


def _guard_startfile(real):
    """``os.startfile`` is the Windows shell-launch primitive.

    ``os.startfile(r"C:\\JARVIS\\_boot_jarvis.ps1")`` runs the file through its
    registered handler and boots a live JARVIS with no ``subprocess.Popen``
    anywhere in the path — the tray's menu items and ``core/actions.py`` both
    use this API. It used to be hooked ONLY by ``tools/browser_guard.py``, as a
    side effect of being the browser guard, so a browser-scoped escape hatch
    silently removed the live-data protection for this route too."""
    def wrapper(path, *a, **k):
        script = _boot_script_in([path])
        if script:
            _refuse("os.startfile", f"shell-launches live JARVIS via {script}: "
                                    f"{path}")
        if _under_live_data(path):
            _refuse("os.startfile", str(path))
        return real(path, *a, **k)
    return _mark(wrapper, real)


def _popen_args(a, k):
    """The command out of a ``Popen(...)`` call, positional or ``args=``."""
    return a[0] if a else k.get("args")


def _guard_popen_init(real):
    def wrapper(self, *a, **k):
        cmd = _popen_args(a, k)
        script = _boot_script_in(cmd)
        if script:
            # Popen.__del__ reads _child_created; set it before raising so the
            # refusal cannot turn into a noisy "Exception ignored in __del__"
            # on a partially constructed object.
            self._child_created = False
            _refuse("subprocess.Popen",
                    f"spawns live JARVIS via {script}: {cmd!r:.300}")
        return real(self, *a, **k)
    return _mark(wrapper, real)


def _guard_popen_callable(real):
    """Degraded path: something replaced ``subprocess.Popen`` with a plain
    callable, so there is no class to patch. Wrap the callable rather than
    silently arm nothing."""
    def wrapper(*a, **k):
        cmd = _popen_args(a, k)
        script = _boot_script_in(cmd)
        if script:
            _refuse("subprocess.Popen",
                    f"spawns live JARVIS via {script}: {cmd!r:.300}")
        return real(*a, **k)
    return _mark(wrapper, real)


# ── arming ──────────────────────────────────────────────────────────────────

def _popen_base(cls):
    """The class that actually spawns: the base-most non-``object`` entry in
    ``subprocess.Popen``'s MRO.

    WHY THE MRO AND NOT ``subprocess.Popen`` ITSELF — MEASURED 2026-08-20
    --------------------------------------------------------------------
    ``tools/browser_guard.py`` arms itself by REBINDING ``subprocess.Popen`` to
    a fresh ``class GuardedPopen(subprocess.Popen)`` (it documents at :565 why
    it must stay a subclassable class), and ``tests/test_browser_guard.py``'s
    ``_unlatched()`` helper un-wraps and re-wraps it mid-run:
    ``_reset_for_tests()`` restores the STOCK class, then ``install()`` builds a
    BRAND NEW subclass. Patching whatever ``subprocess.Popen`` happens to name
    at install time therefore parks the interception on a class object that the
    next rebind throws away — and the guard goes on reporting itself installed.

    Not a theory. With the old code ``tests.test_live_data_guard`` passed ALONE
    and its two spawn tests FAILED under the full CI. A probe (a
    ``unittest.TestCase.run`` hook comparing object identities before every one
    of the 15,005 tests) pinned the drift to
    ``InstallContractTests.test_env_escape_hatch_disables_the_guard_loudly``,
    the first ``_unlatched()`` user: from there on
    ``subprocess.Popen.__init__`` was the stock one and a
    ``Popen([python, "bobert_companion.py"])`` went STRAIGHT THROUGH.

    The base class is the one object no rebind can replace, and every wrapper
    subclasses it, so a patch here composes with guards installed before OR
    after this one and survives a snapshot/restore of the module attribute.
    """
    bases = [c for c in cls.__mro__ if c is not object]
    return bases[-1] if bases else None


def _arm_popen() -> None:
    cls = subprocess.Popen
    if not isinstance(cls, type):        # degraded: already a plain callable
        if not _marked(cls):
            subprocess.Popen = _guard_popen_callable(cls)
        return
    base = _popen_base(cls)
    if base is not None and not _marked(base.__init__):
        base.__init__ = _guard_popen_init(base.__init__)
    # A class between ``cls`` and ``base`` that defines its OWN ``__init__``
    # SHADOWS the base patch for attribute lookup, so the hook can be present
    # and stepped over at the same time.
    #
    # NOT hypothetical (2026-08-20): ``core/no_window_subprocess.install()``
    # wrote exactly such a shadow onto ``tools/browser_guard.py``'s
    # ``GuardedPopen`` on every monolith import. That one chained back to our
    # marked base hook so the spawn was still refused, but a shadow that does
    # NOT chain is a real hole — so repair every unmarked one, deepest first.
    for klass in cls.__mro__:
        if klass is base or klass is object:
            continue
        own = klass.__dict__.get("__init__")
        if own is not None and not _marked(own):
            try:
                klass.__init__ = _guard_popen_init(own)
            except Exception:  # noqa: BLE001 - a guard must never fail a run
                pass


def _arm() -> None:
    """(Re)install every interception that is missing or has been displaced.

    REPAIR-ONLY: a hook still carrying ``_GUARD_MARK`` is left exactly as it is,
    so this is safe to call from any number of entry points in any order and
    never stacks two wrappers on one target."""
    if not _marked(os.remove):
        os.remove = _guard_delete("os.remove", os.remove)
    if not _marked(os.unlink):
        os.unlink = _guard_delete("os.unlink", os.unlink)
    if not _marked(os.rmdir):
        os.rmdir = _guard_delete("os.rmdir", os.rmdir)
    if not _marked(os.removedirs):
        os.removedirs = _guard_delete("os.removedirs", os.removedirs)
    if not _marked(shutil.rmtree):
        shutil.rmtree = _guard_delete("shutil.rmtree", shutil.rmtree)

    if not _marked(pathlib.Path.unlink):
        pathlib.Path.unlink = _guard_path_method(
            "Path.unlink", pathlib.Path.unlink)
    if not _marked(pathlib.Path.rmdir):
        pathlib.Path.rmdir = _guard_path_method(
            "Path.rmdir", pathlib.Path.rmdir)

    if not _marked(os.rename):
        os.rename = _guard_move("os.rename", os.rename)
    if not _marked(os.replace):
        os.replace = _guard_move("os.replace", os.replace)

    # builtins.open and io.open are the SAME function object until the first is
    # replaced, so each is wrapped from its own current binding and neither ends
    # up double-wrapped. io.open is the one that matters for pathlib:
    # Path.open()'s body ends `return io.open(self, ...)` and resolves it as a
    # module attribute at call time, so Path.open / write_text / write_bytes all
    # land here (verified on this interpreter, Python 3.14).
    if not _marked(builtins.open):
        builtins.open = _guard_open(builtins.open)
    if not _marked(io.open):
        io.open = _guard_open(io.open)
    if not _marked(os.open):
        os.open = _guard_os_open(os.open)
    if not _marked(os.truncate):
        os.truncate = _guard_truncate(os.truncate)

    # Windows-only; _arm() must not create the attribute on POSIX, because
    # several tests do mock.patch.object(mod.os, "startfile", create=True) and
    # rely on it being absent there.
    if hasattr(os, "startfile") and not _marked(os.startfile):
        os.startfile = _guard_startfile(os.startfile)

    _arm_popen()


def _popen_interception_is_live() -> bool:
    """Is the spawn interception really in place? Answered BY BEHAVIOUR.

    WHY NOT A STRUCTURAL CHECK — REPRODUCED 2026-08-20
    --------------------------------------------------
    This used to ask "does ``subprocess.Popen.__init__`` carry our marker?".
    ``core/no_window_subprocess.install()`` writes its own ``__init__`` onto
    whatever class ``subprocess.Popen`` names at that moment — during a test run
    that is ``tools/browser_guard.py``'s ``GuardedPopen`` subclass — so from the
    first monolith import onward the answer was "no", even though the shadow
    chained straight back to our marked base hook and the spawn WAS refused:

        python capped_verbose.py "tests.monolith.test_monolith_sec7
            tests.test_live_data_guard" 8 order.log
        -> FAIL test_every_interception_is_still_in_place
           AssertionError: Lists differ: ['subprocess.Popen.__init__'] != []

    The full CI stayed green only because ``tests/test_browser_guard.py`` sorts
    between those two modules and rebinds ``subprocess.Popen`` to a fresh
    subclass, discarding the shadow. The one mechanism built to catch "reports
    armed while disarmed" was itself green for the wrong reason, and one
    alphabetical accident away from red — at which point the honest answer is to
    stop guessing from structure and just TRY it.

    The probe names an executable inside a directory that does not exist, so an
    UNGUARDED run fails at CreateProcess without starting anything: this can
    only ever ask a question, never take an action. Recording is suppressed
    while it runs so the probe never appears in ``violations()``.
    """
    cls = subprocess.Popen
    proc = None
    _PROBING[0] = True
    try:
        proc = cls(list(_PROBE_ARGV))
    except LiveDataGuardError:
        return True
    except Exception:  # noqa: BLE001 - CreateProcess said no: nothing refused it
        return False
    finally:
        _PROBING[0] = False
        if proc is not None:  # pragma: no cover - cannot happen with a fake exe
            for meth in ("kill", "wait"):
                try:
                    getattr(proc, meth)()
                except Exception:  # noqa: BLE001
                    pass
    return False


def unarmed_hooks() -> list[str]:
    """Which interceptions are NOT currently in place, by name.

    The honest answer to "is this run protected?". ``_INSTALLED`` only says
    install() ran once — and that is exactly the lie that let the 2026-08-20
    spawn through.

    Every name listed here is a hook the guard actually installs; the ratchet in
    tests/test_live_data_guard.py displaces each one and checks that it is
    NAMED, because "absent from the list" is what a hook the guard has never
    heard of looks like too.
    """
    bad: list[str] = []
    targets = [("os.remove", os.remove),
               ("os.unlink", os.unlink),
               ("os.rmdir", os.rmdir),
               ("os.removedirs", os.removedirs),
               ("shutil.rmtree", shutil.rmtree),
               ("pathlib.Path.unlink", pathlib.Path.unlink),
               ("pathlib.Path.rmdir", pathlib.Path.rmdir),
               ("os.rename", os.rename),
               ("os.replace", os.replace),
               ("builtins.open", builtins.open),
               ("io.open", io.open),
               ("os.open", os.open),
               ("os.truncate", os.truncate)]
    if hasattr(os, "startfile"):
        targets.append(("os.startfile", os.startfile))
    for name, obj in targets:
        if not _marked(obj):
            bad.append(name)
    if not _popen_interception_is_live():
        bad.append("subprocess.Popen.__init__"
                   if isinstance(subprocess.Popen, type) else "subprocess.Popen")
    return bad


def is_armed() -> bool:
    """True iff install() ran AND every interception is still in place."""
    return _INSTALLED and not unarmed_hooks()


def banner() -> str:
    """The one line every entry point prints — see ``tests/__init__.py``.

    Both siblings emit exactly one loud line and both call that contract out
    explicitly (``tools/mem_guard.py``, ``tools/browser_guard.py``). This guard
    — the one protecting the artefact the owner actually lost — used to print
    NOTHING in either direction, so a run with ``JARVIS_ALLOW_LIVE_DATA=1`` was
    byte-identical in the log to a fully protected one: two armed banners from
    the siblings and silence about live data. Absence of a line is not a signal
    anyone reads.
    """
    if _allowed():
        return (f"[live-data-guard] DISABLED via {_ENV_ESCAPE} — this run CAN "
                f"delete the owner's live data/ and spawn a real JARVIS")
    if not _INSTALLED:
        return ("[live-data-guard] NOT ARMED — this run CAN damage the owner's "
                "live data/")
    missing = unarmed_hooks()
    if missing:
        return (f"[live-data-guard] PARTIALLY armed on {LIVE_DATA_DIR} — these "
                f"interceptions are DISPLACED: {', '.join(missing)}")
    return f"[live-data-guard] armed on {LIVE_DATA_DIR}"


def install() -> bool:
    """Arm the guard. Returns True on the call that first armed it.

    Idempotent AND REPAIRING: a later call re-installs any hook something else
    displaced in the meantime, so the guard cannot end up half-armed while
    still reporting itself installed. Safe to call from several entry points in
    any order (tests/__init__.py does; the tools/ runners may too)."""
    global _INSTALLED
    if _allowed():
        return False
    first = not _INSTALLED
    _INSTALLED = True
    _arm()
    return first


if __name__ == "__main__":  # pragma: no cover - manual smoke
    install()
    print(banner(), file=sys.stderr)
