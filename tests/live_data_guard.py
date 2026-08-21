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
  command names one of ``_BOOT_SCRIPTS`` (``bobert_companion.py``,
  ``_boot_jarvis.ps1``, ``upgrade_jarvis.py``, ``overnight_upgrade.py``,
  ``jarvis_watchdog.py``, ``staging_instance.py``). Matched on each argument's
  BASENAME, never on a substring of the whole command line, so a ``claude``
  prompt that merely mentions the upgrade pipeline (see
  ``tests/test_multi_agent_pipeline.py``) passes straight through.
  This is the one that caused the incident, and it is also the only rule that
  can stop it: the delete happens in another process, minutes later.
* **Deleting anything under the live ``data/``** — ``os.remove`` / ``unlink`` /
  ``rmdir`` / ``removedirs``, ``shutil.rmtree``, ``pathlib.Path.unlink`` /
  ``rmdir``. A tripwire sweep of all 230 test modules on 2026-08-20 recorded
  ZERO legitimate deletes under live ``data/``, so this blocks nothing that
  passes today.
* **Clobbering the flag itself** — ``os.rename`` / ``os.replace`` onto it, or
  ``open()`` of it in a writing mode. The owner's sentinel is prose; a
  production ``_write_clean_shutdown_flag`` landing on it would overwrite that
  with a timestamp and lose the note.

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
``JARVIS_ALLOW_LIVE_DATA=1``.
"""
from __future__ import annotations

import builtins
import os
import pathlib
import shutil
import subprocess
import sys
import traceback

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIVE_DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CLEAN_SHUTDOWN_FLAG = os.path.join(LIVE_DATA_DIR, "clean_shutdown.flag")

# Scripts whose execution IS a live JARVIS (or a live pipeline that boots one).
_BOOT_SCRIPTS = frozenset({
    "bobert_companion.py",
    "_boot_jarvis.ps1",
    "upgrade_jarvis.py",
    "overnight_upgrade.py",
    "jarvis_watchdog.py",
    "staging_instance.py",
})

_ENV_ESCAPE = "JARVIS_ALLOW_LIVE_DATA"

_INSTALLED = False
_VIOLATIONS: list[dict] = []

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


def _mark(fn):
    try:
        setattr(fn, _GUARD_MARK, True)
    except Exception:  # noqa: BLE001 - an immutable wrapper; only a marker
        pass
    return fn


def _marked(fn) -> bool:
    return bool(getattr(fn, _GUARD_MARK, False))


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


def _boot_script_in(args) -> str | None:
    """The live-JARVIS script named by ``args``, or None.

    Matches each argument's BASENAME so a long prompt string that merely
    mentions a script name cannot trip the guard."""
    try:
        if isinstance(args, (str, bytes, os.PathLike)):
            raw = os.fspath(args) if isinstance(args, os.PathLike) else args
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", "replace")
            items = raw.split()
        else:
            items = list(args)
    except Exception:
        return None
    for item in items:
        try:
            if isinstance(item, os.PathLike):
                item = os.fspath(item)
            if isinstance(item, bytes):
                item = item.decode("utf-8", "replace")
            if not isinstance(item, str):
                continue
            base = os.path.basename(item.strip().strip('"').replace("/", os.sep))
            if base.lower() in _BOOT_SCRIPTS:
                return base
        except Exception:
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
    return _mark(wrapper)


def _guard_path_method(op: str, real):
    def wrapper(self, *a, **k):
        if _under_live_data(self):
            _refuse(op, str(self))
        return real(self, *a, **k)
    return _mark(wrapper)


def _guard_move(op: str, real):
    def wrapper(src, dst, *a, **k):
        if _is_clean_flag(dst) or _is_clean_flag(src):
            _refuse(op, f"{src} -> {dst}")
        if _under_live_data(dst):
            _record(op, f"{src} -> {dst}", False)
        return real(src, dst, *a, **k)
    return _mark(wrapper)


def _guard_open(real):
    def wrapper(file, mode="r", *a, **k):
        try:
            writing = isinstance(mode, str) and any(c in mode for c in "wxa+")
        except Exception:
            writing = False
        if writing:
            # Cheap basename test first — this wraps EVERY open() in the run.
            try:
                base = os.path.basename(os.fspath(file)) \
                    if not isinstance(file, int) else ""
            except Exception:
                base = ""
            if base == "clean_shutdown.flag" and _is_clean_flag(file):
                _refuse("open(%s)" % mode, str(file))
            elif base and _under_live_data(file):
                _record("open(%s)" % mode, str(file), False)
        return real(file, mode, *a, **k)
    return _mark(wrapper)


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
    return _mark(wrapper)


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
    return _mark(wrapper)


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
    # A wrapper subclass that defined its OWN __init__ would shadow the base
    # patch. None does today; cover it rather than report armed when it is not.
    own = cls.__dict__.get("__init__")
    if own is not None and not _marked(own):
        cls.__init__ = _guard_popen_init(own)


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

    if not _marked(builtins.open):
        builtins.open = _guard_open(builtins.open)

    _arm_popen()


def unarmed_hooks() -> list[str]:
    """Which interceptions are NOT currently in place, by name.

    The honest answer to "is this run protected?". ``_INSTALLED`` only says
    install() ran once — and that is exactly the lie that let the 2026-08-20
    spawn through."""
    bad: list[str] = []
    for name, obj in (("os.remove", os.remove),
                      ("os.unlink", os.unlink),
                      ("os.rmdir", os.rmdir),
                      ("os.removedirs", os.removedirs),
                      ("shutil.rmtree", shutil.rmtree),
                      ("pathlib.Path.unlink", pathlib.Path.unlink),
                      ("pathlib.Path.rmdir", pathlib.Path.rmdir),
                      ("os.rename", os.rename),
                      ("os.replace", os.replace),
                      ("builtins.open", builtins.open)):
        if not _marked(obj):
            bad.append(name)
    cls = subprocess.Popen
    if isinstance(cls, type):
        if not _marked(cls.__init__):
            bad.append("subprocess.Popen.__init__")
    elif not _marked(cls):
        bad.append("subprocess.Popen")
    return bad


def is_armed() -> bool:
    """True iff install() ran AND every interception is still in place."""
    return _INSTALLED and not unarmed_hooks()


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
    print(f"[live-data-guard] armed on {LIVE_DATA_DIR} "
          f"(unarmed hooks: {unarmed_hooks() or 'none'})", file=sys.stderr)
