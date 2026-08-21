"""Process-wide CREATE_NO_WINDOW safety net for console-subsystem spawns.

JARVIS runs as pythonw (GUI subsystem, no console). On Windows, ANY console
app it spawns without a no-window flag makes the OS allocate a console and
pop a visible window — and on this box the default-terminal delegation
routes that to Windows Terminal, which piles up "ghost" windows the owner
can't close. v2.0.32 fixed the 9 spawn sites an audit confirmed, but the
ghosts came back within hours through sites the audit missed (the unified
HUD's ~3s nvidia-smi utilization poll, the monolith's GPU-temp poll, a
health ping): ~38 windows in 30 minutes. Fixing sites one by one loses to
entropy — every future skill is one bare subprocess.run() away from
re-introducing the leak.

install() patches Popen.__init__ so a spawn that specifies NEITHER
creationflags NOR startupinfo gets CREATE_NO_WINDOW by default:

  • subprocess.run/call/check_output/check_call all route through Popen,
    so one patch covers every stdlib entry point.
  • A caller that passes ANY creationflags (DETACHED_PROCESS,
    CREATE_NEW_CONSOLE, its own CREATE_NO_WINDOW…) is left untouched —
    deliberate console windows stay possible, explicitly.
  • A caller that passes startupinfo is left untouched (it is already
    managing window visibility via STARTF_USESHOWWINDOW).
  • GUI-subsystem children (pythonw, .pyw overlays) ignore the flag, so
    blanket application is harmless to them.
  • No-op on non-Windows and on double-install.

WHICH CLASS GETS PATCHED, AND WHY IT MATTERS (2026-08-20)
--------------------------------------------------------
The patch goes on the BASE-most class in ``subprocess.Popen``'s MRO, not on
whatever ``subprocess.Popen`` happens to NAME at install time.

That used to be the other way round, and it silently uninstalled this net for
most of every test run. ``tools/browser_guard.py`` arms itself by REBINDING
``subprocess.Popen`` to a fresh ``class GuardedPopen(subprocess.Popen)``, and
``tests/test_browser_guard.py``'s ``_unlatched()`` helper restores the STOCK
class and then builds a BRAND NEW subclass. Our ``__init__`` had been written
onto the subclass that just got thrown away — and because ``install()``
short-circuited on "I stored an original once", every later call returned True.
"Already installed" while nothing was installed: this repo's #1 bug class,
aimed at the net whose absence produced the ghost-window incident.

The base class is the one object a rebind cannot replace, and every wrapper
subclasses it, so a patch there composes with guards installed before OR after
this one. ``install()`` is now REPAIR-ONLY: it re-applies whatever is missing
(including a subclass ``__init__`` that SHADOWS the base patch) and leaves an
intact net alone, and ``is_armed()`` answers whether the net is really in the
effective ``Popen.__init__`` chain — never "install() ran once".

Call install() ONCE, as early as possible, in EVERY JARVIS process that can
spawn helpers: the monolith, the HUD/reticle/air-cursor overlays, the tray.
2026-07-10."""
from __future__ import annotations

import os
import subprocess

# The original Popen.__init__ from the FIRST install, kept for tests. A
# single-element list per house style (mutated, never rebound).
_ORIG_INIT = [None]

# Stamped on every wrapper this module installs. ``_marked()`` is how
# ``is_armed()`` tells "our net is still there" from "something replaced it",
# and how install() stays repair-only instead of latching.
_GUARD_MARK = "_jarvis_no_window_guard"

_MISSING = object()

# [(class, previous own __init__ or _MISSING)] in application order, so
# uninstall() puts back exactly what was there — including on a class we do not
# own, which a blanket ``subprocess.Popen.__init__ = saved`` would corrupt.
_APPLIED: list = []


def _marked(fn) -> bool:
    """True iff OUR wrapper is ``fn`` or sits anywhere in its wrapper chain.

    ``object.__getattribute__`` on purpose: a ``MagicMock`` fabricates an
    attribute for every name, so a bare ``getattr`` marker check answers True
    for a hook that has been replaced by a mock — reporting armed while
    disarmed, which is worse than no net at all.

    The ``__wrapped__`` walk exists because SIBLING guards wrap this same
    ``__init__`` (``tests/live_data_guard.py`` does). Without it two guards that
    are cooperating correctly read as one broken one, and each repair pass would
    stack another wrapper.
    """
    depth = 0
    while fn is not None and depth < 12:
        depth += 1
        try:
            if object.__getattribute__(fn, _GUARD_MARK) is True:
                return True
        except Exception:
            pass
        try:
            nxt = object.__getattribute__(fn, "__wrapped__")
        except Exception:
            return False
        if nxt is fn:
            return False
        fn = nxt
    return False


def _popen_base():
    """The base-most non-``object`` class in ``subprocess.Popen``'s MRO — the
    one object no rebind of the module attribute can replace."""
    cls = subprocess.Popen
    if not isinstance(cls, type):
        return None
    bases = [c for c in cls.__mro__ if c is not object]
    return bases[-1] if bases else None


def _wrap(orig):
    create_no_window = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)

    def _no_window_init(self, *args, **kwargs):
        if not kwargs.get("creationflags") and kwargs.get("startupinfo") is None:
            kwargs["creationflags"] = create_no_window
        return orig(self, *args, **kwargs)

    try:
        _no_window_init.__wrapped__ = orig
        setattr(_no_window_init, _GUARD_MARK, True)
    except Exception:  # noqa: BLE001 - an immutable wrapper; only a marker
        pass
    return _no_window_init


def _apply(cls, orig) -> None:
    prev = cls.__dict__.get("__init__", _MISSING)
    try:
        cls.__init__ = _wrap(orig)
    except Exception:  # noqa: BLE001 - a safety net must never fail a spawn
        return
    _APPLIED.append((cls, prev))


def install() -> bool:
    """Activate the safety net. Returns True if it is (now) active.

    Idempotent AND REPAIRING: a later call re-applies whatever was displaced
    since the first one and leaves an intact net alone, so it is safe to call
    from any number of entry points in any order and never double-wraps.
    """
    if os.name != "nt":
        return False
    base = _popen_base()
    if base is None:                      # Popen replaced by a plain callable
        return False
    if not _marked(base.__init__):
        if _ORIG_INIT[0] is None:
            _ORIG_INIT[0] = base.__init__
        _apply(base, base.__init__)
    # A subclass with its OWN __init__ SHADOWS the base patch for attribute
    # lookup, so the net would be present and bypassed at the same time. Repair
    # every such shadow rather than report installed while it is stepped over.
    cls = subprocess.Popen
    if isinstance(cls, type):
        for klass in cls.__mro__:
            if klass is base or klass is object:
                continue
            own = klass.__dict__.get("__init__")
            if own is not None and not _marked(own):
                _apply(klass, own)
    return True


def is_armed() -> bool:
    """True iff the net is really in the effective ``Popen.__init__`` chain.

    ``install()`` returning True is NOT evidence — that was exactly the lie this
    module told for most of a CI run after a ``subprocess.Popen`` rebind threw
    the patched subclass away.
    """
    if os.name != "nt":
        return False
    cls = subprocess.Popen
    try:
        if not isinstance(cls, type):
            return _marked(cls)
        return _marked(cls.__init__)
    except Exception:  # noqa: BLE001 - a diagnostic must never raise
        return False


def uninstall() -> None:
    """Restore the stock Popen (tests only). Undoes every class this module
    patched, newest first, putting back exactly what was there before."""
    while _APPLIED:
        cls, prev = _APPLIED.pop()
        try:
            if prev is _MISSING:
                try:
                    delattr(cls, "__init__")
                except AttributeError:
                    pass
            else:
                cls.__init__ = prev
        except Exception:  # noqa: BLE001
            pass
    _ORIG_INIT[0] = None
