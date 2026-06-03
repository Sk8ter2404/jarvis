"""Typed capability seam between skills and the monolith — M2 Phase 1.

Background
----------
Skills reach back into the ~14.7K-line ``bobert_companion`` monolith through a
single object the loader injects at import time: ``skill_utils`` — a *dict of
~15 bound lambdas* closing over monolith globals (``ask_vision``, ``click``,
``write_hud_state``, the promise helpers, …). That dict is untyped: a skill that
mistypes a key gets a ``KeyError`` at runtime, there's no IDE/`pyflakes` signal,
and every consumer hand-rolls its own ``skill_utils.get(key)`` + ``None`` /
``NameError`` guard.

``docs/design/M2-process-isolation.md`` §"the two seams" calls for formalising
that dict into a **typed ``JarvisServices`` interface** as Phase 1 — the
highest-leverage, lowest-risk slice, landable with *zero* process split. Skills
depend on the *interface*, not the module, so a later phase can move a capability
across a process bus (vision/audio service) without touching a single skill.

What this module is (Phase 1)
-----------------------------
A thin, **additive** wrapper. ``JarvisServices.from_skill_utils(d)`` takes the
*existing* ``skill_utils`` dict and returns an object whose typed methods each
delegate to ``d["<key>"](...)``. No behaviour changes: the dict is still built,
still injected as ``mod.skill_utils``, and every skill that calls
``skill_utils["write_hud_state"](...)`` keeps working unchanged. The monolith
*also* injects ``mod.services = JarvisServices.from_skill_utils(skill_utils)``
alongside it, so a skill can migrate to ``services.write_hud_state(...)`` at its
own pace (Phase 1 migrates exactly one skill as a reference).

Graceful degradation
---------------------
``from_skill_utils`` tolerates a *partial* dict (an older tree, or the isolated
test harness pinning only one key). Each method's missing-key behaviour mirrors
what the monolith itself does when the backing capability is unavailable, so a
skill sees identical semantics whether it goes through the dict or this object:

  * **Side-effecting UI / HUD / app calls** (``write_hud_state``, ``click``,
    ``type_text``, ``press_key``, ``hotkey``, ``scroll``, ``sleep``,
    ``launch_app``, ``open_url``) — **no-op** when absent. The monolith already
    treats these as best-effort (HUD writes are "nice-to-have, not load-bearing";
    the ``ui_*`` helpers ``return`` early when pyautogui is unavailable), so a
    silent no-op is the safe, faithful degradation.
  * **Vision queries that return a value** (``ask_vision`` → ``""``,
    ``take_screenshot`` → ``None``, ``find_click_target`` → ``None``) — return
    the same empty/None sentinel the monolith returns on a vision failure, so
    callers' existing "no answer" handling fires.
  * **Promise helpers** (``make_promise`` → ``None``,
    ``register_promise_condition`` → ``None``, ``fulfil_promise`` → ``False``) —
    exactly the fallbacks the monolith installs when ``core.memory`` (the
    promise store) failed to import.

Stdlib-only by contract
------------------------
This module is on the CI *light-tier* coverage surface (``core/`` is measured on
the bare-Linux runner that lacks numpy/cv2/sounddevice). It must therefore import
nothing heavier than the standard library — and it doesn't: only ``typing``.

See also
--------
``tests/test_services.py`` (CI-safe, mocks everything) and the monolith-tier
loader-injection test in ``tests/monolith/``.
"""
from __future__ import annotations

from typing import Any, Callable, Optional, Protocol, Tuple, runtime_checkable

__all__ = ["JarvisServices", "JarvisServicesProtocol"]


@runtime_checkable
class JarvisServicesProtocol(Protocol):
    """Structural contract for the skill→monolith capability seam.

    Any object exposing these methods (the concrete :class:`JarvisServices`
    below, or a future process-bus RPC proxy) satisfies it. Declared
    ``runtime_checkable`` so a defensive skill can ``isinstance(obj,
    JarvisServicesProtocol)`` before use, mirroring today's
    ``isinstance(skill_utils, dict)`` guard.
    """

    # — vision —
    def ask_vision(self, question: str, png_bytes: Optional[bytes] = None) -> str: ...
    def take_screenshot(self) -> Optional[bytes]: ...
    def find_click_target(self, description: str) -> Optional[Tuple[int, int]]: ...

    # — UI automation —
    def click(self, x: int, y: int, button: str = "left") -> None: ...
    def type_text(self, text: str) -> None: ...
    def press_key(self, key: str) -> None: ...
    def hotkey(self, *keys: str) -> None: ...
    def scroll(self, amount: int) -> None: ...
    def sleep(self, seconds: float) -> None: ...

    # — apps / OS —
    def launch_app(self, name: str) -> Any: ...
    def open_url(self, url: str) -> Any: ...

    # — HUD —
    def write_hud_state(self, **fields: Any) -> None: ...

    # — contextual promises —
    def make_promise(self, message: str, condition: str, **kwargs: Any) -> Optional[int]: ...
    def register_promise_condition(self, *args: Any, **kwargs: Any) -> None: ...
    def fulfil_promise(self, promise_id: int) -> bool: ...


# Sentinel that means "no backing callable was wired for this key". Distinct from
# a real ``None`` value a dict might legitimately hold, so we never mistake an
# explicit ``{"click": None}`` for "key present".
_MISSING = object()


class JarvisServices:
    """Concrete, typed facade over the monolith's ``skill_utils`` dict.

    Construct via :meth:`from_skill_utils`. Each method delegates to the
    correspondingly-named lambda in the wrapped dict and degrades gracefully
    when that key is absent (see the module docstring for per-method behaviour).

    This object is what the loader injects as ``mod.services``. It holds a
    *reference* to the live ``skill_utils`` dict (not a copy), so it stays in
    lockstep if the monolith ever swaps a lambda — there is exactly one source
    of truth.
    """

    __slots__ = ("_utils",)

    def __init__(self, utils: dict) -> None:
        # Stored by reference on purpose — see class docstring.
        self._utils = utils if isinstance(utils, dict) else {}

    # ── construction ──────────────────────────────────────────────────────
    @classmethod
    def from_skill_utils(cls, utils: dict) -> "JarvisServices":
        """Wrap an existing ``skill_utils`` dict. Thin, zero behaviour change.

        ``utils`` may be partial; absent keys degrade per the module docstring.
        A non-dict (defensive) is treated as empty, so *every* method no-ops /
        returns its safe sentinel rather than raising ``AttributeError``.
        """
        return cls(utils)

    # ── internal delegation helpers ───────────────────────────────────────
    def _fn(self, key: str) -> Optional[Callable[..., Any]]:
        """Return the backing callable for ``key``, or ``None`` if the key is
        absent or its value isn't callable (a malformed dict). Never raises."""
        fn = self._utils.get(key, _MISSING)
        if fn is _MISSING or not callable(fn):
            return None
        return fn

    def _call(self, key: str, *args: Any, _default: Any = None, **kwargs: Any) -> Any:
        """Delegate to ``utils[key](*args, **kwargs)``; return ``_default`` if
        the key is absent. Present-but-raising callables propagate — a wired
        capability that throws is a real error the caller should see, exactly as
        ``skill_utils[key](...)`` would surface it today."""
        fn = self._fn(key)
        if fn is None:
            return _default
        return fn(*args, **kwargs)

    # ── vision ────────────────────────────────────────────────────────────
    def ask_vision(self, question: str, png_bytes: Optional[bytes] = None) -> str:
        """Ask Claude vision a question about the screen (or a supplied PNG).

        Returns the model's answer, or ``""`` if vision is unwired — matching
        the monolith's own empty-string return on a vision failure.
        """
        # The monolith lambda is ``lambda *a, **kw: ask_vision(*a, **kw)``; pass
        # png_bytes only when given so we hit ask_vision's own default.
        if png_bytes is None:
            return self._call("ask_vision", question, _default="")
        return self._call("ask_vision", question, png_bytes, _default="")

    def take_screenshot(self) -> Optional[bytes]:
        """PNG bytes of the primary monitor, or ``None`` if unavailable."""
        return self._call("take_screenshot", _default=None)

    def find_click_target(self, description: str) -> Optional[Tuple[int, int]]:
        """Two-pass vision locate of a UI element. ``(x, y)`` or ``None``."""
        return self._call("find_click_target", description, _default=None)

    # ── UI automation (all best-effort no-ops when unwired) ───────────────
    def click(self, x: int, y: int, button: str = "left") -> None:
        """Mouse-click at virtual-desktop ``(x, y)`` (clamped on-screen by the
        backing helper). No-op if UI automation is unavailable."""
        self._call("click", x, y, button)

    def type_text(self, text: str) -> None:
        """Type ``text`` into the focused window. No-op if unavailable."""
        self._call("type_text", text)

    def press_key(self, key: str) -> None:
        """Press a single key (e.g. ``"enter"``). No-op if unavailable."""
        self._call("press_key", key)

    def hotkey(self, *keys: str) -> None:
        """Press a chord (e.g. ``hotkey("ctrl", "shift", "p")``). No-op if
        unavailable."""
        self._call("hotkey", *keys)

    def scroll(self, amount: int) -> None:
        """Scroll the wheel ``amount`` clicks (sign = direction). No-op if
        unavailable."""
        self._call("scroll", amount)

    def sleep(self, seconds: float) -> None:
        """Block the calling skill for ``seconds`` (wraps ``time.sleep``). No-op
        if the helper is unwired — a test harness commonly stubs this so a skill
        never actually sleeps."""
        self._call("sleep", seconds)

    # ── apps / OS ─────────────────────────────────────────────────────────
    def launch_app(self, name: str) -> Any:
        """Launch a desktop app by name. Returns whatever the backing action
        returns (a status string in the monolith); ``None`` if unwired."""
        return self._call("launch_app", name, _default=None)

    def open_url(self, url: str) -> Any:
        """Open ``url`` in the default browser. Returns the backing action's
        result; ``None`` if unwired."""
        return self._call("open_url", url, _default=None)

    # ── HUD ───────────────────────────────────────────────────────────────
    def write_hud_state(self, **fields: Any) -> None:
        """Merge ``fields`` into the shared HUD state via the monolith's
        canonical, lock-guarded ``_write_hud_state`` writer (so concurrent skill
        writers can't clobber each other). No-op if the HUD is unwired —
        faithful to the monolith treating HUD writes as best-effort.
        """
        self._call("write_hud_state", **fields)

    # ── contextual promises ───────────────────────────────────────────────
    def make_promise(self, message: str, condition: str, **kwargs: Any) -> Optional[int]:
        """Register a deferred announcement (``"I'll tell you when X finishes"``).

        Returns the new promise id, or ``None`` if the promise store
        (``core.memory``) isn't loaded — the same fallback the monolith installs
        when that import fails. Extra keyword args (``params``, ``deadline_s``,
        ``source``) pass straight through to ``memory.make_promise``.
        """
        return self._call("make_promise", message, condition, _default=None, **kwargs)

    def register_promise_condition(self, *args: Any, **kwargs: Any) -> None:
        """Register a custom promise-condition predicate. No-op (returns
        ``None``) if the promise store isn't loaded."""
        return self._call("register_promise_condition", *args, **kwargs)

    def fulfil_promise(self, promise_id: int) -> bool:
        """Force-fire a pending promise now. Returns ``True`` on success,
        ``False`` if it couldn't be fired or the store isn't loaded."""
        return bool(self._call("fulfil_promise", promise_id, _default=False))
