"""In-process message bus — M2 Phase 2 groundwork (de-monolith roadmap).

This is the structured IPC layer that will replace the ~50 ``*_state.json`` files
JARVIS currently uses for tray / HUD / blue-green / pending-speech IPC
(see docs/design/M2-process-isolation.md §4). It ships the ABSTRACTION + the
default IN-PROCESS transport (thread-safe synchronous dispatch).

It deliberately ships UNWIRED: nothing in the hot path depends on it yet, per the
M2 design's "each phase ships green, the monolith always boots." A later phase
swaps the transport for named pipes / UDS and migrates the existing JSON-file
subscribers (HUD, tray) onto it, with file IPC kept as a fallback for one release.

Two channels (M2 §4):
  • pub/sub — fire-and-forget events (``wake``, ``gaze``, ``print_progress`` …).
  • request/response — brain↔service RPC (one handler per method).

CI-safety: stdlib only (``threading``), import-light, and every public method is
TOTAL — a raising subscriber is isolated (reported, never propagated) so one bad
subscriber can't crash the bus, the publisher, or the other subscribers.
"""
from __future__ import annotations

import threading
from typing import Any, Callable, Dict, List, Optional, Tuple


class BusError(Exception):
    """Raised by ``request()`` when a method has no handler, or its handler raised."""


class MessageBus:
    """A thread-safe in-process pub/sub + request/response bus."""

    def __init__(self, *,
                 on_error: Optional[Callable[[str, BaseException], None]] = None):
        # topic -> list of (subscription_id, handler)
        self._subs: Dict[str, List[Tuple[int, Callable[[Any], None]]]] = {}
        # method -> single handler
        self._handlers: Dict[str, Callable[[Any], Any]] = {}
        self._lock = threading.RLock()
        self._next_id = 0
        self._on_error = on_error

    # ── pub/sub ────────────────────────────────────────────────────────
    def subscribe(self, topic: str,
                  handler: Callable[[Any], None]) -> Callable[[], None]:
        """Register ``handler`` for ``topic``. Returns an ``unsubscribe()``
        callable (idempotent — safe to call more than once)."""
        with self._lock:
            sid = self._next_id
            self._next_id += 1
            self._subs.setdefault(topic, []).append((sid, handler))

        def _unsub() -> None:
            with self._lock:
                lst = self._subs.get(topic)
                if not lst:
                    return
                remaining = [(i, h) for (i, h) in lst if i != sid]
                if remaining:
                    self._subs[topic] = remaining
                else:
                    del self._subs[topic]

        return _unsub

    def publish(self, topic: str, payload: Any = None) -> int:
        """Deliver ``payload`` to every subscriber of ``topic``; returns the
        count delivered. A subscriber that raises is isolated (reported) and does
        NOT stop the others or the publisher."""
        with self._lock:
            handlers = list(self._subs.get(topic, ()))
        delivered = 0
        for _sid, handler in handlers:
            try:
                handler(payload)
                delivered += 1
            except BaseException as e:  # isolate a bad subscriber
                self._report(f"subscriber for topic {topic!r}", e)
        return delivered

    # ── request/response ───────────────────────────────────────────────
    def register_handler(self, method: str,
                         fn: Callable[[Any], Any]) -> None:
        """Register the single handler for RPC ``method`` (replaces any prior)."""
        with self._lock:
            self._handlers[method] = fn

    def unregister_handler(self, method: str) -> None:
        """Remove the handler for ``method`` if present (idempotent)."""
        with self._lock:
            self._handlers.pop(method, None)

    def request(self, method: str, payload: Any = None) -> Any:
        """Call the handler for ``method`` and return its result. Raises
        ``BusError`` if there is no handler, or wraps a handler exception."""
        with self._lock:
            fn = self._handlers.get(method)
        if fn is None:
            raise BusError(f"no handler registered for method {method!r}")
        try:
            return fn(payload)
        except Exception as e:
            self._report(f"handler for method {method!r}", e)
            raise BusError(f"handler for method {method!r} failed: {e}") from e

    # ── introspection ──────────────────────────────────────────────────
    def topics(self) -> List[str]:
        """Sorted list of topics with at least one subscriber."""
        with self._lock:
            return sorted(self._subs)

    def methods(self) -> List[str]:
        """Sorted list of registered RPC methods."""
        with self._lock:
            return sorted(self._handlers)

    # ── internal ───────────────────────────────────────────────────────
    def _report(self, what: str, e: BaseException) -> None:
        """Route an isolated error to ``on_error`` (or a console line). Never
        raises — an ``on_error`` that itself raises is swallowed."""
        if self._on_error is not None:
            try:
                self._on_error(what, e)
            except Exception:
                pass
        else:
            print(f"  [message-bus] {what} raised: {e!r}")


# Process-wide default bus, created lazily. Opt-in: nothing reads it until a
# later M2 phase wires subscribers onto it.
_DEFAULT: Optional[MessageBus] = None
_DEFAULT_LOCK = threading.Lock()


def default_bus() -> MessageBus:
    """The lazily-created process-wide :class:`MessageBus`."""
    global _DEFAULT
    with _DEFAULT_LOCK:
        if _DEFAULT is None:
            _DEFAULT = MessageBus()
        return _DEFAULT
