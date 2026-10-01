"""Generic web-dashboard panels declared by skills (``WEB_PANELS``).

WHY
===
The owner wants every device JARVIS drives reachable from the one web hub.
Hand-writing a dashboard tab per device would put device-specific (and
sometimes PRIVATE) code into the tracked web server. Instead a skill DECLARES
its panel as data - a module-level ``WEB_PANELS`` list of specs - and the
dashboard renders it with a generic widget renderer. A private skill keeps its
panel private: the spec lives in the (gitignored) skill file, and nothing in
the public tree names it.

HOW IT IS WIRED
===============
* ``bobert_companion.load_skills`` calls ``_collect_skill_web_panels(mod,
  name)`` for every loaded skill, right beside the PROMPT_EXAMPLES /
  SPEAK_VERBATIM_ACTIONS collectors; that calls
  ``REGISTRY.register_from_module(mod, name)``. A reload REPLACES the skill's
  panels (never duplicates them).
* ``tools/web_interface`` serves the process-wide ``REGISTRY``:

    GET  /api/panels                    metadata only - no callables, no state
    GET  /api/panel/<id>/state          cached, TIME-LIMITED state()
    POST /api/panel/<id>/action         {"name", "args": {...}, "confirm"?}
    GET  /api/panel/<id>/stream/<name>  MJPEG from a latest-JPEG callable

THE SPEC (a plain dict; only ``id`` and ``title`` are required)
==============================================================
::

    {
      "id": "desk_device",          # ^[a-z0-9][a-z0-9_-]{0,39}$, unique
      "title": "Desk device",       # nav label, <= 60 chars
      "order": 50,                  # nav position, lower first (default 100)
      "poll_ms": 1000,              # state poll while visible, 250..60000
      "state": get_state,           # callable() -> JSON-safe dict (optional)
      "state_timeout_s": 1.0,       # 0.05..10; a slower state() is served stale
      "stop_action": "stop",        # REQUIRED when a "hold" widget exists
      "layout": [ ...widgets... ],
      "actions": {
         "stop":  stop_fn,                                  # callable(args)
         "drive": {"fn": drive_fn, "hold": True, "rate_hz": 8},
         "reset": {"fn": reset_fn, "confirm": True, "danger": True,
                   "label": "Factory reset"},
      },
      "streams": {"cam": latest_jpeg},   # callable() -> bytes | None
    }

An action value is either a bare callable or a dict: ``fn`` (callable(args:
dict) -> JSON-safe result), ``label``, ``confirm`` (the POST must carry
``"confirm": true``), ``hold`` (may be driven by a hold widget), ``rate_hz``
(the rate a hold widget resends at; default 4, max 30 - the server refuses a
caller going faster than twice that), ``danger`` (styling + confirm prompt
wording). The panel's ``stop_action`` and every ``estop`` widget's action
are NEVER rate-limited and may NOT require confirmation - a stop must always
get through on the first press.

WIDGETS (``type`` + keys; ``label`` is optional everywhere)
-----------------------------------------------------------
  stat     key, unit?                   a value from state[key]
  badge    key, map?                    value -> "ok" | "warn" | "bad" | "info"
  gauge    key, min?=0, max?=100, unit?
  text     key                          multi-line text
  events   key, max?=20                 list of str or {"ts", "text"}
  image    stream, mode?="stream"|"snapshot", refresh_ms?=1000, key?
                                        shows "No picture yet" and requests
                                        nothing until a frame exists: state[key]
                                        truthy (a new value = a new frame; falsy
                                        = none), a user action on the panel, or
                                        one look when the view opens (keyless);
                                        a 404 goes back to the placeholder.
                                        With a frame, snapshot mode refreshes
                                        every refresh_ms
  buttons  buttons=[{"label", "action", "args"?}]
  input    action, arg?="text", placeholder?, button?="Send"
  toggle   key, action, arg?="on"       sends {arg: true|false}
  slider   key, action, arg?="value", min?=0, max?=100, step?=1
  hold     action, args?                resends at the action's rate_hz while
                                        pressed; sends stop_action on release
  estop    action                       pinned on every tab; never throttled

Every ``action`` a widget names must be declared in ``actions``; every
``stream`` in ``streams``. A spec that breaks any rule is REJECTED whole, with
one log line naming the rule, and the rest of the skill loads normally.

VOICE-ONLY ACTIONS (2026-10-01)
===============================
A skill may also define a module-level ``VOICE_ONLY_ACTIONS``: a dict of voice
action name -> a short hint naming what to use instead (e.g. the panel button
that does the same job), or a list of names. These are actions that act only
on a FRESH SPOKEN (or typed) request - they read the owner's last utterance as
proof he asked - so running them from the dashboard's generic Actions list can
never work: they refuse ("binding" / "I need to hear you say ...") and, worse,
mark that utterance as used. The dashboard lists them as voice-only and the
POST refuses them before the handler is called. Collected beside WEB_PANELS
(a reload replaces the skill's entries); a private skill's action names stay
in its gitignored file.

Stdlib only. Nothing here raises into a caller: registration returns False on
a bad spec, and the request-path helpers return an HTTP status + payload.
"""
from __future__ import annotations

import json
import re
import threading
import time

__all__ = ["PanelRegistry", "PanelSpecError", "REGISTRY", "WIDGET_TYPES",
           "json_safe"]

WIDGET_TYPES = frozenset({"stat", "badge", "gauge", "text", "events", "image",
                          "buttons", "input", "toggle", "slider", "hold",
                          "estop"})
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,39}$")
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,39}$")
_BADGE_TONES = frozenset({"ok", "warn", "bad", "info"})
_MAX_WIDGETS = 60
_MAX_ARGS_BYTES = 16 * 1024
_MAX_STATE_BYTES = 512 * 1024
_DEFAULT_RATE_HZ = 4.0
_MAX_RATE_HZ = 30.0


class PanelSpecError(ValueError):
    """A WEB_PANELS spec broke a rule; the message names which one."""


def json_safe(value, limit: int = _MAX_STATE_BYTES):
    """``value`` if it is JSON-serialisable, else a str-coerced copy; a value
    whose encoding exceeds ``limit`` bytes becomes an error dict. Never
    raises."""
    try:
        text = json.dumps(value)
    except Exception:
        try:
            text = json.dumps(value, default=str)
            value = json.loads(text)
        except Exception as e:
            return {"error": "not JSON-serialisable: %s" % type(e).__name__}
    if len(text) > limit:
        return {"error": "too large (%d bytes)" % len(text)}
    return value


def _num(v, name, lo=None, hi=None):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise PanelSpecError("%s must be a number" % name)
    if (lo is not None and v < lo) or (hi is not None and v > hi):
        raise PanelSpecError("%s=%r outside %s..%s" % (name, v, lo, hi))
    return v


class _Action:
    __slots__ = ("name", "fn", "label", "confirm", "hold", "rate_hz",
                 "danger", "exempt")

    def meta(self) -> dict:
        return {"label": self.label, "confirm": self.confirm,
                "hold": self.hold, "rate_hz": self.rate_hz,
                "danger": self.danger, "stop": self.exempt}


class _Panel:
    def __init__(self, owner: str):
        self.owner = owner
        self.lock = threading.Lock()
        self.cache = None
        self.cache_at = None          # clock() of the last GOOD state()
        self.error = None             # last state() failure, or None
        self.inflight = False
        self.inflight_since = 0.0
        self.done = None              # threading.Event of the in-flight fetch
        self.last_call = {}           # action name -> clock()

    def meta(self) -> dict:
        return {"id": self.id, "title": self.title, "order": self.order,
                "poll_ms": self.poll_ms, "has_state": self.state is not None,
                "stop_action": self.stop_action,
                "layout": self.layout,
                "actions": {n: a.meta() for n, a in self.actions.items()},
                "streams": sorted(self.streams)}


def _action_from(name, value) -> _Action:
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise PanelSpecError("action name %r is not a plain identifier" % (name,))
    a = _Action()
    a.name = name
    a.exempt = False
    if callable(value):
        value = {"fn": value}
    if not isinstance(value, dict):
        raise PanelSpecError("action %r must be a callable or a dict" % name)
    unknown = set(value) - {"fn", "label", "confirm", "hold", "rate_hz",
                            "danger"}
    if unknown:
        raise PanelSpecError("action %r has unknown keys %s"
                             % (name, sorted(unknown)))
    if not callable(value.get("fn")):
        raise PanelSpecError("action %r has no callable fn" % name)
    a.fn = value["fn"]
    a.label = str(value.get("label") or name)[:60]
    a.confirm = bool(value.get("confirm", False))
    a.hold = bool(value.get("hold", False))
    a.danger = bool(value.get("danger", False))
    a.rate_hz = float(_num(value.get("rate_hz", _DEFAULT_RATE_HZ),
                           "action %r rate_hz" % name, 0.1, _MAX_RATE_HZ))
    return a


def _need(w, key, i):
    if key not in w or w[key] in (None, ""):
        raise PanelSpecError("widget #%d (%s) needs %r" % (i, w.get("type"), key))
    return w[key]


def _clean_widget(w, i, actions, streams) -> dict:
    if not isinstance(w, dict):
        raise PanelSpecError("widget #%d is not a dict" % i)
    t = w.get("type")
    if t not in WIDGET_TYPES:
        raise PanelSpecError("widget #%d has unknown type %r" % (i, t))
    out = {"type": t}
    if "label" in w:
        out["label"] = str(w["label"])[:80]
    for k in ("key",):
        if k in w:
            out[k] = str(w[k])[:60]
    if t in ("stat", "badge", "gauge", "text", "events", "toggle", "slider"):
        out["key"] = str(_need(w, "key", i))[:60]
    if t in ("stat", "gauge") and "unit" in w:
        out["unit"] = str(w["unit"])[:12]
    if t == "badge" and "map" in w:
        m = w["map"]
        if not isinstance(m, dict) or not all(
                str(v) in _BADGE_TONES for v in m.values()):
            raise PanelSpecError("widget #%d badge map values must be one of %s"
                                 % (i, sorted(_BADGE_TONES)))
        out["map"] = {str(k): str(v) for k, v in m.items()}
    if t in ("gauge", "slider"):
        lo = _num(w.get("min", 0), "widget #%d min" % i)
        hi = _num(w.get("max", 100), "widget #%d max" % i)
        if hi <= lo:
            raise PanelSpecError("widget #%d max must exceed min" % i)
        out["min"], out["max"] = lo, hi
    if t == "slider":
        out["step"] = _num(w.get("step", 1), "widget #%d step" % i, 0)
    if t == "events":
        out["max"] = int(_num(w.get("max", 20), "widget #%d max" % i, 1, 500))
    if t == "image":
        s = str(_need(w, "stream", i))
        if s not in streams:
            raise PanelSpecError("widget #%d names undeclared stream %r" % (i, s))
        out["stream"] = s
        mode = w.get("mode", "stream")
        if mode not in ("stream", "snapshot"):
            raise PanelSpecError("widget #%d mode must be stream|snapshot" % i)
        out["mode"] = mode
        out["refresh_ms"] = int(_num(w.get("refresh_ms", 1000),
                                     "widget #%d refresh_ms" % i, 100, 60000))
    if t == "buttons":
        btns = w.get("buttons")
        if not isinstance(btns, list) or not btns or len(btns) > 24:
            raise PanelSpecError("widget #%d buttons must be a list of 1..24" % i)
        clean = []
        for j, b in enumerate(btns):
            if not isinstance(b, dict) or b.get("action") not in actions:
                raise PanelSpecError("widget #%d button #%d names an undeclared "
                                     "action" % (i, j))
            args = b.get("args", {})
            if not isinstance(args, dict):
                raise PanelSpecError("widget #%d button #%d args must be a dict"
                                     % (i, j))
            clean.append({"label": str(b.get("label") or b["action"])[:40],
                          "action": b["action"], "args": json_safe(args)})
        out["buttons"] = clean
    if t in ("input", "toggle", "slider", "hold", "estop"):
        a = str(_need(w, "action", i))
        if a not in actions:
            raise PanelSpecError("widget #%d names undeclared action %r" % (i, a))
        out["action"] = a
    if t == "input":
        out["arg"] = str(w.get("arg", "text"))[:40]
        out["placeholder"] = str(w.get("placeholder", ""))[:80]
        out["button"] = str(w.get("button", "Send"))[:20]
    if t in ("toggle", "slider"):
        out["arg"] = str(w.get("arg", "on" if t == "toggle" else "value"))[:40]
    if t == "hold":
        if not actions[out["action"]].hold:
            raise PanelSpecError("widget #%d: action %r drives a hold button "
                                 "but is not declared hold=True"
                                 % (i, out["action"]))
        args = w.get("args", {})
        if not isinstance(args, dict):
            raise PanelSpecError("widget #%d args must be a dict" % i)
        out["args"] = json_safe(args)
    return out


class PanelRegistry:
    """Process-wide set of declared panels. Thread-safe; never raises into a
    caller of register_from_module / state / call_action."""

    def __init__(self, *, clock=time.monotonic, log=None):
        self._clock = clock
        self._log = log if log is not None else (
            lambda line: print(line, flush=True))
        self._lock = threading.RLock()
        self._panels = {}            # id -> _Panel
        self._voice_only = {}        # action name -> (owner, hint)

    # ── registration ────────────────────────────────────────────────────
    def _say(self, line: str) -> None:
        try:
            self._log(line)
        except Exception:
            pass

    def validate(self, spec, owner: str = "") -> _Panel:
        """Build a _Panel from ``spec`` or raise PanelSpecError."""
        if not isinstance(spec, dict):
            raise PanelSpecError("a panel spec must be a dict")
        unknown = set(spec) - {"id", "title", "order", "poll_ms", "state",
                               "state_timeout_s", "stop_action", "layout",
                               "actions", "streams"}
        if unknown:
            raise PanelSpecError("unknown keys %s" % sorted(unknown))
        pid = spec.get("id")
        if not isinstance(pid, str) or not _ID_RE.match(pid):
            raise PanelSpecError("id %r must match %s" % (pid, _ID_RE.pattern))
        title = spec.get("title")
        if not isinstance(title, str) or not title.strip():
            raise PanelSpecError("title must be a non-empty string")
        p = _Panel(owner)
        p.id = pid
        p.title = title.strip()[:60]
        p.order = int(_num(spec.get("order", 100), "order", -10000, 10000))
        p.poll_ms = int(_num(spec.get("poll_ms", 1000), "poll_ms", 250, 60000))
        p.state = spec.get("state")
        if p.state is not None and not callable(p.state):
            raise PanelSpecError("state must be callable")
        p.state_timeout_s = float(_num(spec.get("state_timeout_s", 1.0),
                                       "state_timeout_s", 0.05, 10.0))
        acts = spec.get("actions", {}) or {}
        if not isinstance(acts, dict) or len(acts) > 64:
            raise PanelSpecError("actions must be a dict of at most 64")
        p.actions = {n: _action_from(n, v) for n, v in acts.items()}
        streams = spec.get("streams", {}) or {}
        if not isinstance(streams, dict) or len(streams) > 8:
            raise PanelSpecError("streams must be a dict of at most 8")
        for n, fn in streams.items():
            if not isinstance(n, str) or not _NAME_RE.match(n) or not callable(fn):
                raise PanelSpecError("stream %r must be a named callable" % (n,))
        p.streams = dict(streams)
        stop = spec.get("stop_action")
        if stop is not None and stop not in p.actions:
            raise PanelSpecError("stop_action %r is not a declared action"
                                 % (stop,))
        p.stop_action = stop
        layout = spec.get("layout", []) or []
        if not isinstance(layout, list) or len(layout) > _MAX_WIDGETS:
            raise PanelSpecError("layout must be a list of at most %d widgets"
                                 % _MAX_WIDGETS)
        p.layout = [_clean_widget(w, i, p.actions, p.streams)
                    for i, w in enumerate(layout)]
        holds = [w for w in p.layout if w["type"] == "hold"]
        if holds and not stop:
            raise PanelSpecError("a hold widget needs the panel's stop_action")
        # Stops always get through: never throttled, never behind a confirm.
        exempt = {stop} if stop else set()
        exempt |= {w["action"] for w in p.layout if w["type"] == "estop"}
        for n in exempt:
            a = p.actions[n]
            if a.confirm:
                raise PanelSpecError("stop/estop action %r must not require "
                                     "confirmation" % n)
            a.exempt = True
        try:
            json.dumps(p.meta())     # the metadata the dashboard fetches
        except Exception as e:
            raise PanelSpecError("metadata is not JSON-serialisable: %s" % e)
        return p

    def register(self, spec, owner: str = "") -> bool:
        """Validate and add one panel. A bad spec is REJECTED with one log
        line; an id already owned by ANOTHER skill is rejected too (the same
        owner re-registering replaces its own panel)."""
        try:
            p = self.validate(spec, owner)
        except PanelSpecError as e:
            self._say("  [web-panels] %s: REJECTED a panel spec - %s"
                      % (owner or "?", e))
            return False
        except Exception as e:           # a buggy spec must not break loading
            self._say("  [web-panels] %s: REJECTED a panel spec - %s: %s"
                      % (owner or "?", type(e).__name__, e))
            return False
        with self._lock:
            cur = self._panels.get(p.id)
            if cur is not None and cur.owner != owner:
                self._say("  [web-panels] %s: REJECTED panel %r - id already "
                          "registered by another skill" % (owner or "?", p.id))
                return False
            self._panels[p.id] = p
        self._say("  [web-panels] %s: panel %r registered (%d widget(s), %d "
                  "action(s))" % (owner or "?", p.id, len(p.layout),
                                  len(p.actions)))
        return True

    def unregister_owner(self, owner: str) -> int:
        with self._lock:
            ids = [i for i, p in self._panels.items() if p.owner == owner]
            for i in ids:
                del self._panels[i]
            for n in [n for n, (o, _h) in self._voice_only.items()
                      if o == owner]:
                del self._voice_only[n]
        return len(ids)

    def _register_voice_only(self, mod, owner: str) -> int:
        """Collect ``VOICE_ONLY_ACTIONS`` (see the module docstring). A bad
        entry is skipped with one log line. Never raises."""
        try:
            spec = getattr(mod, "VOICE_ONLY_ACTIONS", None)
            if spec is None:
                return 0
            if isinstance(spec, dict):
                items = list(spec.items())
            elif isinstance(spec, (list, tuple, set, frozenset)):
                items = [(n, "") for n in spec]
            else:
                self._say("  [web-panels] %s: REJECTED VOICE_ONLY_ACTIONS - "
                          "must be a dict or a list of names" % owner)
                return 0
            added = 0
            with self._lock:
                for n, hint in items:
                    if not isinstance(n, str) or not n.strip():
                        continue
                    hint = hint.strip()[:80] if isinstance(hint, str) else ""
                    self._voice_only[n.strip()] = (owner, hint)
                    added += 1
            return added
        except Exception as e:
            self._say("  [web-panels] %s: VOICE_ONLY_ACTIONS could not be "
                      "applied: %s" % (owner, e))
            return 0

    def voice_only(self, name):
        """The hint for a voice-only action ("" when it gave none), or None
        when ``name`` is an ordinary action. Never raises."""
        try:
            with self._lock:
                hit = self._voice_only.get(name)
            return None if hit is None else hit[1]
        except Exception:
            return None

    def register_from_module(self, mod, owner: str) -> int:
        """Collect a just-loaded skill's ``WEB_PANELS`` (and its
        ``VOICE_ONLY_ACTIONS``). Replaces whatever the same skill registered
        before (a reload never duplicates). Returns the number of panels
        accepted. Never raises."""
        try:
            self.unregister_owner(owner)
            self._register_voice_only(mod, owner)
            specs = getattr(mod, "WEB_PANELS", None)
            if specs is None:
                return 0
            if not isinstance(specs, (list, tuple)):
                self._say("  [web-panels] %s: REJECTED WEB_PANELS - must be a "
                          "list of specs" % owner)
                return 0
            return sum(1 for s in specs if self.register(s, owner))
        except Exception as e:
            self._say("  [web-panels] %s: WEB_PANELS could not be applied: %s"
                      % (owner, e))
            return 0

    # ── read side ───────────────────────────────────────────────────────
    def get(self, panel_id):
        with self._lock:
            return self._panels.get(panel_id)

    def list_meta(self) -> list:
        with self._lock:
            panels = list(self._panels.values())
        return sorted((p.meta() for p in panels),
                      key=lambda m: (m["order"], m["title"].lower(), m["id"]))

    def ids(self) -> list:
        with self._lock:
            return sorted(self._panels)

    def state(self, panel_id):
        """``(http_code, payload)`` for GET /api/panel/<id>/state.

        state() runs on a helper thread, ONE at a time per panel, and the
        request waits at most ``state_timeout_s``. A slow or hung state()
        therefore never hangs the request: the last good value comes back
        with ``stale: true``, and the fetch keeps running in the background
        (its result lands in the cache when it finishes)."""
        p = self.get(panel_id)
        if p is None:
            return 404, {"error": "unknown panel"}
        if p.state is None:
            return 200, {"id": p.id, "state": {}, "stale": False,
                         "age_s": None, "error": None}
        now = self._clock()
        fresh_window = max(0.1, p.poll_ms / 2000.0)
        with p.lock:
            if (p.cache_at is not None and p.error is None
                    and now - p.cache_at < fresh_window):
                return 200, self._payload(p, now, stale=False)
            if not p.inflight:
                p.inflight = True
                p.inflight_since = now
                p.done = threading.Event()
                threading.Thread(target=self._fetch, args=(p, p.done),
                                 daemon=True,
                                 name="web-panel-state-" + p.id).start()
            done = p.done
        finished = done.wait(p.state_timeout_s)
        now = self._clock()
        with p.lock:
            if finished and p.error is None and p.cache_at is not None:
                return 200, self._payload(p, now, stale=False)
            payload = self._payload(p, now, stale=True)
            if not finished:
                payload["error"] = ("state() still running after %.1fs"
                                    % (now - p.inflight_since))
            return 200, payload

    def _fetch(self, p, done) -> None:
        try:
            value = json_safe(p.state())
            with p.lock:
                p.cache = value
                p.cache_at = self._clock()
                p.error = None
        except Exception as e:
            with p.lock:
                p.error = "%s: %s" % (type(e).__name__, e)
        finally:
            with p.lock:
                p.inflight = False
            done.set()

    @staticmethod
    def _payload(p, now, *, stale) -> dict:
        return {"id": p.id,
                "state": p.cache if p.cache is not None else {},
                "stale": bool(stale),
                "age_s": (None if p.cache_at is None
                          else round(now - p.cache_at, 3)),
                "error": p.error}

    # ── write side ──────────────────────────────────────────────────────
    def call_action(self, panel_id, name, args=None, confirm=False):
        """``(http_code, payload)`` for POST /api/panel/<id>/action. Calls
        the DECLARED callable directly - never through the LLM or the command
        channel. 404 unknown panel/action, 400 bad args, 409 confirm needed,
        429 too fast, 500 the action raised."""
        p = self.get(panel_id)
        if p is None:
            return 404, {"error": "unknown panel"}
        a = p.actions.get(name) if isinstance(name, str) else None
        if a is None:
            return 404, {"error": "unknown action"}
        if args is None:
            args = {}
        if not isinstance(args, dict):
            return 400, {"error": "args must be a JSON object"}
        try:
            if len(json.dumps(args)) > _MAX_ARGS_BYTES:
                return 400, {"error": "args too large"}
        except Exception:
            return 400, {"error": "args must be JSON"}
        if a.confirm and confirm is not True:
            return 409, {"error": "confirmation required",
                         "confirm_required": True, "label": a.label,
                         "danger": a.danger}
        now = self._clock()
        if not a.exempt:
            min_gap = 1.0 / (2.0 * a.rate_hz)
            with p.lock:
                last = p.last_call.get(name)
                if last is not None and (now - last) < min_gap:
                    return 429, {"error": "rate limited",
                                 "retry_after_s": round(min_gap - (now - last), 3)}
                p.last_call[name] = now
        try:
            result = a.fn(dict(args))
        except Exception as e:
            self._say("  [web-panels] %s.%s raised %s: %s"
                      % (p.id, name, type(e).__name__, e))
            return 500, {"ok": False, "error": "%s: %s" % (type(e).__name__, e)}
        return 200, {"ok": True, "result": json_safe(result)}

    def stream_source(self, panel_id, name):
        p = self.get(panel_id)
        if p is None:
            return None
        return p.streams.get(name)


# The one registry the running JARVIS process uses (the skill loader writes
# it, the in-process web server reads it). Tests build their own.
REGISTRY = PanelRegistry()
