"""Sub-agent orchestrator — decompose complex requests into parallel sub-tasks.

A *planner* Claude call breaks a high-level user request ("summarise my
morning") into N atomic sub-tasks, each dispatched in parallel to a cheaper
worker (Haiku or local Ollama) with a restricted tool subset. A *merger*
Claude call synthesises the worker outputs into a single coherent reply
for TTS.

Why this exists
---------------
Long-form requests like "summarise my morning" require multiple
independent reads (email, calendar, news, weather, build status). Today
JARVIS does these sequentially through one expensive Opus/Sonnet turn.
Running them in parallel through Haiku-class workers cuts wall-clock
latency dramatically AND cuts cost — Opus only synthesises.

Design
------
A *sub-agent spec* is a small JSON or dict declaring:
    {
        "name":          "email_reader",
        "description":   "Reads recent unread email and summarises it.",
        "allowed_actions": ["read_email", "ms_graph_search"],
        "model_preference": "haiku",   # haiku | local | auto
        "system_prompt": "You are an email triage assistant. ..."
    }

Specs live in `C:/JARVIS/skills/sub_agents/*.json` (and *.py specs whose
`SPEC` module-level dict matches the shape above). They're loaded at
construction time; reload by calling `Orchestrator.reload_specs()`.

Top-level entrypoint:
    orchestrate(request, actions) -> str

That triggers the full pipeline:
    1. plan_decomposition(request, specs)        -> list of sub_tasks
    2. dispatch_sub_agents(sub_tasks, actions)   -> parallel worker results
    3. merge_results(request, results)           -> final TTS-ready string

Each stage has graceful fallbacks: planner failure returns a single
sub-task targeting the most permissive sub-agent; worker failure logs
the exception and returns an empty string for that slot; merger failure
returns the concatenation of worker outputs.

The orchestrator is OFF by default — flip `ENABLE_ORCHESTRATOR = True`
in bobert_companion.py to wire it in.
"""
from __future__ import annotations

import asyncio
import glob
import importlib.util
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence


# ──────────────────────────────────────────────────────────────────────────
#  CONFIG (defaults — overridable via bobert_companion module attrs)
# ──────────────────────────────────────────────────────────────────────────

_PROJECT_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUB_AGENTS_DIR = os.path.join(_PROJECT_DIR, "skills", "sub_agents")

DEFAULT_PLANNER_MODEL = "claude-sonnet-4-6"
DEFAULT_WORKER_MODEL  = "claude-haiku-4-5"
DEFAULT_MERGER_MODEL  = "claude-sonnet-4-6"

DEFAULT_MAX_PARALLEL  = 4
DEFAULT_WORKER_TIMEOUT_S = 30.0
DEFAULT_PLANNER_TIMEOUT_S = 20.0
DEFAULT_MERGER_TIMEOUT_S  = 20.0

_log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
#  TYPES
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class SubAgentSpec:
    """Declarative description of one sub-agent."""
    name: str
    description: str
    allowed_actions: list[str] = field(default_factory=list)
    model_preference: str = "haiku"          # haiku | local | auto
    system_prompt: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "SubAgentSpec":
        return cls(
            name=str(d.get("name", "")).strip(),
            description=str(d.get("description", "")).strip(),
            allowed_actions=list(d.get("allowed_actions", []) or []),
            model_preference=str(d.get("model_preference", "haiku")).strip() or "haiku",
            system_prompt=str(d.get("system_prompt", "")).strip(),
        )

    def is_valid(self) -> bool:
        return bool(self.name) and bool(self.description)


@dataclass
class SubTask:
    """One unit of work the planner emitted for a specific sub-agent."""
    sub_agent: str          # name of the SubAgentSpec to use
    task: str               # natural-language description of the work
    args: dict = field(default_factory=dict)


@dataclass
class SubTaskResult:
    """Output of one worker dispatch."""
    sub_agent: str
    task: str
    output: str
    duration_s: float
    error: str | None = None
    model_used: str = ""


# ──────────────────────────────────────────────────────────────────────────
#  SPEC LOADING
# ──────────────────────────────────────────────────────────────────────────

def _load_json_spec(path: str) -> SubAgentSpec | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        _log.warning("orchestrator: failed to read %s: %s", path, e)
        return None
    if not isinstance(data, dict):
        _log.warning("orchestrator: %s is not a JSON object", path)
        return None
    spec = SubAgentSpec.from_dict(data)
    if not spec.is_valid():
        _log.warning("orchestrator: %s missing name or description", path)
        return None
    return spec


def _load_py_spec(path: str) -> SubAgentSpec | None:
    """A .py spec exports a module-level `SPEC` dict matching the JSON shape."""
    name = os.path.splitext(os.path.basename(path))[0]
    try:
        spec_obj = importlib.util.spec_from_file_location(f"sub_agent_spec_{name}", path)
        if not spec_obj or not spec_obj.loader:
            return None
        mod = importlib.util.module_from_spec(spec_obj)
        spec_obj.loader.exec_module(mod)
    except Exception as e:
        _log.warning("orchestrator: failed to import %s: %s", path, e)
        return None
    raw = getattr(mod, "SPEC", None)
    if not isinstance(raw, dict):
        return None
    spec = SubAgentSpec.from_dict(raw)
    if not spec.is_valid():
        return None
    return spec


def load_specs(directory: str = SUB_AGENTS_DIR) -> dict[str, SubAgentSpec]:
    """Load every .json / .py spec under `directory`. Files whose name
    starts with `_` are skipped (mirrors the convention skills/ uses)."""
    specs: dict[str, SubAgentSpec] = {}
    if not os.path.isdir(directory):
        return specs
    for path in sorted(glob.glob(os.path.join(directory, "*"))):
        name = os.path.basename(path)
        if name.startswith("_"):
            continue
        if name.endswith(".json"):
            s = _load_json_spec(path)
        elif name.endswith(".py"):
            s = _load_py_spec(path)
        else:
            continue
        if s is not None:
            specs[s.name] = s
    return specs


# ──────────────────────────────────────────────────────────────────────────
#  LLM CALL HELPERS
# ──────────────────────────────────────────────────────────────────────────

def _claude_call(
    model: str,
    system: str,
    user: str,
    max_tokens: int = 600,
    timeout_s: float | None = None,
) -> str:
    """One-shot Claude messages.create. Returns assistant text or raises."""
    import anthropic
    client = anthropic.Anthropic()
    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    if timeout_s is not None:
        # The SDK accepts a per-request `timeout`. Older versions ignore it
        # silently rather than raise.
        kwargs["timeout"] = timeout_s
    msg = client.messages.create(**kwargs)
    parts = []
    for block in getattr(msg, "content", []) or []:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts).strip()


def _ollama_call(
    model: str,
    system: str,
    user: str,
    base_url: str = "http://localhost:11434",
    timeout_s: float = 30.0,
) -> str:
    """One-shot Ollama /api/chat. Returns assistant text or raises."""
    import urllib.request
    payload = json.dumps({
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    }).encode("utf-8")
    req = urllib.request.Request(
        url=f"{base_url.rstrip('/')}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    data = json.loads(body)
    return str(data.get("message", {}).get("content", "")).strip()


def _ollama_reachable(base_url: str = "http://localhost:11434", timeout_s: float = 2.0) -> bool:
    """True if the local Ollama daemon answers /api/tags. Used to gate the
    Claude→Ollama fallback so we only attempt it when a local model could
    actually serve the request."""
    import urllib.request
    try:
        req = urllib.request.Request(
            url=f"{base_url.rstrip('/')}/api/tags", method="GET",
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return 200 <= getattr(resp, "status", 200) < 300
    except Exception:
        return False


def _resolve_local_model(configured: str | None) -> str | None:
    """Pick the Ollama tag the fallback should target.

    Prefer an explicitly configured tag; otherwise ask the rest of JARVIS
    (bobert_companion._get_local_llm_model) for a real installed tag so the
    fallback matches what the main assistant uses. Returns None if neither
    yields a usable name — callers treat that as 'no local fallback'."""
    if configured:
        return configured
    try:
        import importlib
        import sys
        bc = sys.modules.get("bobert_companion") or importlib.import_module("bobert_companion")
        resolver = getattr(bc, "_get_local_llm_model", None)
        if callable(resolver):
            name = str(resolver() or "").strip()
            return name or None
    except Exception as e:
        _log.debug("orchestrator: local-model resolve failed: %s", e)
    return None


# ──────────────────────────────────────────────────────────────────────────
#  PLANNER
# ──────────────────────────────────────────────────────────────────────────

_PLANNER_SYSTEM = (
    "You are a task-decomposition planner for the JARVIS PC assistant. "
    "Given a user request and a catalogue of available sub-agents, you "
    "split the request into independent sub-tasks that can run IN PARALLEL.\n\n"
    "Output STRICT JSON with this shape:\n"
    '  {"sub_tasks": [{"sub_agent": "<spec-name>", "task": "<natural-language instruction>"}, ...]}\n\n'
    "Rules:\n"
    "  • Pick `sub_agent` ONLY from the catalogue.\n"
    "  • Emit no more than 6 sub_tasks.\n"
    "  • If the request is simple and a single sub-agent can answer it, "
    "    emit ONE sub_task — that's fine.\n"
    "  • If NOTHING in the catalogue is relevant, return {\"sub_tasks\": []}.\n"
    "  • Do not include sub-tasks that depend on each other; everything "
    "    you emit will be dispatched concurrently.\n"
    "  • No prose, no markdown — JSON only."
)


def _format_catalogue(specs: dict[str, SubAgentSpec]) -> str:
    lines = []
    for name, s in specs.items():
        lines.append(f"- {name}: {s.description}")
    return "\n".join(lines) if lines else "(catalogue empty)"


def _extract_json_object(text: str) -> dict | None:
    """Pull the first balanced `{...}` substring out of `text` and parse."""
    if not text:
        return None
    text = text.strip()
    # Strip ```json fences if the model wrapped its reply.
    if text.startswith("```"):
        text = text.lstrip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip("` \n")
    start = text.find("{")
    if start < 0:
        return None
    # Walk respecting JSON "..." quotes (with \" escapes) so braces inside
    # string values — e.g. {"task":"set timer {hh}:{mm}"} — don't fool the
    # depth counter into returning a truncated object.
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                blob = text[start:i + 1]
                try:
                    return json.loads(blob)
                except Exception:
                    return None
    return None


def plan_decomposition(
    request: str,
    specs: dict[str, SubAgentSpec],
    planner_model: str = DEFAULT_PLANNER_MODEL,
    timeout_s: float = DEFAULT_PLANNER_TIMEOUT_S,
    local_model: str | None = None,
    local_base_url: str = "http://localhost:11434",
) -> list[SubTask]:
    """Ask the planner LLM to decompose `request` into parallel sub_tasks.

    Returns an empty list if no sub-agent is relevant. If Claude is
    unavailable (e.g. API capped), falls back to a local Ollama model when
    one is reachable; only if that also fails does it degrade to a single
    full-request sub_task targeting the first registered sub-agent — better
    degraded than dark.
    """
    if not specs:
        return []

    catalogue = _format_catalogue(specs)
    user = (
        f"User request:\n  {request}\n\n"
        f"Available sub-agents:\n{catalogue}\n\n"
        "Return the JSON plan now."
    )
    raw: str | None = None
    try:
        raw = _claude_call(
            planner_model,
            _PLANNER_SYSTEM,
            user,
            max_tokens=600,
            timeout_s=timeout_s,
        )
    except Exception as e:
        _log.warning("orchestrator: planner call failed: %s", e)
        # Claude is down/capped — try a local Ollama model before degrading.
        ollama_model = _resolve_local_model(local_model)
        if ollama_model and _ollama_reachable(local_base_url):
            try:
                raw = _ollama_call(
                    ollama_model,
                    _PLANNER_SYSTEM,
                    user,
                    base_url=local_base_url,
                    timeout_s=timeout_s,
                )
                _log.info("orchestrator: planner fell back to ollama %s", ollama_model)
            except Exception as e2:
                _log.warning("orchestrator: planner ollama fallback failed: %s", e2)
                raw = None
        if raw is None:
            # Last resort: hand the whole request to the first valid spec.
            first = next(iter(specs.values()), None)
            return [SubTask(sub_agent=first.name, task=request)] if first else []

    blob = _extract_json_object(raw)
    if not blob or not isinstance(blob.get("sub_tasks"), list):
        _log.warning("orchestrator: planner returned unparseable JSON: %r", raw[:200])
        return []

    out: list[SubTask] = []
    for item in blob["sub_tasks"]:
        if not isinstance(item, dict):
            continue
        name = str(item.get("sub_agent", "")).strip()
        task = str(item.get("task", "")).strip()
        if not name or not task:
            continue
        if name not in specs:
            _log.info("orchestrator: planner picked unknown sub_agent %r — skipping", name)
            continue
        args = item.get("args") if isinstance(item.get("args"), dict) else {}
        out.append(SubTask(sub_agent=name, task=task, args=args or {}))
    return out


# ──────────────────────────────────────────────────────────────────────────
#  WORKER
# ──────────────────────────────────────────────────────────────────────────

_WORKER_SYSTEM_BASE = (
    "You are a sub-agent inside the JARVIS PC assistant. You have access "
    "to a restricted set of tools, listed below. Use them as needed to "
    "complete the task. Reply with a CONCISE plain-text result the "
    "merger can fold into a final answer. No prose decoration.\n\n"
    "Available tools (action name → call by stating ACTION:name(args)):\n"
    "{tool_list}\n\n"
    "If the task asks for something you cannot do with the available "
    "tools, reply with a one-line apology starting 'unavailable: '."
)


def _build_worker_system(spec: SubAgentSpec, actions: Iterable[str]) -> str:
    allowed = [a for a in spec.allowed_actions if a in set(actions)]
    tool_list = "\n".join(f"  • {a}" for a in allowed) if allowed else "  (none)"
    base = _WORKER_SYSTEM_BASE.format(tool_list=tool_list)
    if spec.system_prompt:
        return base + "\n\n" + spec.system_prompt
    return base


def _resolve_worker_model(
    spec: SubAgentSpec,
    worker_model: str,
    local_model: str | None,
) -> tuple[str, str]:
    """Return (backend, model_id). backend is 'claude' or 'ollama'."""
    pref = spec.model_preference.lower()
    if pref == "local" and local_model:
        return "ollama", local_model
    if pref == "auto":
        return "claude", worker_model
    return "claude", worker_model


def _run_worker_sync(
    spec: SubAgentSpec,
    task: SubTask,
    actions: dict[str, Callable[[str], str]],
    worker_model: str,
    local_model: str | None,
    local_base_url: str,
    timeout_s: float,
) -> SubTaskResult:
    """Synchronous worker execution. Wrapped in asyncio.to_thread by the
    parallel dispatcher."""
    started = time.time()
    backend, model_id = _resolve_worker_model(spec, worker_model, local_model)
    system = _build_worker_system(spec, actions.keys())

    # Execute deterministic tool calls FIRST if the task explicitly names
    # one in args — this lets specs that don't need an LLM (pure adapters)
    # skip the model call entirely.
    direct = task.args.get("direct_action") if task.args else None
    if isinstance(direct, str) and direct in actions and direct in spec.allowed_actions:
        try:
            output = actions[direct](str(task.args.get("arg", "")))
            return SubTaskResult(
                sub_agent=spec.name,
                task=task.task,
                output=output if isinstance(output, str) else str(output),
                duration_s=time.time() - started,
                model_used="direct",
            )
        except Exception as e:
            return SubTaskResult(
                sub_agent=spec.name,
                task=task.task,
                output="",
                duration_s=time.time() - started,
                error=f"direct action {direct} failed: {e}",
                model_used="direct",
            )

    # No explicitly-planned direct_action: fetch REAL data by running the
    # sub-agent's first registered read action, then let the worker LLM shape
    # that data per the spec's system prompt. Feeding real tool output (instead
    # of a data-less LLM call) is what stops a worker from FABRICATING — the LLM
    # only summarizes what we actually fetched. If none of the spec's
    # allowed_actions are registered, or the tool yields nothing, return empty
    # so the merger silently omits this sub-agent rather than inventing facts.
    auto = next((a for a in spec.allowed_actions if a in actions), None)
    real_data = ""
    if auto is not None:
        try:
            arg = str(task.args.get("arg", "")) if task.args else ""
            _o = actions[auto](arg)
            real_data = _o if isinstance(_o, str) else str(_o)
        except Exception as e:
            _log.info("orchestrator: worker %s tool %s failed: %s",
                      spec.name, auto, e)
    if not real_data.strip():
        return SubTaskResult(
            sub_agent=spec.name,
            task=task.task,
            output="",
            duration_s=time.time() - started,
            error="no registered tool data",
            model_used=auto or "none",
        )

    worker_user = (
        f"Tool `{auto}` returned this REAL output:\n{real_data}\n\n"
        f"Task: {task.task}\n\n"
        "Summarize ONLY the output above for the merger. Invent nothing — if "
        "the output is empty or unhelpful, say so in one line."
    )
    try:
        if backend == "ollama":
            output = _ollama_call(
                model_id, system, worker_user,
                base_url=local_base_url,
                timeout_s=timeout_s,
            )
        else:
            try:
                output = _claude_call(
                    model_id, system, worker_user,
                    max_tokens=600,
                    timeout_s=timeout_s,
                )
            except Exception as claude_err:
                # Claude down/capped — retry once on a local Ollama model if
                # reachable, else fall back to the RAW real data (still real,
                # never fabricated) rather than failing the whole sub-task.
                ollama_model = _resolve_local_model(local_model)
                if ollama_model and _ollama_reachable(local_base_url):
                    output = _ollama_call(
                        ollama_model, system, worker_user,
                        base_url=local_base_url,
                        timeout_s=timeout_s,
                    )
                    model_id = ollama_model
                    _log.info(
                        "orchestrator: worker %s fell back to ollama %s",
                        spec.name, ollama_model,
                    )
                else:
                    _log.info("orchestrator: worker %s LLM unavailable (%s) — "
                              "returning raw tool data", spec.name, claude_err)
                    output = real_data
    except Exception as e:
        _log.info("orchestrator: worker %s shaping failed (%s) — raw tool data",
                  spec.name, e)
        output = real_data

    return SubTaskResult(
        sub_agent=spec.name,
        task=task.task,
        output=output,
        duration_s=time.time() - started,
        model_used=model_id,
    )


async def dispatch_sub_agents(
    sub_tasks: Sequence[SubTask],
    specs: dict[str, SubAgentSpec],
    actions: dict[str, Callable[[str], str]],
    worker_model: str = DEFAULT_WORKER_MODEL,
    local_model: str | None = None,
    local_base_url: str = "http://localhost:11434",
    max_parallel: int = DEFAULT_MAX_PARALLEL,
    timeout_s: float = DEFAULT_WORKER_TIMEOUT_S,
) -> list[SubTaskResult]:
    """Run every sub_task concurrently, capped at `max_parallel` in flight."""
    if not sub_tasks:
        return []

    sem = asyncio.Semaphore(max(1, max_parallel))

    async def _bounded(task: SubTask) -> SubTaskResult:
        spec = specs.get(task.sub_agent)
        if spec is None:
            return SubTaskResult(
                sub_agent=task.sub_agent,
                task=task.task,
                output="",
                duration_s=0.0,
                error="unknown sub_agent",
            )
        async with sem:
            return await asyncio.to_thread(
                _run_worker_sync,
                spec,
                task,
                actions,
                worker_model,
                local_model,
                local_base_url,
                timeout_s,
            )

    return await asyncio.gather(*[_bounded(t) for t in sub_tasks])


# ──────────────────────────────────────────────────────────────────────────
#  MERGER
# ──────────────────────────────────────────────────────────────────────────

_MERGER_SYSTEM = (
    "You are J.A.R.V.I.S. condensing several sub-agent reports into one "
    "concise, in-character spoken reply for the user. Stay dry, British, "
    "and brief. Three to five short sentences max. If a sub-agent failed "
    "or returned nothing, omit it silently — never apologise for it."
)


def merge_results(
    request: str,
    results: Sequence[SubTaskResult],
    merger_model: str = DEFAULT_MERGER_MODEL,
    timeout_s: float = DEFAULT_MERGER_TIMEOUT_S,
    local_model: str | None = None,
    local_base_url: str = "http://localhost:11434",
) -> str:
    """Synthesise sub-agent outputs into a single TTS-ready reply.

    If Claude is unavailable (e.g. API capped), tries a local Ollama model
    when one is reachable; only if that also fails does it degrade to a
    deterministic concatenation of the worker outputs.
    """
    usable = [r for r in results if r.output and not r.error]
    if not usable:
        return ""

    sections = []
    for r in usable:
        sections.append(f"[{r.sub_agent}]\n{r.output.strip()}")
    bundle = "\n\n".join(sections)

    user = (
        f"Original request: {request}\n\n"
        f"Sub-agent reports:\n{bundle}\n\n"
        "Compose JARVIS's reply now."
    )
    try:
        return _claude_call(
            merger_model,
            _MERGER_SYSTEM,
            user,
            max_tokens=500,
            timeout_s=timeout_s,
        )
    except Exception as e:
        _log.warning("orchestrator: merger call failed: %s", e)
        # Claude is down/capped — try a local Ollama model before degrading.
        ollama_model = _resolve_local_model(local_model)
        if ollama_model and _ollama_reachable(local_base_url):
            try:
                merged = _ollama_call(
                    ollama_model,
                    _MERGER_SYSTEM,
                    user,
                    base_url=local_base_url,
                    timeout_s=timeout_s,
                )
                if merged:
                    _log.info("orchestrator: merger fell back to ollama %s", ollama_model)
                    return merged
            except Exception as e2:
                _log.warning("orchestrator: merger ollama fallback failed: %s", e2)
        # Deterministic fallback — concatenate cleaned outputs.
        return " ".join(r.output.strip() for r in usable)


# ──────────────────────────────────────────────────────────────────────────
#  PUBLIC ORCHESTRATOR
# ──────────────────────────────────────────────────────────────────────────

class Orchestrator:
    """Coordinates planner → parallel workers → merger.

    Construct once at startup, then call `orchestrate(request, actions)`
    from anywhere. Specs are loaded from `skills/sub_agents/` and can be
    reloaded at runtime via `reload_specs()`.
    """

    def __init__(
        self,
        planner_model: str = DEFAULT_PLANNER_MODEL,
        worker_model: str = DEFAULT_WORKER_MODEL,
        merger_model: str = DEFAULT_MERGER_MODEL,
        local_model: str | None = None,
        local_base_url: str = "http://localhost:11434",
        max_parallel: int = DEFAULT_MAX_PARALLEL,
        worker_timeout_s: float = DEFAULT_WORKER_TIMEOUT_S,
        planner_timeout_s: float = DEFAULT_PLANNER_TIMEOUT_S,
        merger_timeout_s: float = DEFAULT_MERGER_TIMEOUT_S,
        specs_dir: str = SUB_AGENTS_DIR,
    ) -> None:
        self.planner_model = planner_model
        self.worker_model = worker_model
        self.merger_model = merger_model
        self.local_model = local_model
        self.local_base_url = local_base_url
        self.max_parallel = max_parallel
        self.worker_timeout_s = worker_timeout_s
        self.planner_timeout_s = planner_timeout_s
        self.merger_timeout_s = merger_timeout_s
        self.specs_dir = specs_dir
        self.specs: dict[str, SubAgentSpec] = load_specs(specs_dir)

    def reload_specs(self) -> int:
        self.specs = load_specs(self.specs_dir)
        return len(self.specs)

    def list_specs(self) -> list[str]:
        return sorted(self.specs.keys())

    def plan(self, request: str) -> list[SubTask]:
        return plan_decomposition(
            request,
            self.specs,
            planner_model=self.planner_model,
            timeout_s=self.planner_timeout_s,
            local_model=self.local_model,
            local_base_url=self.local_base_url,
        )

    async def dispatch(
        self,
        sub_tasks: Sequence[SubTask],
        actions: dict[str, Callable[[str], str]],
    ) -> list[SubTaskResult]:
        return await dispatch_sub_agents(
            sub_tasks,
            self.specs,
            actions,
            worker_model=self.worker_model,
            local_model=self.local_model,
            local_base_url=self.local_base_url,
            max_parallel=self.max_parallel,
            timeout_s=self.worker_timeout_s,
        )

    def merge(self, request: str, results: Sequence[SubTaskResult]) -> str:
        return merge_results(
            request,
            results,
            merger_model=self.merger_model,
            timeout_s=self.merger_timeout_s,
            local_model=self.local_model,
            local_base_url=self.local_base_url,
        )

    async def orchestrate_async(
        self,
        request: str,
        actions: dict[str, Callable[[str], str]],
    ) -> str:
        """Full pipeline. Returns empty string if nothing applicable ran."""
        sub_tasks = self.plan(request)
        if not sub_tasks:
            return ""
        results = await self.dispatch(sub_tasks, actions)
        return self.merge(request, results)

    def orchestrate(
        self,
        request: str,
        actions: dict[str, Callable[[str], str]],
    ) -> str:
        """Sync wrapper around `orchestrate_async`. Safe to call from the
        main turn-based loop, which is itself synchronous."""
        # If a loop is *currently running* in this thread, we can't drive
        # another one — hand off to a worker thread with its own fresh loop.
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is not None:
            import concurrent.futures

            def _runner() -> str:
                loop = asyncio.new_event_loop()
                try:
                    asyncio.set_event_loop(loop)
                    return loop.run_until_complete(
                        self.orchestrate_async(request, actions)
                    )
                finally:
                    loop.close()

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                return ex.submit(_runner).result()

        # No running loop in this thread. Always build a fresh one — never
        # touch asyncio.get_event_loop() (deprecated in 3.12+) or asyncio.run
        # (refuses when another loop has been installed via set_event_loop).
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(
                self.orchestrate_async(request, actions)
            )
        finally:
            loop.close()


# ──────────────────────────────────────────────────────────────────────────
#  MODULE-LEVEL CONVENIENCE
# ──────────────────────────────────────────────────────────────────────────

_default_orchestrator: Orchestrator | None = None


def get_orchestrator(**kwargs) -> Orchestrator:
    """Lazy singleton. Pass kwargs once to override defaults; later calls
    return the same instance regardless of kwargs."""
    global _default_orchestrator
    if _default_orchestrator is None:
        _default_orchestrator = Orchestrator(**kwargs)
    return _default_orchestrator


def orchestrate(
    request: str,
    actions: dict[str, Callable[[str], str]],
    **kwargs,
) -> str:
    """One-line entrypoint that uses the lazy singleton orchestrator."""
    return get_orchestrator(**kwargs).orchestrate(request, actions)
