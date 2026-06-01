# Sub-agent specs

This folder declares the catalogue of sub-agents available to the
orchestrator (`core/orchestrator.py`). The orchestrator's planner LLM
reads these specs to decide how to decompose a request into parallel
work.

Each spec is either a `.json` file or a `.py` file exporting a
module-level `SPEC` dict. Files whose name starts with `_` are skipped
(same convention as `skills/`).

## Shape

```json
{
  "name": "email_reader",
  "description": "One-sentence summary of what this sub-agent can do.",
  "allowed_actions": ["action_a", "action_b"],
  "model_preference": "haiku",
  "system_prompt": "Extra instructions appended to the worker system prompt."
}
```

- `model_preference` is one of `"haiku"`, `"local"`, `"auto"`. The
  orchestrator falls back to the worker default when a preference can't
  be honoured (e.g. no local model configured).
- `allowed_actions` is the subset of the live `ACTIONS` dict this
  sub-agent is permitted to call. Actions not listed are invisible to
  the worker.
- The orchestrator is OFF by default. Flip `ENABLE_ORCHESTRATOR = True`
  in `bobert_companion.py` to wire it in.

## Adding a new sub-agent

1. Drop a `.json` (or `.py` with `SPEC = {...}`) into this folder.
2. Either restart JARVIS or call `Orchestrator.reload_specs()`.
3. Mention something only the new sub-agent would handle and watch the
   planner route to it.
