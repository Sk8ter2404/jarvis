# Contributing to JARVIS

Thanks for trying JARVIS! Bug reports, skill contributions, and portability
fixes are all welcome. It's a beta — rough edges are expected.

## Reporting a bug

Open an issue with:

- What you did (the utterance or action) and what you expected.
- What happened — paste the relevant boot/console log lines (they're verbose by
  design). **Scrub anything personal** before pasting.
- Your environment: OS, `python --version`, GPU (or none), and which optional
  integrations are enabled.
- If it's a crash, the traceback.

## Running the tests

JARVIS uses the standard-library `unittest` (no pytest) so nothing extra is
needed beyond the deps.

```powershell
python tools/run_tests.py             # the full unit suite (~2,100 tests, ~30s)
python tools/run_tests.py <name>      # one file, e.g. `timer` or `skills.test_sh_hue`
python tools/run_tests.py -v          # verbose
python tools/run_coverage.py          # coverage over core/ + skills/ + tools/
python tools/audit_codebase.py        # static auditor — must report 0 findings
python -m pyflakes tests              # lint the test code
python tools/check_no_pii.py          # leak gate — no owner PII / secrets tracked
```

The unit suite loads every skill **in isolation** (no monolith boot) with a fake
`skill_utils` and all I/O mocked, so it needs no hardware, network, or API keys.
On a machine missing a heavy dependency (torch, opencv…), the affected skill
tests **skip** rather than fail. CI (`.github/workflows/ci.yml`) runs exactly
these gates.

For an end-to-end check of the real pipeline (boots a muted staging instance and
asserts non-fabricated replies):

```powershell
python tools/staging_integration.py -v   # needs ANTHROPIC_API_KEY
```

## Adding a skill

A skill is a single file in `skills/<name>.py` that defines a `register()`
function. It's auto-loaded at boot. See [`skills/_example_skill.py`](skills/_example_skill.py)
for the canonical template. The contract:

```python
def my_action(arg: str) -> str:
    # Do something; return a short string JARVIS will speak.
    skill_utils["open_url"]("https://example.com")   # injected helpers
    return "done"

def register(actions: dict):
    actions["my_action_name"] = my_action            # add your action(s)
```

- Handlers take one string arg and **return a string**.
- Use the injected `skill_utils` dict for PC control (click, type, open_url,
  screenshot, …). It's mocked in tests.
- **Degrade gracefully** — if an optional dependency, credential, or device is
  missing, return an informative string; never raise.
- Do all heavy/optional imports **lazily** (inside functions), so the module
  imports cleanly without the dependency installed.

Then add `tests/skills/test_<name>.py` — copy the style of an existing one (e.g.
`tests/skills/test_timer.py`). Use `tests/_skill_harness.load_skill_isolated`
and mock external I/O; assert on real behaviour, not just types.

## Before you open a PR

Run the gates and keep them green:

```powershell
python tools/run_tests.py && python tools/audit_codebase.py && python -m pyflakes tests && python tools/check_no_pii.py
```

- Don't commit secrets or personal data — `check_no_pii.py` is the gate.
- Don't hardcode personal paths, names, or LAN IPs; read them from config/env.
