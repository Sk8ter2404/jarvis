"""Anti-regression guard: every SHIPPED action must be a token the model can
actually emit — and every token the prompt teaches must resolve to a handler.

THE DEFECT CLASS THIS EXISTS TO STOP
  The local brain answers by emitting an ``[ACTION: name]`` token. It only ever
  learns names from the prompt. When a capability is *missing* it does not say
  "I can't" — it emits the nearest wrong-but-plausible token, and the owner
  hears a confident answer to a different question.

  PROVEN 2026-09-04: the owner asked "what microphone are you using right now"
  and got ``[ACTION: system_pulse]`` — a CPU/RAM/GPU read-out.
  ``skills/audio_devices.py`` was then written and correctly registered NINE
  names (``current_mic``, ``what_microphone``, …) with correct speak-set
  declarations. **It did not fix the bug.** None of the nine appeared in
  ``core/prompts.py`` or in a ``PROMPT_EXAMPLES`` block, so the model still had
  no token to emit and still routed to ``system_pulse``. The handler was fixed;
  the routing was left exactly as broken as it was found. Registering an action
  is HALF of shipping it — this test is the other half.

THE TWO ROUTES BY WHICH A NAME BECOMES EMITTABLE (both count here)
  1. It appears in the model-visible prompt text —
     ``core.prompts.BASE_SYSTEM_PROMPT`` + ``PC_CONTROL_PROMPT``, the two
     constants ``bobert_companion.build_system_prompt`` splices.
     CAVEAT, and it is a real one: that is the CLOUD prompt. On the LOCAL
     route the monolith swaps ``PC_CONTROL_PROMPT`` for a per-turn SLIM
     subset, so route 1 alone can credit a name the local brain never sees.
     ``LocalSlimPathVisibilityTests`` below is what stops that credit from
     being a lie — read its comment before touching ``is_reachable``.
  2. It appears in a skill's module-level ``PROMPT_EXAMPLES`` string, collected
     by ``bobert_companion._collect_skill_prompt_examples`` and spliced by
     ``_skill_prompt_examples_block()``. Counting only route 1 would raise
     FALSE failures against skills that legitimately document themselves this
     way (``skills/game_mode.py`` today; the gitignored personal skills by
     design — that is why the mechanism exists at all).

  ``core/dispatcher.py``'s ``_INTENT_RULES`` is a third, pre-LLM route, but it
  is a deterministic regex short-circuit rather than something the MODEL can
  reach for, so it deliberately does NOT count as reachability here. A name
  routable only by a hardcoded regex is still invisible to the brain.

WHAT "REACHABLE" MEANS HERE — generous, with ONE evidence tier
  Base rule: a whole-word occurrence of the name anywhere in that text. The
  prompt documents actions in several shapes (``[ACTION: x]`` citations, table
  rows like ``  name, <arg>  — description``, alias runs like ``a / b / c``,
  and bare comma lists such as ``Status: gate_status, stability_gate_status``),
  and a fully structural pattern fails honest entries — measured 2026-09-05: a
  rendered-text port of ``audit_codebase._extract_prompt_actions`` flipped 46
  registered names, 44 of them genuinely documented.

  The exception, added 2026-09-05 WITH evidence (the docstring's own standing
  instruction: tighten only when you have some). A whole-word hit on a name
  that contains ``_`` or a digit is unambiguous — no English sentence contains
  ``status_ring`` or ``forget_gaze_calibration`` by accident, so prose that
  merely brushes past such a name is still a mention OF that name. A hit on a
  name that is a bare lowercase word is NOT evidence of anything: ``search``
  is a real registration (an alias of ``_act_web_search``) with 11 whole-word
  occurrences in the visible text, and every one is the ordinary English word
  inside a description of a DIFFERENT action —

      "  web_search, <query>          — Google search"
      "  youtube, <query>             — YouTube search"
      "    the browser. There is NO local-library search anymore."

  Same for ``resume`` (skills/focus_mode.py), 13 occurrences, all prose,
  including ``Trigger phrases: 'resume', 'I'm back'`` — which teaches the model
  to emit ``focus_mode_off``, not ``resume``. Both were silently certified as
  shipped. Today they are aliases of documented actions so nothing is visibly
  broken; the failure lands the next time a DISTINCT capability is registered
  as ``mute``, ``open``, ``next``, ``browse``, ``status`` or ``record`` — a
  name the prompt already uses as English — and the guard waves it through.
  That is exactly the accident the identifier-aware lookarounds below claim to
  prevent; they only stop ``hud_on`` matching inside ``show_hud_on_boot``.

  So for a bare-word name the occurrence must additionally be SET OFF from
  prose: punctuated on both sides as a name, with at least one side a hard
  structural boundary (line start, a column gap, a list character, a spaced
  ``a / b`` alias run). See ``_is_set_off`` for why a soft boundary on both
  sides is not enough — it certified ``open`` and ``next`` off wrapped
  description lines ending in ``…workspace: open``.

  Measured over the 19 bare-word registered names, this rejects exactly
  ``search`` and ``resume`` and keeps the other 17 (``click``, ``max``,
  ``press``, ``restart``, ``screenshot``, ``type``, ``upgrade``, ``whoami``
  …), each on a real catalog row, alias run or ``[ACTION: x]`` citation — and
  it closes all six of the hazard names above. If a bare-word name ever fails
  this honestly, the fix is to give it the catalog row it should already have —
  not to widen the rule. ``ActionEvidenceRuleTests`` below pins the behaviour
  against fixtures so it cannot rot silently. Never loosen the base rule.

SCOPE: TRACKED FILES ONLY — three reasons, all load-bearing
  * Shippability. The registry that matters is the one a public install gets.
    ``tools/build_release.py`` exports ``git ls-files``; a local-filesystem scan
    audits the owner's box, not the product. (Same reasoning, and the same
    incident class, as tests/test_prompts.py::ShippedPromptActionInvariantTests
    — see its docstring for the 2026-08-20 case where the local scan was blind.)
  * PII. The gitignored personal skills are gitignored precisely because their
    action names embed a specific person. An allowlist generated from the local
    tree would commit those names into tracked source, straight past
    ``tools/check_no_pii.py`` (a regex word boundary never matches inside an
    identifier, which is how that gate missed this exact shape once already).
  * Reproducibility. The allowlist ratchet asserts an EXACT set. Anything that
    varies between the owner's box and a bare CI runner would make it flap.

THE RATCHET (the half that makes the allowlist shrink instead of rot)
  ``tests/action_reachability_allowlist.txt`` lists every currently-unreachable
  shipped action with a one-line reason. The assertion is EQUALITY, not
  containment, so it cuts both ways:
    * a newly registered-but-undocumented action is not in the file -> FAIL,
      and the fix is to DOCUMENT it, not to append a line;
    * a name that has since become reachable (or was deleted) is still in the
      file -> FAIL as STALE, and must be removed.
  Without the second half an allowlist is a dumping ground that only grows.

Static ``ast`` reads only: nothing here imports the monolith, imports a skill,
touches the network, or writes anywhere in the tree. ``core.prompts`` is
stdlib-only string constants (same import-light tier as tests/test_prompts.py).

CAVEAT, stated plainly: this is a DISK read. A skill that fails to import
registers nothing at runtime, so the live registry can be SMALLER than what is
scanned here. This guard proves reachability of what ships, not of what loaded.
"""
from __future__ import annotations

import argparse
import ast
import glob
import importlib.util
import os
import re
import subprocess
import sys
import unittest

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_TESTS_DIR)
_TOOLS = os.path.join(_PROJECT_ROOT, "tools")

# Under tools/run_tests.py the project root is already the top-level dir; this
# only matters when the file is executed directly for its maintenance mode.
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from core.prompts import BASE_SYSTEM_PROMPT, PC_CONTROL_PROMPT  # noqa: E402
from core import prompt_router  # noqa: E402  (stdlib-only: re + typing)

ALLOWLIST_PATH = os.path.join(_TESTS_DIR, "action_reachability_allowlist.txt")

# Floors that keep this guard from passing BLIND. Every input is scanned, not
# assumed; if an extractor drifts and starts returning almost nothing, these
# fail loudly instead of reporting a green "no gaps found".
_MIN_REGISTERED = 300
_MIN_PROMPT_CHARS = 50_000
_MIN_REACHABLE = 150
_MIN_ACTION_CITATIONS = 100

# ── the registry: WHICH files, WHICH targets ─────────────────────────────────
# Mirrors tools/gen_action_index.py's source set, so this guard and
# docs/ACTION_INDEX.md describe the same registry. The *rule* for recognising a
# registration is NOT re-implemented here: it is imported from
# tools/registration_scan.py, the one home for it (see that module's docstring
# — it previously rotted as four diverging copies and lost the entire browser
# agent from the index). The monolith is scanned for ``ACTIONS`` only; skills
# and core also register into a local ``actions`` parameter.
_MONOLITH_REL = "bobert_companion.py"


def _load_tool(name: str):
    """Import a tools/*.py module by path (tools/ is not a package)."""
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_TOOLS, name + ".py"))
    assert spec and spec.loader, f"could not build import spec for {name}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod      # MUST precede exec for dataclass (3.14)
    spec.loader.exec_module(mod)
    return mod


class _NoGit(Exception):
    """Raised when tracked-vs-local cannot be told apart."""


def _tracked_py() -> set[str]:
    try:
        out = subprocess.run(["git", "ls-files", "-z", "*.py"],
                             cwd=_PROJECT_ROOT, capture_output=True,
                             text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        raise _NoGit(f"git unavailable: {exc}") from exc
    if out.returncode != 0:
        raise _NoGit("not a git checkout — cannot tell shipped from local")
    paths = {p for p in out.stdout.split("\0") if p}
    if not paths:
        raise _NoGit("git ls-files returned nothing")
    return paths


def _rel(path: str) -> str:
    return os.path.relpath(path, _PROJECT_ROOT).replace("\\", "/")


def _registration_sources() -> list[str]:
    return ([os.path.join(_PROJECT_ROOT, _MONOLITH_REL)]
            + sorted(glob.glob(os.path.join(_PROJECT_ROOT, "skills", "*.py")))
            + sorted(glob.glob(os.path.join(_PROJECT_ROOT, "skills", "*",
                                            "__init__.py")))
            + sorted(glob.glob(os.path.join(_PROJECT_ROOT, "core", "*.py"))))


def _skill_sources() -> list[str]:
    return (sorted(glob.glob(os.path.join(_PROJECT_ROOT, "skills", "*.py")))
            + sorted(glob.glob(os.path.join(_PROJECT_ROOT, "skills", "*",
                                            "__init__.py"))))


def _module_level_prompt_examples(path: str) -> str:
    """The module-level ``PROMPT_EXAMPLES`` string of a skill, or ''.

    ast only — a skill is never imported to read it. Module level only, which
    is what the runtime collector reads (``getattr(mod, 'PROMPT_EXAMPLES')``).
    Falls back to the raw source segment when the value is not a foldable
    literal, so a dynamically-built block still counts as coverage rather than
    silently reading as a gap.
    """
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            src = fh.read()
        tree = ast.parse(src)
    except (OSError, SyntaxError):
        return ""
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "PROMPT_EXAMPLES"):
            value = node.value
        elif (isinstance(node, ast.AnnAssign) and node.value is not None
                and isinstance(node.target, ast.Name)
                and node.target.id == "PROMPT_EXAMPLES"):
            value = node.value
        else:
            continue
        try:
            folded = ast.literal_eval(value)
        except (ValueError, SyntaxError, TypeError):
            folded = None
        if isinstance(folded, str) and folded.strip():
            return folded
        return ast.get_source_segment(src, value) or ""
    return ""


# ── what counts as EVIDENCE that the prompt teaches a name ───────────────────
# Identifier-aware whole-word match. \b is WRONG here: there is no word
# boundary between a letter and an underscore, so \bhud_on\b would happily
# match inside show_hud_on_boot and report a gap as covered.
def _occurrences(text: str, name: str) -> list[re.Match[str]]:
    return list(re.finditer(r"(?<![A-Za-z0-9_])" + re.escape(name)
                            + r"(?![A-Za-z0-9_])", text))


# A name is "bare-word" when nothing about its spelling distinguishes it from
# ordinary English — no underscore, no digit. Deliberately over-inclusive
# (``whoami``, ``netflix`` and ``youtube`` are caught too): the only cost is
# that such a name must be documented in one of the shapes the catalog already
# uses, which every one of them already is.
def _is_bare_word(name: str) -> bool:
    return re.fullmatch(r"[a-z]+", name) is not None


# "Set off from prose" — the token is punctuated as a NAME rather than used as
# a word. Boundaries come in two strengths, and BOTH sides must be a boundary
# while at least ONE must be STRONG. The weak-on-both-sides case is what a
# wrapped description line looks like, and it is pure prose:
#
#     "  predictive_morning_setup     — restore the typical workspace: open"
#     "    robot_status               — the REPO Robot build as a whole: next"
#
# Those two ended a line with a colon-introduced English word, and a rule that
# accepted label-colon on the left and end-of-line on the right certified
# ``open`` and ``next`` as documented. They are not.
#
# STRONG — a structural entry boundary: line start, a column gap (2+ spaces),
#   a list/bracket character, or a SPACED slash. ``a / b`` alias runs are a
#   real catalog shape; unspaced ``pause/resume/next`` is prose and never
#   counts. On the right a strong boundary also covers the ``name, <arg>`` row.
# WEAK — a label colon, an em-dash or an arrow on the left; end-of-line, a full
#   stop or a colon on the right. Enough to CONFIRM a name, never to prove one.
#
# Quote characters are boundaries of NEITHER strength, on purpose: the prompt
# is full of quoted TRIGGER PHRASES ("Trigger phrases: 'resume', 'I'm back'"),
# which teach the model what the USER says, never what token to emit.
_STRONG_LEFT_RE = re.compile(
    r"(?:^[ \t]*|[ \t]{2,}|[,;|(\[{][ \t]?|[ \t]/[ \t])$")
_WEAK_LEFT_RE = re.compile(r"[:—→][ \t]?$")
_STRONG_RIGHT_RE = re.compile(r"^(?:[ \t]{2,}|[ \t]?[,;|)\]}]|[ \t]/[ \t])")
_WEAK_RIGHT_RE = re.compile(r"^(?:[ \t]*$|[ \t]?[.:—→])")


def _is_set_off(text: str, match: re.Match[str]) -> bool:
    """Is this occurrence punctuated as a name, within its own line?"""
    line_start = text.rfind("\n", 0, match.start()) + 1
    line_end = text.find("\n", match.end())
    if line_end == -1:
        line_end = len(text)
    before, after = text[line_start:match.start()], text[match.end():line_end]
    left_strong = _STRONG_LEFT_RE.search(before) is not None
    right_strong = _STRONG_RIGHT_RE.match(after) is not None
    left_ok = left_strong or _WEAK_LEFT_RE.search(before) is not None
    right_ok = right_strong or _WEAK_RIGHT_RE.match(after) is not None
    return left_ok and right_ok and (left_strong or right_strong)


def _is_reachable_in(text: str, name: str) -> bool:
    """Can the model learn to emit ``name`` from ``text``? See the module
    docstring's "WHAT 'REACHABLE' MEANS HERE" for why this has two tiers."""
    hits = _occurrences(text, name)
    if not hits:
        return False
    if not _is_bare_word(name):
        return True                      # an identifier cannot be prose
    return any(_is_set_off(text, m) for m in hits)


class _Scan:
    """One immutable snapshot of the tree, computed once per process."""

    _cached: _Scan | None = None
    _error: _NoGit | None = None

    def __init__(self) -> None:
        rs = _load_tool("registration_scan")
        tracked = _tracked_py()

        # 1. every registered action name -> the tracked file registering it
        self.registry: dict[str, str] = {}
        self.unparseable: list[str] = []
        self.skipped_untracked: list[str] = []
        for path in _registration_sources():
            rel = _rel(path)
            if rel not in tracked:
                self.skipped_untracked.append(rel)
                continue
            targets = (("ACTIONS",) if rel == _MONOLITH_REL
                       else rs.DEFAULT_TARGETS)
            try:
                regs = rs.scan_file(path, targets=targets, filename=rel)
            except SyntaxError:
                self.unparseable.append(rel)   # the syntax gate owns this
                continue
            except OSError:
                continue
            for name in regs:
                self.registry.setdefault(name, rel)

        # 2. every name the model can see: prompt text + skill PROMPT_EXAMPLES
        self.prompt_text = BASE_SYSTEM_PROMPT + PC_CONTROL_PROMPT
        self.example_blocks: dict[str, str] = {}
        for path in _skill_sources():
            rel = _rel(path)
            if rel not in tracked:
                continue
            block = _module_level_prompt_examples(path)
            if block.strip():
                self.example_blocks[rel] = block
        self.visible = "\n".join(
            [self.prompt_text]
            + [self.example_blocks[k] for k in sorted(self.example_blocks)])

        self.unreachable = sorted(n for n in self.registry
                                  if not self.is_reachable(n))
        self.cited = set(re.findall(r"ACTION:\s*([A-Za-z_][A-Za-z0-9_]*)",
                                    self.visible))

    def is_reachable(self, name: str) -> bool:
        return _is_reachable_in(self.visible, name)

    def reason_for(self, name: str) -> str:
        owner = self.registry.get(name, "?")
        if _is_bare_word(name) and _occurrences(self.visible, name):
            kind = ("appears in the prompt ONLY as the ordinary English word, "
                    "never set off as an emittable token (2026-09-05 "
                    "bare-word evidence tier)")
        elif name.endswith("_status"):
            kind = ("status read-back with no prompt coverage; the owning "
                    "skill documents on/off but never \"is it on?\" "
                    "(2026-09-04 audit G3)")
        else:
            kind = "no prompt coverage (2026-09-04 audit G2)"
        return f"{owner} — {kind}"

    @classmethod
    def get(cls) -> _Scan:
        if cls._error is not None:
            raise cls._error
        if cls._cached is None:
            try:
                cls._cached = _Scan()
            except _NoGit as exc:
                cls._error = exc
                raise
        return cls._cached


# ── the allowlist file ───────────────────────────────────────────────────────
_ALLOW_LINE_RE = re.compile(r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*::\s*(\S.*?)\s*$")

_ALLOWLIST_HEADER = """\
# Actions that ship but that the model cannot emit — the ratchet allowlist for
# tests/test_audit_action_reachability.py. Read that module's docstring first.
#
# Every name here is REGISTERED (a handler exists and runs) but the local brain
# has no token for it — it appears nowhere in core/prompts.py and in no skill's
# PROMPT_EXAMPLES block, or (added 2026-09-05) only as an ordinary English word
# in prose describing something else, which teaches the model nothing. For most
# entries that means the capability is real and unreachable: asking for it gets
# the nearest wrong-but-plausible action instead of an honest "I can't". For a
# few it is a bare ALIAS of a documented action, so the CAPABILITY is reachable
# through its sibling and only the token is dead — the reason line says which,
# and an alias entry is the one kind that is fine to leave standing.
#
# THIS FILE IS A DEBT LEDGER, NOT A CONFIG. It is asserted as an EXACT set:
#   * a NEW undocumented action fails the build. The fix is to document it in
#     core/prompts.py (or in the owning skill's PROMPT_EXAMPLES) — appending a
#     line here to get green is the failure mode this guard exists to stop.
#   * an entry that became reachable, or whose action was deleted, fails as
#     STALE and must be deleted. That is what makes the list shrink.
#     `python tests/test_audit_action_reachability.py --write-allowlist`
#     prunes stale entries for you; adding new names needs the explicit
#     --accept-new, so it can never quietly absorb a fresh gap.
#
# Generated from the tree at 2026-09-05 (audit 2026-09-04). Reasons are
# per-name and mechanical; the audit's own narrative grouping (browser agent,
# briefings, memory quartet, self-test probes, promise read/cancel, ambient
# subsystem, the *_status read-backs) is the map for paying this down.
#
# format:  <action_name> :: <one-line reason>       (sorted by action name)
"""


def _read_allowlist() -> tuple[dict[str, str], list[str]]:
    """(name -> reason, malformed raw lines)."""
    entries: dict[str, str] = {}
    malformed: list[str] = []
    try:
        with open(ALLOWLIST_PATH, encoding="utf-8") as fh:
            raw = fh.read()
    except OSError:
        return entries, [f"<file missing: {ALLOWLIST_PATH}>"]
    for line in raw.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        m = _ALLOW_LINE_RE.match(line)
        if not m:
            malformed.append(line)
            continue
        name, reason = m.group(1), m.group(2)
        if name in entries:
            malformed.append(f"duplicate entry: {line}")
            continue
        entries[name] = reason
    return entries, malformed


def _write_allowlist(entries: dict[str, str]) -> None:
    body = "".join(f"{n} :: {entries[n]}\n" for n in sorted(entries))
    # utf-8, LF, no BOM — a BOM here would break byte-level readers (and is the
    # standing Windows trap on this box).
    with open(ALLOWLIST_PATH, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(_ALLOWLIST_HEADER + body)


# ── the tests ────────────────────────────────────────────────────────────────
class _ScanCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.scan = _Scan.get()
        except _NoGit as exc:
            raise unittest.SkipTest(str(exc))


class ActionRegistryScanSanityTests(_ScanCase):
    """A guard that cannot see is worse than no guard: it advertises a safety
    net that is not there. These pin the INPUTS before anything is asserted
    about them."""

    def test_registry_scan_is_not_blind(self):
        self.assertGreater(
            len(self.scan.registry), _MIN_REGISTERED,
            f"registration scan found only {len(self.scan.registry)} actions "
            f"(unparseable={self.scan.unparseable[:5]}) — the scanner or the "
            f"source set drifted, so every check below would pass blind")

    def test_no_scanned_source_is_unparseable(self):
        """A file that does not parse contributes ZERO registrations, so its
        gaps vanish from the forward check and its allowlist entries read as
        stale. Found the hard way while proving this guard discriminates: a
        deliberately-broken skill made a KNOWN gap 'pass'. compileall in CI
        would also catch it, but a silent drop here is precisely this
        project's defining bug class, so fail loudly at the point of blindness
        rather than trusting a sibling gate to fire first."""
        self.assertEqual(
            self.scan.unparseable, [],
            "these scanned sources do not parse, so every action they register "
            "is INVISIBLE to this guard (and to docs/ACTION_INDEX.md)")

    def test_prompt_text_is_not_blind(self):
        self.assertGreater(
            len(self.scan.prompt_text), _MIN_PROMPT_CHARS,
            "the model-visible prompt collapsed — every action would look "
            "unreachable")
        self.assertGreater(
            len(self.scan.cited), _MIN_ACTION_CITATIONS,
            "almost no [ACTION: x] citations found — the citation regex "
            "drifted, so the inverse check would pass blind")

    def test_reachable_route_is_not_blind(self):
        reachable = len(self.scan.registry) - len(self.scan.unreachable)
        self.assertGreater(
            reachable, _MIN_REACHABLE,
            f"only {reachable} registered actions are reachable — the "
            f"whole-word match drifted; this would flood the allowlist")

    def test_monolith_and_skills_both_contribute(self):
        # Both halves of the registry must be present, or a whole source set
        # could silently drop out and read as "no gaps".
        owners = set(self.scan.registry.values())
        self.assertIn(_MONOLITH_REL, owners)
        self.assertTrue(any(o.startswith("skills/") for o in owners),
                        "no skill-registered actions found at all")

    def test_prompt_examples_route_is_wired(self):
        # Route 2 must actually collect something. If this ever legitimately
        # goes to zero (every tracked skill moved its block into prompts.py),
        # delete this test rather than weakening it — but confirm first: a
        # silent zero is indistinguishable from a broken extractor, and a
        # broken extractor means false failures for skills documented that way.
        self.assertTrue(
            self.scan.example_blocks,
            "no tracked skill contributed a PROMPT_EXAMPLES block — either the "
            "extractor broke or the last such skill changed")


class ActionEvidenceRuleTests(unittest.TestCase):
    """Pin the reachability rule itself against FIXTURES, not the live tree.

    A guard whose matcher silently rots is the same failure as no guard. These
    need no git checkout and no prompt, so they still run (and still fail) on a
    bare CI box where every _ScanCase above skips.

    Every ACCEPTED string below is copied verbatim from core/prompts.py or a
    skill PROMPT_EXAMPLES block; every REJECTED one is a real occurrence that
    used to be counted as coverage before 2026-09-05."""

    _ACCEPT = [
        # bare-word names in the shapes the catalog actually uses
        ("  screenshot                   — save a screenshot", "screenshot"),
        ("  youtube, <query>             — YouTube search", "youtube"),
        ("  whoami / recognize_face      — say who's in front of the webcam.",
         "whoami"),
        ("  Aliases: python, eval_python, compute, reset_kernel", "compute"),
        ("    'play Succession on Max' -> [ACTION: max, Succession]", "max"),
        ("  restart                      — relaunch JARVIS immediately.",
         "restart"),
        # identifier-shaped names keep the generous rule: a mention in a
        # sentence is still a mention, because prose cannot produce these.
        ("  Status: gaze_status, gaze_stats; forget_gaze_calibration resets it",
         "forget_gaze_calibration"),
        ("    status_ring are TOGGLES — emitting any of them flips the HUD",
         "status_ring"),
    ]
    _REJECT = [
        # the 2026-09-05 defect: the English word describing ANOTHER action
        ("  web_search, <query>          — Google search", "search"),
        ("  youtube, <query>             — YouTube search", "search"),
        ("    the browser. There is NO local-library search anymore.",
         "search"),
        ("   4. click on UI elements by description ('the search box')",
         "search"),
        ("                 dedicated streaming/search actions fit —", "search"),
        ("  Transport (pause/resume/next/previous/now_playing) uses MEDIA KEYS",
         "resume"),
        ("    Trigger phrases: 'resume', 'I'm back', 'end focus mode'.",
         "resume"),
        ("    Example: 'resume, what did I miss' -> [ACTION: focus_mode_off]",
         "resume"),
        ("  resume_print                 — resume a paused print.", "resume"),
        ("  Pause and resume: pause_phone_bridge, resume_phone_bridge",
         "resume"),
        # wrapped description lines ending on a colon-introduced English word.
        # Soft boundary on BOTH sides — the second thing this rule got wrong.
        ("  predictive_morning_setup     — restore the typical workspace: open",
         "open"),
        ("    robot_status             — the REPO Robot build so far: next",
         "next"),
        # the original hazard the lookarounds exist for, still closed
        ("  show_hud_on_boot             — start the HUD at launch", "hud_on"),
    ]

    def test_accepted_shapes(self):
        for text, name in self._ACCEPT:
            with self.subTest(name=name, text=text):
                self.assertTrue(
                    _is_reachable_in(text, name),
                    "this is a real documentation shape from the shipped "
                    "prompt; rejecting it makes the guard cry wolf")

    def test_rejected_shapes(self):
        for text, name in self._REJECT:
            with self.subTest(name=name, text=text):
                self.assertFalse(
                    _is_reachable_in(text, name),
                    "the name is only brushed past as English/another "
                    "action's description — crediting it certifies an action "
                    "as shipped that the model has never been shown")

    def test_bare_word_classifier(self):
        for name in ("search", "resume", "mute", "open", "next", "browse",
                     "status", "record", "whoami"):
            self.assertTrue(_is_bare_word(name), name)
        for name in ("web_search", "hud_on", "status_ring", "media_next",
                     "sh_tuya2"):
            self.assertFalse(_is_bare_word(name), name)

    def test_absent_name_is_never_reachable(self):
        self.assertFalse(_is_reachable_in("  volume_up, <n>  — louder",
                                          "volume_down"))


class ActionReachabilityRatchetTests(_ScanCase):
    """Registered, but emittable by NEITHER route — the audio_devices bug."""

    def setUp(self):
        self.entries, self.malformed = _read_allowlist()

    def test_allowlist_file_is_well_formed(self):
        self.assertEqual(
            self.malformed, [],
            f"{os.path.basename(ALLOWLIST_PATH)}: expected "
            f"'<action_name> :: <one-line reason>' per line")
        self.assertTrue(self.entries,
                        "allowlist parsed to nothing — the format drifted and "
                        "the ratchet would report every entry stale")
        missing_reason = sorted(n for n, r in self.entries.items()
                                if len(r) < 10)
        self.assertEqual(missing_reason, [],
                         "every allowlist entry needs a real one-line reason")

    def test_allowlist_is_sorted(self):
        self.assertEqual(
            list(self.entries), sorted(self.entries),
            "keep the allowlist sorted by action name so its diffs are "
            "readable (python tests/test_audit_action_reachability.py "
            "--write-allowlist)")

    def test_no_new_unreachable_action(self):
        """A registered action the model has no token for. THE regression."""
        new = sorted(set(self.scan.unreachable) - set(self.entries))
        detail = "\n".join(f"    {n}  (registered in {self.scan.registry[n]})"
                           for n in new)
        self.assertEqual(
            new, [],
            "These actions are REGISTERED but appear nowhere in "
            "core/prompts.py and in no skill PROMPT_EXAMPLES block, so the "
            "model can never emit them. It will answer with the nearest "
            "wrong-but-plausible action instead — the 2026-09-04 "
            "'what microphone are you using' -> [ACTION: system_pulse] "
            f"failure:\n{detail}\n"
            "  FIX: document them (core/prompts.py, or the owning skill's "
            "module-level PROMPT_EXAMPLES if the trigger phrases are "
            "personal). Do NOT append them to "
            f"{os.path.basename(ALLOWLIST_PATH)} to get green — that ledger is "
            "for pre-existing debt, and this guard exists because registering "
            "a handler is only half of shipping a capability.")

    def test_allowlist_has_no_stale_entries(self):
        """The half that makes the ledger SHRINK instead of collecting dust."""
        now_reachable = sorted(n for n in self.entries
                               if n in self.scan.registry
                               and self.scan.is_reachable(n))
        gone = sorted(n for n in self.entries if n not in self.scan.registry)
        detail = ""
        if now_reachable:
            detail += ("\n  now documented (delete these lines):\n"
                       + "\n".join(f"    {n}" for n in now_reachable))
        if gone:
            detail += ("\n  no longer registered in any tracked file "
                       "(delete these lines):\n"
                       + "\n".join(f"    {n}" for n in gone))
        self.assertEqual(
            now_reachable + gone, [],
            f"stale entries in {os.path.basename(ALLOWLIST_PATH)}."
            f"{detail}\n"
            "  An allowlist that only ever grows is a dumping ground; removing "
            "an entry the moment it is fixed is what keeps this one honest. "
            "`python tests/test_audit_action_reachability.py "
            "--write-allowlist` prunes them.")


class DocumentedActionsResolveTests(_ScanCase):
    """The inverse: a name the prompt TEACHES that no handler answers.

    The same silent failure wearing the other hat — the monolith's
    unknown-action arm records ``unknown action: <name>`` and strips the token
    from the spoken text, so the owner hears a confident non-answer.

    Deliberately NOT a copy of
    tests/test_prompts.py::ShippedPromptActionInvariantTests: that one applies
    ``audit_codebase._extract_prompt_actions`` (a STRUCTURAL table/list
    extractor) to the SOURCE of core/prompts.py and is the stricter check on
    that file. This one covers the ``[ACTION: x]`` citations across the
    RENDERED prompt PLUS every tracked skill's PROMPT_EXAMPLES block, which
    that extractor never looks at. Both are subset checks against the one
    shared tools/registration_scan registry, so they cannot diverge on what
    "registered" means."""

    def test_every_cited_action_is_registered(self):
        ghosts = sorted(self.scan.cited - set(self.scan.registry))
        detail = "\n".join(f"    {g}" for g in ghosts)
        self.assertEqual(
            ghosts, [],
            "core/prompts.py (or a skill PROMPT_EXAMPLES block) cites "
            "[ACTION: x] for name(s) no tracked file registers. At runtime "
            "these dispatch to 'unknown action', which is swallowed silently "
            f"and never voiced:\n{detail}\n"
            "  FIX: register the handler, or delete the citation.")


# ── the LOCAL slim path: the half "model-visible" does not cover ─────────────
# ``is_reachable`` searches the FULL ``PC_CONTROL_PROMPT``. On the CLOUD route
# that IS what ships (200k ctx, spliced whole), so there the search is the
# truth. On the LOCAL route — ``MODEL_ROUTING.chat == "local"``, the configured
# value on this box — it is not: ``bobert_companion`` replaces
# ``PC_CONTROL_PROMPT`` with ``core.prompt_router.slim_pc_control(user_text,
# …)`` every turn, which keeps the preamble, the sections the turn's words
# implicate, and a one-line INDEX of the section NAMES it dropped. A name whose
# section the router cannot SEE is credited by this file and absent at runtime.
#
# PROVEN 2026-09-05, the second head of the audio_devices bug this module was
# written for. The STATUS READ-BACKS block opened with a PROSE line —
# "STATUS READ-BACKS — 'is it ON?'. Every name here REPORTS state and" — mixed
# case and no trailing colon, so ``prompt_router._HEADER_RE`` did not match it.
# ``split_pc_control`` returned 94 sections with that block folded into SUIT
# DIAGNOSTICS, whose keywords are 'suit diagnostics' / 'full readout' / …, and
# all 17 trigger phrases the block prints for itself dropped their action:
# "is the workshop HUD showing" shipped 12,913 of 122,864 chars with
# ``workshop_hud_status`` nowhere in them. The INDEX could not save it either —
# the INDEX lists section NAMES, and this was not a recognised section. All 22
# read-back names were credited as reachable here the whole time.
#
# Being IN the prompt is not the same as REACHING the model. That is the same
# sentence core/prompts.py's AUDIO DEVICES comment already carries; these tests
# are the executable version of it.
#
# KNOWN LIMIT, stated rather than papered over: this is NOT yet a whole-prompt
# ratchet. Measured 2026-09-05, 16 of 92 extractable "'phrase' → [ACTION: x]"
# examples elsewhere in the prompt still fail the same round-trip (air-mouse,
# keep_music_open, check_for_updates, model_costs, test_vision, …). Each is a
# real gap and none is this card. Reproduce with the extractor below over every
# section; turning that into an exact-set ratchet is the follow-up.
_ARROW_EXAMPLE_RE = re.compile(
    r"(?<!\w)'([^'<>]{4,80})'(?!\w)\s*(?:→|->)\s*"
    r"\[ACTION:\s*([A-Za-z_][A-Za-z0-9_]*)\]")
# Only plain utterance-shaped phrases. A phrase containing an apostrophe is
# un-extractable by the rule above (the contraction closes the quote early), so
# those are skipped rather than fed to the router as garbage — the same
# convention tests/test_prompt_router.py::_quoted_phrases uses.
_PLAIN_UTTERANCE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9 ,\-?]*$")

# A capability block introducer: a column-0 line opening with two or more
# consecutive ALL-CAPS words. Column 0 only — indented sub-headers are promoted
# separately and are covered by
# test_prompt_router.py::test_every_promoted_indented_subheader_has_keywords.
_BLOCK_OPENER_RE = re.compile(r"^[A-Z][A-Z0-9+/&.'-]*(?:[ -][A-Z0-9+/&.'-]+)+")


def _arrow_examples(text: str) -> list[tuple[str, str]]:
    """[(trigger phrase, action it is documented to fire), …] in `text`."""
    out: list[tuple[str, str]] = []
    for phrase, action in _ARROW_EXAMPLE_RE.findall(" ".join(text.split())):
        phrase = phrase.strip()
        if _PLAIN_UTTERANCE_RE.match(phrase):
            out.append((phrase, action))
    return out


class LocalSlimPathVisibilityTests(unittest.TestCase):
    """Route 1 credits the FULL prompt; the local brain gets a slimmed one."""

    @classmethod
    def setUpClass(cls):
        cls.sections = prompt_router.split_pc_control(PC_CONTROL_PROMPT)[1]

    def test_section_split_is_not_blind(self):
        # Floor, same reason as the sanity class above: if the splitter ever
        # returns nothing, every check below would pass by finding nothing.
        self.assertGreater(
            len(self.sections), 50,
            "prompt_router.split_pc_control found almost no sections — the "
            "header parser drifted and both tests below would pass blind")

    def test_no_capability_block_is_invisible_to_the_router(self):
        """A block whose opening line is not a parseable header is folded into
        its predecessor: its actions inherit that section's keywords, load only
        when those unrelated words appear, and its NAME never reaches the
        dropped-section INDEX. That is the exact 2026-09-05 STATUS READ-BACKS
        failure, and this fails on it (verified by re-inserting the prose
        opener: 1 orphan, 94 sections, workshop_hud_status absent from the
        slim prompt for its own trigger phrase)."""
        orphans = []
        for line in prompt_router._join_wrapped_headers(
                PC_CONTROL_PROMPT.split("\n")):
            if not line or line[0] in " \t":
                continue                      # indented: not a block opener
            if prompt_router._HEADER_RE.match(line.strip()):
                continue                      # parses — this is a real section
            m = _BLOCK_OPENER_RE.match(line)
            if m and len(m.group(0)) >= 6:
                orphans.append(line.strip()[:80])
        self.assertEqual(
            orphans, [],
            "these column-0 lines open a capability block but do NOT parse as "
            "a prompt_router section, so the block is folded into the "
            "PRECEDING section and inherits its keywords — the actions it "
            "documents drop out of the local prompt and its name never "
            "reaches the capability INDEX:\n"
            + "\n".join(f"    {o}" for o in orphans)
            + "\n  FIX: give it a parseable header ('HEAD (lowercase "
            "note):' at column 0) and a keyword list in "
            "core/prompt_router.py::_SECTION_KEYWORDS. Do NOT relax "
            "_HEADER_RE to swallow prose — that folds real prose into "
            "sections instead.")

    def test_status_read_back_triggers_reach_the_local_model(self):
        """Every trigger phrase the read-back block prints for itself must
        still carry its action after slimming.

        Located by CONTENT, not by header name, so it survives a rename or a
        move: whichever section documents ``workshop_hud_status`` is the one
        under test. If the block is ever folded into a neighbour again, its
        examples get extracted from that neighbour's body and fail here."""
        homes = [b for h, b in self.sections if "workshop_hud_status" in b]
        self.assertTrue(
            homes, "no prompt section documents workshop_hud_status — the "
                   "read-back block was deleted or renamed its actions")
        examples = [pair for body in homes for pair in _arrow_examples(body)]
        self.assertGreaterEqual(
            len(examples), 15,
            f"expected the read-back block's documented triggers, got "
            f"{examples} — the extractor drifted and this would pass blind")
        misses = []
        for phrase, action in examples:
            slim = prompt_router.slim_pc_control(phrase, PC_CONTROL_PROMPT)
            if not re.search(r"(?<![A-Za-z0-9_])" + re.escape(action)
                             + r"(?![A-Za-z0-9_])", slim):
                misses.append(f"{phrase!r} -> {action} "
                              f"(slim prompt {len(slim)} chars)")
        self.assertEqual(
            misses, [],
            "the prompt prints these trigger phrases next to the action they "
            "should fire, but after slim_pc_control the action name is GONE, "
            "so on the local route the model is asked the question with no "
            "token to answer it:\n"
            + "\n".join(f"    {m}" for m in misses)
            + "\n  FIX: widen the section's keyword list in "
            "core/prompt_router.py::_SECTION_KEYWORDS, or check the section "
            "header still parses. Do NOT delete the example.")


# ── maintenance entry point ──────────────────────────────────────────────────
_MAINT_FLAGS = {"--report", "--write-allowlist", "--accept-new"}


def _main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description="Report or rewrite the action-reachability allowlist.")
    ap.add_argument("--report", action="store_true",
                    help="print the reachability numbers (default action)")
    ap.add_argument("--write-allowlist", action="store_true",
                    help="prune stale entries and rewrite the file sorted")
    ap.add_argument("--accept-new", action="store_true",
                    help="with --write-allowlist, also ADD newly unreachable "
                         "actions. Deliberately separate: documenting the "
                         "action is almost always the right fix.")
    args = ap.parse_args(argv)

    scan = _Scan.get()
    entries, malformed = _read_allowlist()
    unreachable = set(scan.unreachable)
    new = sorted(unreachable - set(entries))
    stale = sorted(set(entries) - unreachable)

    print(f"registered (tracked):  {len(scan.registry)}")
    print(f"reachable:             {len(scan.registry) - len(unreachable)}")
    print(f"unreachable:           {len(unreachable)}")
    print(f"allowlisted:           {len(entries)}")
    print(f"new (not allowlisted): {len(new)}")
    print(f"stale allowlisted:     {len(stale)}")
    if malformed:
        print(f"malformed lines:       {len(malformed)}")
    for n in new:
        print(f"  NEW   {n} :: {scan.reason_for(n)}")
    for n in stale:
        print(f"  STALE {n}")

    if not args.write_allowlist:
        return 1 if (new or stale or malformed) else 0

    kept = {n: entries[n] for n in entries if n in unreachable}
    if args.accept_new:
        for n in new:
            kept[n] = scan.reason_for(n)
    _write_allowlist(kept)
    print(f"wrote {ALLOWLIST_PATH}: {len(kept)} entr(ies)"
          + ("" if args.accept_new else f"; {len(new)} new gap(s) NOT added"))
    return 0


if __name__ == "__main__":
    if set(sys.argv[1:]) & _MAINT_FLAGS:
        sys.exit(_main(sys.argv[1:]))
    unittest.main()
