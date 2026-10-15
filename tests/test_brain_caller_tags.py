"""Brain caller tags (2026-10-15): every call into the brain says WHO is
calling and whether it is foreground or background.

WHY
===
v2.0.126 (speed plan R5) taught the traffic gate to hold a TAGGED background
job (core.local_traffic.background_work) so it cannot evict the local model's
one-slot prefix cache in front of the owner's next turn; v2.0.179 added "no
background brain evictions". What was left was a census: a background caller
that was never tagged posts straight into the owner's turn, and the cost log
could not say who spent what. The mechanism is the one that already existed
(background_work / slot / the gate), extended by one thin layer:

* ``caller=`` (a stable snake_case name) and ``background=`` on every brain
  entry point -- core.llm_client.{create_message, stream_message, complete,
  stream_text} for Claude, the monolith's _llm_quick / _call_local_llm /
  _call_local_vision / _local_complete / _ollama_chat_bounded / _claude_create
  / ask_vision / ask_vision_multi for the local model and the shared helpers,
  and the orchestrator's two stage helpers;
* ``core.local_traffic.call_scope`` turns a declared-background call into a
  ``background_work(opt_in=True)`` job (so it hits the same gate and the same
  BACKGROUND_TAG_STRICT switch), and ``CALLS`` (a per-caller ledger) plus the
  ``[llm] tokens`` / ``[local-llm] served via`` lines carry the tag.

WHAT THESE TESTS PIN
====================
1. RATCHET: an AST scan of the whole tree fails when a brain entry point is
   called without ``caller=`` (a plain snake_case string, or a pass-through
   name inside a function that has its own ``caller`` parameter), when a raw
   Ollama /api/chat POST appears outside the known entry functions, or when a
   raw ``ollama`` / ``messages.create`` appears outside its one home. The
   allowlist below is tiny and every row says why.
2. Foreground by default; a background-tagged call is logged and counted as
   background; a thread already inside background_work is background without
   saying so; an explicit ``background=False`` wins.
3. A declared-background call from a thread with no job runs as an opt-in
   background job (the gate sees it); a foreground call never does.
4. The ledger and the log line carry the caller; the SDK never receives the
   bookkeeping kwargs.
"""
from __future__ import annotations

import ast
import contextlib
import io
import os
import re
import sys
import threading
import types
import unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core import llm_client  # noqa: E402
from core import local_traffic as lt  # noqa: E402

# ── the census ────────────────────────────────────────────────────────────
# Bare function / attribute names whose every call must declare its caller.
ENTRY_NAMES = frozenset({
    # the monolith's brain helpers (also reached as bc.<name>, b.<name>)
    "_llm_quick", "_call_local_llm", "_call_local_vision",
    "_call_local_vision_raw", "_local_complete", "local_complete",
    "_ollama_chat_bounded", "_claude_create", "ask_vision",
    "ask_vision_multi", "_ask_vision_multi_raw",
    # the orchestrator's stage helpers
    "_claude_call", "_ollama_call",
    # the turn paths' wrappers (they hand the caller down to the leaves)
    "_claude_oneshot", "_local_then_cloud_or_honest", "_local_fallback_or",
})
# core.llm_client entry points, matched as <alias>.<name>(...)
CLIENT_NAMES = frozenset({"create_message", "stream_message", "complete",
                          "stream_text"})
CLIENT_ALIASES = frozenset({"llm_client", "_llm_client", "_lc"})
# A call through a local alias of an entry point:
#   fn = getattr(bc, "ask_vision", None) / su.get("ask_vision") / _su("x")
ALIAS_FACTORY_CALLS = frozenset({"getattr", "get", "_su"})
# ... or a helper that returns one.
ALIAS_FACTORY_FUNCS = frozenset({"_ask_vision"})

CALLER_RE = re.compile(r"^[a-z][a-z0-9_]*$")

# Raw Ollama /api/chat POSTs: only inside these functions (file -> funcs).
# Each is either the entry point's own body or model management; none may be
# copied without a caller tag of its own.
RAW_CHAT_ALLOWED = {
    "bobert_companion.py": {
        "_generate": "inside _call_local_llm (caller= declared there)",
        "_call_local_vision_raw": "the vision entry point's own POST",
        "_local_complete": "the skills' plain completion; caller= declared",
        "_post_prime": "idle re-prime: ledger name idle_reprime (background)",
        "_do": "boot warm-up: ledger name boot_warmup (background)",
    },
    "core/orchestrator.py": {
        "_ollama_call": "the orchestrator's local stage helper; caller= "
                        "declared by its callers (this scan)",
    },
    "skills/game_mode.py": {
        "_unload": "model management (keep_alive 0), no generation",
        "_keep_warm": "1-token residency refresh, no answer is used",
    },
    "tools/test_local_prompt.py": {
        "call": "developer tool, run by hand, not part of JARVIS",
    },
}
# `import ollama` / raw Anthropic messages: one home each.
OLLAMA_IMPORT_HOME = ("bobert_companion.py", "_ollama_chat_bounded")
SDK_MESSAGES_HOME = "core/llm_client.py"

_SKIP_DIRS = {"tests", ".git", "__pycache__", "backups", "data_staging",
              "venv", ".venv", "node_modules", "data", "logs",
              ".pytest_cache", "docs", "assets", "native", "memory"}


def _py_files():
    for dirpath, dirs, files in os.walk(_ROOT):
        dirs[:] = [d for d in dirs
                   if d not in _SKIP_DIRS and not d.startswith(".")]
        for f in files:
            if f.endswith(".py"):
                yield os.path.join(dirpath, f)


def _rel(path):
    return os.path.relpath(path, _ROOT).replace(os.sep, "/")


class _Scan(ast.NodeVisitor):
    """Walk one module; collect untagged entry calls and raw-brain sites."""

    def __init__(self, rel):
        self.rel = rel
        self.funcs = []           # stack of FunctionDef / Lambda nodes
        self.aliases = [set()]    # stack of alias-name sets (module, fn...)
        self.problems = []
        self.tagged = []          # (rel, lineno, entry, caller-name)
        self.raw_chat = []        # (lineno, enclosing function names)
        self.ollama_imports = []
        self.sdk_messages = []

    # -- scopes ---------------------------------------------------------
    def _enter(self, node):
        self.funcs.append(node)
        self.aliases.append(set())
        self.generic_visit(node)
        self.aliases.pop()
        self.funcs.pop()

    visit_FunctionDef = _enter
    visit_AsyncFunctionDef = _enter

    def _fn_names(self):
        return [getattr(f, "name", "<lambda>") for f in self.funcs] or [
            "<module>"]

    def _has_caller_param(self):
        for f in self.funcs:
            a = getattr(f, "args", None)
            if a is None:
                continue
            names = [x.arg for x in a.args + a.kwonlyargs]
            if "caller" in names:
                return True
        return False

    # -- aliases --------------------------------------------------------
    def visit_Assign(self, node):
        try:
            if (len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)):
                if self._is_alias_source(node.value):
                    self.aliases[-1].add(node.targets[0].id)
        except Exception:
            pass
        self.generic_visit(node)

    @staticmethod
    def _call_name(call):
        f = call.func
        if isinstance(f, ast.Name):
            return f.id
        if isinstance(f, ast.Attribute):
            return f.attr
        return ""

    def _is_alias_source(self, v):
        if isinstance(v, ast.Call):
            name = self._call_name(v)
            if name in ALIAS_FACTORY_FUNCS:
                return True
            if name in ALIAS_FACTORY_CALLS:
                return any(isinstance(a, ast.Constant)
                           and a.value in ENTRY_NAMES for a in v.args)
        if isinstance(v, ast.Subscript):          # skill_utils["ask_vision"]
            sl = v.slice
            return isinstance(sl, ast.Constant) and sl.value in ENTRY_NAMES
        if isinstance(v, ast.BoolOp):             # a or b
            return any(self._is_alias_source(x) for x in v.values)
        return False

    def _is_alias(self, name):
        return any(name in s for s in self.aliases)

    # -- imports / raw sites -------------------------------------------
    def visit_Import(self, node):
        for a in node.names:
            if a.name.split(".")[0] == "ollama":
                self.ollama_imports.append((node.lineno, self._fn_names()))
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if (node.module or "").split(".")[0] == "ollama":
            self.ollama_imports.append((node.lineno, self._fn_names()))
        self.generic_visit(node)

    # -- calls ----------------------------------------------------------
    def _entry_of(self, call):
        f = call.func
        if isinstance(f, ast.Name):
            if f.id in ENTRY_NAMES or self._is_alias(f.id):
                return f.id
        elif isinstance(f, ast.Attribute):
            if f.attr in ENTRY_NAMES:
                return f.attr
            if (f.attr in CLIENT_NAMES and isinstance(f.value, ast.Name)
                    and f.value.id in CLIENT_ALIASES):
                return f"llm_client.{f.attr}"
            if (isinstance(f.value, ast.Name) and self._is_alias(f.value.id)):
                return None
        return None

    def visit_Call(self, node):
        entry = self._entry_of(node)
        if entry:
            self._check_entry(node, entry)
        # raw /api/chat
        try:
            if self._call_name(node) in ("post", "Request", "urlopen"):
                txt = ast.unparse(node)
                if "/api/chat" in txt or "/api/generate" in txt:
                    self.raw_chat.append((node.lineno, self._fn_names()))
        except Exception:
            pass
        # direct SDK messages.create / .stream
        f = node.func
        if (isinstance(f, ast.Attribute) and f.attr in ("create", "stream")
                and isinstance(f.value, ast.Attribute)
                and f.value.attr == "messages"):
            self.sdk_messages.append((node.lineno, f.attr))
        self.generic_visit(node)

    def _check_entry(self, node, entry):
        kws = {k.arg: k for k in node.keywords if k.arg}
        splat = any(k.arg is None for k in node.keywords) or any(
            isinstance(a, ast.Starred) for a in node.args)
        where = f"{self.rel}:{node.lineno} {entry}"
        if "caller" not in kws:
            if splat:
                return          # forwards *args/**kw: judged at its own site
            self.problems.append(f"{where}: no caller=")
            return
        v = kws["caller"].value
        if isinstance(v, ast.Constant):
            if not (isinstance(v.value, str) and CALLER_RE.match(v.value)):
                self.problems.append(
                    f"{where}: caller={v.value!r} is not a snake_case name")
            else:
                self.tagged.append((self.rel, node.lineno, entry, v.value))
        elif not self._has_caller_param():
            self.problems.append(
                f"{where}: caller= is not a literal name and the enclosing "
                f"function has no `caller` parameter to pass through")
        bg = kws.get("background")
        if bg is not None and isinstance(bg.value, ast.Constant) and not (
                isinstance(bg.value.value, bool) or bg.value.value is None):
            self.problems.append(f"{where}: background= must be a bool")


def _scan_tree():
    out = {}
    for path in _py_files():
        try:
            with open(path, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=path)
        except (SyntaxError, UnicodeDecodeError, OSError):
            continue
        sc = _Scan(_rel(path))
        sc.visit(tree)
        out[sc.rel] = sc
    return out


class TestRatchet(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scans = _scan_tree()

    def test_every_brain_call_declares_its_caller(self):
        problems = []
        for sc in self.scans.values():
            problems.extend(sc.problems)
        self.assertEqual(problems, [], "\n" + "\n".join(problems))

    def test_the_census_is_not_vacuous(self):
        """The scan must actually see the call sites (a refactor that renames
        an entry point would otherwise turn the ratchet into a no-op)."""
        tagged = [t for sc in self.scans.values() for t in sc.tagged]
        self.assertGreaterEqual(len(tagged), 40, len(tagged))
        names = {t[3] for t in tagged}
        for must in ("voice_turn", "voice_followup", "learn_from_turn",
                     "proactive_remark", "overnight_ideas", "ltm_reflect",
                     "notification_triage", "chappie", "credits_monitor"):
            self.assertIn(must, names)
        files = {t[0] for t in tagged}
        for must in ("bobert_companion.py", "core/orchestrator.py",
                     "skills/email_triage.py", "overnight_upgrade.py",
                     "skills/kinect_vision.py"):
            self.assertIn(must, files)

    def test_raw_ollama_chat_posts_only_in_known_functions(self):
        bad = []
        for rel, sc in self.scans.items():
            allowed = RAW_CHAT_ALLOWED.get(rel, {})
            for lineno, fns in sc.raw_chat:
                if not any(fn in allowed for fn in fns):
                    bad.append(f"{rel}:{lineno} in {fns}")
        self.assertEqual(bad, [], "a raw Ollama /api/chat POST outside the "
                         "known entry functions (use _call_local_llm / "
                         "_local_complete with caller=, or add a documented "
                         "row to RAW_CHAT_ALLOWED):\n" + "\n".join(bad))

    def test_allowlist_rows_are_alive(self):
        """A stale allowlist row hides nothing and misleads: every row must
        still match a real raw POST."""
        for rel, rows in RAW_CHAT_ALLOWED.items():
            sc = self.scans.get(rel)
            self.assertIsNotNone(sc, f"{rel} is gone")
            seen = {fn for _l, fns in sc.raw_chat for fn in fns}
            for fn in rows:
                self.assertIn(fn, seen, f"{rel}: {fn} no longer POSTs")

    def test_ollama_sdk_and_messages_create_have_one_home(self):
        home_file, home_fn = OLLAMA_IMPORT_HOME
        bad = []
        for rel, sc in self.scans.items():
            for lineno, fns in sc.ollama_imports:
                if not (rel == home_file and home_fn in fns):
                    bad.append(f"{rel}:{lineno} imports ollama in {fns}")
            if rel != SDK_MESSAGES_HOME:
                for lineno, attr in sc.sdk_messages:
                    bad.append(f"{rel}:{lineno} calls .messages.{attr}() "
                               f"directly")
        self.assertEqual(bad, [], "\n".join(bad))

    def test_the_scanner_catches_an_untagged_call(self):
        """Mutation check: the scanner itself must flag each shape."""
        src = (
            "def f(bc, su):\n"
            "    bc._llm_quick('s', 'u')\n"
            "    llm_client.complete(model='m', messages=[])\n"
            "    ask = getattr(bc, 'ask_vision', None)\n"
            "    ask('q', b'')\n"
            "    g = su.get('ask_vision')\n"
            "    g('q', b'')\n"
            "    bc._call_local_llm('s', [], caller='Bad Name')\n"
            "    bc._call_local_llm('s', [], caller=who)\n"
            "    bc._call_local_llm('s', [], caller='ok_name')\n"
            "    bc._call_local_llm('s', [], **kw)\n"
            "def h(caller=''):\n"
            "    bc._llm_quick('s', 'u', caller=caller)\n"
        )
        sc = _Scan("x.py")
        sc.visit(ast.parse(src))
        self.assertEqual(len(sc.problems), 6, "\n".join(sc.problems))
        self.assertEqual([t[3] for t in sc.tagged], ["ok_name"])


# ── runtime: the one mechanism ────────────────────────────────────────────
class _Usage:
    def __init__(self, i=10, o=5, cr=0, cw=0):
        self.input_tokens = i
        self.output_tokens = o
        self.cache_read_input_tokens = cr
        self.cache_creation_input_tokens = cw


class _Block:
    type = "text"

    def __init__(self, t):
        self.text = t


class _Msg:
    stop_reason = "end_turn"

    def __init__(self, text="ok", usage=None):
        self.content = [_Block(text)]
        self.usage = usage or _Usage()


class _FakeMessages:
    def __init__(self, owner):
        self._o = owner

    def create(self, **kw):
        self._o.sent.append(kw)
        return _Msg()


class _FakeClient:
    def __init__(self):
        self.sent = []
        self.messages = _FakeMessages(self)


class TestCallerResolution(unittest.TestCase):
    def test_foreground_by_default_and_untagged_named(self):
        self.assertEqual(lt.resolve_caller(), (lt.UNTAGGED, False))
        self.assertEqual(lt.resolve_caller("voice_turn"), ("voice_turn", False))

    def test_explicit_background(self):
        self.assertEqual(lt.resolve_caller("learn", True), ("learn", True))

    def test_a_thread_inside_a_job_is_background_and_keeps_its_tag(self):
        with lt.background_work("nightly"):
            self.assertEqual(lt.resolve_caller(), ("nightly", True))
            self.assertEqual(lt.resolve_caller("digest"), ("digest", True))
            # an explicit foreground declaration wins
            self.assertEqual(lt.resolve_caller("x", False), ("x", False))
        self.assertIsNone(lt.current_job())

    def test_names_are_one_token(self):
        self.assertEqual(lt.clean_caller("a b\tc"), "a_b_c")
        self.assertEqual(lt.clean_caller(""), lt.UNTAGGED)
        self.assertLessEqual(len(lt.clean_caller("x" * 200)), 48)


class TestCallScope(unittest.TestCase):
    def test_background_call_runs_as_an_opt_in_job(self):
        seen = {}
        with lt.call_scope("proactive_remark", True) as (name, bg):
            job = lt.current_job()
            seen.update(name=name, bg=bg, tag=job.tag, opt_in=job.opt_in)
        self.assertEqual(seen, {"name": "proactive_remark", "bg": True,
                                "tag": "proactive_remark", "opt_in": True})
        self.assertIsNone(lt.current_job())

    def test_foreground_call_is_never_a_job(self):
        with lt.call_scope("voice_turn") as (name, bg):
            self.assertEqual((name, bg), ("voice_turn", False))
            self.assertIsNone(lt.current_job())

    def test_an_outer_job_is_kept(self):
        with lt.background_work("outer") as outer:
            with lt.call_scope("inner", True):
                self.assertIs(lt.current_job(), outer)

    def test_a_declared_background_call_reaches_the_gate(self):
        """The point of the exercise: the slot sees the job."""
        gate = lt.BackgroundGate(defer_reason=lambda: "turn",
                                 max_defer_s=0.2, hard_grace_s=0.0,
                                 poll_s=0.01, strict=lambda: True)
        out = {}

        def work():
            with lt.call_scope("overnight_ideas", True):
                with lt.slot(gate) as p:
                    out["outcome"] = p.outcome if p else None

        t = threading.Thread(target=work)
        t.start()
        t.join(5)
        self.assertEqual(out["outcome"], "forced")      # it waited, bounded

    def test_a_foreground_call_never_waits(self):
        gate = lt.BackgroundGate(defer_reason=lambda: "turn",
                                 max_defer_s=30.0, strict=lambda: True)
        out = {}

        def work():
            with lt.call_scope("voice_turn"):
                with lt.slot(gate) as p:
                    out["p"] = p

        t = threading.Thread(target=work)
        t.start()
        t.join(5)
        self.assertIsNone(out["p"])

    def test_strict_off_runs_at_once_and_says_so_once(self):
        lines = []
        gate = lt.BackgroundGate(defer_reason=lambda: "conversation",
                                 max_defer_s=30.0, strict=lambda: False,
                                 log=lines.append)
        out = {}

        def work():
            with lt.call_scope("screen_remark", True):
                for _ in range(2):
                    with lt.slot(gate) as p:
                        out.setdefault("o", []).append(p.outcome)

        t = threading.Thread(target=work)
        t.start()
        t.join(5)
        self.assertEqual(out["o"], ["shadow", "shadow"])
        self.assertEqual(len(lines), 1)
        self.assertIn("screen_remark", lines[0])


class TestLedger(unittest.TestCase):
    def test_counts_by_caller_and_side(self):
        L = lt.CallLedger()
        L.note("voice_turn", False, "cloud", {"input": 100, "output": 20})
        L.note("voice_turn", False, "local", {"input": 50, "output": 5})
        L.note("learn_from_turn", True, "local", {"input": 7, "output": 3})
        snap = L.snapshot()
        self.assertEqual(snap["voice_turn"]["calls"], 2)
        self.assertEqual(snap["voice_turn"]["input"], 150)
        self.assertEqual(snap["voice_turn"]["route"], {"cloud": 1, "local": 1})
        self.assertTrue(snap["learn_from_turn"]["background"])
        tot = L.totals()
        self.assertEqual(tot["background"]["calls"], 1)
        self.assertEqual(tot["foreground"]["output"], 25)

    def test_same_name_both_sides_keeps_both_rows(self):
        L = lt.CallLedger()
        L.note("x", False, "cloud")
        L.note("x", True, "cloud")
        snap = L.snapshot()
        self.assertEqual(set(snap), {"x", "x [bg]"})

    def test_bounded_and_junk_proof(self):
        L = lt.CallLedger()
        for i in range(lt.CallLedger.MAX_CALLERS + 20):
            L.note(f"c{i}", False, "cloud", {"input": "junk", "output": -3})
        self.assertLessEqual(len(L.snapshot()), lt.CallLedger.MAX_CALLERS + 1)
        self.assertEqual(L.totals()["foreground"]["input"], 0)
        L.note(None, None, None, "not a dict")      # never raises


class TestCloudClient(unittest.TestCase):
    def setUp(self):
        lt.CALLS.reset()
        self.addCleanup(lt.CALLS.reset)
        # The month tally is its own tested module; keep it off the disk here.
        from core import llm_usage
        orig = llm_usage.note_usage
        llm_usage.note_usage = lambda: None
        self.addCleanup(setattr, llm_usage, "note_usage", orig)

    def test_create_message_foreground(self):
        c = _FakeClient()
        llm_client.create_message(c, purpose="voice", caller="voice_turn",
                                  model="claude-test", max_tokens=5,
                                  messages=[])
        sent = c.sent[0]
        self.assertNotIn("caller", sent)
        self.assertNotIn("background", sent)
        row = lt.CALLS.snapshot()["voice_turn"]
        self.assertFalse(row["background"])
        self.assertEqual((row["calls"], row["input"], row["output"]),
                         (1, 10, 5))
        self.assertEqual(row["route"], {"cloud": 1})

    def test_create_message_background(self):
        c = _FakeClient()
        seen = {}
        orig = c.messages.create

        def spy(**kw):
            job = lt.current_job()
            seen["job"] = (job.tag, job.opt_in) if job else None
            return orig(**kw)
        c.messages.create = spy
        llm_client.create_message(c, purpose="deep", caller="overnight_ideas",
                                  background=True, model="claude-test",
                                  max_tokens=5, messages=[])
        self.assertEqual(seen["job"], ("overnight_ideas", True))
        row = lt.CALLS.snapshot()["overnight_ideas"]
        self.assertTrue(row["background"])
        self.assertNotIn("background", c.sent[0])

    def test_untagged_call_is_named_untagged_foreground(self):
        c = _FakeClient()
        llm_client.create_message(c, purpose="quick", model="m",
                                  max_tokens=5, messages=[])
        self.assertFalse(lt.CALLS.snapshot()[lt.UNTAGGED]["background"])

    def test_a_job_thread_is_background_without_saying_so(self):
        c = _FakeClient()
        with lt.background_work("learn-batch"):
            llm_client.create_message(c, purpose="quick", model="m",
                                      max_tokens=5, messages=[])
        row = lt.CALLS.snapshot()["learn-batch"]
        self.assertTrue(row["background"])

    def test_the_log_line_names_the_caller(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            llm_client._log_cache_usage(_Usage(), "learn_batch", True)
            llm_client._log_cache_usage(_Usage(), "voice_turn", False)
            llm_client._log_cache_usage(_Usage())
        lines = buf.getvalue().splitlines()
        self.assertTrue(lines[0].endswith("caller=learn_batch bg"), lines[0])
        self.assertTrue(lines[1].endswith("caller=voice_turn fg"), lines[1])
        self.assertTrue(lines[2].endswith("cache_write=0"), lines[2])

    def test_complete_and_stream_text_carry_the_tag(self):
        fake = _FakeClient()
        orig = llm_client._client
        llm_client._client = lambda timeout, max_retries=1: fake
        self.addCleanup(setattr, llm_client, "_client", orig)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            text = llm_client.complete(model="m", messages=[], caller="email_draft")
        self.assertEqual(text, "ok")
        self.assertIn("caller=email_draft fg", buf.getvalue())
        self.assertNotIn("caller", fake.sent[0])
        self.assertEqual(lt.CALLS.snapshot()["email_draft"]["calls"], 1)

        class _Stream:
            text_stream = iter(["he", "llo"])

            def get_final_message(self):
                return _Msg("hello", _Usage(3, 2))

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        class _SM:
            def stream(self, **kw):
                self.kw = kw
                return _Stream()

        sm = _SM()
        fake2 = types.SimpleNamespace(messages=sm)
        llm_client._client = lambda timeout, max_retries=1: fake2
        with contextlib.redirect_stdout(io.StringIO()):
            out = llm_client.stream_text(model="m", messages=[],
                                         caller="voice_turn")
        self.assertEqual(out, "hello")
        self.assertNotIn("caller", sm.kw)
        row = lt.CALLS.snapshot()["voice_turn"]
        self.assertEqual((row["calls"], row["input"], row["output"]), (1, 3, 2))
        self.assertFalse(row["background"])

    def test_snapshot_helper_matches_the_ledger(self):
        c = _FakeClient()
        llm_client.create_message(c, purpose="quick", caller="x_y",
                                  background=True, model="m", max_tokens=1,
                                  messages=[])
        self.assertEqual(llm_client.caller_usage_snapshot(),
                         lt.CALLS.snapshot())


if __name__ == "__main__":
    unittest.main()
