"""Per-model request shaping for the Claude 5.5 generation (core.llm_client).

2026-10-01: CLAUDE_MODEL moved to claude-sonnet-5-5, the deep jobs to
claude-opus-5-5. Those models think by default (thinking counts toward
max_tokens), take effort via output_config (sent in extra_body on SDK 0.76),
reject non-default sampling params and forced tool_choice, can open a reply
with a thinking block, and decline with HTTP 200 + stop_reason "refusal".

These tests pin, with NO network (every client is a fake):
  * the effort / max_tokens rules per model family and purpose,
  * that no temperature / top_p / top_k ever reaches a 5.x request,
  * type-safe text extraction (thinking-first replies) and the refusal /
    empty-reply failure path,
  * that unknown / older models get no extra fields,
  * that EVERY production Claude call site goes through the helper (an AST
    audit, with blindness floors so a broken walker can't pass green),
  * the shipped model defaults and the catalog / Settings model lists.
"""
from __future__ import annotations

import ast
import os
import types
import unittest
from unittest import mock

import core.llm_client as llm

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_CURRENT_GEN = ("claude-sonnet-5-5", "claude-opus-5-5", "claude-fable-5-1")
_FIVE_X = _CURRENT_GEN + ("claude-sonnet-5", "claude-opus-5", "claude-fable-5")
_NO_EXTRAS = ("claude-haiku-4-5", "claude-haiku-4-5-20251001",
              "claude-sonnet-4-6", "claude-sonnet-4-5", "claude-opus-4-8",
              "claude-opus-4-6", "claude-test-model", "some-unlisted-model",
              "", None)


def _effort(opts: dict):
    return ((opts.get("extra_body") or {}).get("output_config") or {}).get("effort")


# ── fakes ─────────────────────────────────────────────────────────────────

def _block(btype, **fields):
    return types.SimpleNamespace(type=btype, **fields)


def _msg(*blocks, stop_reason="end_turn", stop_details=None):
    return types.SimpleNamespace(content=list(blocks), stop_reason=stop_reason,
                                 stop_details=stop_details, usage=None)


class _FakeMessages:
    def __init__(self, reply=None, chunks=(), final=None):
        self.calls: list[dict] = []
        self._reply = reply if reply is not None else _msg(
            _block("text", text="ok"))
        self._chunks = list(chunks)
        self._final = final

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._reply

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        outer = self

        class _S:
            text_stream = list(outer._chunks)

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def get_final_message(self):
                return outer._final
        return _S()


class _FakeClient:
    def __init__(self, **kw):
        self.messages = _FakeMessages(**kw)


# ── request_options: the rules per model ─────────────────────────────────

class RequestOptionsTests(unittest.TestCase):
    def test_sonnet_5_5_voice_is_effort_low_with_a_2048_floor(self):
        opts = llm.request_options("claude-sonnet-5-5", "voice", 500)
        self.assertEqual(_effort(opts), "low")
        self.assertEqual(opts["max_tokens"], 2048)

    def test_floor_never_lowers_a_bigger_cap(self):
        opts = llm.request_options("claude-sonnet-5-5", "voice", 4000)
        self.assertNotIn("max_tokens", opts)
        self.assertEqual(_effort(opts), "low")

    def test_opus_5_5_deep_is_effort_medium_with_a_16k_floor(self):
        opts = llm.request_options("claude-opus-5-5", "deep", 2048)
        self.assertEqual(_effort(opts), "medium")
        self.assertEqual(opts["max_tokens"], 16000)

    def test_every_latency_bound_purpose_is_effort_low(self):
        # Sonnet 5.5 at its default effort (high) is ~13 s to the first answer
        # token; at low ~1.2 s. Everything the owner waits on must be low.
        for purpose in ("voice", "quick", "classify", "vision", "compose",
                        "plan", "ping"):
            for model in _CURRENT_GEN:
                with self.subTest(model=model, purpose=purpose):
                    self.assertEqual(
                        _effort(llm.request_options(model, purpose, 100)), "low")

    def test_only_deep_gets_medium(self):
        mediums = [p for p in llm.PURPOSES
                   if _effort(llm.request_options("claude-opus-5-5", p)) != "low"]
        self.assertEqual(mediums, ["deep"])

    def test_planner_floor(self):
        self.assertEqual(
            llm.request_options("claude-sonnet-5-5", "plan", 600)["max_tokens"],
            4096)

    def test_ping_keeps_the_callers_tiny_cap(self):
        opts = llm.request_options("claude-sonnet-5-5", "ping", 1)
        self.assertNotIn("max_tokens", opts)
        self.assertEqual(_effort(opts), "low")

    def test_fable_5_1_gets_effort(self):
        self.assertEqual(_effort(llm.request_options("claude-fable-5-1", "voice")),
                         "low")

    def test_dated_snapshot_and_case_resolve_to_the_family(self):
        for mid in ("claude-sonnet-5-5-20260928", "Claude-Sonnet-5-5",
                    " claude-sonnet-5-5 ", "anthropic.claude-sonnet-5-5"):
            with self.subTest(mid=mid):
                self.assertEqual(_effort(llm.request_options(mid, "voice")), "low")

    def test_prefix_collision_sonnet_5_is_not_sonnet_5_5(self):
        # claude-sonnet-5-5 starts with claude-sonnet-5: the older model must
        # NOT pick up 5.5's effort — only the thinking floor.
        for mid in ("claude-sonnet-5", "claude-opus-5", "claude-fable-5"):
            with self.subTest(mid=mid):
                opts = llm.request_options(mid, "voice", 500)
                self.assertNotIn("extra_body", opts)
                self.assertEqual(opts["max_tokens"], 2048)

    def test_unknown_and_older_models_get_nothing_extra(self):
        for mid in _NO_EXTRAS:
            for purpose in llm.PURPOSES:
                with self.subTest(mid=mid, purpose=purpose):
                    self.assertEqual(llm.request_options(mid, purpose, 8), {})

    def test_unknown_purpose_fails_loudly(self):
        with self.assertRaises(ValueError):
            llm.request_options("claude-sonnet-5-5", "chitchat")

    def test_tables_cover_every_purpose(self):
        for purpose in llm.PURPOSES:
            self.assertIn(purpose, llm._EFFORT_BY_PURPOSE)
            self.assertIn(purpose, llm._MAX_TOKENS_FLOOR)
            self.assertIn(llm._EFFORT_BY_PURPOSE[purpose],
                          ("low", "medium", "high", "xhigh", "max"))

    def test_haiku_never_gets_output_config(self):
        # Haiku 4.5 does not take effort; sending output_config would risk a 400.
        for purpose in llm.PURPOSES:
            self.assertNotIn("extra_body",
                             llm.request_options("claude-haiku-4-5", purpose))

    def test_forced_tool_choice_rejection_table(self):
        for mid in _CURRENT_GEN:
            self.assertTrue(llm.rejects_forced_tool_choice(mid), mid)
        for mid in ("claude-sonnet-5", "claude-haiku-4-5", "claude-opus-4-8", ""):
            self.assertFalse(llm.rejects_forced_tool_choice(mid), mid)


# ── build_request: merging + the sampling-param strip ────────────────────

class BuildRequestTests(unittest.TestCase):
    def test_caller_kwargs_pass_through(self):
        req = llm.build_request(purpose="voice", model="claude-sonnet-5-5",
                                max_tokens=500, system="sys",
                                messages=[{"role": "user", "content": "hi"}],
                                timeout=7.0)
        self.assertEqual(req["model"], "claude-sonnet-5-5")
        self.assertEqual(req["system"], "sys")
        self.assertEqual(req["timeout"], 7.0)
        self.assertEqual(req["max_tokens"], 2048)
        self.assertEqual(req["extra_body"], {"output_config": {"effort": "low"}})

    def test_no_sampling_params_ever_reach_a_5x_model(self):
        for mid in _FIVE_X + ("claude-opus-4-8", "claude-opus-4-7",
                              "claude-sonnet-5-5-20260928"):
            with self.subTest(mid=mid):
                req = llm.build_request(
                    purpose="voice", model=mid, max_tokens=10, messages=[],
                    temperature=0.0, top_p=0.9, top_k=40,
                    extra_body={"temperature": 0.2})
                for p in ("temperature", "top_p", "top_k"):
                    self.assertNotIn(p, req)
                    self.assertNotIn(p, req.get("extra_body") or {})

    def test_unknown_model_has_sampling_stripped_too(self):
        req = llm.build_request(purpose="voice", model="claude-sonnet-9",
                                max_tokens=10, messages=[], temperature=0.0)
        self.assertNotIn("temperature", req)

    def test_models_that_accept_sampling_keep_it(self):
        for mid in ("claude-haiku-4-5", "claude-haiku-4-5-20251001",
                    "claude-sonnet-4-6", "claude-opus-4-6"):
            with self.subTest(mid=mid):
                req = llm.build_request(purpose="classify", model=mid,
                                        max_tokens=8, messages=[],
                                        temperature=0.0)
                self.assertEqual(req["temperature"], 0.0)
                self.assertEqual(req["max_tokens"], 8)
                self.assertNotIn("extra_body", req)

    def test_unknown_model_request_is_otherwise_unchanged(self):
        kw = {"model": "claude-test", "max_tokens": 123, "system": "s",
              "messages": [{"role": "user", "content": "x"}]}
        self.assertEqual(llm.build_request(purpose="voice", **kw), kw)

    def test_caller_extra_body_is_merged_and_explicit_effort_wins(self):
        req = llm.build_request(
            purpose="voice", model="claude-sonnet-5-5", max_tokens=10,
            messages=[], extra_body={"metadata": {"k": "v"},
                                     "output_config": {"effort": "high"}})
        self.assertEqual(req["extra_body"]["metadata"], {"k": "v"})
        self.assertEqual(req["extra_body"]["output_config"]["effort"], "high")

    def test_create_message_sends_the_shaped_request(self):
        client = _FakeClient()
        llm.create_message(client, purpose="deep", model="claude-opus-5-5",
                           max_tokens=2048, messages=[], temperature=0.0)
        sent = client.messages.calls[-1]
        self.assertEqual(sent["extra_body"]["output_config"]["effort"], "medium")
        self.assertEqual(sent["max_tokens"], 16000)
        self.assertNotIn("temperature", sent)
        self.assertNotIn("purpose", sent)   # our routing key never hits the API


# ── response_text: type-safe extraction + refusal ─────────────────────────

class ResponseTextTests(unittest.TestCase):
    def test_thinking_block_first_returns_the_text(self):
        msg = _msg(_block("thinking", thinking="", signature="sig"),
                   _block("text", text="Good evening, sir."))
        self.assertEqual(llm.response_text(msg), "Good evening, sir.")

    def test_redacted_thinking_and_tool_use_are_skipped(self):
        msg = _msg(_block("redacted_thinking", data="xx"),
                   _block("tool_use", id="t", name="n", input={}, text="NO"),
                   _block("text", text="A"), _block("text", text="B"))
        self.assertEqual(llm.response_text(msg), "AB")

    def test_refusal_raises_even_with_partial_text(self):
        msg = _msg(_block("text", text="I can"), stop_reason="refusal",
                   stop_details={"category": "cyber"})
        with self.assertRaises(llm.CloudRefusalError) as cm:
            llm.response_text(msg)
        self.assertEqual(cm.exception.category, "cyber")
        self.assertEqual(cm.exception.stop_reason, "refusal")

    def test_refusal_with_no_content(self):
        with self.assertRaises(llm.CloudRefusalError):
            llm.response_text(_msg(stop_reason="refusal"))

    def test_thinking_ate_the_whole_budget_raises_empty(self):
        msg = _msg(_block("thinking", thinking="", signature="s"),
                   stop_reason="max_tokens")
        with self.assertRaises(llm.CloudEmptyReplyError):
            llm.response_text(msg)

    def test_failures_are_runtime_errors_so_except_exception_falls_back(self):
        self.assertTrue(issubclass(llm.CloudRefusalError, llm.CloudReplyError))
        self.assertTrue(issubclass(llm.CloudEmptyReplyError, llm.CloudReplyError))
        self.assertTrue(issubclass(llm.CloudReplyError, RuntimeError))

    def test_hand_rolled_blocks_without_a_type_still_read(self):
        class Blk:
            def __init__(self, t):
                self.text = t
        msg = types.SimpleNamespace(content=[Blk("hi"), Blk(None), object()])
        self.assertEqual(llm.response_text(msg), "hi")

    def test_mock_stop_reason_is_not_a_refusal(self):
        msg = mock.MagicMock()
        msg.content = [_block("text", text="fine")]
        self.assertEqual(llm.response_text(msg), "fine")

    def test_real_sdk_types(self):
        try:
            from anthropic.types import Message, TextBlock, ThinkingBlock, Usage
        except Exception:  # pragma: no cover - SDK is in the CI dep list
            self.skipTest("anthropic SDK not installed")
        msg = Message.model_construct(
            id="msg_x", type="message", role="assistant",
            model="claude-sonnet-5-5",
            content=[ThinkingBlock.model_construct(type="thinking", thinking="",
                                                   signature="sig"),
                     TextBlock.model_construct(type="text", text="Ready, sir.",
                                               citations=None)],
            stop_reason="end_turn", stop_sequence=None,
            usage=Usage.model_construct(input_tokens=1, output_tokens=1))
        self.assertEqual(llm.response_text(msg), "Ready, sir.")
        refused = Message.model_construct(
            id="msg_y", type="message", role="assistant",
            model="claude-sonnet-5-5", content=[], stop_reason="refusal",
            stop_sequence=None,
            usage=Usage.model_construct(input_tokens=1, output_tokens=0))
        with self.assertRaises(llm.CloudRefusalError):
            llm.response_text(refused)


# ── complete() / stream_text() end to end on a fake client ───────────────

class CompleteAndStreamTests(unittest.TestCase):
    def _patch(self, client):
        return mock.patch.object(llm, "_client", return_value=client)

    def test_complete_sends_effort_and_reads_past_thinking(self):
        client = _FakeClient(reply=_msg(_block("thinking", thinking="", signature="s"),
                                        _block("text", text="Hello, sir.")))
        with self._patch(client):
            out = llm.complete(model="claude-sonnet-5-5", max_tokens=500,
                               messages=[{"role": "user", "content": "hi"}])
        self.assertEqual(out, "Hello, sir.")
        sent = client.messages.calls[-1]
        self.assertEqual(sent["extra_body"], {"output_config": {"effort": "low"}})
        self.assertEqual(sent["max_tokens"], 2048)

    def test_complete_refusal_raises(self):
        client = _FakeClient(reply=_msg(stop_reason="refusal"))
        with self._patch(client), self.assertRaises(llm.CloudRefusalError):
            llm.complete(model="claude-sonnet-5-5", messages=[])

    def test_complete_purpose_is_honoured(self):
        client = _FakeClient()
        with self._patch(client):
            llm.complete(model="claude-opus-5-5", messages=[], purpose="deep")
        self.assertEqual(
            client.messages.calls[-1]["extra_body"]["output_config"]["effort"],
            "medium")

    def test_stream_sends_effort_and_returns_text(self):
        client = _FakeClient(chunks=["Hel", "lo"],
                             final=_msg(_block("text", text="Hello")))
        with self._patch(client):
            out = llm.stream_text(model="claude-sonnet-5-5", messages=[])
        self.assertEqual(out, "Hello")
        self.assertEqual(client.messages.calls[-1]["extra_body"],
                         {"output_config": {"effort": "low"}})

    def test_stream_refusal_raises(self):
        client = _FakeClient(chunks=["I can"], final=_msg(stop_reason="refusal"))
        with self._patch(client), self.assertRaises(llm.CloudRefusalError):
            llm.stream_text(model="claude-sonnet-5-5", messages=[])

    def test_stream_with_no_text_raises_empty(self):
        client = _FakeClient(chunks=[], final=_msg(stop_reason="max_tokens"))
        with self._patch(client), self.assertRaises(llm.CloudEmptyReplyError):
            llm.stream_text(model="claude-sonnet-5-5", messages=[])

    def test_unknown_model_stream_request_unchanged(self):
        client = _FakeClient(chunks=["x"])
        with self._patch(client):
            llm.stream_text(model="m", messages=[], max_tokens=77)
        sent = client.messages.calls[-1]
        self.assertEqual(sent["max_tokens"], 77)
        self.assertNotIn("extra_body", sent)


# ── every production call site goes through the helper (AST audit) ───────

_SKIP_DIRS = {".git", "tests", "backups", "__pycache__", ".venv", "venv",
              "node_modules", "staging", ".claude"}


def _production_py_files():
    for dirpath, dirnames, filenames in os.walk(_ROOT):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS
                       and not d.startswith(".")]
        for fn in filenames:
            if fn.endswith(".py"):
                yield os.path.join(dirpath, fn)


def _parse(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        src = f.read()
    try:
        return ast.parse(src, filename=path)
    except SyntaxError:
        return None


def _call_name(node: ast.Call) -> str:
    f = node.func
    if isinstance(f, ast.Attribute):
        return f.attr
    if isinstance(f, ast.Name):
        return f.id
    return ""


def _rel(path):
    return os.path.relpath(path, _ROOT).replace(os.sep, "/")


class CallSiteAuditTests(unittest.TestCase):
    """No production file may call ``<x>.messages.create(`` / ``.stream(``
    except core/llm_client.py — every other call goes through
    create_message / stream_message / complete / stream_text (or the
    monolith's _claude_create), which apply the per-model shaping."""

    @classmethod
    def setUpClass(cls):
        cls.trees = {}
        for p in _production_py_files():
            t = _parse(p)
            if t is not None:
                cls.trees[_rel(p)] = t

    def test_walker_sees_the_tree(self):
        # Blindness floor: a broken walker must not pass green.
        for must in ("bobert_companion.py", "core/llm_client.py",
                     "core/orchestrator.py", "skills/email_triage.py",
                     "overnight_upgrade.py"):
            self.assertIn(must, self.trees)
        self.assertGreater(len(self.trees), 50)

    def test_no_direct_messages_create_or_stream_outside_llm_client(self):
        offenders = []
        for rel, tree in self.trees.items():
            if rel == "core/llm_client.py":
                continue
            for node in ast.walk(tree):
                if (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr in ("create", "stream", "count_tokens")
                        and isinstance(node.func.value, ast.Attribute)
                        and node.func.value.attr == "messages"):
                    offenders.append(f"{rel}:{node.lineno}")
        self.assertEqual(offenders, [], "direct Claude calls bypass "
                         "core.llm_client's per-model shaping")

    def test_llm_client_itself_has_exactly_the_two_choke_points(self):
        n = 0
        for node in ast.walk(self.trees["core/llm_client.py"]):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr in ("create", "stream")
                    and isinstance(node.func.value, ast.Attribute)
                    and node.func.value.attr == "messages"):
                n += 1
        self.assertEqual(n, 2)

    def test_no_content_zero_text_reads(self):
        # msg.content[0].text is wrong on a thinking model (block 0 can be a
        # thinking block). Production code reads by type via response_text.
        offenders = []
        for rel, tree in self.trees.items():
            for node in ast.walk(tree):
                if (isinstance(node, ast.Attribute) and node.attr == "text"
                        and isinstance(node.value, ast.Subscript)
                        and isinstance(node.value.value, ast.Attribute)
                        and node.value.value.attr == "content"):
                    offenders.append(f"{rel}:{node.lineno}")
        self.assertEqual(offenders, [])

    def _purpose_calls(self):
        """(file, line, purpose-or-None) for every helper call."""
        out = []
        for rel, tree in self.trees.items():
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                name = _call_name(node)
                purpose = "<missing>"
                if name in ("create_message", "stream_message", "_claude_call"):
                    for kw in node.keywords:
                        if kw.arg != "purpose":
                            continue
                        if isinstance(kw.value, ast.Name) and kw.value.id == "purpose":
                            # A thin wrapper forwarding its OWN purpose
                            # parameter (_claude_create, orchestrator._claude_call)
                            # — its callers are audited instead.
                            purpose = "<passthrough>"
                        else:
                            purpose = getattr(kw.value, "value", "<non-literal>")
                elif name == "_claude_create":
                    if node.args and isinstance(node.args[0], ast.Constant):
                        purpose = node.args[0].value
                elif (name in ("complete", "stream_text")
                      and isinstance(node.func, ast.Attribute)
                      and isinstance(node.func.value, ast.Name)
                      and node.func.value.id == "_llm_client"):
                    for kw in node.keywords:
                        if kw.arg == "purpose":
                            purpose = getattr(kw.value, "value", "<non-literal>")
                else:
                    continue
                out.append((rel, node.lineno, name, purpose))
        return out

    def test_every_helper_call_names_a_known_purpose(self):
        bad = []
        for rel, line, name, purpose in self._purpose_calls():
            if rel == "core/llm_client.py":
                continue   # the helpers' own internal plumbing
            if purpose == "<passthrough>" and (rel, name) in (
                    ("bobert_companion.py", "create_message"),
                    ("core/orchestrator.py", "create_message")):
                continue
            if purpose not in llm.PURPOSES:
                bad.append(f"{rel}:{line} {name} purpose={purpose!r}")
        self.assertEqual(bad, [])

    def test_former_call_sites_all_route_through_the_helper(self):
        calls = self._purpose_calls()
        by_file: dict[str, list] = {}
        for rel, _line, name, purpose in calls:
            by_file.setdefault(rel, []).append((name, purpose))
        # The monolith: _llm_quick, _claude_oneshot, the _call_llm direct path,
        # two screen-vision calls, the follow-up round, the boot ping.
        mono = [p for n, p in by_file.get("bobert_companion.py", [])
                if n == "_claude_create"]
        self.assertEqual(sorted(mono), sorted(
            ["quick", "voice", "voice", "vision", "vision", "voice", "ping"]))
        # The monolith's _llm_client.complete/stream_text calls say "voice".
        wrapped = [p for n, p in by_file.get("bobert_companion.py", [])
                   if n in ("complete", "stream_text")]
        self.assertGreaterEqual(len(wrapped), 5)
        self.assertEqual(set(wrapped), {"voice"})
        expect = {
            "core/orchestrator.py": {"plan", "quick", "compose"},
            "core/diagnostic_daemons.py": {"deep"},
            "skills/email_triage.py": {"classify", "compose"},
            "skills/news_briefing.py": {"quick"},
            "skills/notification_triage.py": {"classify"},
            "skills/phone_bridge.py": {"voice"},
            "skills/self_diagnostic.py": {"ping"},
            "overnight_upgrade.py": {"deep"},
        }
        for rel, purposes in expect.items():
            with self.subTest(rel=rel):
                got = {p for _n, p in by_file.get(rel, [])}
                self.assertTrue(purposes <= got, f"{rel}: {got}")

    def test_no_temperature_keyword_on_any_claude_construction(self):
        # browser_agent used to build ChatAnthropic(temperature=0.0); browser-
        # use forwards it to messages.create and 5.x models 400 on it.
        tree = self.trees["skills/browser_agent.py"]
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                keys = [getattr(k, "value", None) for k in node.keys]
                self.assertNotIn("temperature", keys,
                                 f"browser_agent.py:{node.lineno}")


# ── shipped defaults + model lists ───────────────────────────────────────

class ShippedDefaultsTests(unittest.TestCase):
    def test_config_defaults(self):
        import core.config as cfg
        self.assertEqual(cfg.CLAUDE_MODEL, "claude-sonnet-5-5")
        self.assertEqual(cfg.SCREEN_VISION_MODEL, "claude-sonnet-5-5")
        self.assertEqual(cfg.ORCHESTRATOR_PLANNER_MODEL, "claude-sonnet-5-5")
        self.assertEqual(cfg.ORCHESTRATOR_MERGER_MODEL, "claude-sonnet-5-5")
        # 2026-10-02: the workers' blank default follows CLAUDE_FAST_MODEL,
        # the one place the Haiku id is written.
        self.assertEqual(cfg.ORCHESTRATOR_WORKER_MODEL, "")
        self.assertEqual(cfg.CLAUDE_FAST_MODEL, "claude-haiku-4-5")

    def test_orchestrator_defaults_match_config(self):
        import core.config as cfg
        import core.orchestrator as orch
        self.assertEqual(orch.DEFAULT_PLANNER_MODEL, cfg.ORCHESTRATOR_PLANNER_MODEL)
        self.assertEqual(orch.DEFAULT_MERGER_MODEL, cfg.ORCHESTRATOR_MERGER_MODEL)
        self.assertEqual(orch.DEFAULT_WORKER_MODEL, cfg.ORCHESTRATOR_WORKER_MODEL)

    def test_no_voice_path_default_is_opus(self):
        # Opus 5.5 always thinks: ~13 s to the first answer token even at effort
        # low, ~22 s at medium. Never a default for anything the owner waits on.
        import core.config as cfg
        for name in ("CLAUDE_MODEL", "SCREEN_VISION_MODEL",
                     "ORCHESTRATOR_PLANNER_MODEL", "ORCHESTRATOR_MERGER_MODEL",
                     "ORCHESTRATOR_WORKER_MODEL", "CLAUDE_FAST_MODEL"):
            self.assertNotIn("opus", getattr(cfg, name), name)

    def test_deep_jobs_default_to_opus_5_5(self):
        import core.diagnostic_daemons as dd
        if not os.environ.get("JARVIS_DEEP_AUDIT_MODEL"):
            self.assertEqual(dd.DEEP_AUDIT_MODEL, "claude-opus-5-5")
        # Opus 5.5 at medium is ~22 s to the first answer token: the per-attempt
        # timeout must leave room (and stay bounded).
        self.assertGreaterEqual(dd.DEEP_AUDIT_TIMEOUT_S, 90)
        self.assertLessEqual(dd.DEEP_AUDIT_TIMEOUT_S, 120)
        tree = _parse(os.path.join(_ROOT, "overnight_upgrade.py"))
        vals = {t.id: n.value.value for n in tree.body
                if isinstance(n, ast.Assign) for t in n.targets
                if isinstance(t, ast.Name) and isinstance(n.value, ast.Constant)}
        self.assertEqual(vals.get("IDEAS_API_MODEL"), "claude-opus-5-5")

    def test_no_legacy_sonnet_4_5_in_production(self):
        offenders = []
        for p in _production_py_files():
            tree = _parse(p)
            if tree is None:
                continue
            for node in ast.walk(tree):
                if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                        and node.value.strip() == "claude-sonnet-4-5"
                        and _rel(p) != "core/llm_client.py"):
                    offenders.append(f"{_rel(p)}:{node.lineno}")
        self.assertEqual(offenders, [])

    def test_haiku_id_is_written_only_in_config_and_model_tables(self):
        # 2026-10-02: a Haiku swap (its tentative retirement is "not sooner
        # than" 2026-10-15) must be ONE settings line — CLAUDE_FAST_MODEL.
        # Allowed besides core/config.py: the model tables that describe
        # models rather than pick one (the price catalog, the per-model
        # request rules, the Settings model picker's choices). Tracked files
        # only: an owner's gitignored personal skill is not this repo's.
        import subprocess
        allowed = {"core/config.py", "core/model_catalog.py",
                   "core/llm_client.py", "tools/settings_window.py"}
        try:
            listed = subprocess.run(
                ["git", "-C", _ROOT, "ls-files", "*.py"], capture_output=True,
                text=True, timeout=30, check=True).stdout.split()
        except Exception as e:  # pragma: no cover - git is on every runner
            self.skipTest(f"git ls-files unavailable: {e}")
        tracked = [p for p in listed if not p.startswith("tests/")]
        self.assertGreater(len(tracked), 50)        # the scan saw the tree
        offenders = []
        for rel in tracked:
            tree = _parse(os.path.join(_ROOT, rel))
            if tree is None:
                continue
            for node in ast.walk(tree):
                if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                        and node.value.strip().startswith("claude-haiku-4-5")
                        and rel not in allowed):
                    offenders.append(f"{rel}:{node.lineno}")
        self.assertEqual(offenders, [])

    def test_catalog_prices_and_prefix_collision(self):
        import core.model_catalog as mc
        rows = {"claude-sonnet-5-5": ("Claude Sonnet 5.5", 2.0, 10.0),
                "claude-opus-5-5": ("Claude Opus 5.5", 4.0, 20.0),
                "claude-fable-5-1": ("Claude Fable 5.1", 10.0, 50.0),
                "claude-sonnet-5": ("Claude Sonnet 5", 2.0, 10.0),
                "claude-haiku-4-5": ("Claude Haiku", 1.0, 5.0)}
        for mid, (label, i, o) in rows.items():
            with self.subTest(mid=mid):
                m = mc.by_id(mid)
                self.assertEqual((m.label, m.in_price, m.out_price), (label, i, o))
        # Longest prefix wins: a dated 5.5 snapshot is NOT priced as Sonnet 5.
        self.assertEqual(mc.by_id("claude-sonnet-5-5-20260928").id,
                         "claude-sonnet-5-5")
        self.assertEqual(mc.by_id("claude-sonnet-5-20260601").id, "claude-sonnet-5")

    def test_settings_choices_match_the_catalog(self):
        import core.model_catalog as mc
        from tools import settings_window as sw
        spec = sw.SCHEMA["CLAUDE_MODEL"]
        self.assertEqual(spec["default"], "claude-sonnet-5-5")
        choices = set(spec["choices"])
        for mid in ("claude-sonnet-5-5", "claude-opus-5-5", "claude-fable-5-1",
                    "claude-haiku-4-5", "claude-sonnet-5", "claude-opus-4-8"):
            self.assertIn(mid, choices)
        catalog_claude = {m.id for m in mc.catalog() if m.backend == "claude"}
        self.assertEqual(choices, catalog_claude)
        for mid in choices:   # every selectable model is shaped without error
            for purpose in llm.PURPOSES:
                llm.request_options(mid, purpose, 1)


if __name__ == "__main__":
    unittest.main()
