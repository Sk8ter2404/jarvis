"""Unit tests for tools/multi_agent_pipeline.py - the STAGE LIBRARY of the safe
self-upgrade (planner / implementer / reviewer, the Claude wrapper with its
permission lock-down, the risk + budget helpers). The orchestration (worktree,
one commit per task, the gate, the report) is tested in
tests/test_selfup_runner.py.

SAFETY CONTRACT (this suite NEVER runs a real upgrade):
  * No real `claude` CLI / git subprocess work. The module shells out to the
    `claude` CLI via subprocess.run inside `_invoke_claude`, so the LLM is
    mocked by patching `P.subprocess.run` for the low-level wrapper tests and
    `P._invoke_claude` for the stage tests.
  * No real network / LLM / threads.
  * No writes to real source files - PROJECT_DIR and PIPELINE_LOG are
    redirected into a per-test tempdir and restored via addCleanup.

CI-FAITHFUL: multi_agent_pipeline imports only stdlib at module top, so this
suite RUNS under tools/run_tests_ci_sim.py on a bare Linux runner.

PRIVACY: fixtures use only fake identifiers. Any API-key-looking string is
built at runtime via concatenation so the repo's PII gate never trips.
"""
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

import tools.multi_agent_pipeline as P


_IS_WIN = sys.platform.startswith("win")


# ─────────────────────────── shared fakes ────────────────────────────────


class _FakeCompleted:
    """Stand-in for subprocess.CompletedProcess."""

    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _api_key_lookalike() -> str:
    """Build an sk-... looking string at RUNTIME so the literal never appears
    in the test source (the repo's PII gate greps tests/ on CI)."""
    return "sk-" + "x" * 48


# ─────────────────────────── base test case ──────────────────────────────


class _PipeBase(unittest.TestCase):
    """Redirect every module path-constant in multi_agent_pipeline into a fresh
    tempdir for the duration of one test, and stub time.sleep. All overrides
    are restored via addCleanup so the real module globals (and the developer's
    real project tree) are untouched."""

    _PATH_ATTRS = ("PROJECT_DIR", "PIPELINE_LOG")

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp = self._tmpdir.name
        self.addCleanup(self._tmpdir.cleanup)

        self._saved = {a: getattr(P, a) for a in self._PATH_ATTRS}
        self.addCleanup(self._restore_consts)

        p = self.tmp
        P.PROJECT_DIR           = p
        P.PIPELINE_LOG          = os.path.join(p, "data", "pipeline_runs.jsonl")

        # never really sleep
        self._sleep = mock.patch.object(P.time, "sleep", lambda *a, **k: None)
        self._sleep.start()
        self.addCleanup(self._sleep.stop)

        # keep the pipeline env-config deterministic regardless of the host's
        # real environment.
        self._env = mock.patch.dict(os.environ, {}, clear=False)
        self._env.start()
        self.addCleanup(self._env.stop)
        for k in ("JARVIS_PIPELINE_PLANNER_MODEL",
                  "JARVIS_PIPELINE_IMPLEMENTER_MODEL",
                  "JARVIS_PIPELINE_REVIEWER_MODEL",
                  "JARVIS_PIPELINE_MAX_USD"):
            os.environ.pop(k, None)

        # The Claude-account choice (core/claude_accounts) reads the owner's
        # settings and the live state file: pin it to the default profile so
        # these tests see one call per invocation on any machine
        # (tests/test_claude_accounts.py covers the Pro-first path).
        from core import claude_accounts as _ca
        self._acct = mock.patch.object(_ca, "choose",
                                       return_value=(_ca.MAX, "pinned by the test"))
        self._acct.start()
        self.addCleanup(self._acct.stop)

    def _restore_consts(self):
        for a, v in self._saved.items():
            setattr(P, a, v)

    # -- small filesystem helpers (paths relative to the tempdir) --
    def write(self, relpath, text):
        full = os.path.join(self.tmp, relpath)
        os.makedirs(os.path.dirname(full) or self.tmp, exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(text)
        return full

    def read(self, relpath):
        with open(os.path.join(self.tmp, relpath), encoding="utf-8") as f:
            return f.read()

    def write_todo(self, text):
        return self.write("jarvis_todo.md", text)


# ──────────────────────────── rate-limit sniff ────────────────────────────


class LooksRateLimitedTests(unittest.TestCase):
    def test_each_needle_matches(self):
        for needle in P._RATE_LIMIT_NEEDLES:
            self.assertTrue(P._looks_rate_limited(needle, ""),
                            f"needle not detected in stdout: {needle!r}")
            self.assertTrue(P._looks_rate_limited("", needle),
                            f"needle not detected in stderr: {needle!r}")

    def test_case_insensitive(self):
        self.assertTrue(P._looks_rate_limited("USAGE LIMIT REACHED", ""))
        self.assertTrue(P._looks_rate_limited("Rate_Limit_Error", ""))

    def test_clean_output_not_flagged(self):
        self.assertFalse(P._looks_rate_limited(
            "DONE: implemented the feature cleanly", "no errors"))

    def test_removed_needle_not_flagged(self):
        # The comment documents that "please try again later" was deliberately
        # removed because it matched organic prose. Verify it no longer trips.
        self.assertFalse(P._looks_rate_limited(
            "I hit a snag, please try again later with more context.", ""))

    def test_only_first_4kb_inspected(self):
        # Marker buried past 4 KB in BOTH streams must be missed (the function
        # only scans the head of each stream).
        buried = ("." * 5000) + "usage limit reached"
        self.assertFalse(P._looks_rate_limited(buried, buried))

    def test_none_inputs_safe(self):
        self.assertFalse(P._looks_rate_limited(None, None))  # type: ignore[arg-type]


# ──────────────────────────── env config helpers ──────────────────────────


class EnvConfigTests(_PipeBase):
    def test_model_defaults(self):
        self.assertEqual(P._planner_model(), "haiku")
        self.assertEqual(P._implementer_model(), "opus")
        self.assertEqual(P._reviewer_model(), "sonnet")

    def test_model_env_override(self):
        os.environ["JARVIS_PIPELINE_PLANNER_MODEL"] = "sonnet"
        os.environ["JARVIS_PIPELINE_IMPLEMENTER_MODEL"] = "opus-4"
        os.environ["JARVIS_PIPELINE_REVIEWER_MODEL"] = "haiku"
        self.assertEqual(P._planner_model(), "sonnet")
        self.assertEqual(P._implementer_model(), "opus-4")
        self.assertEqual(P._reviewer_model(), "haiku")

    def test_model_blank_env_falls_back(self):
        os.environ["JARVIS_PIPELINE_PLANNER_MODEL"] = "   "
        self.assertEqual(P._planner_model(), "haiku")

    def test_env_model_direct(self):
        self.assertEqual(P._env_model("NOPE_MISSING", "fallback"), "fallback")


# ──────────────────────────── log helper ──────────────────────────────────


class LogPipelineEventTests(_PipeBase):
    def test_writes_jsonl_line_with_ts(self):
        P._log_pipeline_event({"stage": "planner", "ok": True})
        text = self.read(os.path.join("data", "pipeline_runs.jsonl"))
        rec = json.loads(text.strip())
        self.assertEqual(rec["stage"], "planner")
        self.assertIn("ts", rec)  # auto-stamped

    def test_appends_multiple_lines(self):
        P._log_pipeline_event({"n": 1})
        P._log_pipeline_event({"n": 2})
        lines = [json.loads(x) for x in
                 self.read(os.path.join("data", "pipeline_runs.jsonl")).splitlines()]
        self.assertEqual([r["n"] for r in lines], [1, 2])

    def test_non_serializable_uses_default_str(self):
        # default=str means an exotic object is stringified, not raised on.
        P._log_pipeline_event({"obj": object()})
        text = self.read(os.path.join("data", "pipeline_runs.jsonl"))
        self.assertIn("object at 0x", text)

    def test_never_raises_on_write_failure(self):
        with mock.patch("builtins.open", side_effect=OSError("disk full")):
            P._log_pipeline_event({"stage": "x"})  # must swallow


# ──────────────────────────── todo helpers ────────────────────────────────


class StripJsonTests(unittest.TestCase):
    def test_fenced_json_block(self):
        raw = 'prose\n```json\n{"verdict": "approve"}\n```\nmore prose'
        self.assertEqual(P._strip_json(raw), '{"verdict": "approve"}')

    def test_fenced_block_without_lang(self):
        raw = '```\n{"a": 1}\n```'
        self.assertEqual(P._strip_json(raw), '{"a": 1}')

    def test_leading_and_trailing_prose_balanced_braces(self):
        raw = 'Here is the plan: {"files_to_touch": ["x.py"]} hope that helps'
        self.assertEqual(P._strip_json(raw), '{"files_to_touch": ["x.py"]}')

    def test_braces_inside_string_dont_fool_depth(self):
        raw = 'x {"approach": "use {placeholder} carefully"} y'
        out = P._strip_json(raw)
        self.assertEqual(json.loads(out)["approach"], "use {placeholder} carefully")

    def test_escaped_quote_inside_string(self):
        raw = r'{"note": "he said \"hi\" then left"}'
        out = P._strip_json(raw)
        self.assertEqual(json.loads(out)["note"], 'he said "hi" then left')

    def test_no_json_returns_original(self):
        self.assertEqual(P._strip_json("no braces here"), "no braces here")

    def test_unbalanced_returns_from_first_brace(self):
        # missing closing brace -> returns from the first '{' to end of string
        raw = 'prefix {"a": 1'
        self.assertEqual(P._strip_json(raw), '{"a": 1')

    def test_empty_input(self):
        self.assertEqual(P._strip_json(""), "")
        self.assertEqual(P._strip_json(None), "")  # type: ignore[arg-type]

    def test_nested_objects(self):
        raw = 'reply: {"outer": {"inner": {"deep": true}}} done'
        out = P._strip_json(raw)
        self.assertEqual(json.loads(out), {"outer": {"inner": {"deep": True}}})


# ──────────────────────────── claude CLI wrapper ──────────────────────────


class InvokeClaudeTests(_PipeBase):
    def _run(self, **over):
        defaults = dict(prompt="do x", model="haiku",
                        claude_path="/fake/claude", allow_writes=False,
                        project_dir=self.tmp)
        defaults.update(over)
        return P._invoke_claude(**defaults)

    def test_happy_path_returns_streams(self):
        with mock.patch.object(P.subprocess, "run",
                               return_value=_FakeCompleted(0, "out", "err")) as run:
            rc, out, err = self._run()
        self.assertEqual((rc, out, err), (0, "out", "err"))
        cmd = run.call_args.args[0]
        self.assertEqual(cmd[0], "/fake/claude")
        self.assertIn("--print", cmd)
        self.assertIn("--model", cmd)
        self.assertIn("haiku", cmd)
        # never the blanket bypass, in any role
        self.assertNotIn("--dangerously-skip-permissions", cmd)
        # prompt is passed positionally after the "--" terminator
        self.assertEqual(cmd[-1], "do x")
        self.assertEqual(cmd[-2], "--")

    @staticmethod
    def _flag(cmd, name):
        """The value after ``name`` in a claude argv."""
        return cmd[cmd.index(name) + 1]

    def test_planner_and_reviewer_are_read_only(self):
        # allow_writes=False is the planner / reviewer call: the tool set
        # itself is Read, Glob, Grep - nothing that can edit, run a command
        # or reach the network exists - and anything unlisted is refused.
        with mock.patch.object(P.subprocess, "run",
                               return_value=_FakeCompleted(0)) as run:
            self._run(allow_writes=False)
        cmd = run.call_args.args[0]
        self.assertEqual(self._flag(cmd, "--tools"), "Read,Glob,Grep")
        self.assertEqual(self._flag(cmd, "--allowedTools"), "Read,Glob,Grep")
        denied = self._flag(cmd, "--disallowedTools").split(",")
        for tool in ("Edit", "Write", "Bash", "WebFetch"):
            self.assertIn(tool, denied)
        self.assertEqual(self._flag(cmd, "--permission-mode"), "dontAsk")
        self.assertNotIn("--dangerously-skip-permissions", cmd)

    def test_implementer_is_locked_down_not_bypassed(self):
        with mock.patch.object(P.subprocess, "run",
                               return_value=_FakeCompleted(0)) as run:
            self._run(allow_writes=True)
        cmd = run.call_args.args[0]
        self.assertNotIn("--dangerously-skip-permissions", cmd)
        self.assertNotIn("bypassPermissions", cmd)
        self.assertEqual(self._flag(cmd, "--permission-mode"), "dontAsk")
        deny = self._flag(cmd, "--disallowedTools")
        for pattern in ("Bash(git push *)", "Bash(git tag *)", "Bash(git merge *)",
                        "Bash(git reset *)", "Bash(git checkout *)",
                        "Bash(git commit *)", "Edit(data/**)", "Write(data/**)",
                        "Edit(VERSION)", "WebFetch"):
            self.assertIn(pattern, deny)
        allow = self._flag(cmd, "--allowedTools")
        for pattern in ("Edit", "Write", "Bash(python -m py_compile *)"):
            self.assertIn(pattern, allow)
        # nothing that publishes or rewrites history is pre-approved
        for pattern in ("git push", "git commit", "git reset", "git checkout"):
            self.assertNotIn(pattern, allow)
        # the CLI's cwd is the directory it was given (the worktree)
        self.assertEqual(run.call_args.kwargs["cwd"], self.tmp)

    def test_child_env_drops_publishing_credentials(self):
        captured = {}

        def _run(cmd, **kw):
            captured["env"] = kw.get("env")
            return _FakeCompleted(0)

        with mock.patch.dict(os.environ, {"GH_TOKEN": "x", "GITHUB_TOKEN": "y"}),              mock.patch.object(P.subprocess, "run", side_effect=_run):
            self._run(allow_writes=True)
        self.assertNotIn("GH_TOKEN", captured["env"])
        self.assertNotIn("GITHUB_TOKEN", captured["env"])

    def test_extra_args_appended_before_terminator(self):
        with mock.patch.object(P.subprocess, "run",
                               return_value=_FakeCompleted(0)) as run:
            self._run(extra_args=["--add-dir", "/x"])
        cmd = run.call_args.args[0]
        self.assertIn("--add-dir", cmd)
        # still terminated correctly
        self.assertEqual(cmd[-2], "--")

    def test_api_key_stripped_from_child_env(self):
        captured = {}

        def _run(cmd, **kw):
            captured["env"] = kw.get("env")
            return _FakeCompleted(0)

        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": _api_key_lookalike()}), \
             mock.patch.object(P.subprocess, "run", side_effect=_run):
            self._run()
        self.assertIsNotNone(captured["env"])
        self.assertNotIn("ANTHROPIC_API_KEY", captured["env"])

    def test_rate_limit_synthesises_429(self):
        with mock.patch.object(
                P.subprocess, "run",
                return_value=_FakeCompleted(0, "usage limit reached", "")):
            rc, out, err = self._run()
        self.assertEqual(rc, 429)
        self.assertIn("rate-limited", err)

    def test_clean_rc0_not_rewritten(self):
        with mock.patch.object(P.subprocess, "run",
                               return_value=_FakeCompleted(0, "DONE: ok", "")):
            rc, _out, _err = self._run()
        self.assertEqual(rc, 0)

    def test_timeout_returns_124(self):
        exc = P.subprocess.TimeoutExpired(cmd="claude", timeout=600)
        exc.stdout = "partial"
        with mock.patch.object(P.subprocess, "run", side_effect=exc):
            rc, out, err = self._run(timeout_s=600)
        self.assertEqual(rc, 124)
        self.assertIn("timeout after 600s", err)
        self.assertEqual(out, "partial")

    def test_file_not_found_returns_127(self):
        with mock.patch.object(P.subprocess, "run",
                               side_effect=FileNotFoundError("no claude")):
            rc, _out, err = self._run()
        self.assertEqual(rc, 127)
        self.assertIn("not found", err)

    def test_generic_oserror_returns_1(self):
        with mock.patch.object(P.subprocess, "run",
                               side_effect=OSError("spawn fail")):
            rc, _out, err = self._run()
        self.assertEqual(rc, 1)
        self.assertIn("failed to spawn", err)

    def test_no_model_omits_model_flag(self):
        with mock.patch.object(P.subprocess, "run",
                               return_value=_FakeCompleted(0)) as run:
            self._run(model="")
        self.assertNotIn("--model", run.call_args.args[0])


# ──────────────────────────── stage 1: PLANNER ────────────────────────────


class RunPlannerTests(_PipeBase):
    def _invoke(self, rc=0, out="", err=""):
        return mock.patch.object(P, "_invoke_claude",
                                 return_value=(rc, out, err))

    def test_happy_path_parses_and_normalizes(self):
        plan_json = json.dumps({
            "files_to_touch": ["skills/a.py", "tools\\b.py"],
            "regression_risks": ["risk one"],
            "tests_to_run": ["py_compile skills/a.py"],
            "approach": "edit a and b",
        })
        with self._invoke(out=plan_json):
            plan = P._run_planner("do the thing", claude_path="/c",
                                  project_dir=self.tmp)
        self.assertNotIn("error", plan)
        # backslashes normalized to forward slashes
        self.assertEqual(plan["files_to_touch"], ["skills/a.py", "tools/b.py"])
        self.assertEqual(plan["approach"], "edit a and b")

    def test_planner_nonzero_rc_returns_error(self):
        with self._invoke(rc=1, err="boom"):
            plan = P._run_planner("t", claude_path="/c", project_dir=self.tmp)
        self.assertIn("error", plan)
        self.assertIn("rc=1", plan["error"])
        self.assertEqual(plan["rc"], 1)      # the runner needs it to spot a 429

    def test_planner_limit_refusal_keeps_rc_429(self):
        with self._invoke(rc=429, err="rate-limited"):
            plan = P._run_planner("t", claude_path="/c", project_dir=self.tmp)
        self.assertEqual(plan["rc"], 429)

    def test_planner_unparseable_json_returns_error(self):
        with self._invoke(out="not json at all"):
            plan = P._run_planner("t", claude_path="/c", project_dir=self.tmp)
        self.assertIn("error", plan)
        self.assertIn("raw", plan)

    def test_planner_missing_keys_get_defaulted(self):
        with self._invoke(out='{"approach": "just do it"}'):
            plan = P._run_planner("t", claude_path="/c", project_dir=self.tmp)
        self.assertEqual(plan["files_to_touch"], [])
        self.assertEqual(plan["regression_risks"], [])
        self.assertEqual(plan["tests_to_run"], [])

    def test_planner_non_list_files_coerced_to_empty(self):
        with self._invoke(out='{"files_to_touch": "all of them"}'):
            plan = P._run_planner("t", claude_path="/c", project_dir=self.tmp)
        self.assertEqual(plan["files_to_touch"], [])

    def test_planner_caps_files_at_12(self):
        many = [f"f{i}.py" for i in range(30)]
        with self._invoke(out=json.dumps({"files_to_touch": many})):
            plan = P._run_planner("t", claude_path="/c", project_dir=self.tmp)
        self.assertEqual(len(plan["files_to_touch"]), 12)

    def test_planner_prompt_contains_task_and_dir(self):
        captured = {}

        def _fake(**kw):
            captured.update(kw)
            return (0, '{"files_to_touch": []}', "")

        with mock.patch.object(P, "_invoke_claude", side_effect=_fake):
            P._run_planner("UNIQUE_TASK_TOKEN", claude_path="/c",
                           project_dir=self.tmp)
        self.assertIn("UNIQUE_TASK_TOKEN", captured["prompt"])
        self.assertFalse(captured["allow_writes"])  # planner is read-only

    def test_every_stage_prompt_says_the_hard_rules_plainly(self):
        prompts = {}

        def _fake(**kw):
            prompts[kw["allow_writes"]] = prompts.get(kw["allow_writes"], []) + [kw["prompt"]]
            return (0, '{"files_to_touch": [], "verdict": "approve", "risk_score": 1}', "DONE: x")

        with mock.patch.object(P, "_invoke_claude", side_effect=_fake):
            P._run_planner("t", claude_path="/c", project_dir=self.tmp)
            P._run_implementer("t", {"files_to_touch": []}, claude_path="/c",
                               project_dir=self.tmp)
            P._run_reviewer("t", {}, "(no file changes)", claude_path="/c",
                            project_dir=self.tmp)
        planner, reviewer = prompts[False]
        (implementer,) = prompts[True]
        for text in (planner, implementer, reviewer):
            low = text.lower()
            self.assertIn("git push", low)
            self.assertIn("data/", low)
            self.assertIn("secret", low)
        self.assertIn("NO git push, tag, merge", implementer)
        self.assertIn("worktree", implementer)
        self.assertNotIn("--dangerously-skip-permissions", implementer)
        self.assertNotIn("FULL tool access", implementer)


# ──────────────────────────── stage 2: IMPLEMENTER ────────────────────────


class RunImplementerTests(_PipeBase):
    _PLAN = {"files_to_touch": ["skills/a.py"], "regression_risks": ["r1"],
             "approach": "do it"}

    def _invoke(self, rc=0, out="", err=""):
        return mock.patch.object(P, "_invoke_claude",
                                 return_value=(rc, out, err))

    def test_done_marker_parsed(self):
        with self._invoke(out="exploring...\nDONE: wired the feature"):
            res = P._run_implementer("t", self._PLAN, claude_path="/c",
                                     project_dir=self.tmp)
        self.assertEqual(res["note"], "wired the feature")
        self.assertFalse(res["impossible"])
        self.assertFalse(res["already_done"])

    def test_impossible_marker(self):
        with self._invoke(out="IMPOSSIBLE: needs a soldering iron"):
            res = P._run_implementer("t", self._PLAN, claude_path="/c",
                                     project_dir=self.tmp)
        self.assertTrue(res["impossible"])
        self.assertEqual(res["note"], "needs a soldering iron")

    def test_already_done_marker(self):
        with self._invoke(out="ALREADY_DONE: skills/a.py exists, compiles"):
            res = P._run_implementer("t", self._PLAN, claude_path="/c",
                                     project_dir=self.tmp)
        self.assertTrue(res["already_done"])
        self.assertFalse(res["impossible"])

    def test_last_marker_wins_over_earlier_prose(self):
        # model prints exploratory "DONE:" then its real verdict ALREADY_DONE
        out = "DONE: maybe\nthinking more\nALREADY_DONE: it was already there"
        with self._invoke(out=out):
            res = P._run_implementer("t", self._PLAN, claude_path="/c",
                                     project_dir=self.tmp)
        self.assertTrue(res["already_done"])
        self.assertFalse(res["impossible"])
        self.assertEqual(res["note"], "it was already there")

    def test_done_after_impossible_resets_flags(self):
        out = "IMPOSSIBLE: hmm\nactually I can\nDONE: fixed it"
        with self._invoke(out=out):
            res = P._run_implementer("t", self._PLAN, claude_path="/c",
                                     project_dir=self.tmp)
        self.assertFalse(res["impossible"])
        self.assertFalse(res["already_done"])
        self.assertEqual(res["note"], "fixed it")

    def test_rc_passed_through_and_tails_captured(self):
        with self._invoke(rc=1, out="x" * 5000, err="e" * 2000):
            res = P._run_implementer("t", self._PLAN, claude_path="/c",
                                     project_dir=self.tmp)
        self.assertEqual(res["rc"], 1)
        self.assertEqual(len(res["stdout_tail"]), 2000)  # tail -2000
        self.assertEqual(len(res["stderr_tail"]), 500)   # tail -500

    def test_implementer_gets_write_access(self):
        captured = {}

        def _fake(**kw):
            captured.update(kw)
            return (0, "DONE: ok", "")

        with mock.patch.object(P, "_invoke_claude", side_effect=_fake):
            P._run_implementer("t", self._PLAN, claude_path="/c",
                               project_dir=self.tmp)
        self.assertTrue(captured["allow_writes"])
        self.assertEqual(captured["model"], "opus")  # default implementer model
        self.assertEqual(captured["project_dir"], self.tmp)   # the worktree, not live

    def test_no_marker_means_empty_note(self):
        with self._invoke(out="just some output, no marker"):
            res = P._run_implementer("t", self._PLAN, claude_path="/c",
                                     project_dir=self.tmp)
        self.assertEqual(res["note"], "")
        self.assertFalse(res["impossible"])


# ──────────────────────────── stage 3: REVIEWER ───────────────────────────


class RunReviewerTests(_PipeBase):
    _PLAN = {"regression_risks": ["watch the audio loop"]}
    _DIFF = "--- a/skills/a.py\n+++ b/skills/a.py\n@@\n-old\n+new\n"

    def _invoke(self, rc=0, out="", err=""):
        return mock.patch.object(P, "_invoke_claude",
                                 return_value=(rc, out, err))

    def test_approve_verdict_parsed(self):
        review_json = json.dumps({"risk_score": 2, "concerns": ["minor"],
                                  "verdict": "approve"})
        with self._invoke(out=review_json):
            r = P._run_reviewer("t", self._PLAN, self._DIFF, claude_path="/c",
                                project_dir=self.tmp)
        self.assertEqual(r["verdict"], "approve")
        self.assertEqual(r["risk_score"], 2)

    def test_reject_verdict_parsed(self):
        review_json = json.dumps({"risk_score": 8, "concerns": ["regression"],
                                  "verdict": "reject_and_redo"})
        with self._invoke(out=review_json):
            r = P._run_reviewer("t", self._PLAN, self._DIFF, claude_path="/c",
                                project_dir=self.tmp)
        self.assertEqual(r["verdict"], "reject_and_redo")

    def test_unknown_verdict_downgraded_to_warnings(self):
        with self._invoke(out='{"verdict": "ship_it", "risk_score": 1}'):
            r = P._run_reviewer("t", self._PLAN, self._DIFF, claude_path="/c",
                                project_dir=self.tmp)
        self.assertEqual(r["verdict"], "approve_with_warnings")

    def test_infra_error_with_diff_falls_through_to_warnings(self):
        # rc!=0 but there IS a diff (not the no-change case) -> don't stall the
        # queue, return approve_with_warnings with the infra-error flag.
        with self._invoke(rc=1, err="reviewer crashed"):
            r = P._run_reviewer("t", self._PLAN, self._DIFF, claude_path="/c",
                                project_dir=self.tmp)
        self.assertEqual(r["verdict"], "approve_with_warnings")
        self.assertTrue(r["_infra_error"])

    def test_infra_error_with_no_diff_force_rejects(self):
        with self._invoke(rc=1, err="reviewer crashed"):
            r = P._run_reviewer("t", self._PLAN, "(no file changes)",
                                claude_path="/c", project_dir=self.tmp)
        self.assertEqual(r["verdict"], "reject_and_redo")
        self.assertTrue(r["_local_override"])

    def test_unparseable_json_with_diff_warns(self):
        with self._invoke(out="this is not json"):
            r = P._run_reviewer("t", self._PLAN, self._DIFF, claude_path="/c",
                                project_dir=self.tmp)
        self.assertEqual(r["verdict"], "approve_with_warnings")
        self.assertIn("raw", r)

    def test_unparseable_json_no_diff_force_rejects(self):
        with self._invoke(out="garbage"):
            r = P._run_reviewer("t", self._PLAN, "(no file changes)",
                                claude_path="/c", project_dir=self.tmp)
        self.assertEqual(r["verdict"], "reject_and_redo")

    def test_no_change_overrides_approve_to_reject(self):
        # reviewer says approve, but there is no diff and implementer did NOT
        # claim impossible/already_done -> local override to reject.
        with self._invoke(out='{"verdict": "approve", "risk_score": 0}'):
            r = P._run_reviewer("t", self._PLAN, "(no file changes)",
                                claude_path="/c", project_dir=self.tmp,
                                impl_impossible=False, impl_already_done=False)
        self.assertEqual(r["verdict"], "reject_and_redo")
        self.assertGreaterEqual(r["risk_score"], 9)
        self.assertTrue(r["_local_override"])

    def test_no_change_override_survives_a_malformed_risk_score(self):
        # Audit A77: valid JSON whose risk_score is not an int ("high", "7.5",
        # a list, Infinity) crashed the empty-diff override with an unguarded
        # int() -- ValueError / TypeError / OverflowError out of the reviewer,
        # and a traceback on the direct --task CLI path. It must fail closed.
        for raw in ('"high"', '"7.5"', '[7]', '{"n": 7}', 'Infinity', 'NaN'):
            with self.subTest(risk_score=raw):
                out = '{"verdict": "approve", "risk_score": %s}' % raw
                with self._invoke(out=out):
                    r = P._run_reviewer("t", self._PLAN, "(no file changes)",
                                        claude_path="/c", project_dir=self.tmp,
                                        impl_impossible=False,
                                        impl_already_done=False)
                self.assertEqual(r["verdict"], "reject_and_redo")
                self.assertEqual(r["risk_score"], 9)
                self.assertTrue(r["_local_override"])

    def test_no_change_but_impossible_claim_not_overridden(self):
        with self._invoke(out='{"verdict": "approve", "risk_score": 1}'):
            r = P._run_reviewer("t", self._PLAN, "(no file changes)",
                                claude_path="/c", project_dir=self.tmp,
                                impl_impossible=True)
        # impossible claim is legitimate -> approve stands
        self.assertEqual(r["verdict"], "approve")

    def test_already_done_verdict_valid_when_claimed_and_no_diff(self):
        with self._invoke(out='{"verdict": "already_done", "risk_score": 1}'):
            r = P._run_reviewer("t", self._PLAN, "(no file changes)",
                                claude_path="/c", project_dir=self.tmp,
                                impl_already_done=True)
        self.assertEqual(r["verdict"], "already_done")

    def test_already_done_freelanced_on_normal_flow_downgraded(self):
        # reviewer says already_done but implementer never claimed it AND there
        # is a diff -> downgrade to approve_with_warnings.
        with self._invoke(out='{"verdict": "already_done", "risk_score": 0}'):
            r = P._run_reviewer("t", self._PLAN, self._DIFF, claude_path="/c",
                                project_dir=self.tmp, impl_already_done=False)
        self.assertEqual(r["verdict"], "approve_with_warnings")
        self.assertTrue(r["_local_override"])

    def test_reviewer_prompt_includes_already_done_claim(self):
        captured = {}

        def _fake(**kw):
            captured.update(kw)
            return (0, '{"verdict": "already_done"}', "")

        with mock.patch.object(P, "_invoke_claude", side_effect=_fake):
            P._run_reviewer("t", self._PLAN, "(no file changes)", claude_path="/c",
                            project_dir=self.tmp, impl_already_done=True,
                            impl_note="file present at rev abc")
        self.assertIn("ALREADY_DONE", captured["prompt"])
        self.assertIn("file present at rev abc", captured["prompt"])
        self.assertFalse(captured["allow_writes"])  # reviewer read-only

    def test_huge_diff_truncated_in_prompt(self):
        captured = {}

        def _fake(**kw):
            captured.update(kw)
            return (0, '{"verdict": "approve"}', "")

        big_diff = "+" + ("x" * 30_000)
        with mock.patch.object(P, "_invoke_claude", side_effect=_fake):
            P._run_reviewer("t", self._PLAN, big_diff, claude_path="/c",
                            project_dir=self.tmp)
        self.assertIn("diff truncated at 24000 chars", captured["prompt"])


# ──────────────────────────── stage 4: TESTER ─────────────────────────────


class MaxRiskScoreTests(_PipeBase):
    def test_default_is_7(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("JARVIS_PIPELINE_MAX_RISK", None)
            self.assertEqual(P._max_risk_score(), 7)

    def test_custom_value(self):
        with mock.patch.dict(os.environ, {"JARVIS_PIPELINE_MAX_RISK": "5"}):
            self.assertEqual(P._max_risk_score(), 5)

    def test_bad_value_defaults_to_7(self):
        with mock.patch.dict(os.environ, {"JARVIS_PIPELINE_MAX_RISK": "high"}):
            self.assertEqual(P._max_risk_score(), 7)

    def test_clamped_to_0_10(self):
        with mock.patch.dict(os.environ, {"JARVIS_PIPELINE_MAX_RISK": "99"}):
            self.assertEqual(P._max_risk_score(), 10)
        with mock.patch.dict(os.environ, {"JARVIS_PIPELINE_MAX_RISK": "-4"}):
            self.assertEqual(P._max_risk_score(), 0)


class PipelineMaxUsdTests(_PipeBase):
    def test_default_is_25(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("JARVIS_PIPELINE_MAX_USD", None)
            self.assertEqual(P._pipeline_max_usd(), 25.0)

    def test_custom_value(self):
        with mock.patch.dict(os.environ, {"JARVIS_PIPELINE_MAX_USD": "3.5"}):
            self.assertEqual(P._pipeline_max_usd(), 3.5)

    def test_zero_means_no_cap(self):
        with mock.patch.dict(os.environ, {"JARVIS_PIPELINE_MAX_USD": "0"}):
            self.assertEqual(P._pipeline_max_usd(), 0.0)

    def test_negative_clamped_to_zero(self):
        with mock.patch.dict(os.environ, {"JARVIS_PIPELINE_MAX_USD": "-10"}):
            self.assertEqual(P._pipeline_max_usd(), 0.0)

    def test_bad_value_defaults(self):
        with mock.patch.dict(os.environ, {"JARVIS_PIPELINE_MAX_USD": "lots"}):
            self.assertEqual(P._pipeline_max_usd(), 25.0)


class EstimateTaskCostTests(_PipeBase):
    def test_none_result_bills_full_cycle(self):
        # run_pipeline_on_task raised -> bill a full task, never $0.
        self.assertEqual(P._estimate_task_cost_usd(None),
                         P._FULL_TASK_COST_ESTIMATE_USD)

    def test_empty_stages_bills_full_cycle(self):
        self.assertEqual(P._estimate_task_cost_usd({"stages": {}}),
                         P._FULL_TASK_COST_ESTIMATE_USD)

    def test_missing_stages_key_bills_full_cycle(self):
        self.assertEqual(P._estimate_task_cost_usd({"ok": True}),
                         P._FULL_TASK_COST_ESTIMATE_USD)

    def test_planner_only_bills_planner(self):
        # e.g. a planner-marked-IMPOSSIBLE task.
        res = {"stages": {"planner": {}}}
        self.assertEqual(P._estimate_task_cost_usd(res),
                         P._STAGE_COST_ESTIMATE_USD["planner"])

    def test_full_pipeline_sums_billable_stages(self):
        res = {"stages": {"planner": {}, "implementer": {}, "reviewer": {},
                          "tester": {}, "suite": {}}}
        expected = (P._STAGE_COST_ESTIMATE_USD["planner"]
                    + P._STAGE_COST_ESTIMATE_USD["implementer"]
                    + P._STAGE_COST_ESTIMATE_USD["reviewer"])
        self.assertAlmostEqual(P._estimate_task_cost_usd(res), expected)

    def test_local_only_stages_fall_back_not_zero(self):
        # A stages dict with only local (non-Claude) keys must not bill $0.
        res = {"stages": {"tester": {}, "suite": {}}}
        self.assertEqual(P._estimate_task_cost_usd(res),
                         P._FULL_TASK_COST_ESTIMATE_USD)

    def test_implementer_is_most_expensive_stage(self):
        # Documented basis: implementer (Opus, writes) dominates the estimate.
        self.assertGreater(P._STAGE_COST_ESTIMATE_USD["implementer"],
                           P._STAGE_COST_ESTIMATE_USD["planner"])
        self.assertGreater(P._STAGE_COST_ESTIMATE_USD["implementer"],
                           P._STAGE_COST_ESTIMATE_USD["reviewer"])


class FindClaudeCliTests(_PipeBase):
    def test_returns_which_result_when_on_path(self):
        with mock.patch.object(P.shutil, "which", return_value="/usr/bin/claude"):
            self.assertEqual(P._find_claude_cli(), "/usr/bin/claude")

    def test_falls_back_to_known_path(self):
        with mock.patch.object(P.shutil, "which", return_value=None), \
             mock.patch.object(P.os.path, "exists",
                               side_effect=lambda p: p.endswith("claude.exe")
                               or p.endswith("claude.cmd")):
            found = P._find_claude_cli()
        self.assertIsNotNone(found)

    def test_returns_none_when_nothing_found(self):
        with mock.patch.object(P.shutil, "which", return_value=None), \
             mock.patch.object(P.os.path, "exists", return_value=False):
            self.assertIsNone(P._find_claude_cli())


if __name__ == "__main__":
    unittest.main()
