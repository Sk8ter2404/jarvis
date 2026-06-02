"""Unit tests for tools/multi_agent_pipeline.py — the JARVIS multi-agent
self-upgrade pipeline (planner -> implementer -> reviewer -> tester, plus the
loop driver and CLI).

SAFETY CONTRACT (this suite NEVER runs a real upgrade):
  * No real `claude` CLI / git / smoke-test subprocess work. The module shells
    out to the `claude` CLI via subprocess.run inside `_invoke_claude` (it does
    NOT use the `anthropic` SDK), so the LLM is mocked by patching
    `P.subprocess.run` for the low-level wrapper test and by patching
    `P._invoke_claude` (and the stage helpers) for the orchestration tests.
  * No real network / LLM / threads. time.sleep is stubbed in the base class.
  * No writes to real source files — every module path constant (PROJECT_DIR,
    TODO_FILE, PIPELINE_LOG, PIPELINE_BACKUP_ROOT, STABILITY_SCRIPT,
    BOOT_SCRIPT, LOCK_FILE) is redirected into a per-test tempdir and restored
    in tearDown via addCleanup.

ISOLATION: all patches are per-test (context managers / addCleanup). No
module-level sys.modules writes. Path globals are saved/restored around every
test by _PipeBase.

CI-FAITHFUL: multi_agent_pipeline imports only stdlib at module top (difflib,
json, os, re, shutil, subprocess, sys, time, typing), so this suite RUNS (not
skips) under tools/run_tests_ci_sim.py on a bare Linux runner. The handful of
tests that exercise Windows-only code (`_kill_jarvis` powershell calls, the
ctypes.windll ANSI enable in the loop driver) guard on
sys.platform.startswith("win") so they SKIP on the Linux sim rather than fail.

PRIVACY: fixtures use only fake identifiers ("alice", "10.0.0.5"). No real
keys/IPs. Any API-key-looking string is built at runtime via concatenation so
the repo's PII gate that greps tests/ never trips.

Bugs found are annotated with `# BUG:` — the source is NOT modified.
"""
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
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

    _PATH_ATTRS = (
        "PROJECT_DIR", "TODO_FILE", "PIPELINE_LOG", "PIPELINE_BACKUP_ROOT",
        "STABILITY_SCRIPT", "BOOT_SCRIPT", "LOCK_FILE",
    )

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp = self._tmpdir.name
        self.addCleanup(self._tmpdir.cleanup)

        self._saved = {a: getattr(P, a) for a in self._PATH_ATTRS}
        self.addCleanup(self._restore_consts)

        p = self.tmp
        P.PROJECT_DIR           = p
        P.TODO_FILE             = os.path.join(p, "jarvis_todo.md")
        P.PIPELINE_LOG          = os.path.join(p, "data", "pipeline_runs.jsonl")
        P.PIPELINE_BACKUP_ROOT  = os.path.join(p, "backups", "pipeline")
        P.STABILITY_SCRIPT      = os.path.join(p, "tools", "stability_smoke_test.py")
        P.BOOT_SCRIPT           = os.path.join(p, "_boot_jarvis.ps1")
        P.LOCK_FILE             = os.path.join(p, "jarvis.lock")

        # never really sleep (the loop driver backs off on failure)
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
                  "JARVIS_PIPELINE_TESTER_WAIT_S",
                  "JARVIS_PIPELINE_SKIP_TESTER",
                  "JARVIS_PIPELINE_FORCE_CONTINUE_AFTER_REGRESSION"):
            os.environ.pop(k, None)

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

    def test_tester_wait_default(self):
        self.assertEqual(P._tester_wait_s(), 90)

    def test_tester_wait_clamped_to_min_20(self):
        os.environ["JARVIS_PIPELINE_TESTER_WAIT_S"] = "5"
        self.assertEqual(P._tester_wait_s(), 20)

    def test_tester_wait_custom(self):
        os.environ["JARVIS_PIPELINE_TESTER_WAIT_S"] = "150"
        self.assertEqual(P._tester_wait_s(), 150)

    def test_tester_wait_bad_value_defaults(self):
        os.environ["JARVIS_PIPELINE_TESTER_WAIT_S"] = "ninety"
        self.assertEqual(P._tester_wait_s(), 90)

    def test_tester_disabled_only_on_exactly_1(self):
        self.assertFalse(P._tester_disabled())
        os.environ["JARVIS_PIPELINE_SKIP_TESTER"] = "1"
        self.assertTrue(P._tester_disabled())
        os.environ["JARVIS_PIPELINE_SKIP_TESTER"] = "true"
        self.assertFalse(P._tester_disabled())
        os.environ["JARVIS_PIPELINE_SKIP_TESTER"] = " 1 "  # stripped to "1"
        self.assertTrue(P._tester_disabled())


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


class TodoHelperTests(_PipeBase):
    def test_first_unchecked_returns_text_without_prefix(self):
        self.write_todo("# todo\n- [x] done one\n- [ ] do the thing\n- [ ] later\n")
        self.assertEqual(P._first_unchecked_task(), "do the thing")

    def test_first_unchecked_none_when_all_done(self):
        self.write_todo("- [x] a\n- [x] b\n")
        self.assertIsNone(P._first_unchecked_task())

    def test_first_unchecked_missing_file_returns_none(self):
        # no jarvis_todo.md written → OSError swallowed → None
        self.assertIsNone(P._first_unchecked_task())

    def test_count_unchecked(self):
        self.write_todo("- [ ] a\n- [x] b\n- [ ] c\n- [ ] d\n")
        self.assertEqual(P._count_unchecked(), 3)

    def test_count_unchecked_missing_file_zero(self):
        self.assertEqual(P._count_unchecked(), 0)

    def test_tick_task_flips_marker(self):
        self.write_todo("- [ ] do the thing\n- [ ] other\n")
        ok = P._tick_task("do the thing", "all good")
        self.assertTrue(ok)
        text = self.read("jarvis_todo.md")
        self.assertIn("- [x] do the thing", text)
        self.assertIn("DONE", text)
        self.assertIn("all good", text)
        self.assertIn("- [ ] other", text)  # untouched

    def test_tick_task_only_flips_first_occurrence(self):
        self.write_todo("- [ ] dup\n- [ ] dup\n")
        P._tick_task("dup", "note")
        text = self.read("jarvis_todo.md")
        self.assertEqual(text.count("- [x] dup"), 1)
        self.assertEqual(text.count("- [ ] dup"), 1)

    def test_tick_task_missing_needle_returns_false(self):
        self.write_todo("- [ ] something else\n")
        self.assertFalse(P._tick_task("not present", "note"))

    def test_tick_task_missing_file_returns_false(self):
        self.assertFalse(P._tick_task("anything", "note"))

    def test_tick_task_write_failure_returns_false(self):
        self.write_todo("- [ ] do the thing\n")
        import builtins
        real_open = builtins.open

        def _open(path, *a, **k):
            mode = a[0] if a else k.get("mode", "r")
            if "w" in mode:
                raise OSError("read-only fs")
            return real_open(path, *a, **k)

        with mock.patch("builtins.open", _open):
            self.assertFalse(P._tick_task("do the thing", "note"))

    def test_append_regression_task(self):
        self.write_todo("- [ ] original\n")
        P._append_regression_task("original task", "reviewer", "found a leak")
        text = self.read("jarvis_todo.md")
        self.assertIn("**[regression]**", text)
        self.assertIn("reviewer", text)
        self.assertIn("found a leak", text)

    def test_append_regression_truncates_and_strips_newlines(self):
        self.write_todo("- [ ] x\n")
        P._append_regression_task("p" * 500, "tester", "line1\nline2\n" + "z" * 500)
        text = self.read("jarvis_todo.md")
        reg_line = [ln for ln in text.splitlines() if "[regression]" in ln][0]
        # details capped at 400 chars and newlines flattened to spaces
        self.assertNotIn("line1\nline2", reg_line)
        self.assertIn("line1 line2", reg_line)

    def test_append_regression_missing_file_swallowed(self):
        # parent dir exists (tempdir) but we patch open to fail → no raise.
        with mock.patch("builtins.open", side_effect=OSError("denied")):
            P._append_regression_task("t", "stage", "d")  # must not raise


# ──────────────────────────── backup / restore ────────────────────────────


class BackupRestoreTests(_PipeBase):
    def test_backup_roundtrip_and_restore_edit(self):
        self.write("skills/foo.py", "original\n")
        backup = P._backup_files(["skills/foo.py"], "my task")
        self.assertTrue(os.path.isdir(backup))
        # manifest records the file as existed_before
        manifest = json.loads(self.read(
            os.path.relpath(os.path.join(backup, "_manifest.json"), self.tmp)))
        self.assertEqual(manifest["files"][0]["path"].replace("\\", "/"),
                         "skills/foo.py")
        self.assertTrue(manifest["files"][0]["existed_before"])

        # mutate then restore
        self.write("skills/foo.py", "MUTATED\n")
        restored, deleted = P._restore_files(backup)
        self.assertEqual((restored, deleted), (1, 0))
        self.assertEqual(self.read("skills/foo.py"), "original\n")

    def test_backup_records_missing_file_then_restore_deletes_it(self):
        # File does not exist at backup time → recorded existed_before False.
        backup = P._backup_files(["skills/new.py"], "task")
        manifest = json.loads(self.read(
            os.path.relpath(os.path.join(backup, "_manifest.json"), self.tmp)))
        self.assertFalse(manifest["files"][0]["existed_before"])

        # implementer "creates" it, restore should delete it.
        self.write("skills/new.py", "created by implementer\n")
        restored, deleted = P._restore_files(backup)
        self.assertEqual((restored, deleted), (0, 1))
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "skills/new.py")))

    def test_backup_refuses_paths_outside_project_root(self):
        # an absolute path outside PROJECT_DIR -> rel starts with ".." -> skipped
        with tempfile.TemporaryDirectory() as other:
            outside = os.path.join(other, "secret.py")
            with open(outside, "w", encoding="utf-8") as f:
                f.write("x\n")
            backup = P._backup_files([outside], "task")
            manifest = json.loads(self.read(
                os.path.relpath(os.path.join(backup, "_manifest.json"), self.tmp)))
            self.assertEqual(manifest["files"], [])  # nothing backed up

    def test_backup_slug_sanitized(self):
        backup = P._backup_files([], "weird/slug: with*chars")
        base = os.path.basename(backup)
        # slug part (after the timestamp) must contain no path separators / colons
        self.assertNotIn(":", base)
        self.assertNotIn("/", base)
        self.assertNotIn("*", base)

    def test_restore_missing_manifest_returns_zeros(self):
        with tempfile.TemporaryDirectory() as empty:
            self.assertEqual(P._restore_files(empty), (0, 0))

    def test_backup_copy_error_recorded_in_manifest(self):
        self.write("skills/foo.py", "data\n")
        with mock.patch.object(P.shutil, "copy2",
                               side_effect=OSError("perm denied")):
            backup = P._backup_files(["skills/foo.py"], "task")
        manifest = json.loads(self.read(
            os.path.relpath(os.path.join(backup, "_manifest.json"), self.tmp)))
        self.assertIn("backup_error", manifest["files"][0])

    def test_restore_copy_back_error_swallowed(self):
        # existed_before True; copy2 raises during restore -> swallowed, the
        # restored counter is not incremented for that file.
        self.write("skills/foo.py", "original\n")
        backup = P._backup_files(["skills/foo.py"], "task")
        self.write("skills/foo.py", "mutated\n")
        with mock.patch.object(P.shutil, "copy2", side_effect=OSError("ro")):
            restored, deleted = P._restore_files(backup)
        self.assertEqual((restored, deleted), (0, 0))

    def test_restore_delete_error_swallowed(self):
        # existed_before False (implementer-created); os.remove raises during
        # rollback -> swallowed, deleted counter not incremented.
        backup = P._backup_files(["skills/new.py"], "task")
        self.write("skills/new.py", "created\n")
        with mock.patch.object(P.os, "remove", side_effect=OSError("busy")):
            restored, deleted = P._restore_files(backup)
        self.assertEqual((restored, deleted), (0, 0))


# ──────────────────────────── diff computation ────────────────────────────


class ComputeDiffTests(_PipeBase):
    def _backup_with(self, rel, before_text):
        self.write(rel, before_text)
        return P._backup_files([rel], "task")

    def test_no_change_returns_sentinel(self):
        backup = self._backup_with("skills/a.py", "line1\nline2\n")
        self.assertEqual(P._compute_diff(backup, ["skills/a.py"]),
                         "(no file changes)")

    def test_simple_edit_diff(self):
        backup = self._backup_with("skills/a.py", "line1\nline2\n")
        self.write("skills/a.py", "line1\nCHANGED\n")
        diff = P._compute_diff(backup, ["skills/a.py"])
        self.assertIn("-line2", diff)
        self.assertIn("+CHANGED", diff)
        # The diff header embeds os.path.relpath output, which uses the native
        # separator (backslash on Windows) under the forward-slash a/ b/ prefix.
        # Normalize so the assertion holds on both OSes.
        norm = diff.replace("\\", "/")
        self.assertIn("a/skills/a.py", norm)
        self.assertIn("b/skills/a.py", norm)

    def test_new_file_diff_shows_additions(self):
        backup = P._backup_files(["skills/new.py"], "task")  # missing at backup
        self.write("skills/new.py", "fresh1\nfresh2\n")
        diff = P._compute_diff(backup, ["skills/new.py"])
        self.assertIn("+fresh1", diff)
        self.assertIn("+fresh2", diff)

    def test_diff_truncated_at_400_lines(self):
        before = "\n".join(f"line{i}" for i in range(1000)) + "\n"
        backup = self._backup_with("skills/big.py", before)
        after = "\n".join(f"X{i}" for i in range(1000)) + "\n"
        self.write("skills/big.py", after)
        diff = P._compute_diff(backup, ["skills/big.py"])
        self.assertIn("diff truncated at 400 lines", diff)

    def test_multiple_files_concatenated(self):
        self.write("skills/a.py", "a-before\n")
        self.write("skills/b.py", "b-before\n")
        backup = P._backup_files(["skills/a.py", "skills/b.py"], "task")
        self.write("skills/a.py", "a-after\n")
        self.write("skills/b.py", "b-after\n")
        diff = P._compute_diff(backup, ["skills/a.py", "skills/b.py"])
        norm = diff.replace("\\", "/")
        self.assertIn("a/skills/a.py", norm)
        self.assertIn("a/skills/b.py", norm)

    def test_read_error_treated_as_empty_sides(self):
        # If both before and after reads raise OSError, both sides are []
        # (equal) -> the file is skipped -> overall sentinel.
        backup = self._backup_with("skills/a.py", "before\n")
        self.write("skills/a.py", "after\n")
        with mock.patch("builtins.open", side_effect=OSError("io error")):
            diff = P._compute_diff(backup, ["skills/a.py"])
        self.assertEqual(diff, "(no file changes)")


# ──────────────────────────── JSON extraction ─────────────────────────────


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
        # read-only: no skip-permissions
        self.assertNotIn("--dangerously-skip-permissions", cmd)
        # prompt is passed positionally after the "--" terminator
        self.assertEqual(cmd[-1], "do x")
        self.assertEqual(cmd[-2], "--")

    def test_allow_writes_adds_skip_permissions(self):
        with mock.patch.object(P.subprocess, "run",
                               return_value=_FakeCompleted(0)) as run:
            self._run(allow_writes=True)
        self.assertIn("--dangerously-skip-permissions", run.call_args.args[0])

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


class RunTesterTests(_PipeBase):
    def test_disabled_via_env_skips(self):
        os.environ["JARVIS_PIPELINE_SKIP_TESTER"] = "1"
        res = P._run_tester(project_dir=self.tmp)
        self.assertTrue(res["ok"])
        self.assertTrue(res["skipped"])

    def test_missing_smoke_script_skips(self):
        # STABILITY_SCRIPT points into the tempdir but the file doesn't exist.
        res = P._run_tester(project_dir=self.tmp)
        self.assertTrue(res["ok"])
        self.assertTrue(res["skipped"])
        self.assertIn("missing", res["reason"])

    def test_smoke_pass_rc0(self):
        self.write(os.path.join("tools", "stability_smoke_test.py"), "# smoke\n")
        with mock.patch.object(P.subprocess, "run",
                               return_value=_FakeCompleted(0, "PASS", "")):
            res = P._run_tester(project_dir=self.tmp)
        self.assertTrue(res["ok"])
        self.assertEqual(res["rc"], 0)

    def test_smoke_fail_nonzero_rc(self):
        self.write(os.path.join("tools", "stability_smoke_test.py"), "# smoke\n")
        with mock.patch.object(P.subprocess, "run",
                               return_value=_FakeCompleted(1, "FATAL boom", "e")):
            res = P._run_tester(project_dir=self.tmp)
        self.assertFalse(res["ok"])
        self.assertEqual(res["rc"], 1)

    def test_smoke_timeout(self):
        self.write(os.path.join("tools", "stability_smoke_test.py"), "# smoke\n")
        exc = P.subprocess.TimeoutExpired(cmd="smoke", timeout=180)
        exc.stdout = "hung output"
        with mock.patch.object(P.subprocess, "run", side_effect=exc):
            res = P._run_tester(project_dir=self.tmp)
        self.assertFalse(res["ok"])
        self.assertIn("timed out", res["error"])

    def test_smoke_oserror(self):
        self.write(os.path.join("tools", "stability_smoke_test.py"), "# smoke\n")
        with mock.patch.object(P.subprocess, "run",
                               side_effect=OSError("cannot spawn")):
            res = P._run_tester(project_dir=self.tmp)
        self.assertFalse(res["ok"])
        self.assertIn("could not spawn", res["error"])


# ──────────────────────────── _kill_jarvis (Windows) ──────────────────────


class KillJarvisTests(_PipeBase):
    @unittest.skipUnless(_IS_WIN, "uses Win32 powershell process query")
    def test_kills_pids_and_clears_lock(self):
        # first call lists PIDs, subsequent calls Stop-Process per pid.
        self.write("jarvis.lock", "1234")
        calls = []

        def _run(cmd, **kw):
            calls.append(cmd)
            # the first powershell invocation is the Get-CimInstance lister
            joined = " ".join(cmd)
            if "Get-CimInstance" in joined:
                return _FakeCompleted(0, "111\n222\n", "")
            return _FakeCompleted(0, "", "")

        with mock.patch.object(P.subprocess, "run", side_effect=_run):
            n = P._kill_jarvis()
        self.assertEqual(n, 2)
        # lock removed
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "jarvis.lock")))

    @unittest.skipUnless(_IS_WIN, "uses Win32 powershell process query")
    def test_lister_oserror_returns_zero(self):
        with mock.patch.object(P.subprocess, "run",
                               side_effect=OSError("no powershell")):
            self.assertEqual(P._kill_jarvis(), 0)

    @unittest.skipUnless(_IS_WIN, "uses Win32 powershell process query")
    def test_no_pids_no_kills(self):
        with mock.patch.object(P.subprocess, "run",
                               return_value=_FakeCompleted(0, "\n", "")) as run:
            n = P._kill_jarvis()
        self.assertEqual(n, 0)
        # only the lister ran (no Stop-Process calls)
        self.assertEqual(run.call_count, 1)

    @unittest.skipUnless(_IS_WIN, "uses Win32 powershell process query")
    def test_stop_process_error_swallowed_count_still_returned(self):
        # The lister finds one PID; the Stop-Process call raises -> swallowed,
        # and the count of matched PIDs is still returned.
        def _run(cmd, **kw):
            if "Get-CimInstance" in " ".join(cmd):
                return _FakeCompleted(0, "555\n", "")
            raise OSError("stop-process denied")
        with mock.patch.object(P.subprocess, "run", side_effect=_run):
            self.assertEqual(P._kill_jarvis(), 1)

    @unittest.skipUnless(_IS_WIN, "uses Win32 powershell process query")
    def test_lock_remove_error_swallowed(self):
        self.write("jarvis.lock", "x")
        with mock.patch.object(P.subprocess, "run",
                               return_value=_FakeCompleted(0, "\n", "")), \
             mock.patch.object(P.os, "remove", side_effect=OSError("locked")):
            # must not raise even though lock removal fails
            self.assertEqual(P._kill_jarvis(), 0)


# ──────────────────────────── guard surface / prune ───────────────────────


class GuardSurfaceTests(_PipeBase):
    def test_enumerates_root_and_subdir_py_files(self):
        self.write("bobert_companion.py", "# main\n")
        self.write("skills/foo.py", "# skill\n")
        self.write("core/bar.py", "# core\n")
        self.write("README.md", "not python\n")
        self.write(os.path.join("skills", "__pycache__", "foo.cpython.pyc"), "x")
        surface = P._guard_surface(self.tmp, [])
        self.assertIn("bobert_companion.py", surface)
        self.assertIn("skills/foo.py", surface)
        self.assertIn("core/bar.py", surface)
        self.assertNotIn("README.md", surface)
        # __pycache__ pruned
        self.assertFalse(any("__pycache__" in s for s in surface))

    def test_planned_files_unioned_even_if_missing(self):
        self.write("bobert_companion.py", "# main\n")
        surface = P._guard_surface(self.tmp, ["skills/not_yet.py", "tools\\new.py"])
        self.assertIn("skills/not_yet.py", surface)
        self.assertIn("tools/new.py", surface)  # backslash normalized

    def test_result_sorted_and_unique(self):
        self.write("skills/a.py", "x\n")
        surface = P._guard_surface(self.tmp, ["skills/a.py", "skills/a.py"])
        self.assertEqual(surface, sorted(surface))
        self.assertEqual(len(surface), len(set(surface)))

    def test_listdir_error_swallowed_still_unions_planned(self):
        # os.listdir on the root raises -> the root-enumeration is skipped but
        # planned files are still unioned in (the except OSError branch).
        with mock.patch.object(P.os, "listdir", side_effect=OSError("denied")):
            surface = P._guard_surface(self.tmp, ["skills/planned.py"])
        self.assertIn("skills/planned.py", surface)


class PrunePipelineBackupsTests(_PipeBase):
    def test_keeps_only_most_recent_n(self):
        root = P.PIPELINE_BACKUP_ROOT
        os.makedirs(root, exist_ok=True)
        # chronological names sort lexically
        names = [f"2026-06-0{i}_00-00-00_task" for i in range(1, 6)]
        for n in names:
            os.makedirs(os.path.join(root, n), exist_ok=True)
        P._prune_pipeline_backups(keep=2)
        remaining = sorted(os.listdir(root))
        self.assertEqual(remaining, names[-2:])

    def test_missing_root_is_safe(self):
        # PIPELINE_BACKUP_ROOT doesn't exist yet -> swallowed, no raise.
        P._prune_pipeline_backups(keep=5)


# ──────────────────────────── orchestration ───────────────────────────────


class _OrchBase(_PipeBase):
    """Common scaffolding for run_pipeline_on_task tests: a todo file with one
    task, a real source file to diff, and helpers to stub the stage functions.
    The smoke tester is disabled by default (env) so tests opt into it."""

    TASK = "add a cool feature"
    PLAN = {"files_to_touch": ["skills/feature.py"],
            "regression_risks": ["risk"], "tests_to_run": [], "approach": "go"}

    def setUp(self):
        super().setUp()
        self.write_todo(f"- [ ] {self.TASK}\n- [ ] another task\n")
        self.write("skills/feature.py", "original body\n")
        # silence the emit prints by default
        self.emit = lambda *a, **k: None

    def _emit_implementer_change(self):
        """Make the implementer actually mutate the source so _compute_diff
        produces a non-empty diff."""
        def _impl(task_line, plan, **kw):
            self.write("skills/feature.py", "NEW body from implementer\n")
            return {"rc": 0, "stdout_tail": "DONE: did it", "stderr_tail": "",
                    "note": "did it", "impossible": False, "already_done": False}
        return _impl


class RunPipelineOnTaskTests(_OrchBase):
    def test_planner_error_bails_without_ticking(self):
        with mock.patch.object(P, "_run_planner",
                               return_value={"error": "planner exited rc=1"}):
            res = P.run_pipeline_on_task(self.TASK, claude_path="/c",
                                         project_dir=self.tmp, emit=self.emit)
        self.assertFalse(res["ok"])
        self.assertFalse(res["ticked"])
        self.assertEqual(res["stage_failed"], "planner")
        # task still unchecked
        self.assertIn(f"- [ ] {self.TASK}", self.read("jarvis_todo.md"))

    def test_planner_impossible_ticks_without_implementing(self):
        plan = dict(self.PLAN, approach="IMPOSSIBLE: needs hardware")
        with mock.patch.object(P, "_run_planner", return_value=plan), \
             mock.patch.object(P, "_run_implementer") as impl:
            res = P.run_pipeline_on_task(self.TASK, claude_path="/c",
                                         project_dir=self.tmp, emit=self.emit)
        impl.assert_not_called()
        self.assertTrue(res["ok"])
        self.assertTrue(res["ticked"])
        self.assertIn("IMPOSSIBLE (planner)", self.read("jarvis_todo.md"))

    def test_implementer_rc_failure_rolls_back(self):
        def _impl(task_line, plan, **kw):
            self.write("skills/feature.py", "broken partial edit\n")
            return {"rc": 1, "stdout_tail": "", "stderr_tail": "err",
                    "note": "", "impossible": False, "already_done": False}
        with mock.patch.object(P, "_run_planner", return_value=self.PLAN), \
             mock.patch.object(P, "_run_implementer", side_effect=_impl):
            res = P.run_pipeline_on_task(self.TASK, claude_path="/c",
                                         project_dir=self.tmp, emit=self.emit)
        self.assertFalse(res["ok"])
        self.assertEqual(res["stage_failed"], "implementer")
        # rolled back to original content
        self.assertEqual(self.read("skills/feature.py"), "original body\n")

    def test_implementer_impossible_ticks_no_rollback(self):
        def _impl(task_line, plan, **kw):
            self.write("skills/feature.py", "diagnostic state left behind\n")
            return {"rc": 0, "stdout_tail": "", "stderr_tail": "",
                    "note": "no driver", "impossible": True, "already_done": False}
        with mock.patch.object(P, "_run_planner", return_value=self.PLAN), \
             mock.patch.object(P, "_run_implementer", side_effect=_impl):
            res = P.run_pipeline_on_task(self.TASK, claude_path="/c",
                                         project_dir=self.tmp, emit=self.emit)
        self.assertTrue(res["ok"])
        self.assertTrue(res["ticked"])
        self.assertIn("IMPOSSIBLE (implementer)", self.read("jarvis_todo.md"))
        # NOT rolled back
        self.assertEqual(self.read("skills/feature.py"),
                         "diagnostic state left behind\n")

    def test_reviewer_reject_rolls_back_and_queues_regression(self):
        with mock.patch.object(P, "_run_planner", return_value=self.PLAN), \
             mock.patch.object(P, "_run_implementer",
                               side_effect=self._emit_implementer_change()), \
             mock.patch.object(P, "_run_reviewer",
                               return_value={"verdict": "reject_and_redo",
                                             "risk_score": 8,
                                             "concerns": ["introduces a leak"]}):
            res = P.run_pipeline_on_task(self.TASK, claude_path="/c",
                                         project_dir=self.tmp, emit=self.emit)
        self.assertFalse(res["ok"])
        self.assertEqual(res["stage_failed"], "reviewer")
        # rolled back
        self.assertEqual(self.read("skills/feature.py"), "original body\n")
        todo = self.read("jarvis_todo.md")
        self.assertIn("[regression]", todo)
        self.assertIn("introduces a leak", todo)
        # original NOT ticked
        self.assertIn(f"- [ ] {self.TASK}", todo)

    def test_already_done_path_ticks_and_skips_tester(self):
        def _impl(task_line, plan, **kw):
            # no file change; claims already done
            return {"rc": 0, "stdout_tail": "", "stderr_tail": "",
                    "note": "exists already", "impossible": False,
                    "already_done": True}
        with mock.patch.object(P, "_run_planner", return_value=self.PLAN), \
             mock.patch.object(P, "_run_implementer", side_effect=_impl), \
             mock.patch.object(P, "_run_reviewer",
                               return_value={"verdict": "already_done",
                                             "risk_score": 1, "concerns": []}), \
             mock.patch.object(P, "_run_tester") as tester:
            res = P.run_pipeline_on_task(self.TASK, claude_path="/c",
                                         project_dir=self.tmp, emit=self.emit)
        tester.assert_not_called()
        self.assertTrue(res["ok"])
        self.assertTrue(res["ticked"])
        self.assertIn("ALREADY_DONE", self.read("jarvis_todo.md"))

    def test_approve_and_tester_pass_ticks(self):
        with mock.patch.object(P, "_run_planner", return_value=self.PLAN), \
             mock.patch.object(P, "_run_implementer",
                               side_effect=self._emit_implementer_change()), \
             mock.patch.object(P, "_run_reviewer",
                               return_value={"verdict": "approve",
                                             "risk_score": 1, "concerns": []}), \
             mock.patch.object(P, "_run_tester",
                               return_value={"ok": True, "rc": 0}), \
             mock.patch.object(P, "_kill_jarvis", return_value=0):
            res = P.run_pipeline_on_task(self.TASK, claude_path="/c",
                                         project_dir=self.tmp, emit=self.emit)
        self.assertTrue(res["ok"])
        self.assertTrue(res["ticked"])
        todo = self.read("jarvis_todo.md")
        self.assertIn(f"- [x] {self.TASK}", todo)
        self.assertIn("[approve, risk=1]", todo)
        # implementer's edit kept
        self.assertEqual(self.read("skills/feature.py"),
                         "NEW body from implementer\n")

    def test_approve_with_warnings_tag_in_done_note(self):
        with mock.patch.object(P, "_run_planner", return_value=self.PLAN), \
             mock.patch.object(P, "_run_implementer",
                               side_effect=self._emit_implementer_change()), \
             mock.patch.object(P, "_run_reviewer",
                               return_value={"verdict": "approve_with_warnings",
                                             "risk_score": 4, "concerns": ["nit"]}), \
             mock.patch.object(P, "_run_tester",
                               return_value={"ok": True, "rc": 0}), \
             mock.patch.object(P, "_kill_jarvis", return_value=0):
            res = P.run_pipeline_on_task(self.TASK, claude_path="/c",
                                         project_dir=self.tmp, emit=self.emit)
        self.assertTrue(res["ticked"])
        self.assertIn("[approve+warnings, risk=4]", self.read("jarvis_todo.md"))

    def test_tester_fail_rolls_back_and_queues_regression(self):
        with mock.patch.object(P, "_run_planner", return_value=self.PLAN), \
             mock.patch.object(P, "_run_implementer",
                               side_effect=self._emit_implementer_change()), \
             mock.patch.object(P, "_run_reviewer",
                               return_value={"verdict": "approve",
                                             "risk_score": 1, "concerns": []}), \
             mock.patch.object(P, "_run_tester",
                               return_value={"ok": False,
                                             "error": "APPCRASH detected"}), \
             mock.patch.object(P, "_kill_jarvis", return_value=0):
            res = P.run_pipeline_on_task(self.TASK, claude_path="/c",
                                         project_dir=self.tmp, emit=self.emit)
        self.assertFalse(res["ok"])
        self.assertEqual(res["stage_failed"], "tester")
        self.assertEqual(self.read("skills/feature.py"), "original body\n")
        todo = self.read("jarvis_todo.md")
        self.assertIn("[regression]", todo)
        self.assertIn("APPCRASH detected", todo)

    def test_no_change_no_claim_routed_to_reviewer_then_rejected(self):
        # Implementer returns rc=0 but makes NO file change and does NOT claim
        # impossible/already_done -> the "made no file changes" notice fires and
        # the reviewer force-rejects on the empty diff.
        def _impl(task_line, plan, **kw):
            return {"rc": 0, "stdout_tail": "", "stderr_tail": "",
                    "note": "", "impossible": False, "already_done": False}
        lines = []
        with mock.patch.object(P, "_run_planner", return_value=self.PLAN), \
             mock.patch.object(P, "_run_implementer", side_effect=_impl):
            res = P.run_pipeline_on_task(self.TASK, claude_path="/c",
                                         project_dir=self.tmp, emit=lines.append)
        # _run_reviewer's local override force-rejects an empty-diff/no-claim run
        self.assertFalse(res["ok"])
        self.assertEqual(res["stage_failed"], "reviewer")
        self.assertIn("made no file changes", "\n".join(lines))

    def test_tester_stage_reports_killed_stale_processes(self):
        # _kill_jarvis returns >0 -> the "killed N stale process(es)" line fires.
        lines = []
        with mock.patch.object(P, "_run_planner", return_value=self.PLAN), \
             mock.patch.object(P, "_run_implementer",
                               side_effect=self._emit_implementer_change()), \
             mock.patch.object(P, "_run_reviewer",
                               return_value={"verdict": "approve",
                                             "risk_score": 0, "concerns": []}), \
             mock.patch.object(P, "_run_tester",
                               return_value={"ok": True, "skipped": True}), \
             mock.patch.object(P, "_kill_jarvis", return_value=2):
            P.run_pipeline_on_task(self.TASK, claude_path="/c",
                                   project_dir=self.tmp, emit=lines.append)
        self.assertIn("killed 2 stale", "\n".join(lines))

    def test_tester_skipped_still_ticks(self):
        with mock.patch.object(P, "_run_planner", return_value=self.PLAN), \
             mock.patch.object(P, "_run_implementer",
                               side_effect=self._emit_implementer_change()), \
             mock.patch.object(P, "_run_reviewer",
                               return_value={"verdict": "approve",
                                             "risk_score": 0, "concerns": []}), \
             mock.patch.object(P, "_run_tester",
                               return_value={"ok": True, "skipped": True,
                                             "reason": "disabled"}), \
             mock.patch.object(P, "_kill_jarvis", return_value=0):
            res = P.run_pipeline_on_task(self.TASK, claude_path="/c",
                                         project_dir=self.tmp, emit=self.emit)
        self.assertTrue(res["ticked"])

    def test_emit_receives_stage_lines(self):
        lines = []
        with mock.patch.object(P, "_run_planner", return_value=self.PLAN), \
             mock.patch.object(P, "_run_implementer",
                               side_effect=self._emit_implementer_change()), \
             mock.patch.object(P, "_run_reviewer",
                               return_value={"verdict": "approve",
                                             "risk_score": 0, "concerns": []}), \
             mock.patch.object(P, "_run_tester",
                               return_value={"ok": True, "skipped": True}), \
             mock.patch.object(P, "_kill_jarvis", return_value=0):
            P.run_pipeline_on_task(self.TASK, claude_path="/c",
                                   project_dir=self.tmp, emit=lines.append)
        joined = "\n".join(lines)
        self.assertIn("stage 1/4", joined)
        self.assertIn("stage 2/4", joined)
        self.assertIn("stage 3/4", joined)


# ──────────────────────────── loop driver ─────────────────────────────────


class LoopDriverBase(_PipeBase):
    def setUp(self):
        super().setUp()
        self.write("bobert_companion.py", "# main\n")
        # Stub the lazy `from upgrade_jarvis import ...` so the driver doesn't
        # need the real orchestrator. We inject a fake module into sys.modules.
        self._fake_uj = mock.MagicMock()
        self._fake_uj._stability_gate = mock.MagicMock(
            return_value={"verdict": "SKIP", "details": "no gate"})
        self._fake_uj._gate_config = mock.MagicMock(return_value=(999, 300, False))
        self._mods = mock.patch.dict(sys.modules,
                                     {"upgrade_jarvis": self._fake_uj})
        self._mods.start()
        self.addCleanup(self._mods.stop)
        # The final py_compile sweep shells out; stub it.
        self._pc = mock.patch.object(P.subprocess, "run",
                                     return_value=_FakeCompleted(0))
        self._pc.start()
        self.addCleanup(self._pc.stop)

    def _run_loop(self, max_iter=10, task_count=1, argv=None):
        log = os.path.join(self.tmp, "stream.log")
        with mock.patch.object(sys, "argv", argv or ["prog"]):
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = P.run_pipeline_loop_driver(
                    project_dir=self.tmp, claude_path="/c",
                    stream_log=log, max_iter=max_iter, task_count=task_count)
        return rc, buf.getvalue()


class LoopDriverTests(LoopDriverBase):
    def test_drains_queue_and_exits_clean(self):
        self.write_todo("- [ ] only task\n")

        def _on_task(task, **kw):
            # tick the task so the queue empties on the next iteration
            P._tick_task(task, "done")
            return {"ok": True, "ticked": True, "stage_failed": None,
                    "stages": {}}

        with mock.patch.object(P, "run_pipeline_on_task", side_effect=_on_task):
            rc, out = self._run_loop(max_iter=5, task_count=1)
        self.assertEqual(rc, 0)
        self.assertIn("task ticked clean", out)
        self.assertIn("queue empty", out)

    def test_empty_queue_immediate_exit(self):
        self.write_todo("- [x] already done\n")
        with mock.patch.object(P, "run_pipeline_on_task") as on_task:
            rc, out = self._run_loop(max_iter=5, task_count=0)
        on_task.assert_not_called()
        self.assertEqual(rc, 0)
        self.assertIn("queue empty", out)

    def test_no_progress_watchdog_stops_after_limit(self):
        # task never ticks -> NO_PROGRESS_LIMIT (5) consecutive failures -> rc=2
        self.write_todo("- [ ] stuck task\n")
        with mock.patch.object(
                P, "run_pipeline_on_task",
                return_value={"ok": False, "ticked": False,
                              "stage_failed": "reviewer", "stages": {}}):
            rc, out = self._run_loop(max_iter=50, task_count=1)
        self.assertEqual(rc, 2)
        self.assertIn("no task ticked", out)

    def test_pipeline_exception_is_caught(self):
        self.write_todo("- [ ] explode\n")
        calls = {"n": 0}

        def _boom(task, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("stage blew up")
            # second time: tick so we can exit
            P._tick_task(task, "recovered")
            return {"ok": True, "ticked": True, "stage_failed": None, "stages": {}}

        with mock.patch.object(P, "run_pipeline_on_task", side_effect=_boom):
            rc, out = self._run_loop(max_iter=5, task_count=1)
        self.assertIn("pipeline raised on this task", out)
        # recovered on the 2nd iter -> clean exit
        self.assertEqual(rc, 0)

    def test_gate_unavailable_when_import_fails(self):
        # Make the lazy upgrade_jarvis import raise so the gate is disabled.
        self.write_todo("- [ ] t\n")
        broken = mock.MagicMock()
        # Accessing _stability_gate raises -> the `from ... import` fails.
        type(broken)._stability_gate = property(
            lambda self: (_ for _ in ()).throw(ImportError("nope")))

        def _on_task(task, **kw):
            P._tick_task(task, "done")
            return {"ok": True, "ticked": True, "stage_failed": None, "stages": {}}

        with mock.patch.dict(sys.modules, {"upgrade_jarvis": broken}), \
             mock.patch.object(P, "run_pipeline_on_task", side_effect=_on_task):
            rc, out = self._run_loop(max_iter=3, task_count=1)
        self.assertIn("gate disabled this run", out)
        self.assertEqual(rc, 0)

    def test_gate_fail_pauses_without_force_continue(self):
        # 1 completion triggers the gate (interval=1); gate FAILs -> rc=3.
        self.write_todo("- [ ] t1\n- [ ] t2\n- [ ] t3\n")
        self._fake_uj._gate_config = mock.MagicMock(return_value=(1, 300, False))
        self._fake_uj._stability_gate = mock.MagicMock(
            return_value={"verdict": "FAIL", "details": "crashed"})

        def _on_task(task, **kw):
            P._tick_task(task, "done")
            return {"ok": True, "ticked": True, "stage_failed": None, "stages": {}}

        with mock.patch.object(P, "run_pipeline_on_task", side_effect=_on_task):
            rc, out = self._run_loop(max_iter=10, task_count=3)
        self.assertEqual(rc, 3)
        self.assertIn("FAIL", out)
        self.assertIn("pausing pipeline", out)

    def test_gate_fail_with_force_continue_resumes(self):
        self.write_todo("- [ ] t1\n")
        self._fake_uj._gate_config = mock.MagicMock(return_value=(1, 300, False))
        self._fake_uj._stability_gate = mock.MagicMock(
            return_value={"verdict": "FAIL", "details": "crashed"})

        def _on_task(task, **kw):
            P._tick_task(task, "done")
            return {"ok": True, "ticked": True, "stage_failed": None, "stages": {}}

        with mock.patch.object(P, "run_pipeline_on_task", side_effect=_on_task):
            rc, out = self._run_loop(
                max_iter=5, task_count=1,
                argv=["prog", "--force-continue-after-regression"])
        # queue drains after the single task -> clean exit despite FAIL
        self.assertIn("resuming despite FAIL", out)
        self.assertEqual(rc, 0)

    def test_gate_pass_resumes(self):
        self.write_todo("- [ ] t1\n")
        self._fake_uj._gate_config = mock.MagicMock(return_value=(1, 300, False))
        self._fake_uj._stability_gate = mock.MagicMock(
            return_value={"verdict": "PASS"})

        def _on_task(task, **kw):
            P._tick_task(task, "done")
            return {"ok": True, "ticked": True, "stage_failed": None, "stages": {}}

        with mock.patch.object(P, "run_pipeline_on_task", side_effect=_on_task):
            rc, out = self._run_loop(max_iter=5, task_count=1)
        self.assertIn("PASS", out)
        self.assertEqual(rc, 0)

    def test_gate_raises_is_handled(self):
        self.write_todo("- [ ] t1\n")
        self._fake_uj._gate_config = mock.MagicMock(return_value=(1, 300, False))
        self._fake_uj._stability_gate = mock.MagicMock(
            side_effect=RuntimeError("gate crashed"))

        def _on_task(task, **kw):
            P._tick_task(task, "done")
            return {"ok": True, "ticked": True, "stage_failed": None, "stages": {}}

        with mock.patch.object(P, "run_pipeline_on_task", side_effect=_on_task):
            rc, out = self._run_loop(max_iter=5, task_count=1)
        self.assertIn("gate raised", out)
        self.assertEqual(rc, 0)

    def test_final_py_compile_failure_reported(self):
        self.write_todo("- [x] done\n")  # empty queue -> straight to sweep
        # override the module-wide subprocess stub: py_compile returns rc=1
        self._pc.stop()
        try:
            with mock.patch.object(P.subprocess, "run",
                                   return_value=_FakeCompleted(1)):
                rc, out = self._run_loop(max_iter=2, task_count=0)
        finally:
            self._pc.start()  # so addCleanup's stop() has a live patcher
        self.assertIn("SYNTAX ERROR", out)

    def test_final_py_compile_raises_is_reported(self):
        self.write_todo("- [x] done\n")
        self._pc.stop()
        try:
            with mock.patch.object(P.subprocess, "run",
                                   side_effect=OSError("python gone")):
                rc, out = self._run_loop(max_iter=2, task_count=0)
        finally:
            self._pc.start()
        self.assertIn("py_compile failed to run", out)

    def test_unopenable_stream_log_still_runs(self):
        # open(stream_log) raises -> log_fp=None; the loop must still run and
        # emit only to stdout (covers the log_fp-None branches in emit()).
        self.write_todo("- [x] done\n")
        real_open = __import__("builtins").open

        def _open(path, *a, **k):
            if path == os.path.join(self.tmp, "stream.log"):
                raise OSError("cannot open log")
            return real_open(path, *a, **k)

        with mock.patch("builtins.open", _open):
            rc, out = self._run_loop(max_iter=2, task_count=0)
        self.assertEqual(rc, 0)
        self.assertIn("queue empty", out)

    def test_gate_skip_branch(self):
        # gate returns SKIP -> prints the details line, loop continues.
        self.write_todo("- [ ] t1\n")
        self._fake_uj._gate_config = mock.MagicMock(return_value=(1, 300, False))
        self._fake_uj._stability_gate = mock.MagicMock(
            return_value={"verdict": "SKIP", "details": "skipped: prod busy"})

        def _on_task(task, **kw):
            P._tick_task(task, "done")
            return {"ok": True, "ticked": True, "stage_failed": None, "stages": {}}

        with mock.patch.object(P, "run_pipeline_on_task", side_effect=_on_task):
            rc, out = self._run_loop(max_iter=5, task_count=1)
        self.assertIn("skipped: prod busy", out)
        self.assertEqual(rc, 0)

    def test_resolved_but_not_ticked_branch(self):
        # result ok but not ticked (e.g. todo write race) -> "investigate" line,
        # counts as no-progress.
        self.write_todo("- [ ] t\n")
        with mock.patch.object(
                P, "run_pipeline_on_task",
                return_value={"ok": True, "ticked": False,
                              "stage_failed": None, "stages": {}}):
            rc, out = self._run_loop(max_iter=50, task_count=1)
        self.assertIn("not ticked", out)
        # never makes progress -> watchdog trips
        self.assertEqual(rc, 2)

    @unittest.skipUnless(_IS_WIN, "ctypes.windll only exists on Windows")
    def test_ansi_enable_swallows_errors_on_windows(self):
        # Force the windll color-enable path to raise; it must be swallowed.
        self.write_todo("- [x] done\n")
        import ctypes
        with mock.patch.object(ctypes, "windll", create=True) as wd:
            wd.kernel32.GetStdHandle.side_effect = OSError("no console")
            rc, _out = self._run_loop(max_iter=2, task_count=0)
        self.assertEqual(rc, 0)


# ──────────────────────────── CLI / discovery ─────────────────────────────


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


class CliTests(_PipeBase):
    def test_no_claude_returns_127(self):
        with mock.patch.object(P, "_find_claude_cli", return_value=None):
            buf = io.StringIO()
            with redirect_stdout(buf), redirect_stderr(buf):
                rc = P._cli(["--task", "x"])
        self.assertEqual(rc, 127)
        self.assertIn("claude CLI not found", buf.getvalue())

    def test_task_mode_runs_pipeline_and_reports(self):
        with mock.patch.object(P, "_find_claude_cli", return_value="/c"), \
             mock.patch.object(P, "run_pipeline_on_task",
                               return_value={"ok": True, "ticked": True,
                                             "stage_failed": None}) as run:
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = P._cli(["--task", "do a thing"])
        self.assertEqual(rc, 0)
        # the exact task string is forwarded
        self.assertEqual(run.call_args.args[0], "do a thing")
        printed = json.loads(buf.getvalue())
        self.assertTrue(printed["ok"])

    def test_task_mode_failure_returns_1(self):
        with mock.patch.object(P, "_find_claude_cli", return_value="/c"), \
             mock.patch.object(P, "run_pipeline_on_task",
                               return_value={"ok": False, "ticked": False,
                                             "stage_failed": "reviewer"}):
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = P._cli(["--task", "x"])
        self.assertEqual(rc, 1)

    def test_first_unchecked_mode_picks_task(self):
        self.write_todo("- [x] done\n- [ ] pick me\n")
        with mock.patch.object(P, "_find_claude_cli", return_value="/c"), \
             mock.patch.object(P, "run_pipeline_on_task",
                               return_value={"ok": True, "ticked": True,
                                             "stage_failed": None}) as run:
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = P._cli(["--first-unchecked"])
        self.assertEqual(rc, 0)
        self.assertEqual(run.call_args.args[0], "pick me")

    def test_first_unchecked_empty_queue(self):
        self.write_todo("- [x] all done\n")
        with mock.patch.object(P, "_find_claude_cli", return_value="/c"), \
             mock.patch.object(P, "run_pipeline_on_task") as run:
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = P._cli(["--first-unchecked"])
        run.assert_not_called()
        self.assertEqual(rc, 0)
        self.assertIn("QUEUE EMPTY", buf.getvalue())

    def test_loop_mode_invokes_driver_and_relaunch(self):
        fake_uj = mock.MagicMock()
        with mock.patch.object(P, "_find_claude_cli", return_value="/c"), \
             mock.patch.object(P, "run_pipeline_loop_driver",
                               return_value=0) as driver, \
             mock.patch.object(P, "_count_unchecked", return_value=3), \
             mock.patch.dict(sys.modules, {"upgrade_jarvis": fake_uj}):
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = P._cli(["--loop", "--max-iter", "7"])
        self.assertEqual(rc, 0)
        driver.assert_called_once()
        self.assertEqual(driver.call_args.kwargs["max_iter"], 7)
        fake_uj.relaunch_jarvis.assert_called_once()

    def test_loop_mode_relaunch_failure_is_swallowed(self):
        broken = mock.MagicMock()
        broken.relaunch_jarvis.side_effect = RuntimeError("singleton lock")
        with mock.patch.object(P, "_find_claude_cli", return_value="/c"), \
             mock.patch.object(P, "run_pipeline_loop_driver", return_value=0), \
             mock.patch.object(P, "_count_unchecked", return_value=0), \
             mock.patch.dict(sys.modules, {"upgrade_jarvis": broken}):
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = P._cli(["--loop"])
        self.assertEqual(rc, 0)
        self.assertIn("ambient relaunch skipped", buf.getvalue())

    def test_mutually_exclusive_group_required(self):
        # no action flag -> argparse errors out with SystemExit(2)
        with self.assertRaises(SystemExit):
            buf = io.StringIO()
            with redirect_stdout(buf), mock.patch.object(sys, "stderr", buf):
                P._cli([])


if __name__ == "__main__":
    unittest.main()
