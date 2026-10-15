"""Tests for the SAFE self-upgrade: tools/selfup_runner.py (+ the stage library
tools/multi_agent_pipeline.py and the proposals-only overnight engine).

SAFETY CONTRACT (this suite NEVER runs a real upgrade):
  * `claude` is always a fake: the stage functions are patched, or
    `mp.subprocess.run` is replaced by a dispatcher that answers claude argv
    from a script and lets ONLY `git` through.
  * The "repo" is a throwaway one in a temp dir (a bare `origin`, a `live`
    clone, worktrees as siblings of `live`). Nothing touches C:\\JARVIS.
  * The gate (`tools/gate_run.py` + `run_tests.py`) is a fake.
  * Report / ledger / log paths are redirected into the temp dir.
"""
import ast
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import overnight_upgrade as ovn
from core import claude_accounts as ca
from core import config
from tools import multi_agent_pipeline as mp
from tools import selfup_runner as S

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REAL_RUN = subprocess.run
LIMIT_TEXT = "You've hit your session limit \u00b7 resets 9pm"


def git(cwd, *args):
    r = _REAL_RUN(["git", *args], cwd=cwd, capture_output=True, text=True,
                  encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise AssertionError(f"git {args} failed: {r.stderr}")
    return r.stdout.strip()


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


# ── scripted fakes for the three Claude stages ───────────────────────────────
def planner_ok(task, *, claude_path, project_dir):
    return {"files_to_touch": ["skills/feature.py"], "regression_risks": [],
            "tests_to_run": [], "approach": "edit the file"}


def reviewer(verdict="approve", risk=1, **extra):
    def fn(task, plan, diff, **kw):
        out = {"verdict": verdict, "risk_score": risk, "concerns": []}
        out.update(extra)
        return out
    return fn


def implementer(edits, note="did it", rc=0, **extra):
    """An implementer that writes ``edits`` ({relpath: text}) into the cwd it
    is given - which must be the worktree."""
    def fn(task, plan, *, claude_path, project_dir):
        for rel, text in edits.items():
            write(os.path.join(project_dir, rel), text)
        out = {"rc": rc, "stdout_tail": "", "stderr_tail": "", "note": note,
               "impossible": False, "already_done": False}
        out.update(extra)
        return out
    return fn


class _Case(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        t = self.tmp.name
        self.origin = os.path.join(t, "origin.git")
        self.live = os.path.join(t, "JARVIS")
        self.report_dir = os.path.join(t, "reports")
        git(t, "init", "-q", "--bare", "-b", "main", self.origin)
        git(t, "init", "-q", "-b", "main", self.live)
        git(self.live, "config", "user.name", "Test Owner")
        git(self.live, "config", "user.email", "owner@example.com")
        write(os.path.join(self.live, ".gitignore"),
              "data/*\n!data/.gitkeep\njarvis_todo.md\njarvis_todo_proposed.md\n__pycache__/\n")
        write(os.path.join(self.live, "skills", "feature.py"), "VALUE = 1\n")
        write(os.path.join(self.live, "skills", "other.py"), "OTHER = 1\n")
        write(os.path.join(self.live, "VERSION"), "1.0.0\n")
        write(os.path.join(self.live, "data", ".gitkeep"), "")
        git(self.live, "add", "-A")
        git(self.live, "commit", "-q", "-m", "base")
        git(self.live, "remote", "add", "origin", self.origin)
        git(self.live, "push", "-q", "-u", "origin", "main")
        self.todo = os.path.join(self.live, "jarvis_todo.md")
        write(self.todo, "# queue\n- [ ] first task\n- [ ] second task\n- [ ] third task\n")
        self.base_sha = git(self.live, "rev-parse", "HEAD")
        self.speech = []

        # keep every side file in the temp dir
        patch = mock.patch.object(mp, "PIPELINE_LOG",
                                  os.path.join(t, "data", "pipeline_runs.jsonl"))
        patch.start()
        self.addCleanup(patch.stop)
        # a real run needs claude only through the (patched) stage functions
        self.stage_patches = []
        for name, fn in (("_run_planner", planner_ok),
                         ("_run_reviewer", reviewer())):
            p = mock.patch.object(mp, name, side_effect=fn)
            p.start()
            self.stage_patches.append(p)
            self.addCleanup(p.stop)
        self._stamp_n = 0
        self.impl_patches = []

    # -- helpers --------------------------------------------------------
    def use_real_stages(self):
        """Hand the planner / reviewer / implementer back to the real functions
        (the test then scripts a fake `claude` underneath them)."""
        for p in self.stage_patches + self.impl_patches:
            p.stop()

    def next_stamp(self):
        self._stamp_n += 1
        return "20261015-09%04d" % self._stamp_n

    def set_impl(self, fn):
        p = mock.patch.object(mp, "_run_implementer", side_effect=fn)
        p.start()
        self.impl_patches.append(p)
        self.addCleanup(p.stop)

    def set_reviewer(self, fn):
        p = mock.patch.object(mp, "_run_reviewer", side_effect=fn)
        p.start()
        self.addCleanup(p.stop)

    def go(self, gate_ok=True, **kw):
        gate_calls = []

        def gate(wt, report_dir, stamp):
            gate_calls.append((wt, report_dir, stamp))
            return {"ok": gate_ok, "tests_ok": gate_ok, "tests_rc": 0 if gate_ok else 1,
                    "pii_ok": True, "pii_rc": 0, "log": "gate.log",
                    "budget_s": S.GATE_TOTAL_TIMEOUT_S}

        lines = []
        kw.setdefault("stamp", "20261015-010203")
        r = S.run(project_dir=self.live, claude_path="/fake/claude",
                  todo_path=self.todo, report_dir=self.report_dir,
                  emit=lines.append, gate_fn=gate, **kw)
        self.gate_calls, self.lines = gate_calls, lines
        return r

    def commits(self, r):
        return git(self.live, "rev-list", "--count", f"origin/main..{r['branch']}")

    def live_snapshot(self):
        return (git(self.live, "rev-parse", "HEAD"),
                git(self.live, "status", "--porcelain"),
                read(os.path.join(self.live, "skills", "feature.py")))


# ═══════════════════════ 1. request-only: no auto trigger ═══════════════════════
def code_mentions(tree, needle):
    """True when ``needle`` appears in CODE (an import, a name, an attribute or
    a string literal) - docstrings and comments don't count."""
    docs = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef)) and n.body                 and isinstance(n.body[0], ast.Expr)                 and isinstance(getattr(n.body[0], "value", None), ast.Constant):
            docs.add(id(n.body[0].value))
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant) and isinstance(n.value, str)                 and id(n) not in docs and needle in n.value:
            return True
        if isinstance(n, ast.alias) and needle in n.name:
            return True
        if isinstance(n, ast.ImportFrom) and needle in (n.module or ""):
            return True
        if isinstance(n, ast.Attribute) and needle in n.attr:
            return True
        if isinstance(n, ast.Name) and needle in n.id:
            return True
    return False


class NoAutoTriggerTests(unittest.TestCase):
    def _tree(self, path):
        with open(os.path.join(_ROOT, path), encoding="utf-8") as f:
            return ast.parse(f.read())

    def _function_calls(self, path, func):
        with open(os.path.join(_ROOT, path), encoding="utf-8") as f:
            tree = ast.parse(f.read())
        node = next(n for n in ast.walk(tree)
                    if isinstance(n, ast.FunctionDef) and n.name == func)
        names = set()
        for n in ast.walk(node):
            if isinstance(n, ast.Call):
                f = n.func
                names.add(f.id if isinstance(f, ast.Name)
                          else f.attr if isinstance(f, ast.Attribute) else "")
        return names, ast.dump(node)

    def test_the_idle_thread_never_starts_a_run(self):
        calls, dump = self._function_calls("bobert_companion.py",
                                           "_overnight_upgrade_thread")
        for forbidden in ("_act_upgrade", "_act_start_overnight_upgrade", "Popen",
                          "run", "system"):
            self.assertNotIn(forbidden, calls)
        node = next(n for n in ast.walk(self._tree("bobert_companion.py"))
                    if isinstance(n, ast.FunctionDef)
                    and n.name == "_overnight_upgrade_thread")
        for text in ("selfup_runner", "upgrade_jarvis", "_act_upgrade"):
            self.assertFalse(code_mentions(node, text), text)

    def test_the_idle_thread_writes_only_proposals(self):
        _calls, dump = self._function_calls("bobert_companion.py",
                                            "_overnight_upgrade_thread")
        self.assertIn("_append_tasks", dump)           # -> jarvis_todo_proposed.md
        self.assertNotIn("TODO_FILE", dump)

    def test_the_standalone_engine_has_no_upgrade_spawn(self):
        self.assertFalse(hasattr(ovn, "_run_upgrade"))
        self.assertFalse(hasattr(ovn, "UPGRADE_SCRIPT"))
        with open(os.path.join(_ROOT, "overnight_upgrade.py"), encoding="utf-8") as f:
            src = f.read()
        self.assertNotIn("--relaunch", src)
        self.assertNotIn("Popen", src)

    def test_a_cycle_generates_proposals_and_spawns_nothing(self):
        with tempfile.TemporaryDirectory() as td:
            todo, prop = os.path.join(td, "jarvis_todo.md"), os.path.join(td, "p.md")
            with mock.patch.object(ovn, "TODO_FILE", todo), \
                 mock.patch.object(ovn, "PROPOSED_FILE", prop), \
                 mock.patch.object(ovn, "OVERNIGHT_LOG", os.path.join(td, "o.log")), \
                 mock.patch.object(ovn, "_read_memory_summary", return_value=""), \
                 mock.patch.object(ovn, "_read_codebase_features", return_value=""), \
                 mock.patch.object(ovn, "_read_logs_tail", return_value=""), \
                 mock.patch.object(ovn, "_get_recent_logs", return_value=[]), \
                 mock.patch.object(ovn, "_generate_ideas", return_value=["idea A", "idea B"]), \
                 mock.patch.object(subprocess, "Popen",
                                   side_effect=AssertionError("spawned")), \
                 mock.patch.object(subprocess, "run",
                                   side_effect=AssertionError("spawned")):
                self.assertTrue(ovn.run_one_cycle())
            self.assertFalse(os.path.exists(todo))
            self.assertIn("idea A", read(prop))

    def test_the_generator_pauses_while_proposals_pile_up(self):
        with tempfile.TemporaryDirectory() as td:
            prop = os.path.join(td, "p.md")
            write(prop, "".join(f"- [ ] idea {i}\n" for i in range(ovn.MAX_OPEN_PROPOSALS)))
            with mock.patch.object(ovn, "PROPOSED_FILE", prop), \
                 mock.patch.object(ovn, "OVERNIGHT_LOG", os.path.join(td, "o.log")), \
                 mock.patch.object(ovn, "_generate_ideas",
                                   side_effect=AssertionError("asked Claude")):
                self.assertFalse(ovn.run_one_cycle())

    def test_the_runner_is_started_only_from_an_explicit_request(self):
        # Every module that names the runner is a request path: the action
        # handler (voice / tray) and the upgrade_jarvis CLI.
        users = []
        for rel in ("bobert_companion.py", "overnight_upgrade.py", "core/actions.py",
                    "upgrade_jarvis.py", "tray.py", "core/diagnostic_daemons.py"):
            if code_mentions(self._tree(rel), "selfup_runner"):
                users.append(rel)
        self.assertEqual(sorted(users), ["core/actions.py", "upgrade_jarvis.py"])

    def test_no_spawn_site_uses_the_blanket_permission_bypass(self):
        import re
        pat = re.compile(r"""["']--dangerously-skip-permissions["']""")
        for rel in ("tools/multi_agent_pipeline.py", "tools/selfup_runner.py",
                    "upgrade_jarvis.py", "overnight_upgrade.py", "core/actions.py",
                    "bobert_companion.py"):
            with open(os.path.join(_ROOT, rel), encoding="utf-8") as f:
                self.assertIsNone(pat.search(f.read()), rel)


# ═════════════════════════ 2. proposals vs the approved queue ═════════════════════════
class ProposalsTests(unittest.TestCase):
    def test_proposals_never_reach_the_queue(self):
        with tempfile.TemporaryDirectory() as td:
            todo, prop = os.path.join(td, "jarvis_todo.md"), os.path.join(td, "p.md")
            write(todo, "- [ ] approved one\n")
            with mock.patch.object(ovn, "TODO_FILE", todo), \
                 mock.patch.object(ovn, "OVERNIGHT_LOG", os.path.join(td, "o.log")), \
                 mock.patch.object(ovn, "PROPOSED_FILE", prop):
                n = ovn._append_tasks(["fresh idea", "approved one"])   # 2nd = duplicate
            self.assertEqual(n, 1)
            self.assertEqual(read(todo), "- [ ] approved one\n")        # untouched
            self.assertIn("fresh idea", read(prop))
            self.assertEqual(S.approved_tasks(todo), ["approved one"])  # idea not runnable

    def test_machine_written_lines_are_skipped_by_default(self):
        with tempfile.TemporaryDirectory() as td:
            todo = os.path.join(td, "jarvis_todo.md")
            write(todo, "- [ ] human task\n"
                        "- [ ] **[anomaly]** main loop stuck\n"
                        "- [ ] [self-heal] lower VAD threshold\n"
                        "- [ ] **2026-10-01** [Self-Heal] mixed case\n"
                        "- [x] done task\n"
                        "- [ ] another human task\n")
            self.assertEqual(S.approved_tasks(todo), ["human task", "another human task"])
            self.assertEqual(len(S.approved_tasks(todo, include_machine=True)), 5)


# ═════════════════════════════ 3. isolation ═════════════════════════════
class IsolationTests(_Case):
    def test_a_run_works_in_a_fresh_worktree_and_never_touches_the_live_tree(self):
        self.set_impl(implementer({"skills/feature.py": "VALUE = 2\n"}))
        before = self.live_snapshot()
        r = self.go(max_tasks=1)
        self.assertEqual(r["branch"], "auto/selfup-20261015-010203")
        wt = r["worktree"]
        self.assertEqual(os.path.basename(wt), "jarvis_wt_selfup_20261015-010203")
        self.assertEqual(os.path.dirname(wt), os.path.dirname(self.live))
        self.assertEqual(read(os.path.join(wt, "skills", "feature.py")), "VALUE = 2\n")
        self.assertEqual(self.live_snapshot(), before)         # live HEAD, status, file
        self.assertEqual(git(self.live, "rev-parse", "--abbrev-ref", "HEAD"), "main")

    def test_every_subprocess_is_git_and_none_kills_jarvis(self):
        seen = []
        real_popen = subprocess.Popen

        def spy(cmd, *a, **kw):
            seen.append(list(cmd))
            return real_popen(cmd, *a, **kw)

        self.set_impl(implementer({"skills/feature.py": "VALUE = 2\n"}))
        with mock.patch.object(subprocess, "Popen", side_effect=spy):
            self.go(max_tasks=1)
        self.assertTrue(seen)
        for cmd in seen:
            self.assertEqual(cmd[0], "git", cmd)
            text = " ".join(cmd).lower()
            for bad in ("powershell", "taskkill", "stop-process", "_boot_jarvis",
                        "bobert_companion", "upgrade_jarvis"):
                self.assertNotIn(bad, text)
        # the only git verbs a run needs - nothing that publishes or moves refs
        verbs = {c[1] for c in seen}
        self.assertFalse(verbs & {"push", "tag", "merge", "pull", "rebase",
                                  "cherry-pick", "stash"}, verbs)

    def test_destructive_git_refuses_the_live_checkout(self):
        with self.assertRaises(RuntimeError):
            S._revert(self.live, self.live)
        with self.assertRaises(RuntimeError):
            S._revert(os.path.join(self.live, "skills"), self.live)
        self.assertEqual(git(self.live, "rev-parse", "HEAD"), self.base_sha)

    def test_headless_smoke_only_parses_files(self):
        wt = self.live          # any dir: _smoke never imports or executes
        write(os.path.join(wt, "ok.py"), "x = 1\n")
        write(os.path.join(wt, "bad.py"), "def (:\n")
        self.assertEqual(S._smoke(wt, ["ok.py", "notes.md"]), (True, "ok"))
        ok, detail = S._smoke(wt, ["ok.py", "bad.py"])
        self.assertFalse(ok)
        self.assertIn("bad.py", detail)
        self.assertFalse(os.path.exists(os.path.join(wt, "__pycache__")))


# ═══════════════════════ 4. Claude is locked down in the worktree ═══════════════════════
class LockDownIntegrationTests(_Case):
    def test_real_stage_functions_get_the_right_flags_and_the_worktree_cwd(self):
        # Undo the stage patches: drive the REAL planner / implementer /
        # reviewer against a scripted fake claude (only git passes through).
        self.use_real_stages()
        calls = []
        acct = mock.patch.object(ca, "choose", return_value=(ca.MAX, "test"))
        acct.start()
        self.addCleanup(acct.stop)

        def fake(cmd, **kw):
            if cmd[0] == "git":
                return _REAL_RUN(cmd, **kw)
            calls.append((cmd, kw))
            tools = cmd[cmd.index("--tools") + 1]
            if tools == "Read,Glob,Grep" and len(calls) == 1:
                out = json.dumps({"files_to_touch": ["skills/feature.py"],
                                  "regression_risks": [], "tests_to_run": [],
                                  "approach": "edit"})
            elif tools == "Read,Glob,Grep":
                out = json.dumps({"verdict": "approve", "risk_score": 1, "concerns": []})
            else:
                write(os.path.join(kw["cwd"], "skills", "feature.py"), "VALUE = 3\n")
                out = "DONE: bumped VALUE"
            return subprocess.CompletedProcess(cmd, 0, out, "")

        with mock.patch.object(mp.subprocess, "run", side_effect=fake):
            r = self.go(max_tasks=1)
        self.assertEqual(len(r["done"]), 1, r)
        self.assertEqual(len(calls), 3)
        wt = r["worktree"]
        for cmd, kw in calls:
            self.assertEqual(os.path.realpath(kw["cwd"]), os.path.realpath(wt))
            self.assertNotEqual(os.path.realpath(kw["cwd"]), os.path.realpath(self.live))
            self.assertNotIn("--dangerously-skip-permissions", cmd)
            self.assertEqual(cmd[cmd.index("--permission-mode") + 1], "dontAsk")
        planner_cmd, impl_cmd, review_cmd = (c for c, _ in calls)
        for cmd in (planner_cmd, review_cmd):             # read-only roles
            self.assertEqual(cmd[cmd.index("--tools") + 1], "Read,Glob,Grep")
            self.assertEqual(cmd[cmd.index("--allowedTools") + 1], "Read,Glob,Grep")
            self.assertIn("Edit", cmd[cmd.index("--disallowedTools") + 1])
            self.assertIn("Bash", cmd[cmd.index("--disallowedTools") + 1])
        deny = impl_cmd[impl_cmd.index("--disallowedTools") + 1]
        for pattern in ("Bash(git push *)", "Bash(git tag *)", "Bash(git merge *)",
                        "Bash(git reset *)", "Bash(git checkout *)", "Edit(data/**)"):
            self.assertIn(pattern, deny)
        self.assertIn("Edit", impl_cmd[impl_cmd.index("--allowedTools") + 1])


# ═══════════════════ 5 + 6. one commit per task; failures are reverted ═══════════════════
class CommitAndRevertTests(_Case):
    def test_one_commit_per_passing_task(self):
        edits = iter([{"skills/feature.py": "VALUE = 2\n"},
                      {"skills/other.py": "OTHER = 2\n"},
                      {"skills/third.py": "THIRD = 1\n"}])

        def impl(task, plan, *, claude_path, project_dir):
            return implementer(next(edits), note=f"did {task}")(
                task, plan, claude_path=claude_path, project_dir=project_dir)

        self.set_impl(impl)
        r = self.go()
        self.assertEqual([d["task"] for d in r["done"]],
                         ["first task", "second task", "third task"])
        self.assertEqual(self.commits(r), "3")
        shas = git(self.live, "rev-list", "--reverse", f"origin/main..{r['branch']}").split()
        self.assertEqual(shas, [d["sha"] for d in r["done"]])
        for sha, path in zip(shas, ("skills/feature.py", "skills/other.py", "skills/third.py")):
            self.assertEqual(git(self.live, "show", "--name-only", "--format=", sha), path)
            self.assertTrue(git(self.live, "log", "-1", "--format=%s", sha).startswith("selfup: "))
        self.assertEqual(git(self.live, "status", "--porcelain", "-uall"), "")   # live clean
        self.assertEqual(git(r["worktree"], "status", "--porcelain"), "")        # wt clean

    def test_a_rejected_task_is_reverted_with_git_and_logged(self):
        self.set_impl(implementer({"skills/feature.py": "VALUE = 99\n",
                                   "skills/brand_new.py": "NEW = 1\n"}))
        self.set_reviewer(reviewer("reject_and_redo", 9, concerns=["breaks the loop"]))
        r = self.go(max_tasks=1)
        self.assertEqual(r["done"], [])
        self.assertEqual(len(r["failed"]), 1)
        self.assertEqual(r["failed"][0]["stage"], "reviewer")
        self.assertEqual(r["worktree"], "")                      # nothing kept -> cleaned up
        self.assertFalse(os.path.exists(os.path.join(
            os.path.dirname(self.live), "jarvis_wt_selfup_20261015-010203")))
        self.assertEqual(git(self.live, "branch", "--list", "auto/*"), "")
        self.assertEqual(self.gate_calls, [])                    # no gate for an empty branch
        ledger = json.loads(read(os.path.join(self.report_dir, "ledger.json")))
        self.assertEqual(list(ledger.values())[0]["status"], "failed")
        events = [json.loads(ln) for ln in read(mp.PIPELINE_LOG).splitlines()]
        self.assertEqual(events[-1]["stage"], "reviewer")

    def test_a_failure_mid_run_reverts_only_that_task(self):
        seq = iter([{"skills/feature.py": "VALUE = 2\n"},
                    {"skills/other.py": "OTHER = BROKEN(\n"},     # fails the smoke
                    {"skills/third.py": "THIRD = 1\n"}])

        def impl(task, plan, *, claude_path, project_dir):
            return implementer(next(seq))(task, plan, claude_path=claude_path,
                                          project_dir=project_dir)

        self.set_impl(impl)
        r = self.go()
        self.assertEqual([d["task"] for d in r["done"]], ["first task", "third task"])
        self.assertEqual([d["stage"] for d in r["failed"]], ["smoke"])
        self.assertEqual(self.commits(r), "2")
        self.assertEqual(read(os.path.join(r["worktree"], "skills", "other.py")), "OTHER = 1\n")
        self.assertEqual(git(r["worktree"], "status", "--porcelain", "-uall"), "")

    def test_high_risk_and_unparseable_risk_fail_closed(self):
        for score in (7, 9, float("inf"), "high", None, [9]):
            with self.subTest(score=score):
                self.set_impl(implementer({"skills/feature.py": "VALUE = 5\n"}))
                self.set_reviewer(reviewer("approve", score))
                self.todo_reset()
                r = self.go(max_tasks=1, stamp=self.next_stamp())
                self.assertEqual(r["done"], [], score)
                self.assertEqual(r["failed"][0]["stage"], "reviewer")

    def test_risk_just_below_the_threshold_is_committed(self):
        self.set_impl(implementer({"skills/feature.py": "VALUE = 5\n"}))
        self.set_reviewer(reviewer("approve_with_warnings", 6))
        r = self.go(max_tasks=1)
        self.assertEqual(len(r["done"]), 1)

    def todo_reset(self):
        write(self.todo, "- [ ] first task\n")
        try:
            os.remove(os.path.join(self.report_dir, "ledger.json"))
        except OSError:
            pass

    def test_forbidden_paths_fail_the_task(self):
        for rel in ("data/live_state.json", "VERSION", ".claude/settings.json"):
            with self.subTest(path=rel):
                self.todo_reset()
                self.set_impl(implementer({rel: "x\n", "skills/feature.py": "VALUE = 2\n"}))
                r = self.go(max_tasks=1, stamp=self.next_stamp())
                self.assertEqual(r["done"], [])
                self.assertEqual(r["failed"][0]["stage"], "scope", r)
        self.assertEqual(read(os.path.join(self.live, "VERSION")), "1.0.0\n")

    def test_an_ignored_file_written_under_data_is_caught_too(self):
        self.set_impl(implementer({"data/secrets_dump.json": "{}\n",
                                   "skills/feature.py": "VALUE = 2\n"}))
        r = self.go(max_tasks=1)
        self.assertEqual(r["failed"][0]["stage"], "scope")

    def test_an_implementer_that_commits_is_reverted(self):
        def sneaky(task, plan, *, claude_path, project_dir):
            write(os.path.join(project_dir, "skills/feature.py"), "VALUE = 7\n")
            git(project_dir, "add", "-A")
            git(project_dir, "commit", "-q", "-m", "sneaky")
            return {"rc": 0, "stdout_tail": "", "stderr_tail": "", "note": "x",
                    "impossible": False, "already_done": False}

        self.set_impl(sneaky)
        r = self.go(max_tasks=1)
        self.assertEqual(r["done"], [])
        self.assertIn("moved HEAD", r["failed"][0]["note"])
        self.assertEqual(git(self.live, "rev-parse", "origin/main"), self.base_sha)

    def test_impossible_and_already_done_commit_nothing(self):
        self.set_impl(implementer({}, note="needs hardware", impossible=True))
        r = self.go(max_tasks=1)
        self.assertEqual((r["done"], r["failed"], len(r["skipped"])), ([], [], 1))
        self.todo_reset()
        self.set_impl(implementer({}, note="exists", already_done=True))
        self.set_reviewer(reviewer("already_done", 1))
        r = self.go(max_tasks=1, stamp="20261015-040000")
        self.assertEqual((r["done"], r["failed"], len(r["skipped"])), ([], [], 1))

    def test_a_planner_failure_is_a_failed_task_and_leaves_the_tree_clean(self):
        p = mock.patch.object(mp, "_run_planner",
                              return_value={"error": "planner exited rc=1", "rc": 1})
        p.start()
        self.addCleanup(p.stop)
        r = self.go(max_tasks=1)
        self.assertEqual(r["failed"][0]["stage"], "planner")


# ═══════════════════ 5. the real gate, 7. caps + limits, lock, ledger ═══════════════════
class GateTests(unittest.TestCase):
    def test_gate_runs_the_full_suite_with_an_1800_second_budget_then_pii(self):
        calls = []

        def runner(cmd, **kw):
            calls.append((cmd, kw))
            return subprocess.CompletedProcess(cmd, 0, "", "")

        with tempfile.TemporaryDirectory() as td:
            out = S.run_gate(td, os.path.join(td, "reports"), "stamp1", runner=runner)
        (tests_cmd, tests_kw), (pii_cmd, pii_kw) = calls
        self.assertEqual(tests_cmd[1], "tools/gate_run.py")
        self.assertEqual(tests_cmd[2], os.path.join(td, "reports", "gate_stamp1.log"))
        self.assertEqual(tests_cmd[3:5], ["--cwd", td])
        self.assertEqual(tests_cmd[5:], ["--", sys.executable, "tools/run_tests.py"])
        self.assertEqual(tests_kw["cwd"], td)
        self.assertEqual(tests_kw["env"]["JARVIS_TEST_TOTAL_TIMEOUT_S"], "1800")
        self.assertEqual(S.GATE_TOTAL_TIMEOUT_S, 1800)
        self.assertEqual(pii_cmd[1], "tools/check_no_pii.py")
        self.assertEqual(pii_kw["cwd"], td)
        self.assertTrue(out["ok"] and out["tests_ok"] and out["pii_ok"])
        self.assertEqual(out["budget_s"], 1800)

    def test_either_half_failing_fails_the_gate(self):
        for failing, key in (("run_tests.py", "tests_ok"), ("check_no_pii.py", "pii_ok")):
            def runner(cmd, **kw):
                return subprocess.CompletedProcess(
                    cmd, 1 if failing in " ".join(cmd) else 0, "", "")
            with tempfile.TemporaryDirectory() as td:
                out = S.run_gate(td, td, "s", runner=runner)
            self.assertFalse(out["ok"])
            self.assertFalse(out[key])

    def test_a_gate_that_cannot_start_is_a_failure_not_a_crash(self):
        with tempfile.TemporaryDirectory() as td:
            out = S.run_gate(td, td, "s", runner=mock.Mock(side_effect=OSError("nope")))
        self.assertFalse(out["ok"])
        self.assertIn("tests_error", out)


class RunLevelTests(_Case):
    def test_the_gate_runs_once_after_all_tasks_and_no_release_happens(self):
        self.set_impl(implementer({"skills/feature.py": "VALUE = 2\n"}))
        r = self.go(max_tasks=2)
        self.assertEqual(len(self.gate_calls), 1)
        self.assertEqual(self.gate_calls[0][0], r["worktree"])
        # no VERSION bump, no push, no tag, base untouched
        self.assertEqual(read(os.path.join(r["worktree"], "VERSION")), "1.0.0\n")
        self.assertEqual(git(self.live, "diff", "--name-only", f"origin/main..{r['branch']}"),
                         "skills/feature.py")
        self.assertEqual(git(self.origin, "rev-parse", "main"), self.base_sha)
        self.assertNotIn("auto/", git(self.origin, "branch", "--list"))
        self.assertEqual(git(self.origin, "tag", "--list"), "")
        self.assertEqual(git(self.live, "tag", "--list"), "")

    def test_report_and_spoken_sentence(self):
        self.set_impl(implementer({"skills/feature.py": "VALUE = 2\n"}))
        r = self.go(max_tasks=1)
        self.assertTrue(os.path.isfile(r["report"]))
        md = read(r["report"])
        self.assertIn(r["branch"], md)
        self.assertIn(r["done"][0]["sha"][:10], md)
        self.assertIn("first task", md)
        self.assertIn("PASS", md)
        latest = json.loads(read(os.path.join(self.report_dir, "latest.json")))
        self.assertEqual(latest["spoken"],
                         "The upgrade branch is ready for review, sir: 1 task done, tests passed.")
        self.assertTrue(os.path.basename(r["report"]).startswith("selfup_"))
        self.assertEqual(os.path.dirname(r["report"]), self.report_dir)

    def test_failed_gate_is_reported_honestly(self):
        self.set_impl(implementer({"skills/feature.py": "VALUE = 2\n"}))
        r = self.go(gate_ok=False, max_tasks=1)
        self.assertIn("tests failed", r["spoken"])
        self.assertNotIn("passed", r["spoken"])
        self.assertIn("FAIL", read(r["report"]))

    def test_spoken_summary_variants(self):
        f = S.spoken_summary
        self.assertIn("nothing to run", f({"attempted": 0}))
        self.assertIn("2 of 2 tasks failed", f({"attempted": 2, "done": [], "failed": [1, 2]}))
        self.assertIn("could not start", f({"error": "no origin/main"}))
        self.assertIn("all 2 tasks were skipped",
                      f({"attempted": 2, "done": [], "failed": [], "skipped": [1, 2]}))
        two = {"attempted": 2, "done": [1, 2], "failed": [],
               "gate": {"ok": True, "tests_ok": True}}
        self.assertEqual(f(two), "The upgrade branch is ready for review, sir: "
                                 "2 tasks done, tests passed.")
        two["stopped"] = "limit"
        self.assertIn("usage limit", f(two))

    def test_missing_origin_main_reports_and_starts_nothing(self):
        git(self.live, "remote", "remove", "origin")
        r = self.go()
        self.assertIn("origin/main", r["error"])
        self.assertTrue(os.path.isfile(os.path.join(self.report_dir, "latest.json")))
        self.assertIn("could not start", r["spoken"])

    def test_empty_approved_queue_does_nothing(self):
        write(self.todo, "- [x] done\n- [ ] **[anomaly]** machine line\n")
        r = self.go()
        self.assertEqual(r["attempted"], 0)
        self.assertEqual(r["branch"], "")
        self.assertEqual(git(self.live, "worktree", "list").count("\n"), 0)

    def test_done_tasks_are_not_repeated_by_the_next_run(self):
        self.set_impl(implementer({"skills/feature.py": "VALUE = 2\n"}))
        self.go(max_tasks=1)
        r = self.go(max_tasks=1, stamp="20261015-050000")
        self.assertEqual([d["task"] for d in r["done"]], ["second task"])

    def test_a_second_run_is_refused_while_one_is_live(self):
        os.makedirs(self.report_dir, exist_ok=True)
        S._write_json_atomic(os.path.join(self.report_dir, ".running"),
                             {"pid": os.getpid(), "at": __import__("time").time()})
        self.assertTrue(S.is_running(self.report_dir))
        r = self.go()
        self.assertIn("already in progress", r["error"])
        S._release(self.report_dir)
        self.assertFalse(S.is_running(self.report_dir))

    def test_a_stale_lock_is_ignored(self):
        os.makedirs(self.report_dir, exist_ok=True)
        S._write_json_atomic(os.path.join(self.report_dir, ".running"),
                             {"pid": os.getpid(), "at": 1.0})
        self.assertFalse(S.is_running(self.report_dir))

    def test_the_lock_is_released_even_when_a_task_raises(self):
        p = mock.patch.object(S, "run_one_task", side_effect=RuntimeError("boom"))
        p.start()
        self.addCleanup(p.stop)
        r = self.go(max_tasks=1)
        self.assertEqual(r["failed"][0]["stage"], "exception")
        self.assertFalse(S.is_running(self.report_dir))

    # -- cost guard -------------------------------------------------------
    def test_budget_cap_stops_at_a_task_boundary(self):
        self.set_impl(implementer({"skills/feature.py": "VALUE = 2\n"}))
        with mock.patch.dict(os.environ, {"JARVIS_PIPELINE_MAX_USD": "1.0"}):
            r = self.go()
        self.assertEqual(r["stopped"], "budget")
        self.assertEqual(r["attempted"], 1)          # one task (est. $1.75) hit the $1 cap

    def test_consecutive_failures_stop_the_run(self):
        write(self.todo, "".join(f"- [ ] task {i}\n" for i in range(9)))
        self.set_impl(implementer({}, rc=1))
        r = self.go(max_tasks=9)
        self.assertEqual(r["stopped"], "no-progress")
        self.assertEqual(r["attempted"], S.NO_PROGRESS_LIMIT)

    def test_task_cap(self):
        self.set_impl(implementer({"skills/feature.py": "VALUE = 2\n"}))
        r = self.go(max_tasks=2)
        self.assertEqual(r["attempted"], 2)

    # -- Pro -> Max -> stop -------------------------------------------------
    def _account_world(self):
        pro = os.path.join(self.tmp.name, "claude-pro")
        write(os.path.join(pro, ca.SIGN_IN_FILE), "{}")
        for p in (mock.patch.object(ca, "STATE_PATH",
                                    os.path.join(self.tmp.name, "acct_state.json")),
                  mock.patch.multiple(config, CLAUDE_PRO_FIRST_UNTIL="2099-01-01",
                                      CLAUDE_PRO_CONFIG_DIR=pro)):
            p.start()
            self.addCleanup(p.stop)
        self.pro_dir = pro

    def _real_stages_with_fake_claude(self, answer):
        self.use_real_stages()
        calls = []

        def fake(cmd, **kw):
            if cmd[0] == "git":
                return _REAL_RUN(cmd, **kw)
            profile = "pro" if kw["env"].get("CLAUDE_CONFIG_DIR") == self.pro_dir else "max"
            calls.append(profile)
            return answer(profile, cmd, kw)

        q = mock.patch.object(mp.subprocess, "run", side_effect=fake)
        q.start()
        self.addCleanup(q.stop)
        return calls

    def test_pro_limit_falls_back_to_max_and_the_run_continues(self):
        self._account_world()

        def answer(profile, cmd, kw):
            if profile == "pro":
                return subprocess.CompletedProcess(cmd, 0, LIMIT_TEXT, "")
            tools = cmd[cmd.index("--tools") + 1]
            if tools != "Read,Glob,Grep":
                write(os.path.join(kw["cwd"], "skills", "feature.py"), "VALUE = 3\n")
                return subprocess.CompletedProcess(cmd, 0, "DONE: ok", "")
            if "PLANNER" in cmd[-1]:
                out = {"files_to_touch": [], "regression_risks": [], "approach": "x"}
            else:
                out = {"verdict": "approve", "risk_score": 1, "concerns": []}
            return subprocess.CompletedProcess(cmd, 0, json.dumps(out), "")

        calls = self._real_stages_with_fake_claude(answer)
        r = self.go(max_tasks=1)
        self.assertEqual(len(r["done"]), 1, r)
        self.assertEqual(calls[:2], ["pro", "max"])      # Pro refused -> same call on Max
        self.assertEqual(r["stopped"], "")

    def test_when_max_refuses_too_the_run_stops(self):
        self._account_world()
        calls = self._real_stages_with_fake_claude(
            lambda profile, cmd, kw: subprocess.CompletedProcess(cmd, 0, LIMIT_TEXT, ""))
        r = self.go()
        self.assertEqual(calls, ["pro", "max"])          # exactly one Pro try + one Max try
        self.assertEqual(r["stopped"], "limit")
        self.assertEqual(r["attempted"], 1)              # the other two tasks never started
        self.assertEqual(r["done"], [])
        self.assertIn("usage limit", r["spoken"])
        self.assertIn("1 of 1", r["spoken"])


if __name__ == "__main__":
    unittest.main()
