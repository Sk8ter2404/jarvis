"""Unit tests for the FIRST section of the ``bobert_companion`` monolith.

Covers the top-level functions/classes defined between lines ~50 and ~2704 of
bobert_companion.py: the early-boot singleton-lock helpers, conversation-history
trimming, the legacy-memory merge + credential redaction, system-prompt
assembly, the one-shot LLM helper + JSON-array parser, the action
error/history/session-pattern bookkeeping, startup-pattern detection, the
timestamped log Tee, robot ``send`` / ``set_state`` / ``_now_doing_label``, the
HUD/reticle state writers + geometry helpers, the per-camera failure summary,
and the system-tray command dispatcher.

These are LOCAL full-tier tests: the monolith top-level-imports heavy deps
(numpy/sounddevice/cv2/soundfile/requests) that are absent on the light-deps CI
runner, so ``@requires_monolith`` skips the whole module there and runs it here.

Isolation contract (see the task brief + harness docstring):
  * The monolith is imported ONCE (harness-cached). We never re-import it or
    swap ``sys.modules``.
  * Per-test patches use ``mock.patch.object(self.bc, ...)`` which auto-restore.
  * Anything that mutates a monolith global *in place* (e.g. the module-level
    ``conversation_history`` list, the bounded action deques, the single-element
    state-flag lists) is snapshot/restored in ``addCleanup`` so tests don't leak
    state into each other.
  * No real network/LLM/hardware/threads/sleep/filesystem-of-record: external
    I/O is mocked, ``threading.Thread`` is stubbed to run inline or no-op, and
    file writers are pointed at a per-test temp dir.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from collections import deque
from unittest import mock

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith


# ──────────────────────────────────────────────────────────────────────────
#  Small shared helpers
# ──────────────────────────────────────────────────────────────────────────
class _InlineThread:
    """Drop-in for ``threading.Thread(target=..., daemon=...)`` that runs the
    target synchronously on ``.start()`` so background workers are exercised
    deterministically without spawning a real thread."""

    def __init__(self, target=None, args=(), kwargs=None, daemon=None, **_):
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}

    def start(self):
        if self._target is not None:
            self._target(*self._args, **self._kwargs)

    def join(self, *a, **k):
        return None


class _NoopThread(_InlineThread):
    """Like ``_InlineThread`` but ``.start()`` does nothing — for cases where we
    only want to assert a thread *would* have been spawned, not run its body."""

    def start(self):
        return None


@requires_monolith
class _MonolithTestBase(MonolithGlobalsTestCase):
    """Base that loads the cached monolith once for the whole class and
    deep-restores the mutated bobert_companion globals after each test
    (inherited from ``MonolithGlobalsTestCase``)."""

    # -- generic global snapshot/restore helpers -------------------------------
    def _restore_attr_after(self, name):
        """Snapshot ``self.bc.<name>`` (a list/dict global) and restore its
        *contents* in cleanup so in-place mutation can't leak across tests."""
        bc = self.bc
        original = getattr(bc, name)
        if isinstance(original, list):
            snap = list(original)

            def _restore(o=original, s=snap):
                o[:] = s
        elif isinstance(original, dict):
            snap = dict(original)

            def _restore(o=original, s=snap):
                o.clear()
                o.update(s)
        elif isinstance(original, set):
            snap = set(original)

            def _restore(o=original, s=snap):
                o.clear()
                o.update(s)
        elif isinstance(original, deque):
            snap = list(original)

            def _restore(o=original, s=snap):
                o.clear()
                o.extend(s)
        else:  # pragma: no cover - only list/dict/set/deque globals are passed
            raise TypeError(f"unsupported global type for {name}: {type(original)}")
        self.addCleanup(_restore)
        return original


# ──────────────────────────────────────────────────────────────────────────
#  Early-boot singleton-lock helpers (pure-ish, file-backed)
# ──────────────────────────────────────────────────────────────────────────
class LockHelperTests(_MonolithTestBase):
    def test_read_lock_pid_missing_returns_neg1(self):
        path = os.path.join(tempfile.mkdtemp(), "nope.lock")
        self.assertEqual(self.bc._read_lock_pid(path), -1)

    def test_read_lock_pid_valid(self):
        d = tempfile.mkdtemp()
        path = os.path.join(d, "jarvis.lock")
        with open(path, "w", encoding="utf-8") as f:
            f.write("4242")
        self.assertEqual(self.bc._read_lock_pid(path), 4242)

    def test_read_lock_pid_empty_is_stale_zero(self):
        d = tempfile.mkdtemp()
        path = os.path.join(d, "jarvis.lock")
        with open(path, "w", encoding="utf-8") as f:
            f.write("   ")
        # Empty/whitespace → 0 ("truly stale") after the retry budget. Patch
        # sleep so the 10×50ms retry loop doesn't actually wait.
        with mock.patch.object(self.bc.time, "sleep", return_value=None):
            self.assertEqual(self.bc._read_lock_pid(path, max_retries=2), 0)

    def test_read_lock_pid_unparseable_is_stale_zero(self):
        d = tempfile.mkdtemp()
        path = os.path.join(d, "jarvis.lock")
        with open(path, "w", encoding="utf-8") as f:
            f.write("not-a-pid")
        with mock.patch.object(self.bc.time, "sleep", return_value=None):
            self.assertEqual(self.bc._read_lock_pid(path, max_retries=2), 0)

    def test_acquire_os_singleton_lock_success(self):
        # Lock a fresh real file on byte 0; on a free file this must succeed.
        d = tempfile.mkdtemp()
        path = os.path.join(d, "mutex.lock")
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            self.assertTrue(self.bc._acquire_os_singleton_lock(fd))
        finally:
            os.close(fd)

    def test_acquire_os_singleton_lock_fails_open_on_unexpected(self):
        # An invalid fd makes the inner os.lseek/locking raise something other
        # than the caught OSError path... actually OSError is caught and returns
        # False. To exercise the "fail OPEN" outer guard we force a non-OSError
        # by passing an object whose use raises TypeError deep inside.
        sentinel = object()
        # os.lseek(sentinel, ...) raises TypeError -> outer except -> True.
        self.assertTrue(self.bc._acquire_os_singleton_lock(sentinel))

    def test_early_boot_singleton_lock_reentrant_noop(self):
        # The process-wide sentinel already names our PID (the harness set it),
        # so a re-entrant call short-circuits to True without touching files.
        self.assertEqual(
            os.environ.get("_JARVIS_SINGLETON_PID"), str(os.getpid()))
        self.assertTrue(self.bc._early_boot_singleton_lock())


# ──────────────────────────────────────────────────────────────────────────
#  _is_staging + conversation-history trim
# ──────────────────────────────────────────────────────────────────────────
class StagingAndHistoryTests(_MonolithTestBase):
    def test_is_staging_true_when_role_staging(self):
        with mock.patch.object(self.bc, "BLUE_GREEN_ROLE", "staging"):
            self.assertTrue(self.bc._is_staging())

    def test_is_staging_false_for_other_role(self):
        with mock.patch.object(self.bc, "BLUE_GREEN_ROLE", "production"):
            self.assertFalse(self.bc._is_staging())

    def test_trim_keeps_pairs_and_user_first(self):
        hist = self._restore_attr_after("conversation_history")
        hist.clear()
        # 12 messages (6 user/assistant pairs); trim to 8 should drop 2 oldest
        # pairs from the front and leave a user message first.
        for i in range(6):
            hist.append({"role": "user", "content": f"u{i}"})
            hist.append({"role": "assistant", "content": f"a{i}"})
        self.bc._trim_conversation_history(max_history=8)
        self.assertEqual(len(hist), 8)
        self.assertEqual(hist[0]["role"], "user")
        self.assertEqual(hist[0]["content"], "u2")

    def test_trim_noop_when_under_cap(self):
        hist = self._restore_attr_after("conversation_history")
        hist.clear()
        hist.append({"role": "user", "content": "only"})
        self.bc._trim_conversation_history(max_history=20)
        self.assertEqual(len(hist), 1)


# ──────────────────────────────────────────────────────────────────────────
#  merge_memory — dedupe, redaction, trim, topic stamping
# ──────────────────────────────────────────────────────────────────────────
class MergeMemoryTests(_MonolithTestBase):
    def setUp(self):
        # Backing store the patched load/save operate on. merge_memory calls the
        # module-level names load_memory/save_memory, so patch those on bc.
        self._store = {"facts": [], "projects": [], "topics": [], "sessions": []}

        def _fake_load():
            import copy
            return copy.deepcopy(self._store)

        def _fake_save(m):
            self._store = m

        self._p_load = mock.patch.object(self.bc, "load_memory", _fake_load)
        self._p_save = mock.patch.object(self.bc, "save_memory", _fake_save)
        self._p_load.start()
        self._p_save.start()
        self.addCleanup(self._p_load.stop)
        self.addCleanup(self._p_save.stop)

    def test_adds_new_fact_project_topic(self):
        added_f, added_p = self.bc.merge_memory(
            new_facts=["User enjoys hiking"],
            new_projects=["Building a treehouse"],
            new_topic="weekend plans",
        )
        self.assertEqual(added_f, ["User enjoys hiking"])
        self.assertEqual(added_p, ["Building a treehouse"])
        self.assertIn("User enjoys hiking", self._store["facts"])
        self.assertIn("Building a treehouse", self._store["projects"])
        self.assertEqual(self._store["topics"][-1]["topic"], "weekend plans")
        self.assertIn("date", self._store["topics"][-1])

    def test_dedupe_is_case_insensitive(self):
        self._store["facts"] = ["User enjoys hiking"]
        added_f, _ = self.bc.merge_memory(new_facts=["user ENJOYS hiking"])
        self.assertEqual(added_f, [])  # already present, case-insensitively

    def test_secret_facts_are_redacted(self):
        # _is_secret_fact triggers on the "api key" keyword (core/memory_guards
        # is keyword-based), so the value is a harmless placeholder — a real
        # key-shaped token here would trip the check_no_pii leak gate itself.
        added_f, _ = self.bc.merge_memory(
            new_facts=["my api key is <redacted-placeholder>"])
        self.assertEqual(added_f, [])
        self.assertEqual(self._store["facts"], [])

    def test_empty_inputs_short_circuit(self):
        # No new facts/projects/topic → returns empties and never saves.
        with mock.patch.object(self.bc, "save_memory") as msave:
            added_f, added_p = self.bc.merge_memory()
            self.assertEqual((added_f, added_p), ([], []))
            msave.assert_not_called()

    def test_facts_trimmed_to_max(self):
        # Seed just under the cap, then add enough to exceed it; the result is
        # capped to MAX_FACTS keeping the most-recent.
        cap = self.bc.MAX_FACTS
        self._store["facts"] = [f"old fact {i}" for i in range(cap)]
        new = [f"brand new fact {i}" for i in range(5)]
        self.bc.merge_memory(new_facts=new)
        self.assertEqual(len(self._store["facts"]), cap)
        self.assertIn("brand new fact 4", self._store["facts"])
        # The 5 oldest should have been trimmed off the front.
        self.assertNotIn("old fact 0", self._store["facts"])


# ──────────────────────────────────────────────────────────────────────────
#  Standing rules + system prompt
# ──────────────────────────────────────────────────────────────────────────
class SystemPromptTests(_MonolithTestBase):
    def _write_rules(self, payload):
        d = tempfile.mkdtemp()
        path = os.path.join(d, "chappie_standing_rules.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        return path

    def test_standing_rules_formats_entries(self):
        path = self._write_rules({"rules": [
            {"id": "R1", "rule": "Read before send", "severity": "hard-rule"},
            {"id": "R2", "rule": "No impersonation"},  # severity defaults
        ]})
        with mock.patch.object(self.bc, "_CHAPPIE_STANDING_RULES_PATH", path):
            block = self.bc._load_chappie_standing_rules()
        self.assertIn("STANDING RULES", block)
        self.assertIn("R1 (hard-rule): Read before send", block)
        self.assertIn("R2 (rule): No impersonation", block)

    def test_standing_rules_missing_file_returns_empty(self):
        path = os.path.join(tempfile.mkdtemp(), "absent.json")
        with mock.patch.object(self.bc, "_CHAPPIE_STANDING_RULES_PATH", path):
            self.assertEqual(self.bc._load_chappie_standing_rules(), "")

    def test_standing_rules_malformed_returns_empty(self):
        d = tempfile.mkdtemp()
        path = os.path.join(d, "bad.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write("{ not valid json ")
        with mock.patch.object(self.bc, "_CHAPPIE_STANDING_RULES_PATH", path):
            self.assertEqual(self.bc._load_chappie_standing_rules(), "")

    def test_standing_rules_skips_entries_without_id_or_text(self):
        path = self._write_rules({"rules": [
            {"id": "OK1", "rule": "valid"},
            {"id": "NoText"},            # missing rule -> skipped
            {"rule": "NoId"},            # missing id -> skipped
            "not-a-dict",                # skipped
        ]})
        with mock.patch.object(self.bc, "_CHAPPIE_STANDING_RULES_PATH", path):
            block = self.bc._load_chappie_standing_rules()
        self.assertIn("OK1 (rule): valid", block)
        self.assertNotIn("NoText", block)
        self.assertNotIn("NoId", block)

    def test_build_system_prompt_includes_memory_sections(self):
        mem = self.bc._empty_memory()
        mem["facts"] = ["User drinks tea"]
        mem["projects"] = ["Garden shed"]
        mem["topics"] = [{"date": "2026-06-01", "location": "desk",
                          "topic": "carpentry"}]
        mem["sessions"] = [{"date": "2026-06-01", "location": "desk",
                            "summary": "Discussed the shed build."}]
        mem["conversation_count"] = 7
        # Keep the dynamic phrasebook + standing rules deterministic.
        with mock.patch.object(self.bc, "_load_chappie_standing_rules",
                               return_value="RULESBLOCK"), \
             mock.patch.object(self.bc._mcu_phrases, "render_phrasebook_block",
                               return_value="PHRASEBOOK"):
            prompt = self.bc.build_system_prompt(mem)
        self.assertIn("RULESBLOCK", prompt)
        self.assertIn("PHRASEBOOK", prompt)
        self.assertIn("User drinks tea", prompt)
        self.assertIn("Garden shed", prompt)
        self.assertIn("carpentry", prompt)
        self.assertIn("Discussed the shed build.", prompt)
        self.assertIn(self.bc.LOCATION, prompt)
        self.assertIn("7 conversations", prompt)

    def test_build_system_prompt_omits_rules_block_when_empty(self):
        mem = self.bc._empty_memory()
        with mock.patch.object(self.bc, "_load_chappie_standing_rules",
                               return_value=""), \
             mock.patch.object(self.bc._mcu_phrases, "render_phrasebook_block",
                               return_value="PB"):
            prompt = self.bc.build_system_prompt(mem)
        self.assertNotIn("STANDING RULES", prompt)
        self.assertIn("PB", prompt)


# ──────────────────────────────────────────────────────────────────────────
#  _llm_quick / _parse_json_array
# ──────────────────────────────────────────────────────────────────────────
class LlmQuickTests(_MonolithTestBase):
    def setUp(self):
        super().setUp()
        # AMBIENT_LEARNING_FORCE_LOCAL is an OWNER setting (true in this box's
        # user_settings.json); pin it OFF here so the Claude-path tests are
        # deterministic. The two force-local tests re-enable it explicitly.
        import core.config as cfg
        _p = mock.patch.object(cfg, "AMBIENT_LEARNING_FORCE_LOCAL", False)
        _p.start()
        self.addCleanup(_p.stop)

    def test_parse_json_array_extracts_first(self):
        self.assertEqual(self.bc._parse_json_array('noise [1, 2, 3] tail'),
                         [1, 2, 3])

    def test_parse_json_array_none_found(self):
        self.assertEqual(self.bc._parse_json_array("no array here"), [])

    def test_parse_json_array_malformed_returns_empty(self):
        self.assertEqual(self.bc._parse_json_array("[1, 2,]"), [])

    def test_llm_quick_claude_success(self):
        # Build a fake anthropic module whose create() returns a text block.
        fake_block = mock.Mock()
        fake_block.text = "hello there"
        fake_msg = mock.Mock()
        fake_msg.content = [fake_block]
        fake_client = mock.Mock()
        fake_client.messages.create.return_value = fake_msg
        fake_anthropic = mock.Mock()
        fake_anthropic.Anthropic.return_value = fake_client
        with mock.patch.object(self.bc, "AI_BACKEND", "claude"), \
             mock.patch.dict("sys.modules", {"anthropic": fake_anthropic}):
            out = self.bc._llm_quick("sys", "user", max_tokens=10)
        self.assertEqual(out, "hello there")
        fake_client.messages.create.assert_called_once()

    def test_llm_quick_claude_failure_falls_back_to_local(self):
        fake_anthropic = mock.Mock()
        fake_anthropic.Anthropic.side_effect = RuntimeError("cap hit")
        with mock.patch.object(self.bc, "AI_BACKEND", "claude"), \
             mock.patch.dict("sys.modules", {"anthropic": fake_anthropic}), \
             mock.patch.object(self.bc, "_call_local_llm",
                               return_value="local reply") as mlocal:
            out = self.bc._llm_quick("sys", "user")
        self.assertEqual(out, "local reply")
        mlocal.assert_called_once()

    def test_llm_quick_claude_failure_no_local_returns_empty(self):
        fake_anthropic = mock.Mock()
        fake_anthropic.Anthropic.side_effect = RuntimeError("down")
        with mock.patch.object(self.bc, "AI_BACKEND", "claude"), \
             mock.patch.dict("sys.modules", {"anthropic": fake_anthropic}), \
             mock.patch.object(self.bc, "_call_local_llm", return_value=""):
            self.assertEqual(self.bc._llm_quick("sys", "user"), "")

    def test_llm_quick_unknown_backend_returns_empty(self):
        with mock.patch.object(self.bc, "AI_BACKEND", "something-else"):
            self.assertEqual(self.bc._llm_quick("s", "u"), "")

    def test_llm_quick_force_local_never_calls_claude(self):
        # AMBIENT_LEARNING_FORCE_LOCAL short-circuits to the local model so
        # ambient learning is free (Claude is never reached).
        import core.config as cfg
        with mock.patch.object(cfg, "AMBIENT_LEARNING_FORCE_LOCAL", True), \
             mock.patch.object(self.bc, "_call_local_llm",
                               return_value="local fact") as mlocal:
            out = self.bc._llm_quick("sys", "user")
        self.assertEqual(out, "local fact")
        mlocal.assert_called_once()

    def test_llm_quick_force_local_empty_when_local_down(self):
        import core.config as cfg
        with mock.patch.object(cfg, "AMBIENT_LEARNING_FORCE_LOCAL", True), \
             mock.patch.object(self.bc, "_call_local_llm", return_value=""):
            self.assertEqual(self.bc._llm_quick("s", "u"), "")


# ──────────────────────────────────────────────────────────────────────────
#  learn_from_turn — background extraction worker
# ──────────────────────────────────────────────────────────────────────────
class LearnFromTurnTests(_MonolithTestBase):
    def test_disabled_is_noop(self):
        with mock.patch.object(self.bc, "LEARN_EVERY_TURN", False), \
             mock.patch.object(self.bc.threading, "Thread") as mthread:
            self.bc.learn_from_turn("hi", "hello", self.bc._empty_memory())
            mthread.assert_not_called()

    def test_worker_parses_json_and_merges(self):
        mem = self.bc._empty_memory()
        payload = ('{"new_facts": ["User has a cat"], '
                   '"new_projects": [], "topic": "pets"}')
        with mock.patch.object(self.bc, "LEARN_EVERY_TURN", True), \
             mock.patch.object(self.bc.threading, "Thread", _InlineThread), \
             mock.patch.object(self.bc, "_llm_quick", return_value=payload), \
             mock.patch.object(self.bc, "merge_memory",
                               return_value=(["User has a cat"], [])) as mmerge:
            self.bc.learn_from_turn("I have a cat", "Noted.", mem)
        mmerge.assert_called_once()
        _, kwargs = mmerge.call_args
        self.assertEqual(kwargs["new_facts"], ["User has a cat"])
        self.assertEqual(kwargs["new_topic"], "pets")

    def test_worker_handles_no_json_gracefully(self):
        mem = self.bc._empty_memory()
        with mock.patch.object(self.bc, "LEARN_EVERY_TURN", True), \
             mock.patch.object(self.bc.threading, "Thread", _InlineThread), \
             mock.patch.object(self.bc, "_llm_quick", return_value="no json"), \
             mock.patch.object(self.bc, "merge_memory") as mmerge:
            self.bc.learn_from_turn("x", "y", mem)
        mmerge.assert_not_called()  # no "{" → early return before merge


# ──────────────────────────────────────────────────────────────────────────
#  Action error / history / session-action bookkeeping
# ──────────────────────────────────────────────────────────────────────────
class ActionBookkeepingTests(_MonolithTestBase):
    def test_record_and_get_action_errors(self):
        log = self._restore_attr_after("_action_error_log")
        log.clear()
        self.bc.record_action_error("open_url", ValueError("boom"),
                                    traceback_text="TB")
        errs = self.bc.get_recent_action_errors()
        self.assertEqual(len(errs), 1)
        self.assertEqual(errs[0]["action"], "open_url")
        self.assertEqual(errs[0]["exc_class"], "ValueError")
        self.assertEqual(errs[0]["exc_msg"], "boom")
        self.assertEqual(errs[0]["traceback"], "TB")

    def test_get_action_errors_prunes_old_entries(self):
        log = self._restore_attr_after("_action_error_log")
        log.clear()
        now = self.bc.time.time()
        # Insert one ancient + one fresh entry directly.
        log.append({"ts": now - 10_000, "action": "old", "exc_class": "E",
                    "exc_msg": "", "traceback": ""})
        log.append({"ts": now, "action": "new", "exc_class": "E",
                    "exc_msg": "", "traceback": ""})
        recent = self.bc.get_recent_action_errors(window_s=3600)
        self.assertEqual([e["action"] for e in recent], ["new"])

    def test_record_action_error_never_raises(self):
        # Passing a weird action object must be swallowed, not propagated.
        log = self._restore_attr_after("_action_error_log")
        log.clear()
        try:
            self.bc.record_action_error(object(), RuntimeError("x"))
        except Exception as exc:  # pragma: no cover
            self.fail(f"record_action_error raised: {exc!r}")

    def test_record_action_error_captures_default_traceback(self):
        log = self._restore_attr_after("_action_error_log")
        log.clear()
        # No traceback_text passed → format_exc() is used. Raise+catch so a
        # real traceback string is available on the stack.
        try:
            raise KeyError("missing")
        except KeyError as exc:
            self.bc.record_action_error("lookup", exc)
        errs = self.bc.get_recent_action_errors()
        self.assertEqual(len(errs), 1)
        self.assertEqual(errs[0]["exc_class"], "KeyError")
        self.assertIsInstance(errs[0]["traceback"], str)

    def test_record_action_history_bounded(self):
        hist = self._restore_attr_after("_action_history")
        hist.clear()
        for i in range(8):  # deque maxlen is 5
            self.bc.record_action_history("act", str(i), f"res{i}")
        self.assertEqual(len(hist), 5)
        self.assertEqual(hist[-1]["arg"], "7")
        self.assertEqual(hist[0]["arg"], "3")  # oldest 3 evicted

    def test_record_action_history_coerces_nonstring_result(self):
        hist = self._restore_attr_after("_action_history")
        hist.clear()
        self.bc.record_action_history("act", "a", 12345)
        self.assertEqual(hist[-1]["result"], "12345")

    def test_record_session_action_counts_and_apps(self):
        counts = self._restore_attr_after("_session_action_counts")
        apps = self._restore_attr_after("_session_app_names")
        counts.clear()
        apps.clear()
        with mock.patch.dict("sys.modules", {"skill_pattern_learning": None}):
            self.bc.record_session_action("launch_app", "Notepad")
            self.bc.record_session_action("launch_app", "Notepad")
            self.bc.record_session_action("open_url", "https://example.com/page")
        self.assertEqual(counts["launch_app"], 2)
        self.assertEqual(counts["open_url"], 1)
        self.assertIn("notepad", apps)
        self.assertIn("example.com", apps)  # host extracted, scheme stripped

    def test_record_session_action_music_updates_timestamp(self):
        counts = self._restore_attr_after("_session_action_counts")
        played = self._restore_attr_after("_jarvis_played_music_at")
        counts.clear()
        played[0] = 0.0
        a_music_action = next(iter(self.bc.MUSIC_ACTION_NAMES))
        with mock.patch.dict("sys.modules", {"skill_pattern_learning": None}), \
             mock.patch.object(self.bc.time, "time", return_value=999.0):
            self.bc.record_session_action(a_music_action, "")
        self.assertEqual(played[0], 999.0)

    def test_record_session_action_forwards_to_pattern_learning(self):
        counts = self._restore_attr_after("_session_action_counts")
        counts.clear()
        fake_pl = mock.Mock()
        with mock.patch.dict("sys.modules", {"skill_pattern_learning": fake_pl}):
            self.bc.record_session_action("get_time", "")
        fake_pl.log_event.assert_called_once_with("get_time", "")


# ──────────────────────────────────────────────────────────────────────────
#  Patterns persistence + startup-pattern detection
# ──────────────────────────────────────────────────────────────────────────
class PatternsTests(_MonolithTestBase):
    def test_load_patterns_missing_returns_empty(self):
        path = os.path.join(tempfile.mkdtemp(), "patterns.json")
        with mock.patch.object(self.bc, "PATTERNS_FILE", path):
            self.assertEqual(self.bc._load_patterns(), [])

    def test_save_then_load_round_trip(self):
        d = tempfile.mkdtemp()
        path = os.path.join(d, "patterns.json")
        entries = [{"day": "Monday", "top_actions": ["get_time"]}]
        with mock.patch.object(self.bc, "PATTERNS_FILE", path), \
             mock.patch.object(self.bc, "PATTERNS_DIR", d):
            self.bc._save_patterns(entries)
            loaded = self.bc._load_patterns()
        self.assertEqual(loaded, entries)

    def test_load_patterns_non_list_returns_empty(self):
        d = tempfile.mkdtemp()
        path = os.path.join(d, "patterns.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"not": "a list"}, f)
        with mock.patch.object(self.bc, "PATTERNS_FILE", path):
            self.assertEqual(self.bc._load_patterns(), [])

    def test_detect_startup_pattern_too_few_sessions(self):
        with mock.patch.object(self.bc, "_load_patterns", return_value=[]):
            self.assertEqual(self.bc.detect_startup_pattern(), "")

    def test_detect_startup_pattern_streaming_evening(self):
        # Freeze 'now' to a Friday 21:00 so the evening + streaming branch fires.
        # struct_time: (Y, M, D, H, M, S, wday=4(Fri), yday, isdst)
        fixed = time.struct_time((2026, 6, 5, 21, 0, 0, 4, 156, -1))
        entries = []
        for _ in range(6):  # >= MIN_SESSIONS_FOR_PATTERN and >= 3 matching
            entries.append({"day": "Friday", "hour_started": 21,
                            "top_actions": ["netflix", "spotify"]})
        with mock.patch.object(self.bc, "_load_patterns", return_value=entries), \
             mock.patch.object(self.bc.time, "localtime", return_value=fixed), \
             mock.patch.object(self.bc.time, "strftime",
                               side_effect=lambda fmt, *a: "Friday"):
            out = self.bc.detect_startup_pattern()
        self.assertIn("Netflix", out)
        self.assertIn("Friday", out)

    def test_detect_startup_pattern_building(self):
        fixed = time.struct_time((2026, 6, 3, 14, 0, 0, 2, 154, -1))  # Wed 14:00
        entries = []
        for _ in range(6):
            entries.append({"day": "Wednesday", "hour_started": 14,
                            "top_actions": ["upgrade", "queue_task"]})
        with mock.patch.object(self.bc, "_load_patterns", return_value=entries), \
             mock.patch.object(self.bc.time, "localtime", return_value=fixed), \
             mock.patch.object(self.bc.time, "strftime",
                               side_effect=lambda fmt, *a: "Wednesday"):
            out = self.bc.detect_startup_pattern()
        self.assertIn("build session", out)

    def test_detect_startup_pattern_no_strong_match(self):
        fixed = time.struct_time((2026, 6, 3, 14, 0, 0, 2, 154, -1))
        # 6 sessions but none on the current day/hour → no match.
        entries = [{"day": "Sunday", "hour_started": 3,
                    "top_actions": ["get_time"]} for _ in range(6)]
        with mock.patch.object(self.bc, "_load_patterns", return_value=entries), \
             mock.patch.object(self.bc.time, "localtime", return_value=fixed), \
             mock.patch.object(self.bc.time, "strftime",
                               side_effect=lambda fmt, *a: "Wednesday"):
            self.assertEqual(self.bc.detect_startup_pattern(), "")


# ──────────────────────────────────────────────────────────────────────────
#  save_session_pattern / save_session_to_memory
# ──────────────────────────────────────────────────────────────────────────
class SessionSaveTests(_MonolithTestBase):
    def test_save_session_pattern_noop_when_empty(self):
        counts = self._restore_attr_after("_session_action_counts")
        apps = self._restore_attr_after("_session_app_names")
        counts.clear()
        apps.clear()
        with mock.patch.object(self.bc, "_load_patterns") as mload, \
             mock.patch.object(self.bc, "_save_patterns") as msave:
            self.bc.save_session_pattern()
            mload.assert_not_called()
            msave.assert_not_called()

    def test_save_session_pattern_writes_entry(self):
        counts = self._restore_attr_after("_session_action_counts")
        apps = self._restore_attr_after("_session_app_names")
        counts.clear()
        counts.update({"get_time": 3, "open_url": 1})
        apps.clear()
        apps.add("example.com")
        captured = {}
        with mock.patch.object(self.bc, "_load_patterns", return_value=[]), \
             mock.patch.object(self.bc, "_save_patterns",
                               side_effect=lambda e: captured.update(entries=e)):
            self.bc.save_session_pattern()
        self.assertEqual(len(captured["entries"]), 1)
        entry = captured["entries"][0]
        self.assertEqual(entry["top_actions"][0], "get_time")  # highest count
        self.assertEqual(entry["apps"], ["example.com"])

    def test_save_session_to_memory_skips_short_history(self):
        hist = self._restore_attr_after("conversation_history")
        hist.clear()
        hist.append({"role": "user", "content": "hi"})  # < 4 messages
        with mock.patch.object(self.bc, "_llm_quick") as mq:
            self.bc.save_session_to_memory(self.bc._empty_memory())
            mq.assert_not_called()

    def test_save_session_to_memory_writes_summary(self):
        hist = self._restore_attr_after("conversation_history")
        hist.clear()
        for i in range(4):
            hist.append({"role": "user", "content": f"u{i}"})
            hist.append({"role": "assistant", "content": f"a{i}"})
        mem = self.bc._empty_memory()
        # Patch the fresh load/save the summary path uses + the recall index.
        store = {"sessions": []}

        def _fake_load():
            import copy
            return copy.deepcopy(store)

        def _fake_save(m):
            store.update(m)

        with mock.patch.object(self.bc, "_llm_quick",
                               return_value="A productive session.\n"), \
             mock.patch.object(self.bc, "load_memory", _fake_load), \
             mock.patch.object(self.bc, "save_memory", _fake_save), \
             mock.patch.object(self.bc.pattern_memory, "record_session_summary"):
            self.bc.save_session_to_memory(mem)
        self.assertTrue(store["sessions"])
        self.assertEqual(store["sessions"][-1]["summary"], "A productive session.")
        # Local copy kept consistent.
        self.assertEqual(mem["sessions"], store["sessions"])


# ──────────────────────────────────────────────────────────────────────────
#  _TimestampedTee + log cleanup/close
# ──────────────────────────────────────────────────────────────────────────
class LoggingTeeTests(_MonolithTestBase):
    def test_tee_writes_to_console_and_timestamps_file(self):
        import io
        console = io.StringIO()
        logf = io.StringIO()
        tee = self.bc._TimestampedTee(console, logf)
        with mock.patch.object(self.bc.time, "strftime", return_value="[12:00:00] "):
            tee.write("hello\n")
        self.assertEqual(console.getvalue(), "hello\n")
        self.assertEqual(logf.getvalue(), "[12:00:00] hello\n")

    def test_tee_timestamps_each_line(self):
        import io
        console = io.StringIO()
        logf = io.StringIO()
        tee = self.bc._TimestampedTee(console, logf)
        with mock.patch.object(self.bc.time, "strftime", return_value="[T] "):
            tee.write("a\nb\n")
        # Each new line gets its own timestamp prefix.
        self.assertEqual(logf.getvalue(), "[T] a\n[T] b\n")

    def test_tee_flush_swallows_errors(self):
        console = mock.Mock()
        console.flush.side_effect = RuntimeError("x")
        logf = mock.Mock()
        logf.flush.side_effect = RuntimeError("y")
        tee = self.bc._TimestampedTee(console, logf)
        tee.flush()  # must not raise

    def test_tee_write_swallows_console_and_file_errors(self):
        console = mock.Mock()
        console.write.side_effect = RuntimeError("console down")
        logf = mock.Mock()
        logf.write.side_effect = RuntimeError("file down")
        tee = self.bc._TimestampedTee(console, logf)
        tee.write("anything\n")  # both arms raise → both swallowed, no raise

    def test_cleanup_old_logs_noop_when_dir_missing(self):
        missing = os.path.join(tempfile.mkdtemp(), "no_such_subdir")
        with mock.patch.object(self.bc, "LOGS_DIR", missing), \
             mock.patch.object(self.bc.os, "listdir") as mls:
            self.bc._cleanup_old_logs()
            mls.assert_not_called()  # early return before listdir

    def test_cleanup_old_logs_removes_excess(self):
        d = tempfile.mkdtemp()
        # Create more than LOG_KEEP_COUNT .log files with increasing mtimes.
        keep = self.bc.LOG_KEEP_COUNT
        paths = []
        for i in range(keep + 3):
            p = os.path.join(d, f"session_{i}.log")
            with open(p, "w", encoding="utf-8") as f:
                f.write("x")
            os.utime(p, (1000 + i, 1000 + i))  # deterministic order
            paths.append(p)
        with mock.patch.object(self.bc, "LOGS_DIR", d):
            self.bc._cleanup_old_logs()
        remaining = [p for p in paths if os.path.exists(p)]
        self.assertEqual(len(remaining), keep)
        # The newest `keep` survive; the 3 oldest are gone.
        self.assertTrue(os.path.exists(paths[-1]))
        self.assertFalse(os.path.exists(paths[0]))

    def test_close_log_noop_when_no_handle(self):
        with mock.patch.object(self.bc, "_log_file_handle", None):
            self.bc.close_log()  # returns silently

    def test_close_log_footer_content(self):
        # Use a handle that records writes but tolerates close().
        writes = []

        class _Rec:
            closed = False

            def write(self, s):
                writes.append(s)

            def flush(self):
                pass

            def close(self):
                self.closed = True

        rec = _Rec()
        with mock.patch.object(self.bc, "_log_file_handle", rec), \
             mock.patch.object(self.bc.time, "strftime", return_value="TS"):
            self.bc.close_log()
        self.assertTrue(any("Session ended" in w for w in writes))
        self.assertTrue(rec.closed)


# ──────────────────────────────────────────────────────────────────────────
#  Robot send / set_state / _now_doing_label
# ──────────────────────────────────────────────────────────────────────────
class RobotStateTests(_MonolithTestBase):
    def test_send_noop_when_robot_disabled(self):
        with mock.patch.object(self.bc, "ROBOT_ENABLED", False), \
             mock.patch.object(self.bc.requests, "get") as mget:
            self.bc.send(eyes_x=0.5)
            mget.assert_not_called()

    def test_send_calls_requests_when_enabled(self):
        with mock.patch.object(self.bc, "ROBOT_ENABLED", True), \
             mock.patch.object(self.bc.requests, "get") as mget:
            self.bc.send(eyes_x=0.5, leds="white")
            mget.assert_called_once()
            _, kwargs = mget.call_args
            self.assertEqual(kwargs["params"], {"eyes_x": 0.5, "leds": "white"})

    def test_send_swallows_network_error(self):
        with mock.patch.object(self.bc, "ROBOT_ENABLED", True), \
             mock.patch.object(self.bc.requests, "get",
                               side_effect=OSError("net")):
            self.bc.send(x=1)  # must not raise

    def test_now_doing_label_thinking_includes_model(self):
        with mock.patch.object(self.bc, "AI_BACKEND", "claude"), \
             mock.patch.object(self.bc, "CLAUDE_MODEL", "claude-xyz"):
            self.assertEqual(self.bc._now_doing_label("thinking"),
                             "THINKING (claude-xyz)")

    def test_now_doing_label_known_states(self):
        self.assertEqual(self.bc._now_doing_label("listening"), "LISTENING")
        self.assertEqual(self.bc._now_doing_label("speaking"), "SPEAKING")
        self.assertEqual(self.bc._now_doing_label("standby"), "STANDBY")
        self.assertEqual(self.bc._now_doing_label("sleep"), "SLEEP")
        self.assertEqual(self.bc._now_doing_label("anything-else"), "IDLE")
        self.assertEqual(self.bc._now_doing_label(""), "IDLE")

    def test_set_state_listening_sends_and_writes_hud(self):
        label = self._restore_attr_after("_current_state_label")
        standby = self._restore_attr_after("_standby_mode")
        standby[0] = False  # not in standby → normal listening path
        with mock.patch.object(self.bc, "send") as msend, \
             mock.patch.object(self.bc, "_write_hud_state") as mhud:
            self.bc.set_state("listening")
        msend.assert_called_once()
        mhud.assert_called_once()
        _, kwargs = mhud.call_args
        self.assertEqual(kwargs["state"], "Listening")
        self.assertEqual(kwargs["now_doing"], "LISTENING")
        self.assertEqual(label[0], "Listening")

    def test_set_state_each_state_sends_expected(self):
        standby = self._restore_attr_after("_standby_mode")
        self._restore_attr_after("_current_state_label")
        standby[0] = False
        for state in ("idle", "thinking", "speaking", "sleep"):
            with mock.patch.object(self.bc, "send") as msend, \
                 mock.patch.object(self.bc, "_write_hud_state"):
                self.bc.set_state(state)
            msend.assert_called_once()  # each non-standby branch sends once

    def test_set_state_idle_in_standby_shows_standby(self):
        label = self._restore_attr_after("_current_state_label")
        standby = self._restore_attr_after("_standby_mode")
        standby[0] = True
        with mock.patch.object(self.bc, "send") as msend, \
             mock.patch.object(self.bc, "_write_hud_state") as mhud:
            self.bc.set_state("idle")
        # Standby branch returns early after one send + one hud write.
        msend.assert_called_once()
        mhud.assert_called_once()
        _, kwargs = mhud.call_args
        self.assertEqual(kwargs["state"], "Standby")
        self.assertEqual(label[0], "Standby")


# ──────────────────────────────────────────────────────────────────────────
#  HUD state writer + launch/shutdown
# ──────────────────────────────────────────────────────────────────────────
class HudStateTests(_MonolithTestBase):
    def test_write_hud_state_noop_when_disabled(self):
        with mock.patch.object(self.bc, "HUD_ENABLED", False):
            # Should return immediately without touching the cache.
            cache_before = dict(self.bc._hud_state_cache)
            self.bc._write_hud_state(state="ZZZ")
            self.assertEqual(self.bc._hud_state_cache, cache_before)

    def test_write_hud_state_merges_and_writes_file(self):
        cache = self._restore_attr_after("_hud_state_cache")
        d = tempfile.mkdtemp()
        path = os.path.join(d, "hud_state.json")
        with mock.patch.object(self.bc, "HUD_ENABLED", True), \
             mock.patch.object(self.bc, "HUD_STATE_FILE", path):
            self.bc._write_hud_state(state="Listening", mic_level=0.7)
        self.assertEqual(cache["state"], "Listening")
        self.assertEqual(cache["mic_level"], 0.7)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["state"], "Listening")
        self.assertIn("updated_at", data)

    def test_write_hud_state_swallows_write_error(self):
        # Point HUD_STATE_FILE at a path whose dir can't be made → silent.
        with mock.patch.object(self.bc, "HUD_ENABLED", True), \
             mock.patch.object(self.bc.tempfile, "mkstemp",
                               side_effect=OSError("nope")):
            self.bc._write_hud_state(state="x")  # must not raise

    def test_launch_hud_noop_when_disabled(self):
        with mock.patch.object(self.bc, "HUD_ENABLED", False), \
             mock.patch.object(self.bc.subprocess, "Popen") as mpop:
            self.bc._launch_hud()
            mpop.assert_not_called()

    def test_launch_hud_missing_script_skips(self):
        with mock.patch.object(self.bc, "HUD_ENABLED", True), \
             mock.patch.object(self.bc.os.path, "exists", return_value=False), \
             mock.patch.object(self.bc.subprocess, "Popen") as mpop:
            self.bc._launch_hud()
            mpop.assert_not_called()

    def test_shutdown_hud_terminates_process(self):
        fake_proc = mock.Mock()
        with mock.patch.object(self.bc, "_hud_process", fake_proc):
            self.bc._shutdown_hud()
            fake_proc.terminate.assert_called_once()
        self.assertIsNone(self.bc._hud_process)

    def test_shutdown_hud_noop_when_none(self):
        with mock.patch.object(self.bc, "_hud_process", None):
            self.bc._shutdown_hud()  # returns silently


# ──────────────────────────────────────────────────────────────────────────
#  Reticle publish + geometry helpers + launch/shutdown
# ──────────────────────────────────────────────────────────────────────────
class ReticleTests(_MonolithTestBase):
    def test_publish_reticle_noop_when_disabled(self):
        with mock.patch.object(self.bc, "RETICLE_OVERLAY_ENABLED", False):
            d = tempfile.mkdtemp()
            path = os.path.join(d, "ret.json")
            with mock.patch.object(self.bc, "RETICLE_STATE_FILE", path):
                self.bc._publish_reticle(10, 20, "x")
            self.assertFalse(os.path.exists(path))

    def test_publish_reticle_writes_entry(self):
        d = tempfile.mkdtemp()
        path = os.path.join(d, "ret.json")
        with mock.patch.object(self.bc, "RETICLE_OVERLAY_ENABLED", True), \
             mock.patch.object(self.bc, "RETICLE_STATE_FILE", path):
            self.bc._publish_reticle(100, 200, "target-label-that-is-long" * 3)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(len(data["reticles"]), 1)
        r = data["reticles"][0]
        self.assertEqual((r["x"], r["y"]), (100, 200))
        self.assertLessEqual(len(r["label"]), 24)  # label truncated to 24

    def test_publish_reticle_invalid_coords_returns(self):
        d = tempfile.mkdtemp()
        path = os.path.join(d, "ret.json")
        with mock.patch.object(self.bc, "RETICLE_OVERLAY_ENABLED", True), \
             mock.patch.object(self.bc, "RETICLE_STATE_FILE", path):
            self.bc._publish_reticle("not-an-int", 5)
        self.assertFalse(os.path.exists(path))

    def test_publish_reticle_recovers_from_corrupt_state(self):
        d = tempfile.mkdtemp()
        path = os.path.join(d, "ret.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write("{ corrupt not json")
        with mock.patch.object(self.bc, "RETICLE_OVERLAY_ENABLED", True), \
             mock.patch.object(self.bc, "RETICLE_STATE_FILE", path):
            self.bc._publish_reticle(5, 6, "fresh")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Corrupt prior content discarded; only the new entry remains.
        self.assertEqual([r["label"] for r in data["reticles"]], ["fresh"])

    def test_publish_reticle_prunes_expired(self):
        d = tempfile.mkdtemp()
        path = os.path.join(d, "ret.json")
        old = {"reticles": [{"x": 1, "y": 1, "label": "old",
                             "created_at": 0.0}]}  # far past → pruned
        with open(path, "w", encoding="utf-8") as f:
            json.dump(old, f)
        with mock.patch.object(self.bc, "RETICLE_OVERLAY_ENABLED", True), \
             mock.patch.object(self.bc, "RETICLE_STATE_FILE", path), \
             mock.patch.object(self.bc.time, "time", return_value=10_000.0):
            self.bc._publish_reticle(2, 2, "new")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        labels = [r["label"] for r in data["reticles"]]
        self.assertEqual(labels, ["new"])  # old one pruned

    def test_virtual_screen_bounds_from_monitors(self):
        fake_mons = {"a": (0, 0, 100, 100), "b": (100, 50, 100, 100)}
        with mock.patch.object(self.bc, "MONITORS", fake_mons):
            self.assertEqual(self.bc._virtual_screen_bounds(), (0, 0, 200, 150))

    def test_virtual_screen_bounds_empty_default(self):
        with mock.patch.object(self.bc, "MONITORS", {}):
            self.assertEqual(self.bc._virtual_screen_bounds(),
                             (0, 0, 2560, 1440))

    def test_active_window_center_no_pygetwindow(self):
        # Force the import to fail → returns None.
        with mock.patch.dict("sys.modules", {"pygetwindow": None}):
            self.assertIsNone(self.bc._active_window_center())

    def test_shutdown_reticle_overlay_terminates(self):
        fake_proc = mock.Mock()
        with mock.patch.object(self.bc, "_reticle_process", fake_proc):
            self.bc._shutdown_reticle_overlay()
            fake_proc.terminate.assert_called_once()
        self.assertIsNone(self.bc._reticle_process)

    def test_launch_reticle_noop_when_disabled(self):
        with mock.patch.object(self.bc, "RETICLE_OVERLAY_ENABLED", False), \
             mock.patch.object(self.bc.subprocess, "Popen") as mpop:
            self.bc._launch_reticle_overlay()
            mpop.assert_not_called()


# ──────────────────────────────────────────────────────────────────────────
#  Camera failure summary
# ──────────────────────────────────────────────────────────────────────────
class CameraFailureSummaryTests(_MonolithTestBase):
    def test_note_failure_tracks_consecutive_and_max(self):
        summ = self._restore_attr_after("_camera_failure_summary")
        summ.clear()
        self.bc._note_camera_read_attempt(0, ok=False, fails=3, error="read err")
        self.bc._note_camera_read_attempt(0, ok=False, fails=5, error="read err")
        snap = self.bc.get_camera_failure_summary()
        self.assertEqual(snap[0]["consecutive_fails"], 5)
        self.assertEqual(snap[0]["max_consecutive_fails"], 5)
        self.assertEqual(snap[0]["total_fails"], 2)
        self.assertEqual(snap[0]["last_error"], "read err")

    def test_note_ok_resets_consecutive_keeps_max(self):
        summ = self._restore_attr_after("_camera_failure_summary")
        summ.clear()
        self.bc._note_camera_read_attempt(1, ok=False, fails=4, error="e")
        self.bc._note_camera_read_attempt(1, ok=True)
        snap = self.bc.get_camera_failure_summary()
        self.assertEqual(snap[1]["consecutive_fails"], 0)
        self.assertEqual(snap[1]["max_consecutive_fails"], 4)  # max preserved
        self.assertGreater(snap[1]["last_ok_at"], 0.0)

    def test_get_summary_is_a_copy(self):
        summ = self._restore_attr_after("_camera_failure_summary")
        summ.clear()
        self.bc._note_camera_read_attempt(2, ok=False, fails=1)
        snap = self.bc.get_camera_failure_summary()
        snap[2]["consecutive_fails"] = 999  # mutate the copy
        # Internal state must be unaffected.
        self.assertEqual(
            self.bc.get_camera_failure_summary()[2]["consecutive_fails"], 1)


# ──────────────────────────────────────────────────────────────────────────
#  get_session_log_path
# ──────────────────────────────────────────────────────────────────────────
class SessionLogPathTests(_MonolithTestBase):
    def test_returns_path_when_set(self):
        with mock.patch.object(self.bc, "_log_file_path", "C:/logs/x.log"):
            self.assertEqual(self.bc.get_session_log_path(), "C:/logs/x.log")

    def test_returns_none_when_unset(self):
        with mock.patch.object(self.bc, "_log_file_path", None):
            self.assertIsNone(self.bc.get_session_log_path())

    def test_returns_none_when_empty_string(self):
        with mock.patch.object(self.bc, "_log_file_path", ""):
            self.assertIsNone(self.bc.get_session_log_path())


# ──────────────────────────────────────────────────────────────────────────
#  Tray command dispatch (_dispatch_tray_command + _process_inflight +
#  _drain_tray_commands_once)
# ──────────────────────────────────────────────────────────────────────────
class TrayDispatchTests(_MonolithTestBase):
    def test_enter_standby_sets_flags(self):
        sleep = self._restore_attr_after("_sleep_mode")
        standby = self._restore_attr_after("_standby_mode")
        sleep[0] = False
        standby[0] = False
        with mock.patch.object(self.bc, "_write_hud_state"):
            self.bc._dispatch_tray_command("enter_standby", {})
        self.assertTrue(sleep[0])
        self.assertTrue(standby[0])

    def test_audio_toggles_flip_flags(self):
        master = self._restore_attr_after("_audio_master_enabled")
        master[0] = True
        with mock.patch.object(self.bc, "_publish_audio_state"):
            self.bc._dispatch_tray_command("audio_processing_toggle", {})
        self.assertFalse(master[0])

    def test_mute_tts_toggle(self):
        muted = self._restore_attr_after("_tts_muted")
        muted[0] = False
        with mock.patch.object(self.bc, "_write_hud_state"):
            self.bc._dispatch_tray_command("mute_tts_toggle", {})
        self.assertTrue(muted[0])

    def test_debug_mode_toggle(self):
        dbg = self._restore_attr_after("_debug_mode")
        start = dbg[0]
        with mock.patch.object(self.bc, "_write_hud_state"):
            self.bc._dispatch_tray_command("debug_mode_toggle", {})
        self.assertEqual(dbg[0], not start)

    def test_generic_command_routes_to_actions(self):
        fake_fn = mock.Mock(return_value="done")
        fake_actions = {"some_action": fake_fn}
        with mock.patch.object(self.bc, "ACTIONS", fake_actions), \
             mock.patch.object(self.bc, "_HEAVY_ACTIONS", frozenset()):
            self.bc._dispatch_tray_command("some_action", {"arg": "payload"})
        fake_fn.assert_called_once_with("payload")

    def test_generic_switch_llm_uses_backend_field(self):
        fake_fn = mock.Mock(return_value="ok")
        with mock.patch.object(self.bc, "ACTIONS", {"switch_llm": fake_fn}), \
             mock.patch.object(self.bc, "_HEAVY_ACTIONS", frozenset()):
            self.bc._dispatch_tray_command("switch_llm", {"backend": "ollama"})
        fake_fn.assert_called_once_with("ollama")

    def test_generic_unknown_command_is_ignored(self):
        with mock.patch.object(self.bc, "ACTIONS", {}):
            # Unknown → prints + returns, no raise.
            self.bc._dispatch_tray_command("does_not_exist", {})

    def test_heavy_action_dispatched_async(self):
        fake_fn = mock.Mock(return_value="slow")
        with mock.patch.object(self.bc, "ACTIONS", {"run_diagnostic": fake_fn}), \
             mock.patch.object(self.bc, "_HEAVY_ACTIONS",
                               frozenset({"run_diagnostic"})), \
             mock.patch.object(self.bc, "_tray_async") as masync:
            self.bc._dispatch_tray_command("run_diagnostic", {"arg": ""})
        masync.assert_called_once()
        # The underlying fn shouldn't have been called synchronously.
        fake_fn.assert_not_called()

    def test_ambient_mode_toggle_invokes_registered_action(self):
        active = self._restore_attr_after("_ambient_mode_active")
        active[0] = False
        fake_start = mock.Mock()
        with mock.patch.object(self.bc, "ACTIONS",
                               {"ambient_listen_start": fake_start}), \
             mock.patch.object(self.bc, "_write_hud_state"):
            self.bc._dispatch_tray_command("ambient_mode_toggle", {})
        self.assertTrue(active[0])
        fake_start.assert_called_once_with("")

    def test_force_wake_clears_flags_and_speaks(self):
        sleep = self._restore_attr_after("_sleep_mode")
        standby = self._restore_attr_after("_standby_mode")
        wake_date = self._restore_attr_after("_last_wake_date")
        sleep[0] = True
        standby[0] = True
        with mock.patch.object(self.bc, "_write_hud_state"), \
             mock.patch.object(self.bc, "_speak") as mspeak, \
             mock.patch.object(self.bc.os.path, "exists", return_value=False):
            self.bc._dispatch_tray_command("force_wake", {})
        self.assertFalse(sleep[0])
        self.assertFalse(standby[0])
        self.assertIsNotNone(wake_date[0])  # day's first-wake stamped
        mspeak.assert_called_once()

    def test_force_wake_removes_overnight_flag(self):
        sleep = self._restore_attr_after("_sleep_mode")
        standby = self._restore_attr_after("_standby_mode")
        sleep[0] = True
        standby[0] = True
        with mock.patch.object(self.bc, "_write_hud_state"), \
             mock.patch.object(self.bc, "_speak"), \
             mock.patch.object(self.bc.os.path, "exists", return_value=True), \
             mock.patch.object(self.bc.os, "remove") as mremove:
            self.bc._dispatch_tray_command("force_wake", {})
        mremove.assert_called_once_with(self.bc.OVERNIGHT_FLAG_FILE)

    def test_open_hud_relaunches(self):
        with mock.patch.object(self.bc, "_shutdown_hud") as mdown, \
             mock.patch.object(self.bc, "_launch_hud") as mup:
            self.bc._dispatch_tray_command("open_hud", {})
        mdown.assert_called_once()
        mup.assert_called_once()

    def test_restart_command_calls_act_restart(self):
        with mock.patch.object(self.bc, "_act_restart") as mrestart:
            self.bc._dispatch_tray_command("restart", {})
        mrestart.assert_called_once()

    def test_trigger_overnight_calls_action(self):
        with mock.patch.object(self.bc,
                               "_act_start_overnight_upgrade") as mover:
            self.bc._dispatch_tray_command("trigger_overnight", {})
        mover.assert_called_once()

    def test_each_audio_subtoggle_flips_its_flag(self):
        for cmd, attr in [
            ("audio_echo_cancel_toggle", "_audio_aec_enabled"),
            ("audio_noise_suppress_toggle", "_audio_ns_enabled"),
            ("audio_agc_toggle", "_audio_agc_enabled"),
        ]:
            flag = self._restore_attr_after(attr)
            start = flag[0]
            with mock.patch.object(self.bc, "_publish_audio_state"):
                self.bc._dispatch_tray_command(cmd, {})
            self.assertEqual(flag[0], not start, msg=f"{cmd}/{attr}")

    def test_pause_daemons_toggle_flips_and_propagates(self):
        paused = self._restore_attr_after("_daemons_paused")
        paused[0] = False
        # diagnostic_daemons + ambient_listen propagation are wrapped in
        # try/except; provide a fake ambient module + a stub diag module.
        fake_diag = mock.Mock()
        fake_al = mock.Mock()
        import core
        # ``from core import diagnostic_daemons`` resolves via the ``core``
        # package attr once a sibling imports the real module, so patch BOTH
        # sys.modules and the parent-package attr (see sibling test). The
        # skill_ambient_listen fake is a flat module — sys.modules alone is
        # correct for it.
        with mock.patch.object(self.bc, "_write_hud_state"), \
             mock.patch.dict("sys.modules",
                             {"core.diagnostic_daemons": fake_diag,
                              "skill_ambient_listen": fake_al}), \
             mock.patch.object(core, "diagnostic_daemons", fake_diag,
                               create=True):
            self.bc._dispatch_tray_command("pause_daemons_toggle", {})
        self.assertTrue(paused[0])
        fake_al.set_paused.assert_called_once_with(True)

    def test_generic_action_raising_is_swallowed(self):
        boom = mock.Mock(side_effect=RuntimeError("kaboom"))
        with mock.patch.object(self.bc, "ACTIONS", {"x": boom}), \
             mock.patch.object(self.bc, "_HEAVY_ACTIONS", frozenset()):
            # Must not propagate.
            self.bc._dispatch_tray_command("x", {"arg": "a"})
        boom.assert_called_once_with("a")

    def test_process_inflight_dispatches_commands(self):
        d = tempfile.mkdtemp()
        inflight = os.path.join(d, "tray.json.inflight")
        with open(inflight, "w", encoding="utf-8") as f:
            json.dump([{"cmd": "a"}, {"cmd": "b"}], f)
        with mock.patch.object(self.bc, "_dispatch_tray_command") as mdisp:
            n = self.bc._process_inflight(inflight)
        self.assertEqual(n, 2)
        self.assertEqual(mdisp.call_count, 2)
        # File removed before dispatch so relaunch commands can't re-fire.
        self.assertFalse(os.path.exists(inflight))

    def test_process_inflight_empty_file(self):
        d = tempfile.mkdtemp()
        inflight = os.path.join(d, "tray.json.inflight")
        with open(inflight, "w", encoding="utf-8") as f:
            f.write("   ")
        with mock.patch.object(self.bc, "_dispatch_tray_command") as mdisp:
            self.assertEqual(self.bc._process_inflight(inflight), 0)
            mdisp.assert_not_called()
        self.assertFalse(os.path.exists(inflight))

    def test_process_inflight_corrupt_json(self):
        d = tempfile.mkdtemp()
        inflight = os.path.join(d, "tray.json.inflight")
        with open(inflight, "w", encoding="utf-8") as f:
            f.write("{ broken")
        with mock.patch.object(self.bc, "_dispatch_tray_command") as mdisp:
            self.assertEqual(self.bc._process_inflight(inflight), 0)
            mdisp.assert_not_called()
        self.assertFalse(os.path.exists(inflight))

    def test_drain_commands_claims_and_processes(self):
        d = tempfile.mkdtemp()
        cmd_file = os.path.join(d, "tray_commands.json")
        with open(cmd_file, "w", encoding="utf-8") as f:
            json.dump([{"cmd": "x"}], f)
        with mock.patch.object(self.bc, "TRAY_COMMANDS_FILE", cmd_file), \
             mock.patch.object(self.bc, "_dispatch_tray_command") as mdisp:
            n = self.bc._drain_tray_commands_once()
        self.assertEqual(n, 1)
        mdisp.assert_called_once()
        # The claim file (.inflight) is consumed/removed.
        self.assertFalse(os.path.exists(cmd_file + ".inflight"))
        self.assertFalse(os.path.exists(cmd_file))

    def test_drain_commands_no_file_returns_zero(self):
        d = tempfile.mkdtemp()
        cmd_file = os.path.join(d, "tray_commands.json")  # does not exist
        with mock.patch.object(self.bc, "TRAY_COMMANDS_FILE", cmd_file):
            self.assertEqual(self.bc._drain_tray_commands_once(), 0)


# ──────────────────────────────────────────────────────────────────────────
#  _shutdown_tray (sets stop events, terminates process)
# ──────────────────────────────────────────────────────────────────────────
class TrayShutdownTests(_MonolithTestBase):
    def test_shutdown_tray_sets_events_and_terminates(self):
        fake_proc = mock.Mock()
        # Use real Event objects so .set()/.is_set() behave; restore after.
        drain = self.bc._tray_drain_stop
        pub = self.bc._tray_publisher_stop
        drain.clear()
        pub.clear()
        self.addCleanup(drain.clear)
        self.addCleanup(pub.clear)
        with mock.patch.object(self.bc, "_tray_process", fake_proc):
            self.bc._shutdown_tray()
            fake_proc.terminate.assert_called_once()
        self.assertTrue(drain.is_set())
        self.assertTrue(pub.is_set())
        self.assertIsNone(self.bc._tray_process)

    def test_shutdown_tray_noop_process_still_sets_events(self):
        drain = self.bc._tray_drain_stop
        pub = self.bc._tray_publisher_stop
        drain.clear()
        pub.clear()
        self.addCleanup(drain.clear)
        self.addCleanup(pub.clear)
        with mock.patch.object(self.bc, "_tray_process", None):
            self.bc._shutdown_tray()
        self.assertTrue(drain.is_set())
        self.assertTrue(pub.is_set())


# ──────────────────────────────────────────────────────────────────────────
#  Subprocess launch paths (HUD / tray / reticle) — success branch
# ──────────────────────────────────────────────────────────────────────────
class LaunchPathTests(_MonolithTestBase):
    def test_launch_hud_spawns_subprocess(self):
        fake_proc = mock.Mock()
        fake_proc.pid = 4321
        with mock.patch.object(self.bc, "HUD_ENABLED", True), \
             mock.patch.object(self.bc.os.path, "exists", return_value=True), \
             mock.patch.object(self.bc.subprocess, "Popen",
                               return_value=fake_proc) as mpop, \
             mock.patch.object(self.bc, "_hud_process", None):
            self.bc._launch_hud()
            mpop.assert_called_once()
            self.assertIs(self.bc._hud_process, fake_proc)
        # restore
        self.bc._hud_process = None

    def test_launch_hud_popen_failure_is_swallowed(self):
        with mock.patch.object(self.bc, "HUD_ENABLED", True), \
             mock.patch.object(self.bc.os.path, "exists", return_value=True), \
             mock.patch.object(self.bc.subprocess, "Popen",
                               side_effect=OSError("spawn fail")), \
             mock.patch.object(self.bc, "_hud_process", "sentinel"):
            self.bc._launch_hud()  # must not raise
            self.assertIsNone(self.bc._hud_process)
        self.bc._hud_process = None

    def test_launch_tray_spawns_and_seeds_audio_state(self):
        fake_proc = mock.Mock()
        fake_proc.pid = 99
        with mock.patch.object(self.bc, "TRAY_ENABLED", True), \
             mock.patch.object(self.bc.os.path, "exists", return_value=True), \
             mock.patch.object(self.bc.os, "remove"), \
             mock.patch.object(self.bc.subprocess, "Popen",
                               return_value=fake_proc) as mpop, \
             mock.patch.object(self.bc, "_publish_audio_state") as maudio, \
             mock.patch.object(self.bc, "_tray_process", None):
            self.bc._launch_tray()
            mpop.assert_called_once()
            maudio.assert_called_once()
        self.bc._tray_process = None

    def test_launch_tray_noop_when_disabled(self):
        with mock.patch.object(self.bc, "TRAY_ENABLED", False), \
             mock.patch.object(self.bc.subprocess, "Popen") as mpop:
            self.bc._launch_tray()
            mpop.assert_not_called()

    def test_launch_reticle_spawns_subprocess(self):
        fake_proc = mock.Mock()
        fake_proc.pid = 7
        d = tempfile.mkdtemp()
        state = os.path.join(d, "ret.json")
        with mock.patch.object(self.bc, "RETICLE_OVERLAY_ENABLED", True), \
             mock.patch.object(self.bc, "RETICLE_STATE_FILE", state), \
             mock.patch.object(self.bc.os.path, "exists", return_value=True), \
             mock.patch.object(self.bc.subprocess, "Popen",
                               return_value=fake_proc) as mpop, \
             mock.patch.object(self.bc, "_reticle_process", None):
            self.bc._launch_reticle_overlay()
            mpop.assert_called_once()
            self.assertIs(self.bc._reticle_process, fake_proc)
        self.bc._reticle_process = None

    def test_launch_reticle_missing_script_skips(self):
        with mock.patch.object(self.bc, "RETICLE_OVERLAY_ENABLED", True), \
             mock.patch.object(self.bc.os.path, "exists", return_value=False), \
             mock.patch.object(self.bc.subprocess, "Popen") as mpop:
            self.bc._launch_reticle_overlay()
            mpop.assert_not_called()


# ──────────────────────────────────────────────────────────────────────────
#  setup_logging — installs the Tee + excepthook
# ──────────────────────────────────────────────────────────────────────────
class SetupLoggingTests(_MonolithTestBase):
    def test_setup_logging_noop_when_disabled(self):
        with mock.patch.object(self.bc, "LOGGING_ENABLED", False), \
             mock.patch.object(self.bc.os, "makedirs") as mmk:
            self.bc.setup_logging()
            mmk.assert_not_called()

    def test_setup_logging_creates_log_and_installs_tee(self):
        d = tempfile.mkdtemp()
        # Save originals so we can restore stdout/stderr/excepthook/global.
        orig_out, orig_err = self.bc.sys.stdout, self.bc.sys.stderr
        orig_hook = self.bc.sys.excepthook
        orig_handle = self.bc._log_file_handle
        orig_path = self.bc._log_file_path

        def _restore():
            self.bc.sys.stdout = orig_out
            self.bc.sys.stderr = orig_err
            self.bc.sys.excepthook = orig_hook
            try:
                if self.bc._log_file_handle is not None:
                    self.bc._log_file_handle.close()
            except Exception:
                pass
            self.bc._log_file_handle = orig_handle
            self.bc._log_file_path = orig_path

        self.addCleanup(_restore)
        with mock.patch.object(self.bc, "LOGGING_ENABLED", True), \
             mock.patch.object(self.bc, "LOGS_DIR", d), \
             mock.patch.object(self.bc, "_cleanup_old_logs"):
            self.bc.setup_logging()
        # A log file was opened and the Tee installed.
        self.assertIsNotNone(self.bc._log_file_path)
        self.assertTrue(os.path.exists(self.bc._log_file_path))
        self.assertIsInstance(self.bc.sys.stdout, self.bc._TimestampedTee)


# ──────────────────────────────────────────────────────────────────────────
#  _active_window_center — success path with a fake pygetwindow
# ──────────────────────────────────────────────────────────────────────────
class ActiveWindowCenterTests(_MonolithTestBase):
    def _fake_gw(self, win):
        mod = mock.Mock()
        mod.getActiveWindow.return_value = win
        return mod

    def test_center_computed_from_active_window(self):
        win = mock.Mock(left=100, top=50, width=200, height=100)
        with mock.patch.dict("sys.modules",
                             {"pygetwindow": self._fake_gw(win)}):
            self.assertEqual(self.bc._active_window_center(), (200, 100))

    def test_none_when_no_active_window(self):
        with mock.patch.dict("sys.modules",
                             {"pygetwindow": self._fake_gw(None)}):
            self.assertIsNone(self.bc._active_window_center())

    def test_none_when_degenerate_zero_size(self):
        win = mock.Mock(left=0, top=0, width=0, height=0)
        with mock.patch.dict("sys.modules",
                             {"pygetwindow": self._fake_gw(win)}):
            self.assertIsNone(self.bc._active_window_center())


# ──────────────────────────────────────────────────────────────────────────
#  _process_inflight non-list + _drain orphan recovery
# ──────────────────────────────────────────────────────────────────────────
class TrayInflightEdgeTests(_MonolithTestBase):
    def test_process_inflight_non_list_payload(self):
        d = tempfile.mkdtemp()
        inflight = os.path.join(d, "tray.json.inflight")
        with open(inflight, "w", encoding="utf-8") as f:
            json.dump({"cmd": "x"}, f)  # dict, not a list → 0, file removed
        with mock.patch.object(self.bc, "_dispatch_tray_command") as mdisp:
            self.assertEqual(self.bc._process_inflight(inflight), 0)
            mdisp.assert_not_called()
        self.assertFalse(os.path.exists(inflight))

    def test_process_inflight_skips_non_dict_entries(self):
        d = tempfile.mkdtemp()
        inflight = os.path.join(d, "tray.json.inflight")
        with open(inflight, "w", encoding="utf-8") as f:
            json.dump([{"cmd": "a"}, "not-a-dict", {"cmd": "b"}], f)
        with mock.patch.object(self.bc, "_dispatch_tray_command") as mdisp:
            n = self.bc._process_inflight(inflight)
        self.assertEqual(n, 2)  # the string entry is skipped
        self.assertEqual(mdisp.call_count, 2)

    def test_process_inflight_read_failure_removes_file(self):
        d = tempfile.mkdtemp()
        inflight = os.path.join(d, "tray.json.inflight")
        with open(inflight, "w", encoding="utf-8") as f:
            json.dump([{"cmd": "a"}], f)
        # Force the open()/read to raise so the read-failure arm runs.
        real_open = open

        def _boom(path, *a, **k):
            if path == inflight:
                raise OSError("read denied")
            return real_open(path, *a, **k)

        with mock.patch("builtins.open", _boom), \
             mock.patch.object(self.bc, "_dispatch_tray_command") as mdisp:
            self.assertEqual(self.bc._process_inflight(inflight), 0)
            mdisp.assert_not_called()
        self.assertFalse(os.path.exists(inflight))

    def test_process_inflight_dispatch_failure_counts_others(self):
        d = tempfile.mkdtemp()
        inflight = os.path.join(d, "tray.json.inflight")
        with open(inflight, "w", encoding="utf-8") as f:
            json.dump([{"cmd": "bad"}, {"cmd": "good"}], f)

        def _disp(cmd, entry):
            if cmd == "bad":
                raise RuntimeError("nope")

        with mock.patch.object(self.bc, "_dispatch_tray_command",
                               side_effect=_disp):
            # 'bad' raises (caught, not counted); 'good' succeeds → n == 1.
            self.assertEqual(self.bc._process_inflight(inflight), 1)

    def test_drain_recovers_orphaned_inflight_first(self):
        d = tempfile.mkdtemp()
        cmd_file = os.path.join(d, "tray_commands.json")
        inflight = cmd_file + ".inflight"
        # An orphaned inflight from a previous crash, plus a fresh inbox.
        with open(inflight, "w", encoding="utf-8") as f:
            json.dump([{"cmd": "orphan"}], f)
        with open(cmd_file, "w", encoding="utf-8") as f:
            json.dump([{"cmd": "fresh"}], f)
        seen = []
        with mock.patch.object(self.bc, "TRAY_COMMANDS_FILE", cmd_file), \
             mock.patch.object(self.bc, "_dispatch_tray_command",
                               side_effect=lambda c, e: seen.append(c)):
            n = self.bc._drain_tray_commands_once()
        # Both the orphan and the freshly-claimed command run.
        self.assertEqual(n, 2)
        self.assertEqual(seen, ["orphan", "fresh"])


# ══════════════════════════════════════════════════════════════════════════
#  COVERAGE-EXTENSION PASS — additional error/edge branches in lines 50-2704.
#  Same harness + rules as the suites above: inherit _MonolithTestBase, use
#  self.bc, mock ALL I/O, restore any directly-mutated global, never touch real
#  hardware/network/LLM/threads.
# ══════════════════════════════════════════════════════════════════════════


# ──────────────────────────────────────────────────────────────────────────
#  _read_lock_pid / _acquire_os_singleton_lock — remaining retry/error arms
# ──────────────────────────────────────────────────────────────────────────
class LockHelperEdgeTests(_MonolithTestBase):
    def test_read_lock_pid_oserror_retries_then_stale(self):
        # An OSError other than FileNotFoundError on read → retry path (77-79).
        # After the budget is exhausted with no readable PID it returns 0.
        d = tempfile.mkdtemp()
        path = os.path.join(d, "jarvis.lock")
        with open(path, "w", encoding="utf-8") as f:
            f.write("12345")
        real_open = open
        calls = {"n": 0}

        def _boom_open(p, *a, **k):
            if p == path:
                calls["n"] += 1
                raise OSError("transient share violation")
            return real_open(p, *a, **k)

        with mock.patch("builtins.open", _boom_open), \
             mock.patch.object(self.bc.time, "sleep", return_value=None):
            self.assertEqual(self.bc._read_lock_pid(path, max_retries=3), 0)
        # Every attempt hit the OSError arm (slept + continued), not a one-shot.
        self.assertEqual(calls["n"], 3)

    def test_read_lock_pid_oserror_then_recovers(self):
        # First read raises OSError (retry), second read returns a valid PID.
        d = tempfile.mkdtemp()
        path = os.path.join(d, "jarvis.lock")
        with open(path, "w", encoding="utf-8") as f:
            f.write("777")
        real_open = open
        state = {"n": 0}

        def _flaky_open(p, *a, **k):
            if p == path:
                state["n"] += 1
                if state["n"] == 1:
                    raise OSError("locked")
            return real_open(p, *a, **k)

        with mock.patch("builtins.open", _flaky_open), \
             mock.patch.object(self.bc.time, "sleep", return_value=None):
            self.assertEqual(self.bc._read_lock_pid(path, max_retries=5), 777)

    def test_early_boot_acquires_lock_and_writes_pid(self):
        # Drive the full happy path of _early_boot_singleton_lock with every
        # filesystem touch mocked so nothing is written to the real repo and the
        # process never exits. Covers the acquire + PID-write success arms
        # (167-250, 299-303).
        bc = self.bc
        written = {}
        real_open = open

        def _fake_open(path, mode="r", *a, **k):
            if str(path).endswith(".lock"):
                import io
                buf = io.StringIO()

                def _close(_b=buf, _p=path):
                    written[_p] = _b.getvalue()
                buf.close = _close  # capture on context-manager exit
                return buf
            return real_open(path, mode, *a, **k)

        self.addCleanup(setattr, bc, "_SINGLETON_HELD_FD",
                        bc._SINGLETON_HELD_FD)
        orig_env = os.environ.get("_JARVIS_SINGLETON_PID")

        def _restore_env(v=orig_env):
            if v is None:
                os.environ.pop("_JARVIS_SINGLETON_PID", None)
            else:
                os.environ["_JARVIS_SINGLETON_PID"] = v
        self.addCleanup(_restore_env)
        os.environ.pop("_JARVIS_SINGLETON_PID", None)  # bypass re-entrancy guard

        with mock.patch.object(bc.os.path, "exists", return_value=False), \
             mock.patch.object(bc.os, "open", return_value=4242), \
             mock.patch.object(bc.os, "close"), \
             mock.patch.object(bc, "_acquire_os_singleton_lock",
                               return_value=True), \
             mock.patch("builtins.open", _fake_open), \
             mock.patch.object(bc.sys.stdout, "flush"):
            self.assertTrue(bc._early_boot_singleton_lock())
        # The plain PID file was written with our PID; the held fd was kept.
        self.assertTrue(any(v == str(os.getpid()) for v in written.values()))
        self.assertEqual(bc._SINGLETON_HELD_FD, 4242)

    def test_early_boot_duplicate_instance_exits(self):
        # Lock acquisition fails AND the PID file names a DIFFERENT pid → another
        # live instance → sys.exit(0). Covers the duplicate-refusal arm
        # (205-235).
        bc = self.bc
        self.addCleanup(setattr, bc, "_SINGLETON_HELD_FD",
                        bc._SINGLETON_HELD_FD)
        orig_env = os.environ.get("_JARVIS_SINGLETON_PID")

        def _restore_env(v=orig_env):
            if v is None:
                os.environ.pop("_JARVIS_SINGLETON_PID", None)
            else:
                os.environ["_JARVIS_SINGLETON_PID"] = v
        self.addCleanup(_restore_env)
        os.environ.pop("_JARVIS_SINGLETON_PID", None)

        with mock.patch.object(bc.os.path, "exists", return_value=False), \
             mock.patch.object(bc.os, "open", return_value=99), \
             mock.patch.object(bc.os, "close"), \
             mock.patch.object(bc, "_acquire_os_singleton_lock",
                               return_value=False), \
             mock.patch.object(bc, "_read_lock_pid", return_value=1234567), \
             mock.patch.object(bc.sys.stdout, "flush"):
            with self.assertRaises(SystemExit) as ctx:
                bc._early_boot_singleton_lock()
        self.assertEqual(ctx.exception.code, 0)

    def test_early_boot_lock_held_by_self_returns_true(self):
        # Lock acquisition fails but the PID file names US → just a second
        # module-identity of our own process → treat as success (211-217).
        bc = self.bc
        self.addCleanup(setattr, bc, "_SINGLETON_HELD_FD",
                        bc._SINGLETON_HELD_FD)
        orig_env = os.environ.get("_JARVIS_SINGLETON_PID")

        def _restore_env(v=orig_env):
            if v is None:
                os.environ.pop("_JARVIS_SINGLETON_PID", None)
            else:
                os.environ["_JARVIS_SINGLETON_PID"] = v
        self.addCleanup(_restore_env)
        os.environ.pop("_JARVIS_SINGLETON_PID", None)

        with mock.patch.object(bc.os.path, "exists", return_value=False), \
             mock.patch.object(bc.os, "open", return_value=55), \
             mock.patch.object(bc.os, "close"), \
             mock.patch.object(bc, "_acquire_os_singleton_lock",
                               return_value=False), \
             mock.patch.object(bc, "_read_lock_pid", return_value=os.getpid()), \
             mock.patch.object(bc.sys.stdout, "flush"):
            self.assertTrue(bc._early_boot_singleton_lock())
        self.assertEqual(os.environ.get("_JARVIS_SINGLETON_PID"),
                         str(os.getpid()))

    def test_early_boot_lock_write_failure_exits_1(self):
        # Mutex acquired but writing the plain PID file raises → fast-fail
        # sys.exit(1) after dropping the boot-error marker + JSONL record
        # (252-298). All file writes + makedirs are mocked away.
        bc = self.bc
        self.addCleanup(setattr, bc, "_SINGLETON_HELD_FD",
                        bc._SINGLETON_HELD_FD)
        orig_env = os.environ.get("_JARVIS_SINGLETON_PID")

        def _restore_env(v=orig_env):
            if v is None:
                os.environ.pop("_JARVIS_SINGLETON_PID", None)
            else:
                os.environ["_JARVIS_SINGLETON_PID"] = v
        self.addCleanup(_restore_env)
        os.environ.pop("_JARVIS_SINGLETON_PID", None)

        real_open = open

        def _open_pid_fails(path, mode="r", *a, **k):
            sp = str(path)
            if sp.endswith(".lock") and ("w" in mode):
                raise OSError("pid write denied")
            if sp.endswith("boot_error.txt") or sp.endswith(".jsonl"):
                import io
                return io.StringIO()  # swallow the marker writes
            return real_open(path, mode, *a, **k)

        with mock.patch.object(bc.os.path, "exists", return_value=False), \
             mock.patch.object(bc.os, "open", return_value=66), \
             mock.patch.object(bc.os, "close"), \
             mock.patch.object(bc.os, "makedirs"), \
             mock.patch.object(bc, "_acquire_os_singleton_lock",
                               return_value=True), \
             mock.patch("builtins.open", _open_pid_fails), \
             mock.patch.object(bc.sys.stdout, "flush"), \
             mock.patch.object(bc.sys.stderr, "write"), \
             mock.patch.object(bc.sys.stderr, "flush"):
            with self.assertRaises(SystemExit) as ctx:
                bc._early_boot_singleton_lock()
        self.assertEqual(ctx.exception.code, 1)

    def test_acquire_os_singleton_lock_busy_returns_false(self):
        # Hold the byte-0 lock on a real file, then a second fd's non-blocking
        # acquire must return False (the platform-specific OSError arm: 125/131).
        d = tempfile.mkdtemp()
        path = os.path.join(d, "mutex.lock")
        fd1 = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
        fd2 = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            self.assertTrue(self.bc._acquire_os_singleton_lock(fd1))
            # fd1 owns byte 0; fd2 cannot lock it → caught OSError → False.
            self.assertFalse(self.bc._acquire_os_singleton_lock(fd2))
        finally:
            os.close(fd1)
            os.close(fd2)


# ──────────────────────────────────────────────────────────────────────────
#  merge_memory — internal-noise drop + project dedupe
# ──────────────────────────────────────────────────────────────────────────
class MergeMemoryEdgeTests(_MonolithTestBase):
    def setUp(self):
        self._store = {"facts": [], "projects": [], "topics": [], "sessions": []}

        def _fake_load():
            import copy
            return copy.deepcopy(self._store)

        def _fake_save(m):
            self._store = m

        p_load = mock.patch.object(self.bc, "load_memory", _fake_load)
        p_save = mock.patch.object(self.bc, "save_memory", _fake_save)
        p_load.start()
        p_save.start()
        self.addCleanup(p_load.stop)
        self.addCleanup(p_save.stop)

    def test_internal_noise_fact_is_dropped(self):
        # "n/a" is a placeholder the internal-noise classifier rejects, so the
        # candidate is dropped before storage (1161-1163) — no save of it.
        added_f, _ = self.bc.merge_memory(new_facts=["n/a"])
        self.assertEqual(added_f, [])
        self.assertEqual(self._store["facts"], [])

    def test_duplicate_project_is_skipped(self):
        # Seed an existing project; a case-different duplicate hits the
        # dedupe `continue` (1192-1193) and is not re-added.
        self._store["projects"] = ["Building a treehouse"]
        _, added_p = self.bc.merge_memory(new_projects=["building A TREEHOUSE"])
        self.assertEqual(added_p, [])
        self.assertEqual(self._store["projects"], ["Building a treehouse"])


# ──────────────────────────────────────────────────────────────────────────
#  _load_chappie_standing_rules — empty-list + all-filtered → ""
# ──────────────────────────────────────────────────────────────────────────
class StandingRulesEdgeTests(_MonolithTestBase):
    def _write(self, payload):
        d = tempfile.mkdtemp()
        path = os.path.join(d, "rules.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        return path

    def test_empty_rules_list_returns_empty(self):
        # "rules": [] → falsy list → early "" (line 1248).
        path = self._write({"rules": []})
        with mock.patch.object(self.bc, "_CHAPPIE_STANDING_RULES_PATH", path):
            self.assertEqual(self.bc._load_chappie_standing_rules(), "")

    def test_all_entries_filtered_returns_empty(self):
        # Every entry is missing id or text (or not a dict), so `lines` stays
        # empty after the loop → final "" guard (line 1260), NOT the formatted
        # block.
        path = self._write({"rules": [
            {"id": "NoText"},        # missing rule
            {"rule": "NoId"},        # missing id
            "not-a-dict",            # skipped
            {},                      # both missing
        ]})
        with mock.patch.object(self.bc, "_CHAPPIE_STANDING_RULES_PATH", path):
            self.assertEqual(self.bc._load_chappie_standing_rules(), "")


# ──────────────────────────────────────────────────────────────────────────
#  build_system_prompt — malformed first_meeting falls through
# ──────────────────────────────────────────────────────────────────────────
class SystemPromptEdgeTests(_MonolithTestBase):
    def test_malformed_first_meeting_swallowed(self):
        # A non-ISO first_meeting makes the days-known parse raise → except/pass
        # (1295-1296); days_known stays 0 and the prompt still builds.
        mem = self.bc._empty_memory()
        mem["first_meeting"] = "not-a-date"
        with mock.patch.object(self.bc, "_load_chappie_standing_rules",
                               return_value=""), \
             mock.patch.object(self.bc._mcu_phrases, "render_phrasebook_block",
                               return_value="PB"):
            prompt = self.bc.build_system_prompt(mem)
        self.assertIn("0 day(s)", prompt)
        self.assertIn("PB", prompt)


# ──────────────────────────────────────────────────────────────────────────
#  _llm_quick — ollama backend branch
# ──────────────────────────────────────────────────────────────────────────
class LlmQuickOllamaTests(_MonolithTestBase):
    def setUp(self):
        super().setUp()
        # Pin the owner's force-local setting OFF so the ollama-backend path is
        # what gets exercised (see LlmQuickTests.setUp).
        import core.config as cfg
        _p = mock.patch.object(cfg, "AMBIENT_LEARNING_FORCE_LOCAL", False)
        _p.start()
        self.addCleanup(_p.stop)

    def test_ollama_backend_returns_message_content(self):
        fake_ollama = mock.Mock()
        fake_ollama.chat.return_value = {"message": {"content": "ollama says hi"}}
        with mock.patch.object(self.bc, "AI_BACKEND", "ollama"), \
             mock.patch.object(self.bc, "OLLAMA_MODEL", "llama3"), \
             mock.patch.dict("sys.modules", {"ollama": fake_ollama}):
            out = self.bc._llm_quick("sys", "user", max_tokens=33)
        self.assertEqual(out, "ollama says hi")
        fake_ollama.chat.assert_called_once()
        _, kwargs = fake_ollama.chat.call_args
        self.assertEqual(kwargs["model"], "llama3")
        self.assertEqual(kwargs["messages"][0]["role"], "system")


# ──────────────────────────────────────────────────────────────────────────
#  learn_from_turn worker — bad-JSON + worker-exception arms
# ──────────────────────────────────────────────────────────────────────────
class LearnFromTurnEdgeTests(_MonolithTestBase):
    def test_worker_malformed_json_after_brace_returns(self):
        # _llm_quick yields a '{' but the rest isn't valid JSON → raw_decode
        # raises JSONDecodeError → early return before merge (1440-1442).
        mem = self.bc._empty_memory()
        with mock.patch.object(self.bc, "LEARN_EVERY_TURN", True), \
             mock.patch.object(self.bc.threading, "Thread", _InlineThread), \
             mock.patch.object(self.bc, "_llm_quick", return_value="{ broken json"), \
             mock.patch.object(self.bc, "merge_memory") as mmerge:
            self.bc.learn_from_turn("x", "y", mem)
        mmerge.assert_not_called()

    def test_worker_exception_is_swallowed(self):
        # _llm_quick raising inside the worker must be caught (1460-1462), not
        # propagate out of learn_from_turn.
        mem = self.bc._empty_memory()
        with mock.patch.object(self.bc, "LEARN_EVERY_TURN", True), \
             mock.patch.object(self.bc.threading, "Thread", _InlineThread), \
             mock.patch.object(self.bc, "_llm_quick",
                               side_effect=RuntimeError("llm boom")), \
             mock.patch.object(self.bc, "merge_memory") as mmerge:
            self.bc.learn_from_turn("x", "y", mem)  # must not raise
        mmerge.assert_not_called()


# ──────────────────────────────────────────────────────────────────────────
#  record_action_error — prune-in-record + traceback/outer error arms
# ──────────────────────────────────────────────────────────────────────────
class RecordActionErrorEdgeTests(_MonolithTestBase):
    def test_record_prunes_expired_front_entry(self):
        # An entry older than the window is popped during the record insert
        # itself (the while-popleft at 1540-1541), not just on snapshot.
        log = self._restore_attr_after("_action_error_log")
        log.clear()
        now = self.bc.time.time()
        window = self.bc._ACTION_ERROR_LOG_WINDOW_S
        log.append({"ts": now - window - 100, "action": "ancient",
                    "exc_class": "E", "exc_msg": "", "traceback": ""})
        self.bc.record_action_error("fresh", ValueError("x"), traceback_text="")
        actions = [e["action"] for e in log]
        self.assertNotIn("ancient", actions)   # pruned in-record
        self.assertIn("fresh", actions)

    def test_record_default_traceback_capture_failure(self):
        # traceback.format_exc() raising is caught and traceback_text becomes ""
        # (1528-1529); the entry is still recorded.
        log = self._restore_attr_after("_action_error_log")
        log.clear()
        with mock.patch.object(self.bc.traceback, "format_exc",
                               side_effect=RuntimeError("no tb")):
            self.bc.record_action_error("act", ValueError("boom"))
        self.assertEqual(len(log), 1)
        self.assertEqual(log[0]["traceback"], "")

    def test_record_outer_failure_swallowed(self):
        # An error inside the outer try (here: time.time raising) is swallowed
        # by the 1542-1544 guard so recording never breaks the dispatcher.
        log = self._restore_attr_after("_action_error_log")
        log.clear()
        with mock.patch.object(self.bc.time, "time",
                               side_effect=RuntimeError("clock dead")):
            self.bc.record_action_error("act", ValueError("boom"))  # no raise
        self.assertEqual(len(log), 0)  # nothing appended


# ──────────────────────────────────────────────────────────────────────────
#  _note_camera_read_attempt — outer except guard
# ──────────────────────────────────────────────────────────────────────────
class CameraNoteEdgeTests(_MonolithTestBase):
    def test_note_outer_failure_swallowed(self):
        # time.time raising trips the outer except (1609-1610); no raise, no
        # entry created.
        summ = self._restore_attr_after("_camera_failure_summary")
        summ.clear()
        with mock.patch.object(self.bc.time, "time",
                               side_effect=RuntimeError("clock dead")):
            self.bc._note_camera_read_attempt(0, ok=False, fails=1)  # no raise
        self.assertEqual(self.bc.get_camera_failure_summary(), {})


# ──────────────────────────────────────────────────────────────────────────
#  _load_patterns corrupt-JSON / _save_patterns write-failure arms
# ──────────────────────────────────────────────────────────────────────────
class PatternsEdgeTests(_MonolithTestBase):
    def test_load_patterns_corrupt_json_returns_empty(self):
        # Invalid JSON makes json.load raise → except arm (1627-1628) → [].
        d = tempfile.mkdtemp()
        path = os.path.join(d, "patterns.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write("{ not valid json ][")
        with mock.patch.object(self.bc, "PATTERNS_FILE", path):
            self.assertEqual(self.bc._load_patterns(), [])

    def test_save_patterns_failure_is_swallowed(self):
        # makedirs raising trips the except (1638-1639); no propagation.
        with mock.patch.object(self.bc.os, "makedirs",
                               side_effect=OSError("read-only fs")):
            self.bc._save_patterns([{"day": "Monday"}])  # must not raise


# ──────────────────────────────────────────────────────────────────────────
#  record_session_action — pattern_learning log_event failure arm
# ──────────────────────────────────────────────────────────────────────────
class RecordSessionActionEdgeTests(_MonolithTestBase):
    def test_pattern_learning_log_event_failure_swallowed(self):
        # A pattern_learning module whose log_event raises must be caught
        # (1681-1682) — the action still counts.
        counts = self._restore_attr_after("_session_action_counts")
        counts.clear()
        fake_pl = mock.Mock()
        fake_pl.log_event.side_effect = RuntimeError("pl down")
        with mock.patch.dict("sys.modules", {"skill_pattern_learning": fake_pl}):
            self.bc.record_session_action("get_time", "")  # must not raise
        self.assertEqual(counts["get_time"], 1)
        fake_pl.log_event.assert_called_once()


# ──────────────────────────────────────────────────────────────────────────
#  detect_startup_pattern — neutral match (no strong category) → ""
# ──────────────────────────────────────────────────────────────────────────
class DetectStartupPatternEdgeTests(_MonolithTestBase):
    def test_matching_but_no_strong_signal_returns_empty(self):
        # 6 sessions, current day/hour match (>=3 matching, tally non-empty) but
        # the dominant action is neither streaming-evening nor a build action,
        # so execution reaches the final `return ""` (line 1765).
        fixed = time.struct_time((2026, 6, 3, 14, 0, 0, 2, 154, -1))  # Wed 14:00
        entries = [{"day": "Wednesday", "hour_started": 14,
                    "top_actions": ["get_time", "weather"]} for _ in range(6)]
        with mock.patch.object(self.bc, "_load_patterns", return_value=entries), \
             mock.patch.object(self.bc.time, "localtime", return_value=fixed), \
             mock.patch.object(self.bc.time, "strftime",
                               side_effect=lambda fmt, *a: "Wednesday"):
            self.assertEqual(self.bc.detect_startup_pattern(), "")

    def test_matching_but_empty_top_actions_returns_empty(self):
        # >=3 matching sessions but every top_actions is empty → tally stays
        # empty → the `if not tally` guard (line 1745) returns "".
        fixed = time.struct_time((2026, 6, 3, 14, 0, 0, 2, 154, -1))
        entries = [{"day": "Wednesday", "hour_started": 14,
                    "top_actions": []} for _ in range(6)]
        with mock.patch.object(self.bc, "_load_patterns", return_value=entries), \
             mock.patch.object(self.bc.time, "localtime", return_value=fixed), \
             mock.patch.object(self.bc.time, "strftime",
                               side_effect=lambda fmt, *a: "Wednesday"):
            self.assertEqual(self.bc.detect_startup_pattern(), "")


# ──────────────────────────────────────────────────────────────────────────
#  save_session_to_memory — recall-index failure + outer failure arms
# ──────────────────────────────────────────────────────────────────────────
class SaveSessionEdgeTests(_MonolithTestBase):
    def _seed_history(self):
        hist = self._restore_attr_after("conversation_history")
        hist.clear()
        for i in range(4):
            hist.append({"role": "user", "content": f"u{i}"})
            hist.append({"role": "assistant", "content": f"a{i}"})
        return hist

    def test_recall_index_failure_swallowed_but_session_saved(self):
        # record_session_summary raising is caught (1816-1817); the session
        # summary is still persisted via the fresh locked write.
        self._seed_history()
        mem = self.bc._empty_memory()
        store = {"sessions": []}

        def _fake_load():
            import copy
            return copy.deepcopy(store)

        def _fake_save(m):
            store.update(m)

        with mock.patch.object(self.bc, "_llm_quick",
                               return_value="A good session."), \
             mock.patch.object(self.bc, "load_memory", _fake_load), \
             mock.patch.object(self.bc, "save_memory", _fake_save), \
             mock.patch.object(self.bc.pattern_memory, "record_session_summary",
                               side_effect=RuntimeError("recall down")):
            self.bc.save_session_to_memory(mem)  # must not raise
        self.assertTrue(store["sessions"])
        self.assertEqual(store["sessions"][-1]["summary"], "A good session.")

    def test_outer_failure_swallowed(self):
        # _llm_quick raising trips the outer except (1818-1819); no propagation
        # and nothing is written.
        self._seed_history()
        mem = self.bc._empty_memory()
        with mock.patch.object(self.bc, "_llm_quick",
                               side_effect=RuntimeError("summary boom")), \
             mock.patch.object(self.bc, "save_memory") as msave:
            self.bc.save_session_to_memory(mem)  # must not raise
            msave.assert_not_called()


# ──────────────────────────────────────────────────────────────────────────
#  _session_summary_checkpoint_thread — single-iteration body
# ──────────────────────────────────────────────────────────────────────────
class SessionCheckpointThreadTests(_MonolithTestBase):
    def _drive_one_iteration(self, *, last_len_val):
        """Run the daemon body exactly once: the leading sleep is a no-op and
        the trailing in-loop sleep raises a sentinel so the ``while True`` loop
        exits after one pass instead of looping forever."""
        class _StopLoop(Exception):
            pass

        sleeps = {"n": 0}

        def _sleep(_secs):
            sleeps["n"] += 1
            if sleeps["n"] >= 2:   # the trailing sleep inside the while loop
                raise _StopLoop()
            return None            # the leading pre-loop sleep

        last = self._restore_attr_after("_session_checkpoint_last_len")
        last[0] = last_len_val
        hist = self._restore_attr_after("conversation_history")
        hist.clear()
        for i in range(4):
            hist.append({"role": "user", "content": f"u{i}"})
            hist.append({"role": "assistant", "content": f"a{i}"})
        with mock.patch.object(self.bc.time, "sleep", _sleep):
            try:
                self.bc._session_summary_checkpoint_thread()
            except _StopLoop:
                pass
        return last

    def test_checkpoint_writes_summary_when_history_grew(self):
        captured = {}
        with mock.patch.object(self.bc, "_llm_quick",
                               return_value="Mid-session summary.\nextra"), \
             mock.patch.object(self.bc.pattern_memory, "record_session_summary",
                               side_effect=lambda s, **k: captured.update(s=s)):
            last = self._drive_one_iteration(last_len_val=0)
        self.assertEqual(captured["s"], "Mid-session summary.")
        self.assertEqual(last[0], 8)   # checkpoint stamps the new length

    def test_checkpoint_skips_when_history_unchanged(self):
        # last_len already equals the current history length (8) → the
        # hist_len != last guard is False, so no LLM call / no recall write.
        with mock.patch.object(self.bc, "_llm_quick") as mq, \
             mock.patch.object(self.bc.pattern_memory,
                               "record_session_summary") as mrec:
            self._drive_one_iteration(last_len_val=8)
        mq.assert_not_called()
        mrec.assert_not_called()

    def test_checkpoint_summary_failure_swallowed(self):
        # record_session_summary raising is caught by the inner except
        # (1869-1870); the loop continues to the trailing sleep (our sentinel).
        with mock.patch.object(self.bc, "_llm_quick",
                               return_value="A summary."), \
             mock.patch.object(self.bc.pattern_memory, "record_session_summary",
                               side_effect=RuntimeError("recall boom")):
            last = self._drive_one_iteration(last_len_val=0)
        # Failed write → last_len NOT advanced (still 0).
        self.assertEqual(last[0], 0)

    def test_checkpoint_outer_loop_error_swallowed(self):
        # An error in the snapshot/length logic (outside the inner try) is
        # caught by the outer loop guard (1871-1872). Here list(conversation_
        # history) raises because the global is a bad object for one iteration.
        class _StopLoop(Exception):
            pass

        sleeps = {"n": 0}

        def _sleep(_secs):
            sleeps["n"] += 1
            if sleeps["n"] >= 2:
                raise _StopLoop()
            return None

        class _BadHist:
            def __iter__(self):
                raise RuntimeError("snapshot boom")

        with mock.patch.object(self.bc, "conversation_history", _BadHist()), \
             mock.patch.object(self.bc.time, "sleep", _sleep), \
             mock.patch.object(self.bc, "_llm_quick") as mq:
            try:
                self.bc._session_summary_checkpoint_thread()
            except _StopLoop:
                pass
        # The outer except absorbed the snapshot error before the LLM was hit.
        mq.assert_not_called()


# ──────────────────────────────────────────────────────────────────────────
#  _cleanup_old_logs — inner unlink + outer listdir error arms
# ──────────────────────────────────────────────────────────────────────────
class CleanupOldLogsEdgeTests(_MonolithTestBase):
    def test_unlink_failure_per_file_swallowed(self):
        # os.unlink raising for an excess log is caught per-file (1928-1929);
        # the function completes without raising.
        d = tempfile.mkdtemp()
        keep = self.bc.LOG_KEEP_COUNT
        for i in range(keep + 2):
            p = os.path.join(d, f"session_{i}.log")
            with open(p, "w", encoding="utf-8") as f:
                f.write("x")
            os.utime(p, (1000 + i, 1000 + i))
        with mock.patch.object(self.bc, "LOGS_DIR", d), \
             mock.patch.object(self.bc.os, "unlink",
                               side_effect=OSError("locked")):
            self.bc._cleanup_old_logs()  # must not raise

    def test_listdir_failure_swallowed(self):
        # os.listdir raising trips the outer except (1930-1931).
        d = tempfile.mkdtemp()
        with mock.patch.object(self.bc, "LOGS_DIR", d), \
             mock.patch.object(self.bc.os, "listdir",
                               side_effect=OSError("denied")):
            self.bc._cleanup_old_logs()  # must not raise


# ──────────────────────────────────────────────────────────────────────────
#  setup_logging — installed excepthook body + faulthandler failure arm
# ──────────────────────────────────────────────────────────────────────────
class SetupLoggingEdgeTests(_MonolithTestBase):
    def _install_with_restore(self, d, **extra_patches):
        orig_out, orig_err = self.bc.sys.stdout, self.bc.sys.stderr
        orig_hook = self.bc.sys.excepthook
        orig_handle = self.bc._log_file_handle
        orig_path = self.bc._log_file_path

        def _restore():
            self.bc.sys.stdout = orig_out
            self.bc.sys.stderr = orig_err
            self.bc.sys.excepthook = orig_hook
            try:
                if self.bc._log_file_handle is not None:
                    self.bc._log_file_handle.close()
            except Exception:
                pass
            self.bc._log_file_handle = orig_handle
            self.bc._log_file_path = orig_path

        self.addCleanup(_restore)

    def test_excepthook_writes_fatal_to_stderr(self):
        # Drive the excepthook installed by setup_logging (1961-1965) with a
        # synthetic exception and confirm it writes the [FATAL] banner.
        d = tempfile.mkdtemp()
        self._install_with_restore(d)
        with mock.patch.object(self.bc, "LOGGING_ENABLED", True), \
             mock.patch.object(self.bc, "LOGS_DIR", d), \
             mock.patch.object(self.bc, "_cleanup_old_logs"):
            self.bc.setup_logging()
        import io as _io
        sink = _io.StringIO()
        # Point the freshly-installed Tee at a capturable stderr, then fire the
        # hook with a real (already-raised) exception so the traceback is valid.
        with mock.patch.object(self.bc.sys, "stderr", sink):
            try:
                raise ValueError("synthetic fatal")
            except ValueError:
                import sys as _sys
                self.bc.sys.excepthook(*_sys.exc_info())
        self.assertIn("[FATAL]", sink.getvalue())
        self.assertIn("synthetic fatal", sink.getvalue())

    def test_faulthandler_failure_is_swallowed(self):
        # faulthandler.enable raising must be caught (1985-1986); the Tee is
        # still installed and the log file still created.
        d = tempfile.mkdtemp()
        self._install_with_restore(d)
        import faulthandler as _fh
        with mock.patch.object(self.bc, "LOGGING_ENABLED", True), \
             mock.patch.object(self.bc, "LOGS_DIR", d), \
             mock.patch.object(self.bc, "_cleanup_old_logs"), \
             mock.patch.object(_fh, "enable",
                               side_effect=RuntimeError("no faulthandler")):
            self.bc.setup_logging()  # must not raise
        self.assertIsInstance(self.bc.sys.stdout, self.bc._TimestampedTee)
        self.assertTrue(os.path.exists(self.bc._log_file_path))


# ──────────────────────────────────────────────────────────────────────────
#  close_log — write-failure arm
# ──────────────────────────────────────────────────────────────────────────
class CloseLogEdgeTests(_MonolithTestBase):
    def test_close_log_write_failure_swallowed(self):
        # A handle whose write raises must be caught (1997-1998); no raise.
        class _Bad:
            def write(self, s):
                raise OSError("disk full")

            def flush(self):
                pass

            def close(self):
                pass

        with mock.patch.object(self.bc, "_log_file_handle", _Bad()), \
             mock.patch.object(self.bc.time, "strftime", return_value="TS"):
            self.bc.close_log()  # must not raise


# ──────────────────────────────────────────────────────────────────────────
#  _write_hud_state — inner replace-failure cleans up temp then re-raises
# ──────────────────────────────────────────────────────────────────────────
class WriteHudStateEdgeTests(_MonolithTestBase):
    def test_replace_failure_removes_temp_and_swallows(self):
        # os.replace raising runs the inner except (2137-2143): the temp file is
        # removed and the error re-raised into the outer pass. Net: no leftover
        # temp, no raise.
        d = tempfile.mkdtemp()
        path = os.path.join(d, "hud_state.json")
        with mock.patch.object(self.bc, "HUD_ENABLED", True), \
             mock.patch.object(self.bc, "HUD_STATE_FILE", path), \
             mock.patch.object(self.bc.os, "replace",
                               side_effect=OSError("replace failed")):
            self.bc._write_hud_state(state="x")  # must not raise
        # No .hud_* temp left behind in the target dir.
        leftovers = [f for f in os.listdir(d) if f.startswith(".hud_")]
        self.assertEqual(leftovers, [])

    def test_replace_and_cleanup_both_fail_swallowed(self):
        # os.replace AND the temp-cleanup os.remove both raising exercises the
        # nested except (2141-2142) before the re-raise lands in the outer pass.
        d = tempfile.mkdtemp()
        path = os.path.join(d, "hud_state.json")
        with mock.patch.object(self.bc, "HUD_ENABLED", True), \
             mock.patch.object(self.bc, "HUD_STATE_FILE", path), \
             mock.patch.object(self.bc.os, "replace",
                               side_effect=OSError("replace failed")), \
             mock.patch.object(self.bc.os, "remove",
                               side_effect=OSError("remove failed")):
            self.bc._write_hud_state(state="x")  # must not raise


# ──────────────────────────────────────────────────────────────────────────
#  _launch_hud — monitor-pick fallback + blue/green except
# ──────────────────────────────────────────────────────────────────────────
class LaunchHudEdgeTests(_MonolithTestBase):
    def test_unknown_monitor_falls_back_to_default(self):
        # MONITORS lacks the requested HUD_MONITOR but has a "middle" fallback,
        # so the defensive `mon = MONITORS.get("top") or ...` line (2182) runs
        # and the launch still proceeds.
        fake_proc = mock.Mock()
        fake_proc.pid = 11
        with mock.patch.object(self.bc, "HUD_ENABLED", True), \
             mock.patch.object(self.bc, "HUD_MONITOR", "does_not_exist"), \
             mock.patch.object(self.bc, "MONITORS",
                               {"middle": (0, 0, 1920, 1080)}), \
             mock.patch.object(self.bc.os.path, "exists", return_value=True), \
             mock.patch.object(self.bc, "_bgm", None), \
             mock.patch.object(self.bc.subprocess, "Popen",
                               return_value=fake_proc) as mpop, \
             mock.patch.object(self.bc, "_hud_process", None):
            self.bc._launch_hud()
            mpop.assert_called_once()
        self.bc._hud_process = None

    def test_blue_green_monitor_lookup_failure_swallowed(self):
        # With _bgm active but _BLUE_GREEN_PATHS.get raising, the per-role
        # monitor pick is caught (2176-2177) and launch continues on HUD_MONITOR.
        fake_proc = mock.Mock()
        fake_proc.pid = 12

        class _BadPaths(dict):
            # Only the in-try monitor-name lookup explodes (2174); the later
            # out-of-try hud_state_file lookup (2201) must behave normally or it
            # would propagate past the function instead of exercising 2176-2177.
            def get(self, key, default=None):
                if key == "monitor_name":
                    raise RuntimeError("paths exploded")
                return default

        with mock.patch.object(self.bc, "HUD_ENABLED", True), \
             mock.patch.object(self.bc, "HUD_MONITOR", "left"), \
             mock.patch.object(self.bc, "MONITORS",
                               {"left": (0, 0, 1920, 1080)}), \
             mock.patch.object(self.bc.os.path, "exists", return_value=True), \
             mock.patch.object(self.bc, "_bgm", object()), \
             mock.patch.object(self.bc, "_BLUE_GREEN_PATHS", _BadPaths()), \
             mock.patch.object(self.bc.subprocess, "Popen",
                               return_value=fake_proc) as mpop, \
             mock.patch.object(self.bc, "_hud_process", None):
            self.bc._launch_hud()  # must not raise
            mpop.assert_called_once()
        self.bc._hud_process = None


# ──────────────────────────────────────────────────────────────────────────
#  _publish_reticle — non-list "reticles" payload is reset
# ──────────────────────────────────────────────────────────────────────────
class PublishReticleEdgeTests(_MonolithTestBase):
    def test_non_list_reticles_payload_reset(self):
        # Valid JSON but "reticles" is a string (not a list) → the
        # isinstance guard resets entries to [] (line 2263) before appending.
        d = tempfile.mkdtemp()
        path = os.path.join(d, "ret.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"reticles": "not-a-list"}, f)
        with mock.patch.object(self.bc, "RETICLE_OVERLAY_ENABLED", True), \
             mock.patch.object(self.bc, "RETICLE_STATE_FILE", path):
            self.bc._publish_reticle(3, 4, "fresh")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual([r["label"] for r in data["reticles"]], ["fresh"])

    def test_final_write_failure_swallowed(self):
        # The final atomic os.replace failing trips the outer except (2280-2281)
        # — best-effort write, no raise.
        d = tempfile.mkdtemp()
        path = os.path.join(d, "ret.json")
        with mock.patch.object(self.bc, "RETICLE_OVERLAY_ENABLED", True), \
             mock.patch.object(self.bc, "RETICLE_STATE_FILE", path), \
             mock.patch.object(self.bc.os, "replace",
                               side_effect=OSError("replace failed")):
            self.bc._publish_reticle(1, 2, "x")  # must not raise


# ──────────────────────────────────────────────────────────────────────────
#  _active_window_center — inner geometry error + outer getActiveWindow error
# ──────────────────────────────────────────────────────────────────────────
class ActiveWindowCenterEdgeTests(_MonolithTestBase):
    def _fake_gw_module(self, **kw):
        mod = mock.Mock()
        for k, v in kw.items():
            setattr(mod.getActiveWindow, k, v)
        return mod

    def test_non_numeric_geometry_returns_none(self):
        # left/width that can't do arithmetic make the int(...) computation
        # raise → inner except returns None (2299-2300).
        win = mock.Mock(left=object(), top=object(), width=object(),
                        height=object())
        mod = mock.Mock()
        mod.getActiveWindow.return_value = win
        with mock.patch.dict("sys.modules", {"pygetwindow": mod}):
            self.assertIsNone(self.bc._active_window_center())

    def test_get_active_window_raises_returns_none(self):
        # getActiveWindow itself raising trips the outer except (2306-2307).
        mod = mock.Mock()
        mod.getActiveWindow.side_effect = RuntimeError("win32 hiccup")
        with mock.patch.dict("sys.modules", {"pygetwindow": mod}):
            self.assertIsNone(self.bc._active_window_center())


# ──────────────────────────────────────────────────────────────────────────
#  _virtual_screen_bounds — malformed monitor tuple is skipped
# ──────────────────────────────────────────────────────────────────────────
class VirtualScreenBoundsEdgeTests(_MonolithTestBase):
    def test_malformed_monitor_entry_skipped(self):
        # A 2-tuple can't unpack to (mx,my,mw,mh) → that monitor is skipped
        # (2320-2321); the well-formed one still defines the bounds.
        with mock.patch.object(self.bc, "MONITORS",
                               {"bad": (0, 0), "good": (10, 20, 100, 200)}):
            self.assertEqual(self.bc._virtual_screen_bounds(),
                             (10, 20, 100, 200))

    def test_all_malformed_returns_default(self):
        # Every monitor entry is malformed → xs/ys empty → safe default.
        with mock.patch.object(self.bc, "MONITORS", {"bad": (0, 0)}):
            self.assertEqual(self.bc._virtual_screen_bounds(),
                             (0, 0, 2560, 1440))


# ──────────────────────────────────────────────────────────────────────────
#  _launch_reticle_overlay — state-file reset error + Popen failure
# ──────────────────────────────────────────────────────────────────────────
class LaunchReticleEdgeTests(_MonolithTestBase):
    def test_state_reset_failure_swallowed_then_launches(self):
        # open() for the state-file reset raising is caught (2347-2348); launch
        # still proceeds via Popen.
        fake_proc = mock.Mock()
        fake_proc.pid = 5
        state = os.path.join(tempfile.mkdtemp(), "ret.json")
        real_open = open

        def _boom_open(p, *a, **k):
            if p == state:
                raise OSError("state locked")
            return real_open(p, *a, **k)

        with mock.patch.object(self.bc, "RETICLE_OVERLAY_ENABLED", True), \
             mock.patch.object(self.bc, "RETICLE_STATE_FILE", state), \
             mock.patch.object(self.bc.os.path, "exists", return_value=True), \
             mock.patch("builtins.open", _boom_open), \
             mock.patch.object(self.bc.subprocess, "Popen",
                               return_value=fake_proc) as mpop, \
             mock.patch.object(self.bc, "_reticle_process", None):
            self.bc._launch_reticle_overlay()
            mpop.assert_called_once()
        self.bc._reticle_process = None

    def test_popen_failure_swallowed(self):
        # Popen raising is caught (2359-2361) and _reticle_process reset to None.
        state = os.path.join(tempfile.mkdtemp(), "ret.json")
        with mock.patch.object(self.bc, "RETICLE_OVERLAY_ENABLED", True), \
             mock.patch.object(self.bc, "RETICLE_STATE_FILE", state), \
             mock.patch.object(self.bc.os.path, "exists", return_value=True), \
             mock.patch.object(self.bc.subprocess, "Popen",
                               side_effect=OSError("spawn fail")), \
             mock.patch.object(self.bc, "_reticle_process", "sentinel"):
            self.bc._launch_reticle_overlay()  # must not raise
            self.assertIsNone(self.bc._reticle_process)
        self.bc._reticle_process = None


# ──────────────────────────────────────────────────────────────────────────
#  Subprocess shutdown helpers — None-return + terminate-failure arms
# ──────────────────────────────────────────────────────────────────────────
class ShutdownHelperEdgeTests(_MonolithTestBase):
    def test_shutdown_reticle_noop_when_none(self):
        with mock.patch.object(self.bc, "_reticle_process", None):
            self.bc._shutdown_reticle_overlay()  # early return (2367-2368)

    def test_shutdown_reticle_terminate_failure_swallowed(self):
        proc = mock.Mock()
        proc.terminate.side_effect = OSError("already dead")
        with mock.patch.object(self.bc, "_reticle_process", proc):
            self.bc._shutdown_reticle_overlay()  # except 2371-2372
        self.assertIsNone(self.bc._reticle_process)

    def test_shutdown_hud_terminate_failure_swallowed(self):
        # terminate raising hits the except (2384-2385). NOTE: unlike
        # _shutdown_reticle, _shutdown_hud clears _hud_process *inside* the try
        # (line 2383), so a terminate failure leaves it non-None — asserted
        # here as the documented current behaviour, then restored.
        proc = mock.Mock()
        proc.terminate.side_effect = OSError("already dead")
        with mock.patch.object(self.bc, "_hud_process", proc):
            self.bc._shutdown_hud()  # must not raise
            self.assertIs(self.bc._hud_process, proc)  # not cleared on failure
        self.bc._hud_process = None

    def test_shutdown_tray_terminate_failure_swallowed(self):
        proc = mock.Mock()
        proc.terminate.side_effect = OSError("already dead")
        drain = self.bc._tray_drain_stop
        pub = self.bc._tray_publisher_stop
        drain.clear()
        pub.clear()
        self.addCleanup(drain.clear)
        self.addCleanup(pub.clear)
        with mock.patch.object(self.bc, "_tray_process", proc):
            self.bc._shutdown_tray()  # except 2452-2453
        self.assertTrue(drain.is_set())
        self.assertTrue(pub.is_set())
        self.assertIsNone(self.bc._tray_process)


# ──────────────────────────────────────────────────────────────────────────
#  _launch_tray — missing script, stale-cleanup error, Popen failure
# ──────────────────────────────────────────────────────────────────────────
class LaunchTrayEdgeTests(_MonolithTestBase):
    def test_missing_script_skips(self):
        with mock.patch.object(self.bc, "TRAY_ENABLED", True), \
             mock.patch.object(self.bc.os.path, "exists", return_value=False), \
             mock.patch.object(self.bc.subprocess, "Popen") as mpop:
            self.bc._launch_tray()  # early return (2414-2416)
            mpop.assert_not_called()

    def test_stale_cleanup_error_swallowed_then_launches(self):
        # A stale inbox "exists" but os.remove raises → per-file except
        # (2424-2425); launch still proceeds.
        fake_proc = mock.Mock()
        fake_proc.pid = 8
        with mock.patch.object(self.bc, "TRAY_ENABLED", True), \
             mock.patch.object(self.bc.os.path, "exists", return_value=True), \
             mock.patch.object(self.bc.os, "remove",
                               side_effect=OSError("locked")), \
             mock.patch.object(self.bc.subprocess, "Popen",
                               return_value=fake_proc) as mpop, \
             mock.patch.object(self.bc, "_publish_audio_state"), \
             mock.patch.object(self.bc, "_tray_process", None):
            self.bc._launch_tray()  # must not raise
            mpop.assert_called_once()
        self.bc._tray_process = None

    def test_popen_failure_swallowed(self):
        with mock.patch.object(self.bc, "TRAY_ENABLED", True), \
             mock.patch.object(self.bc.os.path, "exists", return_value=False), \
             mock.patch.object(self.bc.os, "remove"), \
             mock.patch.object(self.bc.subprocess, "Popen",
                               side_effect=OSError("spawn fail")), \
             mock.patch.object(self.bc, "_tray_process", "sentinel"):
            # exists=False short-circuits before Popen, so force the spawn path:
            pass
        # Drive the actual Popen-failure arm with a present script.
        with mock.patch.object(self.bc, "TRAY_ENABLED", True), \
             mock.patch.object(self.bc.os.path, "exists", return_value=True), \
             mock.patch.object(self.bc.os, "remove"), \
             mock.patch.object(self.bc.subprocess, "Popen",
                               side_effect=OSError("spawn fail")), \
             mock.patch.object(self.bc, "_tray_process", "sentinel"):
            self.bc._launch_tray()  # except 2437-2439
            self.assertIsNone(self.bc._tray_process)
        self.bc._tray_process = None


# ──────────────────────────────────────────────────────────────────────────
#  _process_inflight — os.remove failures on each delete arm
# ──────────────────────────────────────────────────────────────────────────
class ProcessInflightRemoveEdgeTests(_MonolithTestBase):
    def _write(self, payload_text):
        d = tempfile.mkdtemp()
        path = os.path.join(d, "tray.json.inflight")
        with open(path, "w", encoding="utf-8") as f:
            f.write(payload_text)
        return path

    def test_empty_file_remove_failure_swallowed(self):
        path = self._write("   ")
        with mock.patch.object(self.bc.os, "remove",
                               side_effect=OSError("locked")), \
             mock.patch.object(self.bc, "_dispatch_tray_command") as mdisp:
            self.assertEqual(self.bc._process_inflight(path), 0)  # 2465-2466
            mdisp.assert_not_called()

    def test_corrupt_file_remove_failure_swallowed(self):
        path = self._write("{ broken json")
        with mock.patch.object(self.bc.os, "remove",
                               side_effect=OSError("locked")), \
             mock.patch.object(self.bc, "_dispatch_tray_command") as mdisp:
            self.assertEqual(self.bc._process_inflight(path), 0)  # 2472-2473
            mdisp.assert_not_called()

    def test_non_list_remove_failure_swallowed(self):
        path = self._write(json.dumps({"cmd": "x"}))
        with mock.patch.object(self.bc.os, "remove",
                               side_effect=OSError("locked")), \
             mock.patch.object(self.bc, "_dispatch_tray_command") as mdisp:
            self.assertEqual(self.bc._process_inflight(path), 0)  # 2476-2477
            mdisp.assert_not_called()

    def test_read_failure_remove_failure_swallowed(self):
        path = self._write(json.dumps([{"cmd": "a"}]))
        real_open = open

        def _boom_open(p, *a, **k):
            if p == path:
                raise OSError("read denied")
            return real_open(p, *a, **k)

        with mock.patch("builtins.open", _boom_open), \
             mock.patch.object(self.bc.os, "remove",
                               side_effect=OSError("locked")), \
             mock.patch.object(self.bc, "_dispatch_tray_command") as mdisp:
            self.assertEqual(self.bc._process_inflight(path), 0)  # 2481-2482
            mdisp.assert_not_called()

    def test_predispatch_remove_failure_still_dispatches(self):
        # The pre-dispatch claim-file remove failing (2488-2489) must not stop
        # the commands from running.
        path = self._write(json.dumps([{"cmd": "a"}, {"cmd": "b"}]))
        with mock.patch.object(self.bc.os, "remove",
                               side_effect=OSError("locked")), \
             mock.patch.object(self.bc, "_dispatch_tray_command") as mdisp:
            self.assertEqual(self.bc._process_inflight(path), 2)
            self.assertEqual(mdisp.call_count, 2)


# ──────────────────────────────────────────────────────────────────────────
#  _drain_tray_commands_once — claim (os.replace) failure
# ──────────────────────────────────────────────────────────────────────────
class DrainClaimFailureTests(_MonolithTestBase):
    def test_claim_replace_failure_returns_orphan_count(self):
        # os.replace claim failing is caught (2536-2538); the function returns
        # whatever the orphan pass produced (0 here) without raising.
        d = tempfile.mkdtemp()
        cmd_file = os.path.join(d, "tray_commands.json")
        with open(cmd_file, "w", encoding="utf-8") as f:
            json.dump([{"cmd": "x"}], f)
        with mock.patch.object(self.bc, "TRAY_COMMANDS_FILE", cmd_file), \
             mock.patch.object(self.bc.os, "replace",
                               side_effect=OSError("claim failed")), \
             mock.patch.object(self.bc, "_dispatch_tray_command") as mdisp:
            self.assertEqual(self.bc._drain_tray_commands_once(), 0)
            mdisp.assert_not_called()


# ──────────────────────────────────────────────────────────────────────────
#  _dispatch_tray_command — remaining error/branch arms
# ──────────────────────────────────────────────────────────────────────────
class DispatchTrayCommandEdgeTests(_MonolithTestBase):
    def test_force_wake_wake_date_failure_swallowed(self):
        # Making _last_wake_date non-subscriptable forces the wake-date stamp to
        # raise → caught (2569-2570); the rest of force_wake still runs.
        sleep = self._restore_attr_after("_sleep_mode")
        standby = self._restore_attr_after("_standby_mode")
        sleep[0] = True
        standby[0] = True
        with mock.patch.object(self.bc, "_write_hud_state"), \
             mock.patch.object(self.bc, "_speak"), \
             mock.patch.object(self.bc, "_last_wake_date", object()), \
             mock.patch.object(self.bc.os.path, "exists", return_value=False):
            self.bc._dispatch_tray_command("force_wake", {})  # must not raise
        self.assertFalse(sleep[0])
        self.assertFalse(standby[0])

    def test_force_wake_overnight_remove_failure_swallowed(self):
        # os.remove raising during overnight-flag cleanup is caught (2577-2578).
        sleep = self._restore_attr_after("_sleep_mode")
        standby = self._restore_attr_after("_standby_mode")
        sleep[0] = True
        standby[0] = True
        with mock.patch.object(self.bc, "_write_hud_state"), \
             mock.patch.object(self.bc, "_speak"), \
             mock.patch.object(self.bc.os.path, "exists", return_value=True), \
             mock.patch.object(self.bc.os, "remove",
                               side_effect=OSError("locked")):
            self.bc._dispatch_tray_command("force_wake", {})  # must not raise
        self.assertFalse(sleep[0])

    def test_force_wake_speak_failure_swallowed(self):
        # _speak raising is caught by the 2580-2581 guard.
        sleep = self._restore_attr_after("_sleep_mode")
        standby = self._restore_attr_after("_standby_mode")
        sleep[0] = True
        standby[0] = True
        with mock.patch.object(self.bc, "_write_hud_state"), \
             mock.patch.object(self.bc, "_speak",
                               side_effect=RuntimeError("tts dead")), \
             mock.patch.object(self.bc.os.path, "exists", return_value=False):
            self.bc._dispatch_tray_command("force_wake", {})  # must not raise
        self.assertFalse(sleep[0])

    def test_open_hud_enables_flag_when_disabled(self):
        # HUD_ENABLED False → the `if not HUD_ENABLED: HUD_ENABLED = True` body
        # (2590-2591) runs before relaunch. mock.patch.object restores the flag.
        with mock.patch.object(self.bc, "HUD_ENABLED", False), \
             mock.patch.object(self.bc, "_shutdown_hud"), \
             mock.patch.object(self.bc, "_launch_hud"):
            self.bc._dispatch_tray_command("open_hud", {})
            self.assertTrue(self.bc.HUD_ENABLED)

    def test_restart_action_failure_swallowed(self):
        with mock.patch.object(self.bc, "_act_restart",
                               side_effect=RuntimeError("restart boom")):
            self.bc._dispatch_tray_command("restart", {})  # except 2597-2598

    def test_trigger_overnight_action_failure_swallowed(self):
        with mock.patch.object(self.bc, "_act_start_overnight_upgrade",
                               side_effect=RuntimeError("overnight boom")):
            self.bc._dispatch_tray_command("trigger_overnight", {})  # 2602-2603

    def test_ambient_toggle_action_raises_swallowed(self):
        # The registered ambient action raising is caught (2636-2637).
        active = self._restore_attr_after("_ambient_mode_active")
        active[0] = False
        boom = mock.Mock(side_effect=RuntimeError("ambient boom"))
        with mock.patch.object(self.bc, "ACTIONS",
                               {"ambient_listen_start": boom}), \
             mock.patch.object(self.bc, "_write_hud_state"):
            self.bc._dispatch_tray_command("ambient_mode_toggle", {})
        self.assertTrue(active[0])
        boom.assert_called_once_with("")

    def test_ambient_toggle_action_not_registered(self):
        # No matching ACTIONS entry → the else-print branch (2638-2639).
        active = self._restore_attr_after("_ambient_mode_active")
        active[0] = False
        with mock.patch.object(self.bc, "ACTIONS", {}), \
             mock.patch.object(self.bc, "_write_hud_state"):
            self.bc._dispatch_tray_command("ambient_mode_toggle", {})
        self.assertTrue(active[0])

    def test_pause_daemons_resume_branch_and_failure(self):
        # Start paused=True so the toggle flips to False → the `else: resume`
        # branch (2652-2653) runs; resume_diagnostics raising hits 2654-2655.
        paused = self._restore_attr_after("_daemons_paused")
        paused[0] = True
        fake_diag = mock.Mock()
        fake_diag.resume_diagnostics.side_effect = RuntimeError("diag boom")
        import core
        # The monolith does ``from core import diagnostic_daemons``, which
        # resolves via the ``core`` package attribute once any sibling test has
        # imported the real ``core.diagnostic_daemons`` — so the sys.modules
        # fake alone is bypassed (passes alone, FAILS in the full suite). Patch
        # BOTH sys.modules and the parent-package attr.
        with mock.patch.object(self.bc, "_write_hud_state"), \
             mock.patch.dict("sys.modules",
                             {"core.diagnostic_daemons": fake_diag}), \
             mock.patch.object(core, "diagnostic_daemons", fake_diag,
                               create=True):
            self.bc._dispatch_tray_command("pause_daemons_toggle", {})
        self.assertFalse(paused[0])
        fake_diag.resume_diagnostics.assert_called_once()

    def test_pause_daemons_ambient_set_paused_failure(self):
        # skill_ambient_listen.set_paused raising is caught (2660-2661).
        paused = self._restore_attr_after("_daemons_paused")
        paused[0] = False
        fake_diag = mock.Mock()
        fake_al = mock.Mock()
        fake_al.set_paused.side_effect = RuntimeError("al boom")
        import core
        # ``from core import diagnostic_daemons`` resolves via the ``core``
        # package attr once a sibling imports the real module, so patch BOTH
        # sys.modules and the parent-package attr (see sibling test). The
        # skill_ambient_listen fake is a flat module — sys.modules alone is
        # correct for it.
        with mock.patch.object(self.bc, "_write_hud_state"), \
             mock.patch.dict("sys.modules",
                             {"core.diagnostic_daemons": fake_diag,
                              "skill_ambient_listen": fake_al}), \
             mock.patch.object(core, "diagnostic_daemons", fake_diag,
                               create=True):
            self.bc._dispatch_tray_command("pause_daemons_toggle", {})
        self.assertTrue(paused[0])
        fake_al.set_paused.assert_called_once_with(True)

    def test_generic_long_result_is_truncated(self):
        # A registered action returning a >120-char string exercises the
        # truncation branch (2698-2699).
        long_line = "Z" * 200
        fn = mock.Mock(return_value=long_line)
        with mock.patch.object(self.bc, "ACTIONS", {"verbose": fn}), \
             mock.patch.object(self.bc, "_HEAVY_ACTIONS", frozenset()):
            self.bc._dispatch_tray_command("verbose", {"arg": ""})
        fn.assert_called_once_with("")

    def test_generic_nonstring_result_dispatched_branch(self):
        # A non-string result yields head == "" → the `else: dispatched`
        # branch (2701-2702).
        fn = mock.Mock(return_value=12345)   # not a str
        with mock.patch.object(self.bc, "ACTIONS", {"numbery": fn}), \
             mock.patch.object(self.bc, "_HEAVY_ACTIONS", frozenset()):
            self.bc._dispatch_tray_command("numbery", {"arg": ""})
        fn.assert_called_once_with("")


if __name__ == "__main__":
    unittest.main()
