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
        with mock.patch.object(self.bc, "_write_hud_state"), \
             mock.patch.dict("sys.modules",
                             {"core.diagnostic_daemons": fake_diag,
                              "skill_ambient_listen": fake_al}):
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


if __name__ == "__main__":
    unittest.main()
