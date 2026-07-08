"""Logic tests for skills/notification_triage.py.

Covers the tunable rules engine: schema validation, priority-ordered rule
selection, regex matching (incl. focus-mode-bypass priority), the family of
content/timestamp/announce dedup keys, the JARVIS-voice speech formatter, the
LLM-verdict→action mapping, and the registered voice actions
(triage_status / add / remove / recent / pause / resume).

It also exercises the heavy listener-side logic offline: the full
_handle_notification triage pipeline (every dedup gate + every action branch),
the four on-disk dedup caches (load/save/prune), rule persistence, the Haiku /
Ollama classifier fallbacks, the winsdk import probe, and the async-op +
poll-loop wrappers.

No monolith boot, no winsdk, no real listener thread (neutered by the harness).
Disk writes (rules file, dedup caches, jsonl log) are redirected to a temp dir
in setUp and reset in tearDown so nothing real is touched and module-global
caches never leak between tests.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


# A realistic Unix epoch base. The content-dedup gate compares
# ``now_ts - prev_ts < TTL`` WITHOUT a ``prev_ts > 0`` guard, so a tiny mocked
# clock (e.g. 2000.0) makes a never-seen key look "within TTL" and wrongly
# suppresses the first toast. Real timestamps are ~1.7e9 ≫ any TTL, so the
# bug never bites in production; tests mock time around this base to match.
_T0 = 1_700_000_000.0


# ─────────────────────────────────────────────────────────────────────────
# Fakes for the WinRT UserNotification object graph the listener walks.
# ─────────────────────────────────────────────────────────────────────────
class _FakeTextElement:
    def __init__(self, text):
        self.text = text


class _FakeBinding:
    def __init__(self, texts):
        self._texts = [_FakeTextElement(t) for t in texts]

    def get_text_elements(self):
        return self._texts


class _FakeVisual:
    def __init__(self, bindings):
        self.bindings = bindings


class _FakeToast:
    def __init__(self, visual):
        self.visual = visual


class _FakeDisplayInfo:
    def __init__(self, display_name):
        self.display_name = display_name


class _FakeAppInfo:
    def __init__(self, display_name):
        self.display_info = _FakeDisplayInfo(display_name)


class _FakeNotification:
    """Mimics a WinRT UserNotification: ``.id``, ``.app_info`` and the nested
    ``.notification.visual.bindings[*].get_text_elements()`` text graph."""

    def __init__(self, nid=1, app="Microsoft Teams", texts=("Sam", "are you around"),
                 *, app_info_missing=False, toast_raises=False):
        self.id = nid
        self.app_info = None if app_info_missing else _FakeAppInfo(app)
        self._toast_raises = toast_raises
        self._texts = list(texts)

    @property
    def notification(self):
        if self._toast_raises:
            raise RuntimeError("visual marshalling blew up")
        return _FakeToast(_FakeVisual([_FakeBinding(self._texts)]))


def _make_notification(**kw):
    return _FakeNotification(**kw)


# ─────────────────────────────────────────────────────────────────────────
# Async-op stand-ins for _request_access / _get_notifications.
# ─────────────────────────────────────────────────────────────────────────
class _AsyncOpGet:
    """IAsyncOperation whose .get() returns a value (the happy winsdk path)."""

    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value


class _AsyncOpManualPoll:
    """IAsyncOperation whose .get() raises, forcing the manual completed/
    get_results() fallback path in the wrappers. ``completed`` flips to True
    after ``pending`` checks so the ``while not op.completed: sleep`` body runs
    at least once before resolving."""

    def __init__(self, value, pending=0):
        self._value = value
        self._pending = pending

    @property
    def completed(self):
        if self._pending > 0:
            self._pending -= 1
            return False
        return True

    def get(self):
        raise RuntimeError("no .get on this awaitable")

    def get_results(self):
        return self._value


class _AccessResult:
    def __init__(self, name):
        self.name = name


class NotificationTriageTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("notification_triage")
        # Start every test from a known, small rule set so ordering assertions
        # are deterministic regardless of what register()'s _load_rules read.
        with self.mod._state_lock:
            self.mod._rules.clear()
            self.mod._recent.clear()
            self.mod._snooze.clear()
        self.mod._pause_flag[0] = False

    # ── _validate_rule ───────────────────────────────────────────────────
    def test_validate_rule_accepts_minimal(self):
        self.assertTrue(self.mod._validate_rule({"id": "x", "action": "drop"}))

    def test_validate_rule_rejects_bad_action(self):
        self.assertFalse(self.mod._validate_rule({"id": "x", "action": "explode"}))

    def test_validate_rule_rejects_missing_id(self):
        self.assertFalse(self.mod._validate_rule({"action": "drop"}))

    def test_validate_rule_rejects_non_dict_match(self):
        self.assertFalse(
            self.mod._validate_rule({"id": "x", "action": "log", "match": "nope"}))

    def test_validate_rule_rejects_non_dict_rule(self):
        self.assertFalse(self.mod._validate_rule("not a dict"))
        self.assertFalse(self.mod._validate_rule(None))

    # ── _sort_rules / _rule_matches ──────────────────────────────────────
    def test_sort_rules_priority_descending(self):
        ordered = self.mod._sort_rules(
            [{"id": "lo", "priority": 10}, {"id": "hi", "priority": 50},
             {"id": "mid", "priority": 30}])
        self.assertEqual([r["id"] for r in ordered], ["hi", "mid", "lo"])

    def test_rule_matches_teams_dm_excludes_channel(self):
        rule = {"match": {"app_pattern": r"(?i)teams",
                          "title_pattern": r"(?i)^(?!.*\bchannel\b).+"},
                "action": "read_aloud"}
        self.assertTrue(self.mod._rule_matches(rule, "Microsoft Teams", "Sam", "hi"))
        self.assertFalse(
            self.mod._rule_matches(rule, "Microsoft Teams", "channel ping", "hi"))

    def test_rule_matches_empty_match_is_catchall(self):
        self.assertTrue(self.mod._rule_matches({"action": "log"}, "a", "b", "c"))

    def test_rule_matches_body_pattern(self):
        rule = {"match": {"body_pattern": r"(?i)unsubscribe"}, "action": "drop"}
        self.assertTrue(self.mod._rule_matches(rule, "App", "Sale", "click unsubscribe here"))
        self.assertFalse(self.mod._rule_matches(rule, "App", "Sale", "real message"))

    def test_rule_matches_none_value_uses_empty_string(self):
        # A None body must be coerced to "" — pattern simply doesn't match.
        rule = {"match": {"body_pattern": r"x"}, "action": "log"}
        self.assertFalse(self.mod._rule_matches(rule, "App", "T", None))

    def test_rule_matches_bad_regex_returns_false(self):
        # An invalid pattern must not crash — it disqualifies the rule.
        self.assertFalse(
            self.mod._rule_matches({"match": {"app_pattern": "("}, "action": "log"},
                                   "anything", "t", "b"))

    # ── _select_action (priority order) ──────────────────────────────────
    def test_select_action_highest_priority_wins(self):
        self.mod._rules = self.mod._sort_rules([
            {"id": "lo", "match": {"app_pattern": "slack"}, "action": "log",
             "priority": 1},
            {"id": "hi", "match": {"app_pattern": "slack"}, "action": "drop",
             "priority": 9},
        ])
        action, rule = self.mod._select_action("slack", "t", "b")
        self.assertEqual(action, "drop")
        self.assertEqual(rule["id"], "hi")

    def test_select_action_no_match_defers_to_classify(self):
        self.mod._rules = [{"id": "x", "match": {"app_pattern": "teams"},
                            "action": "read_aloud"}]
        action, rule = self.mod._select_action("steam", "t", "b")
        self.assertEqual(action, "classify")
        self.assertIsNone(rule)

    def test_select_action_rule_without_action_defaults_classify(self):
        # A matching rule with no explicit action falls back to classify.
        self.mod._rules = [{"id": "noact", "match": {}}]
        action, rule = self.mod._select_action("a", "t", "b")
        self.assertEqual(action, "classify")
        self.assertEqual(rule["id"], "noact")

    def test_default_rules_cover_build_failure_high_priority(self):
        # The shipped defaults must keep build/CI failures above the
        # focus-mode bypass floor so "the build failed" always cuts through.
        build_rule = next(r for r in self.mod.DEFAULT_RULES
                          if r["id"] == "build_or_ci_failure")
        self.assertEqual(build_rule["action"], "read_aloud")
        self.assertGreaterEqual(build_rule["priority"], self.mod.HIGH_PRIORITY_FLOOR)

    # ── LLM verdict mapping ──────────────────────────────────────────────
    def test_llm_verdict_to_action_map(self):
        m = self.mod.LLM_VERDICTS_TO_ACTION
        self.assertEqual(m["urgent"], "read_aloud")
        self.assertEqual(m["spam"], "drop")
        self.assertEqual(m["newsletter"], "log")
        self.assertEqual(m["fyi"], "log")

    # ── dedup keys ───────────────────────────────────────────────────────
    def test_content_dedupe_key_case_insensitive_and_stable(self):
        a = self.mod._content_dedupe_key("Teams", "Sam Smith", "are you around")
        b = self.mod._content_dedupe_key("teams", "Sam Smith", "Are You Around")
        self.assertEqual(a, b)

    def test_content_dedupe_key_differs_on_body(self):
        a = self.mod._content_dedupe_key("Teams", "Sam", "message one")
        b = self.mod._content_dedupe_key("Teams", "Sam", "message two")
        self.assertNotEqual(a, b)

    def test_content_dedupe_key_empty_title_blank_sender(self):
        # No title → sender component is empty, but key is still produced.
        k = self.mod._content_dedupe_key("Teams", "", "body only")
        self.assertTrue(k.startswith("teams||"))

    def test_timestamp_bucket_groups_within_minute(self):
        # bucket = int(ts // 60): 120-179 share bucket 2; 180 rolls to bucket 3.
        self.assertEqual(self.mod._timestamp_bucket(120.0),
                         self.mod._timestamp_bucket(179.0))
        self.assertNotEqual(self.mod._timestamp_bucket(120.0),
                            self.mod._timestamp_bucket(180.0))

    def test_body_snippet_falls_back_to_title(self):
        # Empty body must not collapse distinct senders onto the empty hash.
        self.assertEqual(self.mod._body_snippet("", "Standup reminder"),
                         "standup reminder")

    def test_body_snippet_truncates_to_limit(self):
        snip = self.mod._body_snippet("y" * 500)
        self.assertEqual(len(snip), self.mod._NOTIF_TS_BODY_SNIPPET_CH)

    def test_notification_timestamp_key_varies_by_bucket(self):
        a = self.mod._notification_timestamp_dedupe_key("Sam hi", "body", 1)
        b = self.mod._notification_timestamp_dedupe_key("Sam hi", "body", 2)
        self.assertNotEqual(a, b)

    def test_announce_dedup_key_normalizes(self):
        a = self.mod._announce_dedup_key("Teams", "Sam", "Hello There")
        b = self.mod._announce_dedup_key("teams", "sam", "hello   there")
        self.assertEqual(a, b)
        self.assertEqual(len(a), 40)

    def test_announce_sha256_key_coerces_window_id(self):
        # window_id may be int / str / None — all coerced to str for hashing.
        none_key = self.mod._announce_sha256_dedupe_key("Sam", "hi", None)
        zero_key = self.mod._announce_sha256_dedupe_key("Sam", "hi", "")
        self.assertEqual(none_key, zero_key)   # None → "" same as "".

    def test_announce_sha256_key_distinct_from_sha1(self):
        sha1 = self.mod._announce_dedup_key("Teams", "Sam", "hi")
        sha256 = self.mod._announce_sha256_dedupe_key("Sam", "hi", 42)
        self.assertEqual(len(sha1), 40)      # SHA-1 hex
        self.assertEqual(len(sha256), 64)    # SHA-256 hex
        self.assertNotEqual(sha1, sha256)

    # ── _format_for_speech ───────────────────────────────────────────────
    def test_format_for_speech_includes_app_and_text(self):
        out = self.mod._format_for_speech("Teams", "Sam", "are you around")
        self.assertIn("Teams", out)
        self.assertIn("Sam", out)
        self.assertIn("are you around", out)
        self.assertTrue(out.startswith("Sir,"))

    def test_format_for_speech_truncates_long_body(self):
        long_body = "x" * 400
        out = self.mod._format_for_speech("App", "Title", long_body)
        # MAX_SPOKEN_BODY_CHARS caps the body; the whole line stays bounded.
        self.assertLessEqual(len(out), self.mod.MAX_SPOKEN_BODY_CHARS + 60)
        self.assertIn("…", out)

    def test_format_for_speech_empty_is_graceful(self):
        out = self.mod._format_for_speech("Slack", "", "")
        self.assertIn("Slack", out)
        self.assertIn("notification", out.lower())

    def test_format_for_speech_empty_no_app(self):
        # No app, no title/body → the "your system" fallback fires.
        out = self.mod._format_for_speech("", "", "")
        self.assertIn("your system", out)

    def test_format_for_speech_no_app_head_with_text(self):
        # Empty app but real text → the head-less "Sir, notification:" branch.
        out = self.mod._format_for_speech("", "Title", "")
        self.assertTrue(out.startswith("Sir, notification:"))
        self.assertIn("Title", out)

    def test_format_for_speech_body_same_as_title_not_duplicated(self):
        out = self.mod._format_for_speech("App", "same", "same")
        # Body identical to title is dropped → "same" appears once.
        self.assertEqual(out.count("same"), 1)

    # ── recent_notifications helper ──────────────────────────────────────
    def test_recent_notifications_returns_tail(self):
        with self.mod._state_lock:
            for i in range(5):
                self.mod._recent.append({"app": f"a{i}", "title": "t", "ts": i})
        out = self.mod.recent_notifications(2)
        self.assertEqual([r["app"] for r in out], ["a3", "a4"])

    def test_recent_notifications_floor_of_one(self):
        with self.mod._state_lock:
            self.mod._recent.append({"app": "only", "ts": 0})
        # n<=0 is floored to 1 (max(1, int(n))).
        out = self.mod.recent_notifications(0)
        self.assertEqual(len(out), 1)

    # ── registered actions ───────────────────────────────────────────────
    def test_add_rule_requires_json(self):
        self.assertIn("json", self.actions["add_notification_rule"]("").lower())

    def test_add_rule_rejects_bad_json(self):
        out = self.actions["add_notification_rule"]("{not valid")
        self.assertIn("could not parse", out.lower())

    def test_add_rule_rejects_invalid_schema(self):
        out = self.actions["add_notification_rule"](
            json.dumps({"id": "x", "action": "boom"}))
        self.assertIn("missing required", out.lower())

    def test_add_rule_then_remove(self):
        rule = {"id": "slack_general",
                "match": {"app_pattern": "(?i)slack"},
                "action": "drop", "priority": 25}
        # Patch the persistence so nothing is written to the real rules file.
        with mock.patch.object(self.mod, "_save_rules"):
            added = self.actions["add_notification_rule"](json.dumps(rule))
            self.assertIn("slack_general", added)
            self.assertTrue(any(r["id"] == "slack_general" for r in self.mod._rules))
            removed = self.actions["remove_notification_rule"]("slack_general")
            self.assertIn("removed", removed.lower())
            self.assertFalse(any(r["id"] == "slack_general"
                                 for r in self.mod._rules))

    def test_add_rule_replaces_existing_by_id(self):
        with mock.patch.object(self.mod, "_save_rules"):
            self.actions["add_notification_rule"](
                json.dumps({"id": "dup", "action": "log"}))
            self.actions["add_notification_rule"](
                json.dumps({"id": "dup", "action": "drop"}))
        dups = [r for r in self.mod._rules if r["id"] == "dup"]
        self.assertEqual(len(dups), 1)
        self.assertEqual(dups[0]["action"], "drop")

    def test_remove_unknown_rule(self):
        self.assertIn("no rule", self.actions["remove_notification_rule"]("ghost").lower())

    def test_remove_rule_requires_id(self):
        self.assertIn("which rule", self.actions["remove_notification_rule"]("").lower())

    def test_list_rules_empty(self):
        self.assertIn("no notification rules",
                      self.actions["list_notification_rules"]("").lower())

    def test_list_rules_renders_entries(self):
        self.mod._rules = [{"id": "teams_dm",
                            "match": {"app_pattern": "(?i)teams"},
                            "action": "read_aloud", "priority": 50}]
        out = self.actions["list_notification_rules"]("")
        self.assertIn("teams_dm", out)
        self.assertIn("read_aloud", out)

    def test_list_rules_truncates_over_25(self):
        self.mod._rules = [{"id": f"r{i}", "match": {}, "action": "log",
                            "priority": i} for i in range(30)]
        out = self.actions["list_notification_rules"]("")
        self.assertIn("and 5 more", out)

    def test_recent_summary_empty(self):
        self.assertIn("nothing in the notification log",
                      self.actions["recent_notifications_summary"]("").lower())

    def test_recent_summary_renders_records(self):
        with self.mod._state_lock:
            self.mod._recent.append(
                {"app": "Teams", "title": "Sam pinged", "action": "read_aloud",
                 "ts": time.time()})
        out = self.actions["recent_notifications_summary"]("5")
        self.assertIn("Teams", out)
        self.assertIn("Sam pinged", out)
        self.assertIn("read_aloud", out)

    def test_recent_summary_clamps_and_handles_bad_arg(self):
        # Non-numeric arg defaults to 5; record with no title shows "(no title)".
        with self.mod._state_lock:
            self.mod._recent.append({"app": "", "title": "", "action": "log",
                                     "ts": time.time()})
        out = self.actions["recent_notifications_summary"]("not a number")
        self.assertIn("(no title)", out)

    def test_pause_and_resume_toggle_flag(self):
        self.assertIn("paused", self.actions["pause_notification_triage"]("").lower())
        self.assertTrue(self.mod._pause_flag[0])
        # resume tries to (re)start the listener thread; harness neutered start.
        self.assertIn("resumed", self.actions["resume_notification_triage"]("").lower())
        self.assertFalse(self.mod._pause_flag[0])

    def test_triage_status_mentions_rule_count(self):
        self.mod._rules = [{"id": "a", "action": "log"},
                           {"id": "b", "action": "drop"}]
        out = self.actions["triage_status"]("")
        self.assertIn("2 rules", out)
        self.assertTrue(out.endswith("sir."))

    # ── _format_rule_for_voice ───────────────────────────────────────────
    def test_format_rule_for_voice_all_patterns(self):
        r = {"id": "full", "priority": 7, "action": "drop",
             "match": {"app_pattern": "slack", "title_pattern": "#general",
                       "body_pattern": "noise"}}
        out = self.mod._format_rule_for_voice(r)
        self.assertIn("full", out)
        self.assertIn("p=7", out)
        self.assertIn("app~slack", out)
        self.assertIn("title~#general", out)
        self.assertIn("body~noise", out)

    def test_format_rule_for_voice_no_match_uses_star(self):
        out = self.mod._format_rule_for_voice({"id": "any", "action": "log"})
        self.assertIn("when *", out)


# ─────────────────────────────────────────────────────────────────────────
# Isolation base: redirect every on-disk path to a temp dir and reset all
# module-global caches before AND after each test, so the heavy save-on-every-
# change pipeline never touches the real project tree or leaks across tests.
# ─────────────────────────────────────────────────────────────────────────
class _IsolatedTriageBase(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("notification_triage")
        self.tmp = tempfile.mkdtemp(prefix="notif_triage_test_")
        self.addCleanup(self._cleanup_tmp)

        # Redirect every file the module writes.
        self._patch_attr("_DEDUPE_FILE", os.path.join(self.tmp, "dedup.json"))
        self._patch_attr("_CONTENT_DEDUPE_FILE",
                         os.path.join(self.tmp, "content.json"))
        self._patch_attr("_NOTIF_TS_DEDUPE_FILE", os.path.join(self.tmp, "ts.json"))
        self._patch_attr("_ANNOUNCE_DEDUPE_FILE",
                         os.path.join(self.tmp, "announce.json"))
        self._patch_attr("_ANNOUNCE_SHA256_DEDUPE_FILE",
                         os.path.join(self.tmp, "announce256.json"))
        self._patch_attr("_RULES_FILE", os.path.join(self.tmp, "rules.json"))
        self._patch_attr("_DATA_DIR", os.path.join(self.tmp, "notifications"))

        # Snapshot + clear every mutable global cache.
        self._reset_state()
        self.addCleanup(self._reset_state)

    def _patch_attr(self, name, value):
        p = mock.patch.object(self.mod, name, value)
        p.start()
        self.addCleanup(p.stop)

    def _reset_state(self):
        with self.mod._state_lock:
            self.mod._rules.clear()
            self.mod._recent.clear()
            self.mod._snooze.clear()
            self.mod._seen_ids.clear()
            self.mod._content_dedupe.clear()
            self.mod._notification_timestamp_dedup.clear()
            self.mod._announce_dedup.clear()
            self.mod._announce_sha256_dedup.clear()
        self.mod._pause_flag[0] = False
        self.mod._last_dedupe_save[0] = 0.0

    def _cleanup_tmp(self):
        for root, _dirs, files in os.walk(self.tmp, topdown=False):
            for fn in files:
                try:
                    os.unlink(os.path.join(root, fn))
                except OSError:
                    pass
            try:
                os.rmdir(root)
            except OSError:
                pass


# ─────────────────────────────────────────────────────────────────────────
# _handle_notification — the core triage pipeline.
# ─────────────────────────────────────────────────────────────────────────
class HandleNotificationTests(_IsolatedTriageBase):
    def _drop_all_rule(self):
        return [{"id": "drop_all", "match": {}, "action": "drop", "priority": 0}]

    def test_read_aloud_end_to_end_announces(self):
        self.mod._rules = [{"id": "teams", "match": {"app_pattern": "(?i)teams"},
                            "action": "read_aloud", "priority": 50}]
        with mock.patch.object(self.mod, "_proactive_announce") as ann, \
             mock.patch.object(self.mod, "_focus_mode_active", return_value=False):
            self.mod._handle_notification(
                _make_notification(nid=11, app="Microsoft Teams",
                                   texts=("Sam", "are you around")))
        ann.assert_called_once()
        msg = ann.call_args[0][0]
        self.assertIn("Sam", msg)
        # Record landed in the ring buffer with the resolved action.
        self.assertEqual(self.mod._recent[-1]["action"], "read_aloud")
        self.assertEqual(self.mod._recent[-1]["rule_id"], "teams")

    def test_seen_id_short_circuits(self):
        self.mod._rules = self._drop_all_rule()
        n = _make_notification(nid=77, texts=("A", "b"))
        with mock.patch.object(self.mod, "_proactive_announce"):
            self.mod._handle_notification(n)
        before = len(self.mod._recent)
        # Same id again → returns immediately, nothing appended.
        self.mod._handle_notification(_make_notification(nid=77, texts=("A", "b")))
        self.assertEqual(len(self.mod._recent), before)

    def test_seen_id_set_pruned_when_huge(self):
        # Pre-seed >4000 ids so the prune branch (drop oldest 1000) runs.
        with self.mod._state_lock:
            self.mod._seen_ids.update(range(100000, 104100))
        self.mod._rules = self._drop_all_rule()
        self.mod._handle_notification(_make_notification(nid=5, texts=("x", "y")))
        self.assertLessEqual(len(self.mod._seen_ids), 4000 + 1)

    def test_snooze_dict_pruned_of_stale_stamps(self):
        # _snooze must not grow unbounded: stamps older than SNOOZE_SECONDS can
        # never suppress again, so _prune_snooze evicts them. Seed an old stamp +
        # a fresh one, prune at 'now', and confirm only the fresh one survives.
        now = 1_000_000.0
        with self.mod._state_lock:
            self.mod._snooze.clear()
            self.mod._snooze["app|old"] = now - self.mod.SNOOZE_SECONDS - 1  # expired
            self.mod._snooze["app|fresh"] = now - 1                          # still live
        self.mod._prune_snooze(now)
        self.assertNotIn("app|old", self.mod._snooze)
        self.assertIn("app|fresh", self.mod._snooze)

    def test_drop_action_not_logged(self):
        self.mod._rules = self._drop_all_rule()
        with mock.patch.object(self.mod, "_persist_to_log") as plog:
            self.mod._handle_notification(_make_notification(nid=3, texts=("Spam", "buy")))
        plog.assert_not_called()
        self.assertEqual(self.mod._recent[-1]["action"], "drop")

    def test_log_action_persists_but_silent(self):
        self.mod._rules = [{"id": "logger", "match": {}, "action": "log",
                            "priority": 0}]
        with mock.patch.object(self.mod, "_persist_to_log") as plog, \
             mock.patch.object(self.mod, "_proactive_announce") as ann:
            self.mod._handle_notification(_make_notification(nid=4, texts=("Hi", "x")))
        plog.assert_called_once()
        ann.assert_not_called()

    def test_timestamp_dedupe_suppresses_rapid_repeat(self):
        self.mod._rules = [{"id": "logger", "match": {}, "action": "log"}]
        with mock.patch.object(self.mod.time, "time", return_value=_T0), \
             mock.patch.object(self.mod._time, "time", return_value=_T0), \
             mock.patch.object(self.mod, "_persist_to_log"):
            # Two different ids, identical (sender, body) inside the same bucket.
            self.mod._handle_notification(_make_notification(nid=1, texts=("Sam", "ping")))
            n_before = len(self.mod._recent)
            self.mod._handle_notification(_make_notification(nid=2, texts=("Sam", "ping")))
        # Second one suppressed by the timestamp-bucket cache.
        self.assertEqual(len(self.mod._recent), n_before)

    def test_content_dedupe_suppresses_across_buckets(self):
        self.mod._rules = [{"id": "logger", "match": {}, "action": "log"}]
        with mock.patch.object(self.mod, "_persist_to_log"):
            t0 = _T0
            with mock.patch.object(self.mod._time, "time", return_value=t0), \
                 mock.patch.object(self.mod.time, "time", return_value=t0):
                self.mod._handle_notification(
                    _make_notification(nid=1, texts=("Sam", "dupbody")))
            n_before = len(self.mod._recent)
            # Far enough ahead to clear the 10-min timestamp window but well
            # inside the 4-hour content window → content cache suppresses.
            t1 = t0 + (20 * 60)
            with mock.patch.object(self.mod._time, "time", return_value=t1), \
                 mock.patch.object(self.mod.time, "time", return_value=t1):
                self.mod._handle_notification(
                    _make_notification(nid=2, texts=("Sam", "dupbody")))
        self.assertEqual(len(self.mod._recent), n_before)

    def test_content_dedupe_read_write_atomic_single_lock_hold(self):
        # Regression: the content-dedupe get + set must happen within ONE
        # _state_lock hold. The old split (read under lock, RELEASE, test,
        # re-acquire to write) let two listener threads both observe a stale
        # dedupe and double-announce the same backlog toast. Instrument the lock
        # + the dedupe dict and assert the lock is never fully released between
        # the dedupe read and the dedupe write.
        import threading as _th
        self.mod._rules = [{"id": "logger", "match": {}, "action": "log"}]
        backing = _th.RLock()
        depth = {"n": 0}
        state = {"watching": False, "released_between": False}

        class _DepthLock:
            def __enter__(self_):
                self_.acquire()
                return self_

            def __exit__(self_, *a):
                self_.release()
                return False

            def acquire(self_, *a, **k):
                r = backing.acquire(*a, **k)
                depth["n"] += 1
                return r

            def release(self_):
                depth["n"] -= 1
                if state["watching"] and depth["n"] == 0:
                    state["released_between"] = True
                backing.release()

        class _SpyDedupe(dict):
            def get(self_, k, d=None):
                state["watching"] = True          # window opens at the read
                return dict.get(self_, k, d)

            def __setitem__(self_, k, v):
                state["watching"] = False          # window closes at the write
                dict.__setitem__(self_, k, v)

        with mock.patch.object(self.mod, "_state_lock", _DepthLock()), \
             mock.patch.object(self.mod, "_content_dedupe", _SpyDedupe()), \
             mock.patch.object(self.mod, "_persist_to_log"):
            self.mod._handle_notification(
                _make_notification(nid=1, texts=("Sam", "atomicbody")))

        self.assertFalse(
            state["released_between"],
            "content-dedupe released _state_lock between its read and write — "
            "two listener threads could pass the freshness test and double-announce")

    def test_classify_action_uses_llm_verdict(self):
        # No rule matches → classify → LLM says "urgent" → read_aloud.
        self.mod._rules = []
        with mock.patch.object(self.mod, "_classify_with_llm",
                               return_value="urgent") as clf, \
             mock.patch.object(self.mod, "_focus_mode_active", return_value=False), \
             mock.patch.object(self.mod, "_proactive_announce") as ann:
            self.mod._handle_notification(
                _make_notification(nid=9, app="Outlook", texts=("Boss", "call me")))
        clf.assert_called_once()
        ann.assert_called_once()
        self.assertEqual(self.mod._recent[-1]["llm_verdict"], "urgent")
        self.assertEqual(self.mod._recent[-1]["action"], "read_aloud")

    def test_classify_action_llm_none_falls_back_to_log(self):
        self.mod._rules = []
        with mock.patch.object(self.mod, "_classify_with_llm", return_value=None), \
             mock.patch.object(self.mod, "_persist_to_log") as plog:
            self.mod._handle_notification(
                _make_notification(nid=8, app="X", texts=("a", "b")))
        # LLM unavailable → silent log.
        self.assertEqual(self.mod._recent[-1]["action"], "log")
        plog.assert_called_once()

    def test_focus_mode_suppresses_low_priority_read_aloud(self):
        self.mod._rules = [{"id": "teams", "match": {"app_pattern": "(?i)teams"},
                            "action": "read_aloud", "priority": 50}]
        with mock.patch.object(self.mod, "_focus_mode_active", return_value=True), \
             mock.patch.object(self.mod, "_proactive_announce") as ann:
            self.mod._handle_notification(
                _make_notification(nid=12, app="Teams", texts=("Sam", "hi")))
        ann.assert_not_called()   # focus mode held it.

    def test_focus_mode_bypassed_by_high_priority(self):
        self.mod._rules = [{"id": "build", "match": {}, "action": "read_aloud",
                            "priority": self.mod.HIGH_PRIORITY_FLOOR + 5}]
        with mock.patch.object(self.mod, "_focus_mode_active", return_value=True), \
             mock.patch.object(self.mod, "_proactive_announce") as ann:
            self.mod._handle_notification(
                _make_notification(nid=13, app="CI", texts=("build failed", "")))
        ann.assert_called_once()   # critical rule cut through focus mode.

    def test_snooze_suppresses_second_announce(self):
        self.mod._rules = [{"id": "teams", "match": {}, "action": "read_aloud",
                            "priority": 50}]
        with mock.patch.object(self.mod, "_focus_mode_active", return_value=False), \
             mock.patch.object(self.mod, "_proactive_announce") as ann:
            t = _T0
            with mock.patch.object(self.mod.time, "time", return_value=t), \
                 mock.patch.object(self.mod._time, "time", return_value=t):
                self.mod._handle_notification(
                    _make_notification(nid=1, app="Teams", texts=("Sam", "msg one")))
            # Different id/body so dedup caches don't fire, but same app+title
            # within SNOOZE_SECONDS → snooze gate suppresses the announce.
            t2 = t + 10
            with mock.patch.object(self.mod.time, "time", return_value=t2), \
                 mock.patch.object(self.mod._time, "time", return_value=t2):
                self.mod._handle_notification(
                    _make_notification(nid=2, app="Teams", texts=("Sam", "msg two")))
        self.assertEqual(ann.call_count, 1)

    def test_announce_sha1_dedupe_gate(self):
        # Pre-seed the SHA-1 announce cache with this toast's key so the gate
        # fires even though snooze/focus are clear.
        self.mod._rules = [{"id": "teams", "match": {}, "action": "read_aloud",
                            "priority": 50}]
        now = _T0
        key = self.mod._announce_dedup_key("Teams", "Sam", "body")
        with self.mod._state_lock:
            self.mod._announce_dedup[key] = now
        with mock.patch.object(self.mod, "_focus_mode_active", return_value=False), \
             mock.patch.object(self.mod, "_proactive_announce") as ann, \
             mock.patch.object(self.mod.time, "time", return_value=now), \
             mock.patch.object(self.mod._time, "time", return_value=now):
            self.mod._handle_notification(
                _make_notification(nid=21, app="Teams", texts=("Sam", "body")))
        ann.assert_not_called()

    def test_announce_sha256_dedupe_gate(self):
        # SHA-1 cache empty, but SHA-256 (sender, body, window_id) pre-seeded.
        self.mod._rules = [{"id": "teams", "match": {}, "action": "read_aloud",
                            "priority": 50}]
        now = _T0
        sha_key = self.mod._announce_sha256_dedupe_key("Sam", "body", 31)
        with self.mod._state_lock:
            self.mod._announce_sha256_dedup[sha_key] = now
        with mock.patch.object(self.mod, "_focus_mode_active", return_value=False), \
             mock.patch.object(self.mod, "_proactive_announce") as ann, \
             mock.patch.object(self.mod.time, "time", return_value=now), \
             mock.patch.object(self.mod._time, "time", return_value=now):
            self.mod._handle_notification(
                _make_notification(nid=31, app="Teams", texts=("Sam", "body")))
        ann.assert_not_called()

    def test_empty_toast_skips_dedupe_but_records(self):
        # All-empty (no app/title/body) → dedup gates skipped; default classify.
        self.mod._rules = []
        with mock.patch.object(self.mod, "_classify_with_llm", return_value=None), \
             mock.patch.object(self.mod, "_persist_to_log"):
            self.mod._handle_notification(
                _make_notification(nid=99, app="", texts=()))
        self.assertEqual(self.mod._recent[-1]["action"], "log")
        self.assertEqual(self.mod._recent[-1]["title"], "")

    def test_id_extraction_failure_uses_zero(self):
        # A notification whose .id raises → nid falls back to 0, seen-id logic
        # skipped, but the toast is still processed.
        class _BadId(_FakeNotification):
            @property
            def id(self):
                raise RuntimeError("no id")

            @id.setter
            def id(self, v):
                pass

        self.mod._rules = [{"id": "logger", "match": {}, "action": "log"}]
        with mock.patch.object(self.mod, "_persist_to_log"):
            self.mod._handle_notification(_BadId(nid=0, texts=("T", "b")))
        self.assertTrue(self.mod._recent)

    def test_app_info_missing_leaves_app_blank(self):
        self.mod._rules = [{"id": "logger", "match": {}, "action": "log"}]
        with mock.patch.object(self.mod, "_persist_to_log"):
            self.mod._handle_notification(
                _make_notification(nid=44, app_info_missing=True,
                                   texts=("Title", "Body")))
        self.assertEqual(self.mod._recent[-1]["app"], "")

    def test_app_info_access_raises_swallowed(self):
        # ``.app_info`` raising → the app-extraction try/except swallows it and
        # the toast is still processed with a blank app.
        class _RaisingAppInfo(_FakeNotification):
            @property
            def app_info(self):
                raise RuntimeError("marshalling boom")

            @app_info.setter
            def app_info(self, v):
                pass

        self.mod._rules = [{"id": "logger", "match": {}, "action": "log"}]
        with mock.patch.object(self.mod, "_persist_to_log"):
            self.mod._handle_notification(
                _RaisingAppInfo(nid=46, texts=("Title", "Body")))
        self.assertEqual(self.mod._recent[-1]["app"], "")
        self.assertEqual(self.mod._recent[-1]["title"], "Title")

    def test_text_extraction_failure_still_records(self):
        # The toast.visual access raises → title/body stay empty, handler
        # swallows it and still records.
        self.mod._rules = []
        with mock.patch.object(self.mod, "_classify_with_llm", return_value=None), \
             mock.patch.object(self.mod, "_persist_to_log"):
            self.mod._handle_notification(
                _make_notification(nid=55, app="App", texts=("x",),
                                   toast_raises=True))
        self.assertEqual(self.mod._recent[-1]["title"], "")
        self.assertEqual(self.mod._recent[-1]["app"], "App")

    def test_three_text_elements_concatenate_into_body(self):
        # First element → title, second → body, third+ appended to body.
        self.mod._rules = [{"id": "logger", "match": {}, "action": "log"}]
        with mock.patch.object(self.mod, "_persist_to_log"):
            self.mod._handle_notification(
                _make_notification(nid=66, app="App",
                                   texts=("Title", "line one", "line two")))
        rec = self.mod._recent[-1]
        self.assertEqual(rec["title"], "Title")
        self.assertIn("line one", rec["body"])
        self.assertIn("line two", rec["body"])

    def test_recent_ring_buffer_trims_to_max(self):
        self.mod._rules = [{"id": "logger", "match": {}, "action": "log"}]
        # Pre-fill the ring above the cap, then push one more.
        with self.mod._state_lock:
            for i in range(self.mod.MAX_NOTIFICATION_LOG + 5):
                self.mod._recent.append({"app": "x", "title": str(i), "ts": i})
        with mock.patch.object(self.mod, "_persist_to_log"):
            self.mod._handle_notification(_make_notification(nid=70, texts=("New", "b")))
        self.assertLessEqual(len(self.mod._recent), self.mod.MAX_NOTIFICATION_LOG)


# ─────────────────────────────────────────────────────────────────────────
# _classify_with_llm — Haiku primary + Ollama fallback.
# ─────────────────────────────────────────────────────────────────────────
class ClassifyWithLLMTests(_IsolatedTriageBase):
    def test_disabled_returns_none(self):
        with mock.patch.object(self.mod, "ENABLE_LLM_CLASSIFIER", False):
            self.assertIsNone(self.mod._classify_with_llm("App", "T", "B"))

    def test_anthropic_path_parses_verdict(self):
        block = types.SimpleNamespace(text="urgent")
        msg = types.SimpleNamespace(content=[block])
        client = mock.MagicMock()
        client.messages.create.return_value = msg
        fake_anthropic = types.ModuleType("anthropic")
        fake_anthropic.Anthropic = mock.MagicMock(return_value=client)
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}), \
             mock.patch.dict(sys.modules, {"anthropic": fake_anthropic}):
            out = self.mod._classify_with_llm("Teams", "Sam", "call me")
        self.assertEqual(out, "urgent")

    def test_anthropic_garbage_verdict_falls_through_to_local(self):
        block = types.SimpleNamespace(text="banana")   # not a valid label
        msg = types.SimpleNamespace(content=[block])
        client = mock.MagicMock()
        client.messages.create.return_value = msg
        fake_anthropic = types.ModuleType("anthropic")
        fake_anthropic.Anthropic = mock.MagicMock(return_value=client)
        bc = types.ModuleType("bobert_companion")
        bc._call_local_llm = mock.MagicMock(return_value="spam")
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}), \
             mock.patch.dict(sys.modules, {"anthropic": fake_anthropic,
                                           "bobert_companion": bc}):
            out = self.mod._classify_with_llm("App", "T", "buy now")
        self.assertEqual(out, "spam")   # local fallback won.

    def test_anthropic_raises_falls_back_to_local(self):
        fake_anthropic = types.ModuleType("anthropic")
        fake_anthropic.Anthropic = mock.MagicMock(side_effect=RuntimeError("boom"))
        bc = types.ModuleType("bobert_companion")
        bc._call_local_llm = mock.MagicMock(return_value="fyi")
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}), \
             mock.patch.dict(sys.modules, {"anthropic": fake_anthropic,
                                           "bobert_companion": bc}):
            out = self.mod._classify_with_llm("App", "T", "status update")
        self.assertEqual(out, "fyi")

    def test_no_api_key_uses_local_only(self):
        bc = types.ModuleType("bobert_companion")
        bc._call_local_llm = mock.MagicMock(return_value="newsletter")
        with mock.patch.dict(os.environ, {}, clear=True), \
             mock.patch.dict(sys.modules, {"bobert_companion": bc}):
            out = self.mod._classify_with_llm("App", "Digest", "weekly roundup")
        self.assertEqual(out, "newsletter")
        bc._call_local_llm.assert_called_once()

    def test_local_returns_garbage_yields_none(self):
        bc = types.ModuleType("bobert_companion")
        bc._call_local_llm = mock.MagicMock(return_value="???not-a-label???")
        with mock.patch.dict(os.environ, {}, clear=True), \
             mock.patch.dict(sys.modules, {"bobert_companion": bc}):
            self.assertIsNone(self.mod._classify_with_llm("App", "T", "B"))

    def test_local_raises_yields_none(self):
        bc = types.ModuleType("bobert_companion")
        bc._call_local_llm = mock.MagicMock(side_effect=RuntimeError("ollama down"))
        with mock.patch.dict(os.environ, {}, clear=True), \
             mock.patch.dict(sys.modules, {"bobert_companion": bc}):
            self.assertIsNone(self.mod._classify_with_llm("App", "T", "B"))

    def test_local_empty_string_yields_none(self):
        # _call_local_llm returns "" → falsy → no verdict.
        bc = types.ModuleType("bobert_companion")
        bc._call_local_llm = mock.MagicMock(return_value="")
        with mock.patch.dict(os.environ, {}, clear=True), \
             mock.patch.dict(sys.modules, {"bobert_companion": bc}):
            self.assertIsNone(self.mod._classify_with_llm("App", "T", "B"))


# ─────────────────────────────────────────────────────────────────────────
# _proactive_announce / _focus_mode_active integration shims.
# ─────────────────────────────────────────────────────────────────────────
class AnnounceAndFocusShimTests(_IsolatedTriageBase):
    def test_proactive_announce_uses_bobert(self):
        bc = types.ModuleType("bobert_companion")
        bc.proactive_announce = mock.MagicMock()
        with mock.patch.dict(sys.modules, {"bobert_companion": bc}):
            self.mod._proactive_announce("hello sir")
        bc.proactive_announce.assert_called_once()
        # source kw routes the announcement category.
        self.assertEqual(bc.proactive_announce.call_args.kwargs.get("source"),
                         "notification_triage")

    def test_proactive_announce_no_announcer_prints(self):
        # bobert present but without a callable proactive_announce → print path.
        bc = types.ModuleType("bobert_companion")
        bc.proactive_announce = None
        with mock.patch.dict(sys.modules, {"bobert_companion": bc}), \
             mock.patch("builtins.print") as pr:
            self.mod._proactive_announce("fallback msg")
        self.assertTrue(pr.called)

    def test_proactive_announce_import_failure_prints(self):
        # importlib.import_module raising on bobert_companion → print branch.
        # Delegate every other name so mock's own internal import_module calls
        # (e.g. resolving builtins.print) keep working.
        real_import_module = self.mod.importlib.import_module

        def _imp(name, *a, **k):
            if name == "bobert_companion":
                raise ImportError("no bobert")
            return real_import_module(name, *a, **k)

        with mock.patch("builtins.print") as pr:
            with mock.patch.object(self.mod.importlib, "import_module",
                                   side_effect=_imp):
                self.mod._proactive_announce("x")
        self.assertTrue(pr.called)

    def test_focus_mode_active_no_skill_loaded(self):
        with mock.patch.dict(sys.modules, {}, clear=False):
            sys.modules.pop("skill_dnd_focus_mode", None)
            self.assertFalse(self.mod._focus_mode_active())

    def test_focus_mode_active_true(self):
        fm = types.ModuleType("skill_dnd_focus_mode")
        fm.is_focus_mode_active = lambda: True
        with mock.patch.dict(sys.modules, {"skill_dnd_focus_mode": fm}):
            self.assertTrue(self.mod._focus_mode_active())

    def test_focus_mode_active_raises_is_false(self):
        fm = types.ModuleType("skill_dnd_focus_mode")
        fm.is_focus_mode_active = mock.MagicMock(side_effect=RuntimeError("x"))
        with mock.patch.dict(sys.modules, {"skill_dnd_focus_mode": fm}):
            self.assertFalse(self.mod._focus_mode_active())

    def test_focus_mode_active_missing_attr_is_false(self):
        fm = types.ModuleType("skill_dnd_focus_mode")   # no is_focus_mode_active
        with mock.patch.dict(sys.modules, {"skill_dnd_focus_mode": fm}):
            self.assertFalse(self.mod._focus_mode_active())


# ─────────────────────────────────────────────────────────────────────────
# _persist_to_log — append-mode jsonl writer.
# ─────────────────────────────────────────────────────────────────────────
class PersistToLogTests(_IsolatedTriageBase):
    def test_persist_writes_jsonl_line(self):
        rec = {"ts": 1_700_000_000.0, "app": "Teams", "title": "Sam",
               "body": "hi", "action": "log"}
        self.mod._persist_to_log(rec)
        day = time.strftime("%Y-%m-%d", time.localtime(rec["ts"]))
        path = os.path.join(self.mod._DATA_DIR, f"{day}.jsonl")
        self.assertTrue(os.path.exists(path))
        with open(path, encoding="utf-8") as f:
            line = f.readline()
        self.assertEqual(json.loads(line)["app"], "Teams")

    def test_persist_appends_multiple(self):
        rec = {"ts": 1_700_000_000.0, "app": "A", "action": "log"}
        self.mod._persist_to_log(rec)
        self.mod._persist_to_log({**rec, "app": "B"})
        day = time.strftime("%Y-%m-%d", time.localtime(rec["ts"]))
        path = os.path.join(self.mod._DATA_DIR, f"{day}.jsonl")
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 2)

    def test_persist_swallows_write_error(self):
        # makedirs raising must not propagate (best-effort logging).
        with mock.patch.object(self.mod.os, "makedirs",
                               side_effect=OSError("denied")):
            self.mod._persist_to_log({"ts": 1.0, "app": "x"})  # no raise


# ─────────────────────────────────────────────────────────────────────────
# Rule persistence: _load_rules / _save_rules.
# ─────────────────────────────────────────────────────────────────────────
class RulePersistenceTests(_IsolatedTriageBase):
    def test_load_rules_seeds_defaults_when_missing(self):
        self.assertFalse(os.path.exists(self.mod._RULES_FILE))
        rules = self.mod._load_rules()
        # File seeded and the in-memory copy is sorted desc by priority.
        self.assertTrue(os.path.exists(self.mod._RULES_FILE))
        prios = [int(r.get("priority", 0)) for r in rules]
        self.assertEqual(prios, sorted(prios, reverse=True))

    def test_load_rules_reads_existing_and_filters_invalid(self):
        good = {"id": "ok", "action": "log", "priority": 5}
        bad = {"id": "", "action": "nope"}     # invalid → dropped
        with open(self.mod._RULES_FILE, "w", encoding="utf-8") as f:
            json.dump([good, bad], f)
        rules = self.mod._load_rules()
        ids = [r["id"] for r in rules]
        self.assertIn("ok", ids)
        self.assertEqual(len(rules), 1)

    def test_load_rules_corrupt_file_falls_back_to_defaults(self):
        with open(self.mod._RULES_FILE, "w", encoding="utf-8") as f:
            f.write("{ not json")
        rules = self.mod._load_rules()
        # Falls back to the shipped defaults rather than crashing.
        self.assertTrue(any(r["id"] == "build_or_ci_failure" for r in rules))

    def test_load_rules_non_list_json_falls_back(self):
        with open(self.mod._RULES_FILE, "w", encoding="utf-8") as f:
            json.dump({"not": "a list"}, f)
        rules = self.mod._load_rules()
        self.assertTrue(any(r["id"] == "build_or_ci_failure" for r in rules))

    def test_load_rules_seed_write_failure_still_returns_defaults(self):
        with mock.patch.object(self.mod, "_atomic_write_json",
                               side_effect=OSError("ro fs")):
            rules = self.mod._load_rules()
        self.assertTrue(any(r["id"] == "build_or_ci_failure" for r in rules))

    def test_save_rules_writes_snapshot(self):
        self.mod._rules = [{"id": "a", "action": "log", "priority": 1}]
        self.mod._save_rules()
        with open(self.mod._RULES_FILE, encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data[0]["id"], "a")

    def test_save_rules_swallows_write_error(self):
        self.mod._rules = [{"id": "a", "action": "log"}]
        with mock.patch.object(self.mod, "_atomic_write_json",
                               side_effect=OSError("denied")):
            self.mod._save_rules()   # no raise

    def test_add_rule_action_persists_to_disk(self):
        # End-to-end through the registered action (no _save_rules patch) so
        # the real atomic write lands in the temp rules file.
        out = self.actions["add_notification_rule"](
            json.dumps({"id": "disk_rule", "action": "drop", "priority": 3}))
        self.assertIn("disk_rule", out)
        with open(self.mod._RULES_FILE, encoding="utf-8") as f:
            data = json.load(f)
        self.assertTrue(any(r["id"] == "disk_rule" for r in data))


# ─────────────────────────────────────────────────────────────────────────
# Dedup cache persistence: load / save / prune for all four caches.
# ─────────────────────────────────────────────────────────────────────────
class DedupePersistenceTests(_IsolatedTriageBase):
    # ── id + snooze dedupe ────────────────────────────────────────────────
    def test_save_then_load_dedupe_state_roundtrips(self):
        now = time.time()
        with self.mod._state_lock:
            self.mod._seen_ids.update({101, 102})
            self.mod._snooze["Teams|Sam"] = now
        self.mod._save_dedupe_state()
        # Clear, then hydrate from disk.
        with self.mod._state_lock:
            self.mod._seen_ids.clear()
            self.mod._snooze.clear()
        self.mod._load_dedupe_state()
        self.assertIn(101, self.mod._seen_ids)
        self.assertIn("Teams|Sam", self.mod._snooze)

    def test_load_dedupe_state_missing_file_is_noop(self):
        self.assertFalse(os.path.exists(self.mod._DEDUPE_FILE))
        self.mod._load_dedupe_state()   # no raise, nothing added
        self.assertEqual(self.mod._seen_ids, set())

    def test_load_dedupe_state_stale_ts_skipped(self):
        stale = time.time() - (self.mod._DEDUPE_TTL_SEC + 100)
        with open(self.mod._DEDUPE_FILE, "w", encoding="utf-8") as f:
            json.dump({"ts": stale, "seen_ids": [1, 2], "snooze": {}}, f)
        self.mod._load_dedupe_state()
        self.assertEqual(self.mod._seen_ids, set())   # too old → ignored

    def test_load_dedupe_state_non_dict_ignored(self):
        with open(self.mod._DEDUPE_FILE, "w", encoding="utf-8") as f:
            json.dump([1, 2, 3], f)
        self.mod._load_dedupe_state()   # not a dict → bail
        self.assertEqual(self.mod._seen_ids, set())

    def test_load_dedupe_state_skips_stale_snooze_entries(self):
        now = time.time()
        payload = {
            "ts": now,
            "seen_ids": ["nope", 5],      # "nope" int() fails → skipped
            "snooze": {"fresh": now, "old": now - (self.mod._DEDUPE_TTL_SEC + 50),
                       "bad": "x"},
        }
        with open(self.mod._DEDUPE_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        self.mod._load_dedupe_state()
        self.assertIn(5, self.mod._seen_ids)
        self.assertIn("fresh", self.mod._snooze)
        self.assertNotIn("old", self.mod._snooze)
        self.assertNotIn("bad", self.mod._snooze)

    def test_save_dedupe_state_swallows_error(self):
        with mock.patch.object(self.mod._os, "makedirs",
                               side_effect=OSError("denied")):
            self.mod._save_dedupe_state()   # no raise

    def test_maybe_save_dedupe_state_stamps_and_saves(self):
        with mock.patch.object(self.mod, "_save_dedupe_state") as sv:
            self.mod._maybe_save_dedupe_state()
        sv.assert_called_once()
        self.assertGreater(self.mod._last_dedupe_save[0], 0.0)

    # ── content dedupe ────────────────────────────────────────────────────
    def test_content_dedupe_save_load_roundtrip(self):
        now = time.time()
        with self.mod._state_lock:
            self.mod._content_dedupe["teams|sam|hash"] = now
        self.mod._save_content_dedupe()
        with self.mod._state_lock:
            self.mod._content_dedupe.clear()
        self.mod._load_content_dedupe()
        self.assertIn("teams|sam|hash", self.mod._content_dedupe)

    def test_content_dedupe_load_missing_file(self):
        self.mod._load_content_dedupe()   # no file → no raise
        self.assertEqual(self.mod._content_dedupe, {})

    def test_content_dedupe_load_bad_shapes(self):
        # Non-dict top-level → bail.
        with open(self.mod._CONTENT_DEDUPE_FILE, "w", encoding="utf-8") as f:
            json.dump([1], f)
        self.mod._load_content_dedupe()
        self.assertEqual(self.mod._content_dedupe, {})
        # entries not a dict → bail.
        with open(self.mod._CONTENT_DEDUPE_FILE, "w", encoding="utf-8") as f:
            json.dump({"entries": "nope"}, f)
        self.mod._load_content_dedupe()
        self.assertEqual(self.mod._content_dedupe, {})

    def test_content_dedupe_load_skips_stale_and_bad_values(self):
        now = time.time()
        entries = {"fresh": now, "stale": now - (self.mod._CONTENT_DEDUPE_TTL_SEC + 1),
                   "bad": "x"}
        with open(self.mod._CONTENT_DEDUPE_FILE, "w", encoding="utf-8") as f:
            json.dump({"entries": entries}, f)
        self.mod._load_content_dedupe()
        self.assertIn("fresh", self.mod._content_dedupe)
        self.assertNotIn("stale", self.mod._content_dedupe)
        self.assertNotIn("bad", self.mod._content_dedupe)

    def test_prune_content_dedupe_drops_old(self):
        now = 10_000.0
        with self.mod._state_lock:
            self.mod._content_dedupe["old"] = now - (self.mod._CONTENT_DEDUPE_TTL_SEC + 1)
            self.mod._content_dedupe["new"] = now
        self.mod._prune_content_dedupe(now)
        self.assertNotIn("old", self.mod._content_dedupe)
        self.assertIn("new", self.mod._content_dedupe)

    def test_prune_content_dedupe_defaults_now(self):
        # now=None path uses _time.time(); seed something fresh so it survives.
        with self.mod._state_lock:
            self.mod._content_dedupe["fresh"] = time.time()
        self.mod._prune_content_dedupe()
        self.assertIn("fresh", self.mod._content_dedupe)

    def test_content_dedupe_save_swallows_error(self):
        with mock.patch.object(self.mod._os, "makedirs",
                               side_effect=OSError("denied")):
            self.mod._save_content_dedupe()   # no raise

    # ── timestamp-bucket dedupe ───────────────────────────────────────────
    def test_timestamp_dedupe_save_load_roundtrip(self):
        now = time.time()
        with self.mod._state_lock:
            self.mod._notification_timestamp_dedup["sam|h|5"] = now
        self.mod._save_notification_timestamp_dedup()
        with self.mod._state_lock:
            self.mod._notification_timestamp_dedup.clear()
        self.mod._load_notification_timestamp_dedup()
        self.assertIn("sam|h|5", self.mod._notification_timestamp_dedup)

    def test_timestamp_dedupe_load_missing_file(self):
        self.mod._load_notification_timestamp_dedup()
        self.assertEqual(self.mod._notification_timestamp_dedup, {})

    def test_timestamp_dedupe_load_bad_shapes(self):
        with open(self.mod._NOTIF_TS_DEDUPE_FILE, "w", encoding="utf-8") as f:
            json.dump("scalar", f)
        self.mod._load_notification_timestamp_dedup()
        self.assertEqual(self.mod._notification_timestamp_dedup, {})
        with open(self.mod._NOTIF_TS_DEDUPE_FILE, "w", encoding="utf-8") as f:
            json.dump({"entries": 5}, f)
        self.mod._load_notification_timestamp_dedup()
        self.assertEqual(self.mod._notification_timestamp_dedup, {})

    def test_timestamp_dedupe_load_skips_stale_and_bad(self):
        now = time.time()
        entries = {"fresh": now,
                   "stale": now - (self.mod._NOTIF_TS_DEDUPE_TTL_SEC + 1),
                   "bad": "x"}
        with open(self.mod._NOTIF_TS_DEDUPE_FILE, "w", encoding="utf-8") as f:
            json.dump({"entries": entries}, f)
        self.mod._load_notification_timestamp_dedup()
        self.assertIn("fresh", self.mod._notification_timestamp_dedup)
        self.assertNotIn("stale", self.mod._notification_timestamp_dedup)
        self.assertNotIn("bad", self.mod._notification_timestamp_dedup)

    def test_prune_timestamp_dedupe(self):
        now = 20_000.0
        with self.mod._state_lock:
            self.mod._notification_timestamp_dedup["old"] = \
                now - (self.mod._NOTIF_TS_DEDUPE_TTL_SEC + 1)
            self.mod._notification_timestamp_dedup["new"] = now
        self.mod._prune_notification_timestamp_dedup(now)
        self.assertNotIn("old", self.mod._notification_timestamp_dedup)
        self.assertIn("new", self.mod._notification_timestamp_dedup)

    def test_prune_timestamp_dedupe_defaults_now(self):
        with self.mod._state_lock:
            self.mod._notification_timestamp_dedup["fresh"] = time.time()
        self.mod._prune_notification_timestamp_dedup()
        self.assertIn("fresh", self.mod._notification_timestamp_dedup)

    def test_timestamp_dedupe_save_swallows_error(self):
        with mock.patch.object(self.mod._os, "makedirs",
                               side_effect=OSError("denied")):
            self.mod._save_notification_timestamp_dedup()

    # ── announce SHA-1 dedupe ─────────────────────────────────────────────
    def test_announce_dedupe_save_load_roundtrip(self):
        now = time.time()
        with self.mod._state_lock:
            self.mod._announce_dedup["abc"] = now
        self.mod._save_announce_dedup()
        with self.mod._state_lock:
            self.mod._announce_dedup.clear()
        self.mod._load_announce_dedup()
        self.assertIn("abc", self.mod._announce_dedup)

    def test_announce_dedupe_load_missing_file(self):
        self.mod._load_announce_dedup()
        self.assertEqual(self.mod._announce_dedup, {})

    def test_announce_dedupe_load_bad_shapes(self):
        with open(self.mod._ANNOUNCE_DEDUPE_FILE, "w", encoding="utf-8") as f:
            json.dump(42, f)
        self.mod._load_announce_dedup()
        self.assertEqual(self.mod._announce_dedup, {})
        with open(self.mod._ANNOUNCE_DEDUPE_FILE, "w", encoding="utf-8") as f:
            json.dump({"entries": []}, f)
        self.mod._load_announce_dedup()
        self.assertEqual(self.mod._announce_dedup, {})

    def test_announce_dedupe_load_skips_stale_and_bad(self):
        now = time.time()
        entries = {"fresh": now,
                   "stale": now - (self.mod._ANNOUNCE_DEDUPE_TTL_SEC + 1),
                   "bad": "x"}
        with open(self.mod._ANNOUNCE_DEDUPE_FILE, "w", encoding="utf-8") as f:
            json.dump({"entries": entries}, f)
        self.mod._load_announce_dedup()
        self.assertIn("fresh", self.mod._announce_dedup)
        self.assertNotIn("stale", self.mod._announce_dedup)
        self.assertNotIn("bad", self.mod._announce_dedup)

    def test_prune_announce_dedupe(self):
        now = 30_000.0
        with self.mod._state_lock:
            self.mod._announce_dedup["old"] = now - (self.mod._ANNOUNCE_DEDUPE_TTL_SEC + 1)
            self.mod._announce_dedup["new"] = now
        self.mod._prune_announce_dedup(now)
        self.assertNotIn("old", self.mod._announce_dedup)
        self.assertIn("new", self.mod._announce_dedup)

    def test_prune_announce_dedupe_defaults_now(self):
        with self.mod._state_lock:
            self.mod._announce_dedup["fresh"] = time.time()
        self.mod._prune_announce_dedup()
        self.assertIn("fresh", self.mod._announce_dedup)

    def test_announce_dedupe_save_swallows_error(self):
        with mock.patch.object(self.mod._os, "makedirs",
                               side_effect=OSError("denied")):
            self.mod._save_announce_dedup()

    # ── announce SHA-256 dedupe ───────────────────────────────────────────
    def test_announce_sha256_save_load_roundtrip(self):
        now = time.time()
        with self.mod._state_lock:
            self.mod._announce_sha256_dedup["deadbeef"] = now
        self.mod._save_announce_sha256_dedup()
        with self.mod._state_lock:
            self.mod._announce_sha256_dedup.clear()
        self.mod._load_announce_sha256_dedup()
        self.assertIn("deadbeef", self.mod._announce_sha256_dedup)

    def test_announce_sha256_load_missing_file(self):
        self.mod._load_announce_sha256_dedup()
        self.assertEqual(self.mod._announce_sha256_dedup, {})

    def test_announce_sha256_load_bad_shapes(self):
        with open(self.mod._ANNOUNCE_SHA256_DEDUPE_FILE, "w", encoding="utf-8") as f:
            json.dump("nope", f)
        self.mod._load_announce_sha256_dedup()
        self.assertEqual(self.mod._announce_sha256_dedup, {})
        with open(self.mod._ANNOUNCE_SHA256_DEDUPE_FILE, "w", encoding="utf-8") as f:
            json.dump({"entries": 0}, f)
        self.mod._load_announce_sha256_dedup()
        self.assertEqual(self.mod._announce_sha256_dedup, {})

    def test_announce_sha256_load_skips_stale_and_bad(self):
        now = time.time()
        entries = {"fresh": now,
                   "stale": now - (self.mod._ANNOUNCE_SHA256_DEDUPE_TTL_SEC + 1),
                   "bad": "x"}
        with open(self.mod._ANNOUNCE_SHA256_DEDUPE_FILE, "w", encoding="utf-8") as f:
            json.dump({"entries": entries}, f)
        self.mod._load_announce_sha256_dedup()
        self.assertIn("fresh", self.mod._announce_sha256_dedup)
        self.assertNotIn("stale", self.mod._announce_sha256_dedup)
        self.assertNotIn("bad", self.mod._announce_sha256_dedup)

    def test_prune_announce_sha256(self):
        now = 40_000.0
        with self.mod._state_lock:
            self.mod._announce_sha256_dedup["old"] = \
                now - (self.mod._ANNOUNCE_SHA256_DEDUPE_TTL_SEC + 1)
            self.mod._announce_sha256_dedup["new"] = now
        self.mod._prune_announce_sha256_dedup(now)
        self.assertNotIn("old", self.mod._announce_sha256_dedup)
        self.assertIn("new", self.mod._announce_sha256_dedup)

    def test_prune_announce_sha256_defaults_now(self):
        with self.mod._state_lock:
            self.mod._announce_sha256_dedup["fresh"] = time.time()
        self.mod._prune_announce_sha256_dedup()
        self.assertIn("fresh", self.mod._announce_sha256_dedup)

    def test_announce_sha256_save_swallows_error(self):
        with mock.patch.object(self.mod._os, "makedirs",
                               side_effect=OSError("denied")):
            self.mod._save_announce_sha256_dedup()

    # ── replace-fails cleanup branch (inner os.remove of the temp file) ──
    def _assert_no_tmp_left(self, prefix):
        leftovers = [f for f in os.listdir(self.tmp) if f.startswith(prefix)]
        self.assertEqual(leftovers, [])

    def test_save_dedupe_state_replace_fails_cleans_tmp(self):
        with self.mod._state_lock:
            self.mod._seen_ids.add(1)
        with mock.patch.object(self.mod._os, "replace",
                               side_effect=OSError("rename boom")):
            self.mod._save_dedupe_state()
        self._assert_no_tmp_left(".notif_")

    def test_save_content_dedupe_replace_fails_cleans_tmp(self):
        with self.mod._state_lock:
            self.mod._content_dedupe["k"] = time.time()
        with mock.patch.object(self.mod._os, "replace",
                               side_effect=OSError("rename boom")):
            self.mod._save_content_dedupe()
        self._assert_no_tmp_left(".notif_content_")

    def test_save_timestamp_dedupe_replace_fails_cleans_tmp(self):
        with self.mod._state_lock:
            self.mod._notification_timestamp_dedup["k"] = time.time()
        with mock.patch.object(self.mod._os, "replace",
                               side_effect=OSError("rename boom")):
            self.mod._save_notification_timestamp_dedup()
        self._assert_no_tmp_left(".notif_ts_")

    def test_save_announce_dedupe_replace_fails_cleans_tmp(self):
        with self.mod._state_lock:
            self.mod._announce_dedup["k"] = time.time()
        with mock.patch.object(self.mod._os, "replace",
                               side_effect=OSError("rename boom")):
            self.mod._save_announce_dedup()
        self._assert_no_tmp_left(".notif_announce_")

    def test_save_announce_sha256_replace_fails_cleans_tmp(self):
        with self.mod._state_lock:
            self.mod._announce_sha256_dedup["k"] = time.time()
        with mock.patch.object(self.mod._os, "replace",
                               side_effect=OSError("rename boom")):
            self.mod._save_announce_sha256_dedup()
        self._assert_no_tmp_left(".notif_announce_sha256_")

    # ── corrupt-JSON load → outermost except: pass (json.load raises) ────
    def test_load_dedupe_state_corrupt_json_swallowed(self):
        with open(self.mod._DEDUPE_FILE, "w", encoding="utf-8") as f:
            f.write("{ not valid json :::")
        self.mod._load_dedupe_state()   # json.load raises → swallowed
        self.assertEqual(self.mod._seen_ids, set())

    def test_load_content_dedupe_corrupt_json_swallowed(self):
        with open(self.mod._CONTENT_DEDUPE_FILE, "w", encoding="utf-8") as f:
            f.write("{ corrupt")
        self.mod._load_content_dedupe()
        self.assertEqual(self.mod._content_dedupe, {})

    def test_load_timestamp_dedupe_corrupt_json_swallowed(self):
        with open(self.mod._NOTIF_TS_DEDUPE_FILE, "w", encoding="utf-8") as f:
            f.write("}{ broken")
        self.mod._load_notification_timestamp_dedup()
        self.assertEqual(self.mod._notification_timestamp_dedup, {})

    def test_load_announce_dedupe_corrupt_json_swallowed(self):
        with open(self.mod._ANNOUNCE_DEDUPE_FILE, "w", encoding="utf-8") as f:
            f.write("not json at all")
        self.mod._load_announce_dedup()
        self.assertEqual(self.mod._announce_dedup, {})

    def test_load_announce_sha256_corrupt_json_swallowed(self):
        with open(self.mod._ANNOUNCE_SHA256_DEDUPE_FILE, "w", encoding="utf-8") as f:
            f.write("<<>>")
        self.mod._load_announce_sha256_dedup()
        self.assertEqual(self.mod._announce_sha256_dedup, {})

    # ── replace AND remove both fail → inner except: pass ────────────────
    def test_save_dedupe_state_replace_and_remove_both_fail(self):
        with self.mod._state_lock:
            self.mod._seen_ids.add(1)
        with mock.patch.object(self.mod._os, "replace",
                               side_effect=OSError("rename")), \
             mock.patch.object(self.mod._os, "remove",
                               side_effect=OSError("remove")):
            self.mod._save_dedupe_state()   # both raise → fully swallowed

    def test_save_content_dedupe_replace_and_remove_both_fail(self):
        with self.mod._state_lock:
            self.mod._content_dedupe["k"] = time.time()
        with mock.patch.object(self.mod._os, "replace",
                               side_effect=OSError("rename")), \
             mock.patch.object(self.mod._os, "remove",
                               side_effect=OSError("remove")):
            self.mod._save_content_dedupe()

    def test_save_timestamp_dedupe_replace_and_remove_both_fail(self):
        with self.mod._state_lock:
            self.mod._notification_timestamp_dedup["k"] = time.time()
        with mock.patch.object(self.mod._os, "replace",
                               side_effect=OSError("rename")), \
             mock.patch.object(self.mod._os, "remove",
                               side_effect=OSError("remove")):
            self.mod._save_notification_timestamp_dedup()

    def test_save_announce_dedupe_replace_and_remove_both_fail(self):
        with self.mod._state_lock:
            self.mod._announce_dedup["k"] = time.time()
        with mock.patch.object(self.mod._os, "replace",
                               side_effect=OSError("rename")), \
             mock.patch.object(self.mod._os, "remove",
                               side_effect=OSError("remove")):
            self.mod._save_announce_dedup()

    def test_save_announce_sha256_replace_and_remove_both_fail(self):
        with self.mod._state_lock:
            self.mod._announce_sha256_dedup["k"] = time.time()
        with mock.patch.object(self.mod._os, "replace",
                               side_effect=OSError("rename")), \
             mock.patch.object(self.mod._os, "remove",
                               side_effect=OSError("remove")):
            self.mod._save_announce_sha256_dedup()


# ─────────────────────────────────────────────────────────────────────────
# winsdk probe + async-op wrappers + poll loop.
# ─────────────────────────────────────────────────────────────────────────
class WinsdkProbeTests(_IsolatedTriageBase):
    def tearDown(self):
        # _probe_winsdk mutates the module-global _winsdk_modules; clear it so
        # nothing leaks into other tests.
        self.mod._winsdk_modules.clear()

    def test_probe_winsdk_primary_path(self):
        fake_mgmt = types.ModuleType("winsdk.windows.ui.notifications.management")
        fake_mgmt.UserNotificationListener = object()
        fake_notif = types.ModuleType("winsdk.windows.ui.notifications")
        fake_notif.NotificationKinds = object()
        fake_notif.KnownNotificationBindings = object()

        def _imp(path):
            return {"winsdk.windows.ui.notifications.management": fake_mgmt,
                    "winsdk.windows.ui.notifications": fake_notif}[path]

        with mock.patch.object(self.mod.importlib, "import_module",
                               side_effect=_imp):
            ok = self.mod._probe_winsdk()
        self.assertTrue(ok)
        self.assertIn("Listener", self.mod._winsdk_modules)

    def test_probe_winsdk_falls_back_to_winrt(self):
        fake_mgmt = types.ModuleType("winrt.windows.ui.notifications.management")
        fake_mgmt.UserNotificationListener = object()
        fake_notif = types.ModuleType("winrt.windows.ui.notifications")
        fake_notif.NotificationKinds = object()
        fake_notif.KnownNotificationBindings = object()

        def _imp(path):
            if path.startswith("winsdk."):
                raise ImportError("no winsdk")
            return {"winrt.windows.ui.notifications.management": fake_mgmt,
                    "winrt.windows.ui.notifications": fake_notif}[path]

        with mock.patch.object(self.mod.importlib, "import_module",
                               side_effect=_imp):
            ok = self.mod._probe_winsdk()
        self.assertTrue(ok)

    def test_probe_winsdk_all_fail(self):
        with mock.patch.object(self.mod.importlib, "import_module",
                               side_effect=ImportError("nothing")):
            self.assertFalse(self.mod._probe_winsdk())

    def test_request_access_get_path(self):
        listener = mock.MagicMock()
        listener.request_access_async.return_value = _AsyncOpGet(_AccessResult("Allowed"))
        self.assertEqual(self.mod._request_access(listener), "Allowed")

    def test_request_access_manual_poll_path(self):
        listener = mock.MagicMock()
        # pending=1 → the `while not op.completed: time.sleep(0.05)` body runs
        # once before get_results() resolves.
        listener.request_access_async.return_value = \
            _AsyncOpManualPoll(_AccessResult("Denied"), pending=1)
        with mock.patch.object(self.mod.time, "sleep") as slp:
            self.assertEqual(self.mod._request_access(listener), "Denied")
        slp.assert_called()

    def test_request_access_pascalcase_fallback(self):
        # request_access_async missing → AttributeError → PascalCase variant.
        listener = mock.MagicMock(spec=["RequestAccessAsync"])
        listener.RequestAccessAsync.return_value = _AsyncOpGet(_AccessResult("Allowed"))
        self.assertEqual(self.mod._request_access(listener), "Allowed")

    def test_request_access_result_without_name(self):
        listener = mock.MagicMock()
        listener.request_access_async.return_value = _AsyncOpGet("Allowed")  # plain str
        self.assertEqual(self.mod._request_access(listener), "Allowed")

    def test_request_access_manual_poll_raises_wraps(self):
        class _BadOp:
            completed = True

            def get(self):
                raise RuntimeError("no get")

            def get_results(self):
                raise RuntimeError("results boom")

        listener = mock.MagicMock()
        listener.request_access_async.return_value = _BadOp()
        with self.assertRaises(RuntimeError):
            self.mod._request_access(listener)

    def test_get_notifications_no_enum_raises(self):
        self.mod._winsdk_modules.clear()
        with self.assertRaises(RuntimeError):
            self.mod._get_notifications(mock.MagicMock())

    def test_get_notifications_get_path(self):
        self.mod._winsdk_modules["NotificationKinds"] = \
            types.SimpleNamespace(TOAST=1)
        listener = mock.MagicMock()
        listener.get_notifications_async.return_value = _AsyncOpGet(["n1", "n2"])
        out = self.mod._get_notifications(listener)
        self.assertEqual(out, ["n1", "n2"])

    def test_get_notifications_manual_poll_path(self):
        self.mod._winsdk_modules["NotificationKinds"] = \
            types.SimpleNamespace(TOAST=1)
        listener = mock.MagicMock()
        listener.get_notifications_async.return_value = \
            _AsyncOpManualPoll(["x"], pending=1)
        with mock.patch.object(self.mod.time, "sleep") as slp:
            self.assertEqual(self.mod._get_notifications(listener), ["x"])
        slp.assert_called()

    def test_get_notifications_pascalcase_fallback(self):
        self.mod._winsdk_modules["NotificationKinds"] = \
            types.SimpleNamespace(TOAST=1)
        listener = mock.MagicMock(spec=["GetNotificationsAsync"])
        listener.GetNotificationsAsync.return_value = _AsyncOpGet(["p"])
        self.assertEqual(self.mod._get_notifications(listener), ["p"])

    def test_get_notifications_manual_poll_raises_wraps(self):
        self.mod._winsdk_modules["NotificationKinds"] = \
            types.SimpleNamespace(TOAST=1)

        class _BadOp:
            completed = True

            def get(self):
                raise RuntimeError("no get")

            def get_results(self):
                raise RuntimeError("boom")

        listener = mock.MagicMock()
        listener.get_notifications_async.return_value = _BadOp()
        with self.assertRaises(RuntimeError):
            self.mod._get_notifications(listener)


# ─────────────────────────────────────────────────────────────────────────
# _listener_loop — exercised via the early-return branches (never enters the
# infinite poll loop). time.sleep is neutered so INITIAL_DELAY is instant.
# ─────────────────────────────────────────────────────────────────────────
class ListenerLoopTests(_IsolatedTriageBase):
    def setUp(self):
        super().setUp()
        self._sleep = mock.patch.object(self.mod.time, "sleep",
                                        lambda *_a, **_k: None)
        self._sleep.start()
        self.addCleanup(self._sleep.stop)
        # Reset the status dict so assertions are clean.
        self.mod._subsystem_status.update({
            "winsdk_available": False, "listener_access": None,
            "listening": False, "last_error": None, "started_at": None,
            "last_poll_at": None,
        })

    def tearDown(self):
        self.mod._winsdk_modules.clear()

    def test_loop_bails_when_winsdk_unavailable(self):
        with mock.patch.object(self.mod, "_probe_winsdk", return_value=False):
            self.mod._listener_loop()
        self.assertFalse(self.mod._subsystem_status["winsdk_available"])
        self.assertIn("winsdk not installed",
                      self.mod._subsystem_status["last_error"])

    def test_loop_bails_when_listener_unobtainable(self):
        class _Listener:
            # both .current and .Current raise → "could not obtain listener".
            @property
            def current(self):
                raise RuntimeError("no current")

            @property
            def Current(self):
                raise RuntimeError("no Current")

        with mock.patch.object(self.mod, "_probe_winsdk", return_value=True):
            self.mod._winsdk_modules["Listener"] = _Listener()
            self.mod._listener_loop()
        self.assertIn("could not obtain listener",
                      self.mod._subsystem_status["last_error"])

    def test_loop_bails_on_request_access_failure(self):
        listener = object()
        Listener = types.SimpleNamespace(current=listener)
        with mock.patch.object(self.mod, "_probe_winsdk", return_value=True), \
             mock.patch.object(self.mod, "_request_access",
                               side_effect=RuntimeError("denied req")):
            self.mod._winsdk_modules["Listener"] = Listener
            self.mod._listener_loop()
        self.assertIn("RequestAccessAsync failed",
                      self.mod._subsystem_status["last_error"])

    def test_loop_bails_when_access_not_allowed(self):
        listener = object()
        Listener = types.SimpleNamespace(current=listener)
        with mock.patch.object(self.mod, "_probe_winsdk", return_value=True), \
             mock.patch.object(self.mod, "_request_access", return_value="Denied"):
            self.mod._winsdk_modules["Listener"] = Listener
            self.mod._listener_loop()
        self.assertEqual(self.mod._subsystem_status["listener_access"], "Denied")
        self.assertFalse(self.mod._subsystem_status["listening"])

    def test_loop_processes_one_batch_then_stops(self):
        # Allowed access; the poll loop runs once, handles a batch, then we
        # raise from time.sleep to break out of the infinite while.
        listener = object()
        Listener = types.SimpleNamespace(current=listener)
        handled = []

        sleep_calls = {"n": 0}

        def _sleep(_secs):
            # First sleep is the INITIAL_DELAY (before probe). Allow it; the
            # second sleep is the end-of-iteration pause — raise to exit.
            sleep_calls["n"] += 1
            if sleep_calls["n"] >= 2:
                raise KeyboardInterrupt("stop loop")

        with mock.patch.object(self.mod.time, "sleep", _sleep), \
             mock.patch.object(self.mod, "_probe_winsdk", return_value=True), \
             mock.patch.object(self.mod, "_request_access", return_value="Allowed"), \
             mock.patch.object(self.mod, "_get_notifications",
                               return_value=["n1", "n2"]), \
             mock.patch.object(self.mod, "_handle_notification",
                               side_effect=lambda n: handled.append(n)):
            self.mod._winsdk_modules["Listener"] = Listener
            with self.assertRaises(KeyboardInterrupt):
                self.mod._listener_loop()
        self.assertEqual(handled, ["n1", "n2"])
        self.assertTrue(self.mod._subsystem_status["listening"])

    def test_loop_paused_skips_polling(self):
        listener = object()
        Listener = types.SimpleNamespace(current=listener)
        self.mod._pause_flag[0] = True
        sleep_calls = {"n": 0}

        def _sleep(_secs):
            # #1 INITIAL_DELAY, #2 pause-sleep (then `continue`), #3 pause-sleep
            # of the 2nd iteration → raise to exit. This exercises the
            # pause-branch `continue`.
            sleep_calls["n"] += 1
            if sleep_calls["n"] >= 3:
                raise KeyboardInterrupt("stop")

        with mock.patch.object(self.mod.time, "sleep", _sleep), \
             mock.patch.object(self.mod, "_probe_winsdk", return_value=True), \
             mock.patch.object(self.mod, "_request_access", return_value="Allowed"), \
             mock.patch.object(self.mod, "_get_notifications") as getn:
            self.mod._winsdk_modules["Listener"] = Listener
            with self.assertRaises(KeyboardInterrupt):
                self.mod._listener_loop()
        getn.assert_not_called()   # paused → never polled.

    def test_loop_backoff_after_many_consecutive_errors(self):
        # >5 consecutive poll failures → the backoff sleep + continue branch.
        listener = object()
        Listener = types.SimpleNamespace(current=listener)
        iters = {"n": 0}

        def _getn(_listener):
            iters["n"] += 1
            raise RuntimeError(f"poll fail {iters['n']}")

        def _sleep(_secs):
            # Bail once we've driven enough iterations to pass the >5 threshold
            # and hit the backoff branch at least once.
            if iters["n"] >= 7:
                raise KeyboardInterrupt("stop")

        with mock.patch.object(self.mod.time, "sleep", _sleep), \
             mock.patch.object(self.mod, "_probe_winsdk", return_value=True), \
             mock.patch.object(self.mod, "_request_access", return_value="Allowed"), \
             mock.patch.object(self.mod, "_get_notifications", side_effect=_getn):
            self.mod._winsdk_modules["Listener"] = Listener
            with self.assertRaises(KeyboardInterrupt):
                self.mod._listener_loop()
        self.assertGreaterEqual(iters["n"], 6)
        self.assertIn("poll fail", self.mod._subsystem_status["last_error"])

    def test_loop_handler_exception_swallowed(self):
        listener = object()
        Listener = types.SimpleNamespace(current=listener)
        sleep_calls = {"n": 0}

        def _sleep(_secs):
            sleep_calls["n"] += 1
            if sleep_calls["n"] >= 2:
                raise KeyboardInterrupt("stop")

        with mock.patch.object(self.mod.time, "sleep", _sleep), \
             mock.patch.object(self.mod, "_probe_winsdk", return_value=True), \
             mock.patch.object(self.mod, "_request_access", return_value="Allowed"), \
             mock.patch.object(self.mod, "_get_notifications", return_value=["bad"]), \
             mock.patch.object(self.mod, "_handle_notification",
                               side_effect=RuntimeError("handler crash")):
            self.mod._winsdk_modules["Listener"] = Listener
            with self.assertRaises(KeyboardInterrupt):
                self.mod._listener_loop()
        # Loop survived the handler crash (reached the 2nd sleep).
        self.assertGreaterEqual(sleep_calls["n"], 2)

    def test_loop_poll_exception_sets_last_error(self):
        listener = object()
        Listener = types.SimpleNamespace(current=listener)
        sleep_calls = {"n": 0}

        def _sleep(_secs):
            sleep_calls["n"] += 1
            if sleep_calls["n"] >= 2:
                raise KeyboardInterrupt("stop")

        with mock.patch.object(self.mod.time, "sleep", _sleep), \
             mock.patch.object(self.mod, "_probe_winsdk", return_value=True), \
             mock.patch.object(self.mod, "_request_access", return_value="Allowed"), \
             mock.patch.object(self.mod, "_get_notifications",
                               side_effect=RuntimeError("poll boom")):
            self.mod._winsdk_modules["Listener"] = Listener
            with self.assertRaises(KeyboardInterrupt):
                self.mod._listener_loop()
        self.assertIn("poll boom", self.mod._subsystem_status["last_error"])

    def test_loop_listener_via_pascalcase_current(self):
        # .current raises but .Current works → PascalCase listener path, then
        # access denied to bail quickly.
        class _Listener:
            @property
            def current(self):
                raise RuntimeError("no lower")

            Current = object()

        with mock.patch.object(self.mod, "_probe_winsdk", return_value=True), \
             mock.patch.object(self.mod, "_request_access", return_value="Denied"):
            self.mod._winsdk_modules["Listener"] = _Listener()
            self.mod._listener_loop()
        self.assertEqual(self.mod._subsystem_status["listener_access"], "Denied")


# ─────────────────────────────────────────────────────────────────────────
# _start_listener + triage_status branch coverage.
# ─────────────────────────────────────────────────────────────────────────
class StartListenerAndStatusTests(_IsolatedTriageBase):
    def test_start_listener_spawns_thread_when_none(self):
        self.mod._listener_thread[0] = None
        created = {}

        class _FakeThread:
            def __init__(self, *a, **k):
                created["made"] = True
                self._alive = False

            def start(self):
                self._alive = True

            def is_alive(self):
                return self._alive

        with mock.patch.object(self.mod.threading, "Thread", _FakeThread):
            self.mod._start_listener()
        self.assertTrue(created.get("made"))
        self.assertIsNotNone(self.mod._listener_thread[0])

    def test_start_listener_noop_when_alive(self):
        alive = mock.MagicMock()
        alive.is_alive.return_value = True
        self.mod._listener_thread[0] = alive
        with mock.patch.object(self.mod.threading, "Thread") as T:
            self.mod._start_listener()
        T.assert_not_called()   # already running → no new thread.

    def test_triage_status_winsdk_unavailable(self):
        self.mod._subsystem_status["winsdk_available"] = False
        self.mod._rules = [{"id": "a", "action": "log"}]
        out = self.actions["triage_status"]("")
        self.assertIn("winsdk not installed", out)

    def test_triage_status_listening_and_focus_and_paused(self):
        self.mod._subsystem_status.update({
            "winsdk_available": True, "listener_access": "Allowed",
            "listening": True, "last_error": "some boom error",
        })
        self.mod._pause_flag[0] = True
        with self.mod._state_lock:
            self.mod._recent.append({"app": "Teams", "action": "read_aloud",
                                     "ts": time.time()})
        with mock.patch.object(self.mod, "_focus_mode_active", return_value=True):
            out = self.actions["triage_status"]("")
        self.assertIn("access=Allowed", out)
        self.assertIn("listening", out)
        self.assertIn("focus mode engaged", out)
        self.assertIn("paused", out)
        self.assertIn("last error", out)
        self.assertIn("last: Teams", out)

    def test_triage_status_idle_when_not_listening(self):
        self.mod._subsystem_status.update({
            "winsdk_available": True, "listener_access": "Allowed",
            "listening": False, "last_error": None,
        })
        with mock.patch.object(self.mod, "_focus_mode_active", return_value=False):
            out = self.actions["triage_status"]("")
        self.assertIn("idle", out)

    def test_resume_restarts_when_thread_dead(self):
        dead = mock.MagicMock()
        dead.is_alive.return_value = False
        self.mod._listener_thread[0] = dead
        with mock.patch.object(self.mod, "_start_listener") as start:
            out = self.actions["resume_notification_triage"]("")
        start.assert_called_once()
        self.assertIn("resumed", out.lower())

    def test_recent_summary_exception_in_int_defaults(self):
        # An arg that int() chokes on after .strip() → caught → n=5.
        with self.mod._state_lock:
            self.mod._recent.append({"app": "A", "title": "t", "action": "log",
                                     "ts": time.time()})
        out = self.actions["recent_notifications_summary"]("12.5xyz")
        self.assertIn("Recent notifications", out)


class NotificationTriageImportGuardTests(unittest.TestCase):
    def test_path_bootstrap_inserts_project_root(self):
        mod, _ = load_skill_isolated("notification_triage")
        path = mod.__file__
        proj = os.path.dirname(os.path.dirname(path))
        spec = importlib.util.spec_from_file_location("notif_triage_reexec", path)
        m = importlib.util.module_from_spec(spec)
        m.skill_utils = {}
        saved = list(sys.path)
        try:
            sys.path[:] = [p for p in sys.path
                           if os.path.abspath(p) != os.path.abspath(proj)]
            spec.loader.exec_module(m)
            self.assertIn(m._PROJECT_DIR, sys.path)
        finally:
            sys.path[:] = saved


if __name__ == "__main__":
    unittest.main()
