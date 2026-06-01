"""Logic tests for skills/notification_triage.py.

Covers the tunable rules engine: schema validation, priority-ordered rule
selection, regex matching (incl. focus-mode-bypass priority), the family of
content/timestamp/announce dedup keys, the JARVIS-voice speech formatter, the
LLM-verdict→action mapping, and the registered voice actions
(triage_status / add / remove / recent / pause / resume).

No monolith boot, no winsdk, no real listener thread (neutered by the harness).
Disk writes (rules file, dedup caches) are redirected to temp paths so nothing
real is touched.
"""
from __future__ import annotations

import json
import time
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


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

    # ── recent_notifications helper ──────────────────────────────────────
    def test_recent_notifications_returns_tail(self):
        with self.mod._state_lock:
            for i in range(5):
                self.mod._recent.append({"app": f"a{i}", "title": "t", "ts": i})
        out = self.mod.recent_notifications(2)
        self.assertEqual([r["app"] for r in out], ["a3", "a4"])

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


if __name__ == "__main__":
    unittest.main()
