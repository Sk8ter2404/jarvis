"""Logic tests for skills/email_triage.py.

Unified Gmail + Outlook triage. The skill persists pending drafts / an inbox
index to real files at the project root, so setUp redirects ALL four path
constants into a fresh temp dir — nothing real is read or written. Backends
(ms_graph + the Gmail client) and the Haiku classifier are mocked, so no
network / API / LLM is ever hit. We focus on the parsing/normalisation
helpers, the numeric/ordinal handle resolver, the pending-draft state machine,
and the registered voice actions' content + credential-absent degradation.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


class EmailTriageTestBase(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("email_triage")
        # Redirect every on-disk path into a throwaway temp dir.
        self._tmp = tempfile.TemporaryDirectory()
        d = self._tmp.name
        self.mod.PENDING_DRAFTS_FILE = os.path.join(d, "pending.json")
        self.mod.INBOX_INDEX_FILE = os.path.join(d, "index.json")
        self.mod.GMAIL_TOKEN_FILE = os.path.join(d, "gmail_token.json")
        self.mod.GMAIL_CREDENTIALS_FILE = os.path.join(d, "gmail_creds.json")
        self.addCleanup(self._tmp.cleanup)


class SplitFromTests(EmailTriageTestBase):
    def test_name_and_address(self):
        self.assertEqual(self.mod._split_from("Jane Doe <jane@x.com>"),
                         ("Jane Doe", "jane@x.com"))

    def test_quoted_name(self):
        self.assertEqual(self.mod._split_from('"Doe, Jane" <jane@x.com>'),
                         ("Doe, Jane", "jane@x.com"))

    def test_bare_address(self):
        self.assertEqual(self.mod._split_from("bob@y.com"), ("", "bob@y.com"))

    def test_name_only(self):
        self.assertEqual(self.mod._split_from("Mailer Daemon"), ("Mailer Daemon", ""))

    def test_empty(self):
        self.assertEqual(self.mod._split_from(""), ("", ""))


class GmailHeaderTests(EmailTriageTestBase):
    def test_case_insensitive_lookup(self):
        headers = [{"name": "From", "value": "a@b.com"},
                   {"name": "Subject", "value": "Hi there"}]
        self.assertEqual(self.mod._gmail_header(headers, "subject"), "Hi there")
        self.assertEqual(self.mod._gmail_header(headers, "FROM"), "a@b.com")

    def test_missing_header_empty(self):
        self.assertEqual(self.mod._gmail_header([{"name": "X", "value": "y"}], "From"), "")

    def test_empty_headers(self):
        self.assertEqual(self.mod._gmail_header([], "From"), "")


class GmailBodyDecodeTests(EmailTriageTestBase):
    def test_decode_base64url_part(self):
        import base64
        text = "Hello, body!"
        data = base64.urlsafe_b64encode(text.encode()).decode().rstrip("=")
        part = {"body": {"data": data}}
        self.assertEqual(self.mod._gmail_decode_part(part), text)

    def test_decode_empty_part(self):
        self.assertEqual(self.mod._gmail_decode_part({"body": {}}), "")

    def test_extract_prefers_plain_over_html(self):
        import base64

        def enc(s):
            return base64.urlsafe_b64encode(s.encode()).decode().rstrip("=")
        payload = {
            "mimeType": "multipart/alternative",
            "parts": [
                {"mimeType": "text/plain", "body": {"data": enc("plain text wins")}},
                {"mimeType": "text/html",
                 "body": {"data": enc("<p>html version</p>")}},
            ],
        }
        plain, html = self.mod._gmail_extract_body(payload)
        self.assertEqual(plain, "plain text wins")
        self.assertIn("html version", html)


class ShapeGmailMessageTests(EmailTriageTestBase):
    def test_normalises_message(self):
        msg = {
            "id": "g1",
            "threadId": "t1",
            "snippet": "  preview text  ",
            "labelIds": ["UNREAD", "INBOX", "Label_42"],
            "payload": {"headers": [
                {"name": "From", "value": "Jane Doe <jane@x.com>"},
                {"name": "Subject", "value": "Re: lunch"},
                {"name": "Date", "value": "Mon, 01 Jun 2026 10:00:00 +0000"},
            ]},
        }
        shaped = self.mod._shape_gmail_message(msg)
        self.assertEqual(shaped["backend"], "gmail")
        self.assertEqual(shaped["from_name"], "Jane Doe")
        self.assertEqual(shaped["from_addr"], "jane@x.com")
        self.assertEqual(shaped["subject"], "Re: lunch")
        self.assertEqual(shaped["snippet"], "preview text")
        self.assertTrue(shaped["unread"])           # UNREAD label present
        self.assertIn("Label_42", shaped["categories"])

    def test_read_message(self):
        msg = {"id": "g2", "labelIds": ["INBOX"], "payload": {"headers": []}}
        self.assertFalse(self.mod._shape_gmail_message(msg)["unread"])


class ParseReceivedTests(EmailTriageTestBase):
    def test_iso8601(self):
        self.assertGreater(self.mod._parse_received("2026-06-01T10:00:00Z"), 0)

    def test_rfc2822(self):
        self.assertGreater(
            self.mod._parse_received("Mon, 01 Jun 2026 10:00:00 +0000"), 0)

    def test_unparseable_zero(self):
        self.assertEqual(self.mod._parse_received("whenever"), 0.0)

    def test_empty_zero(self):
        self.assertEqual(self.mod._parse_received(""), 0.0)


class ResolveHandleTests(EmailTriageTestBase):
    def _index(self):
        return [
            {"index": 1, "backend": "outlook", "id": "OUT1"},
            {"index": 2, "backend": "gmail", "id": "GM2"},
            {"index": 3, "backend": "gmail", "id": "GM3"},
        ]

    def test_latest_maps_to_first(self):
        with mock.patch.object(self.mod, "_load_inbox_index", return_value=self._index()):
            self.assertEqual(self.mod._resolve_handle("latest"),
                             {"backend": "outlook", "id": "OUT1"})

    def test_numeric_index(self):
        with mock.patch.object(self.mod, "_load_inbox_index", return_value=self._index()):
            self.assertEqual(self.mod._resolve_handle("2"),
                             {"backend": "gmail", "id": "GM2"})

    def test_ordinal_word(self):
        with mock.patch.object(self.mod, "_load_inbox_index", return_value=self._index()):
            self.assertEqual(self.mod._resolve_handle("third"),
                             {"backend": "gmail", "id": "GM3"})

    def test_out_of_range_falls_through_to_raw_id(self):
        with mock.patch.object(self.mod, "_load_inbox_index", return_value=self._index()):
            self.assertEqual(self.mod._resolve_handle("99"),
                             {"backend": "auto", "id": "99"})

    def test_raw_id_when_no_index(self):
        with mock.patch.object(self.mod, "_load_inbox_index", return_value=[]):
            self.assertEqual(self.mod._resolve_handle("AAMkRAWID"),
                             {"backend": "auto", "id": "AAMkRAWID"})

    def test_empty_token_none(self):
        self.assertIsNone(self.mod._resolve_handle(""))


class FormattingTests(EmailTriageTestBase):
    def test_format_msg_line_with_index(self):
        m = {"from_name": "Jane", "subject": "Lunch", "backend": "gmail"}
        line = self.mod._format_msg_line(m, 2)
        self.assertTrue(line.startswith("2. "))
        self.assertIn("Jane", line)
        self.assertIn("Lunch", line)
        self.assertIn("Gmail", line)

    def test_format_msg_line_fallbacks(self):
        line = self.mod._format_msg_line({"from_addr": "x@y.com"}, None)
        self.assertIn("x@y.com", line)
        self.assertIn("(no subject)", line)

    def test_format_for_speech_truncates(self):
        out = self.mod._format_for_speech("x" * 1000)
        self.assertLessEqual(len(out), self.mod.MAX_SPOKEN_BODY_CHARS)
        self.assertTrue(out.endswith("…"))

    def test_format_for_speech_short_unchanged(self):
        self.assertEqual(self.mod._format_for_speech("short"), "short")


class PendingDraftStateTests(EmailTriageTestBase):
    def test_set_get_clear_roundtrip(self):
        rec = {"backend": "gmail", "draft_id": "d1", "to": "a@b.com", "body": "hi"}
        self.mod._set_pending(rec)
        got = self.mod._get_pending()
        self.assertEqual(got["draft_id"], "d1")
        self.assertEqual(got["to"], "a@b.com")
        self.mod._clear_pending()
        self.assertIsNone(self.mod._get_pending())

    def test_set_pending_moves_previous_to_history(self):
        self.mod._set_pending({"draft_id": "first", "body": "1"})
        self.mod._set_pending({"draft_id": "second", "body": "2"})
        state = self.mod._load_pending()
        self.assertEqual(state["active"]["draft_id"], "second")
        self.assertTrue(any(h.get("draft_id") == "first" for h in state["history"]))

    def test_public_accessor_matches_private(self):
        self.mod._set_pending({"draft_id": "x"})
        self.assertEqual(self.mod.get_pending_draft(), self.mod._get_pending())

    def test_load_pending_missing_file_default(self):
        # File path points at a nonexistent temp file → default shape.
        state = self.mod._load_pending()
        self.assertEqual(state, {"active": None, "history": []})


class ListUnreadActionTests(EmailTriageTestBase):
    def test_no_backends_configured(self):
        with mock.patch.object(self.mod, "_outlook_configured", return_value=False), \
             mock.patch.object(self.mod, "is_gmail_available", return_value=False):
            out = self.actions["list_unread"]("")
        self.assertIn("No email backends configured", out)
        self.assertIn("--auth", out)

    def test_inbox_clear_when_configured_but_empty(self):
        with mock.patch.object(self.mod, "_outlook_configured", return_value=True), \
             mock.patch.object(self.mod, "is_gmail_available", return_value=False), \
             mock.patch.object(self.mod, "_outlook_list_unread", return_value=[]):
            out = self.actions["list_unread"]("")
        self.assertIn("Inbox is clear", out)

    def test_lists_and_numbers_messages(self):
        msgs = [
            {"backend": "outlook", "id": "o1", "from_name": "Alice",
             "subject": "Budget", "received": "2026-06-01T10:00:00Z"},
            {"backend": "gmail", "id": "g1", "from_name": "Bob",
             "subject": "Lunch", "received": "2026-06-01T09:00:00Z"},
        ]
        with mock.patch.object(self.mod, "_outlook_configured", return_value=True), \
             mock.patch.object(self.mod, "is_gmail_available", return_value=False), \
             mock.patch.object(self.mod, "_outlook_list_unread", return_value=msgs):
            out = self.actions["list_unread"]("")
        self.assertIn("2 unread", out)
        self.assertIn("1. Alice", out)   # newest first (10:00 before 09:00)
        self.assertIn("Budget", out)
        self.assertIn("Bob", out)
        # Inbox index must have been persisted for later numeric resolution.
        self.assertTrue(os.path.exists(self.mod.INBOX_INDEX_FILE))
        with open(self.mod.INBOX_INDEX_FILE, encoding="utf-8") as f:
            idx = json.load(f)
        self.assertEqual(idx["messages"][0]["id"], "o1")


class ReadThreadActionTests(EmailTriageTestBase):
    def test_reads_and_marks_read(self):
        thread = {"backend": "gmail", "id": "g1", "from_name": "Carol",
                  "subject": "Report", "body_text": "Here is the report."}
        with mock.patch.object(self.mod, "_resolve_handle",
                               return_value={"backend": "gmail", "id": "g1"}), \
             mock.patch.object(self.mod, "_get_thread", return_value=thread), \
             mock.patch.object(self.mod, "_mark_read", return_value=True) as mark:
            out = self.actions["read_thread"]("1")
        self.assertIn("Carol", out)
        self.assertIn("Report", out)
        self.assertIn("Here is the report.", out)
        mark.assert_called_once()   # read receipt set so it drops off unread

    def test_not_found(self):
        with mock.patch.object(self.mod, "_resolve_handle",
                               return_value={"backend": "auto", "id": "zzz"}), \
             mock.patch.object(self.mod, "_get_thread", return_value=None):
            out = self.actions["read_thread"]("zzz")
        self.assertIn("not found", out)


class DraftReplyActionTests(EmailTriageTestBase):
    def test_draft_generation_unavailable(self):
        thread = {"backend": "gmail", "id": "g1", "from_addr": "a@b.com"}
        with mock.patch.object(self.mod, "_resolve_handle",
                               return_value={"backend": "gmail", "id": "g1"}), \
             mock.patch.object(self.mod, "_get_thread", return_value=thread), \
             mock.patch.object(self.mod, "_generate_draft_reply", return_value=None):
            out = self.actions["draft_reply"]("1")
        self.assertIn("Draft generation unavailable", out)
        self.assertIn("ANTHROPIC_API_KEY", out)

    def test_draft_created_and_pending_set(self):
        thread = {"backend": "gmail", "id": "g1", "from_name": "Dana",
                  "from_addr": "dana@x.com", "subject": "Q3"}
        with mock.patch.object(self.mod, "_resolve_handle",
                               return_value={"backend": "gmail", "id": "g1"}), \
             mock.patch.object(self.mod, "_get_thread", return_value=thread), \
             mock.patch.object(self.mod, "_generate_draft_reply",
                               return_value="Sounds good. — B"), \
             mock.patch.object(self.mod, "_gmail_create_draft", return_value="draft-1"):
            out = self.actions["draft_reply"]("1 keep it short")
        self.assertIn("draft reply ready", out.lower())
        self.assertIn("Sounds good", out)
        pending = self.mod._get_pending()
        self.assertEqual(pending["draft_id"], "draft-1")
        self.assertEqual(pending["to"], "dana@x.com")


class ConfirmScrapEditActionTests(EmailTriageTestBase):
    def test_confirm_no_pending(self):
        self.assertIn("No draft waiting", self.actions["confirm_pending_draft"](""))

    def test_confirm_sends_and_clears(self):
        self.mod._set_pending({"backend": "gmail", "draft_id": "d1", "to": "a@b.com"})
        with mock.patch.object(self.mod, "_gmail_send_draft", return_value=True):
            out = self.actions["confirm_pending_draft"]("")
        self.assertIn("Sent the reply", out)
        self.assertIsNone(self.mod._get_pending())   # cleared after send

    def test_confirm_send_failure_keeps_pending(self):
        self.mod._set_pending({"backend": "gmail", "draft_id": "d1", "to": "a@b.com"})
        with mock.patch.object(self.mod, "_gmail_send_draft", return_value=False):
            out = self.actions["confirm_pending_draft"]("")
        self.assertIn("refused to send", out)
        self.assertIsNotNone(self.mod._get_pending())

    def test_scrap_clears(self):
        self.mod._set_pending({"backend": "gmail", "draft_id": "d1"})
        out = self.actions["scrap_pending_draft"]("")
        self.assertIn("Scrapped", out)
        self.assertIsNone(self.mod._get_pending())

    def test_scrap_nothing(self):
        self.assertIn("Nothing pending", self.actions["scrap_pending_draft"](""))

    def test_edit_requires_body(self):
        self.assertIn("new body text", self.actions["edit_pending_draft"](""))

    def test_edit_updates_local_and_backend(self):
        self.mod._set_pending({"backend": "gmail", "draft_id": "d1",
                               "message_id": "m1", "body": "old"})
        with mock.patch.object(self.mod, "_gmail_update_draft", return_value=True):
            out = self.actions["edit_pending_draft"]("new improved text")
        self.assertIn("Draft updated", out)
        self.assertEqual(self.mod._get_pending()["body"], "new improved text")


class CategorizeInboxActionTests(EmailTriageTestBase):
    def test_clear_inbox(self):
        with mock.patch.object(self.mod, "list_unread", return_value=[]):
            out = self.actions["categorize_inbox"]("")
        self.assertIn("Inbox is clear", out)

    def test_no_classifier_available(self):
        msgs = [{"backend": "gmail", "id": "g1", "subject": "x"}]
        with mock.patch.object(self.mod, "list_unread", return_value=msgs), \
             mock.patch.object(self.mod, "_triage_message", return_value=None):
            out = self.actions["categorize_inbox"]("")
        self.assertIn("Triaged nothing", out)
        self.assertIn("ANTHROPIC_API_KEY", out)

    def test_counts_verdicts_and_applies_categories(self):
        msgs = [{"backend": "gmail", "id": f"g{i}", "subject": str(i)}
                for i in range(3)]
        verdicts = iter(["urgent", "spam", "urgent"])
        with mock.patch.object(self.mod, "list_unread", return_value=msgs), \
             mock.patch.object(self.mod, "_triage_message",
                               side_effect=lambda m: next(verdicts)), \
             mock.patch.object(self.mod, "_apply_category", return_value=True) as applied:
            out = self.actions["categorize_inbox"]("")
        self.assertIn("Triaged 3 messages", out)
        self.assertIn("2 urgent", out)
        self.assertIn("1 spam", out)
        self.assertEqual(applied.call_count, 3)


class EmailBriefingActionTests(EmailTriageTestBase):
    def test_clear(self):
        with mock.patch.object(self.mod, "list_unread", return_value=[]):
            self.assertIn("Inbox is clear", self.actions["email_briefing"](""))

    def test_urgent_first_then_tail_counts(self):
        msgs = [
            {"backend": "gmail", "id": "g1", "from_name": "Boss", "subject": "Sign now"},
            {"backend": "gmail", "id": "g2", "from_name": "News", "subject": "Weekly"},
            {"backend": "gmail", "id": "g3", "from_name": "Ad", "subject": "Sale"},
        ]
        verdicts = {"g1": "urgent", "g2": "newsletter", "g3": "spam"}
        with mock.patch.object(self.mod, "list_unread", return_value=msgs), \
             mock.patch.object(self.mod, "_triage_message",
                               side_effect=lambda m: verdicts[m["id"]]):
            out = self.actions["email_briefing"]("")
        self.assertIn("urgent message from Boss", out)
        self.assertIn("Sign now", out)
        self.assertIn("1 newsletter", out)
        self.assertIn("1 spam", out)


class StatusActionTests(EmailTriageTestBase):
    def test_status_reports_backends_and_pending(self):
        self.mod._set_pending({"to": "a@b.com"})
        with mock.patch.object(self.mod, "_outlook_configured", return_value=True), \
             mock.patch.object(self.mod, "is_gmail_available", return_value=False):
            out = self.actions["email_triage_status"]("")
        self.assertIn("Outlook: configured", out)
        self.assertIn("Gmail: unavailable", out)
        self.assertIn("pending draft to a@b.com", out)
        self.assertTrue(out.rstrip().endswith("sir."))


if __name__ == "__main__":
    unittest.main()
