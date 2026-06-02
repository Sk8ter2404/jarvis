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

import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated

def _spec_present(name: str) -> bool:
    # find_spec on a DOTTED name imports the parent package to locate the
    # submodule; when the parent is absent (bare CI runner) it RAISES
    # ModuleNotFoundError rather than returning None. Catch that so this probe
    # degrades to "absent" instead of erroring at module-import time.
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


_HAS_GOOGLE_API = all(
    _spec_present(n)
    for n in ("googleapiclient.discovery", "googleapiclient.errors",
              "google.oauth2.credentials", "google.auth.transport.requests",
              "google_auth_oauthlib.flow")
)


# ─────────────────────────────────────────────────────────────────────────
# Fakes for the two backends + the LLM clients the skill talks to.
#
# ISOLATION CONTRACT (see wave-1/2 lessons in the task brief):
#   • Nothing is written to sys.modules at import/module level. Every fake
#     (anthropic, bobert_companion, a stand-in skills.ms_graph) is installed
#     via ``mock.patch.dict(sys.modules, {...})`` INSIDE a with-block or via
#     addCleanup, so the real modules — and the *absence* of the optional
#     google libs — are restored after each test.
#   • The local-LLM fallback in _triage_message / _generate_draft_reply does
#     ``sys.modules.get("bobert_companion") or import_module("bobert_companion")``.
#     The real bobert_companion is a 14k-line monolith, so EVERY test that can
#     reach that fallback injects a fake bobert_companion first — the
#     ``sys.modules.get`` short-circuits before import_module ever runs.
#   • The module-level Gmail-service / unavailable-reason singletons are reset
#     in setUp so a service handle built by one test never leaks into another.
# ─────────────────────────────────────────────────────────────────────────


def _fake_anthropic(text="urgent", raise_on_construct=None,
                    raise_on_create=None):
    """A stand-in ``anthropic`` module. ``client.messages.create`` returns a
    response whose ``.content`` is a list of blocks each carrying ``.text``."""
    mod = types.ModuleType("anthropic")
    if raise_on_construct is not None:
        mod.Anthropic = mock.MagicMock(side_effect=raise_on_construct)
        return mod
    client = mock.MagicMock(name="anthropic.client")
    if raise_on_create is not None:
        client.messages.create.side_effect = raise_on_create
    else:
        block = types.SimpleNamespace(text=text)
        client.messages.create.return_value = types.SimpleNamespace(
            content=[block])
    mod.Anthropic = mock.MagicMock(return_value=client)
    return mod


def _fake_bobert(local_return="", local_raises=None):
    """A stand-in ``bobert_companion`` exposing only ``_call_local_llm`` — the
    Ollama fallback both LLM helpers consult when Claude is unavailable."""
    bc = types.ModuleType("bobert_companion")
    if local_raises is not None:
        bc._call_local_llm = mock.MagicMock(side_effect=local_raises)
    else:
        bc._call_local_llm = mock.MagicMock(return_value=local_return)
    return bc


class _GmailReq:
    """A chainable Gmail-API request stub: ``.execute()`` returns a preset
    payload or raises a preset exception. Mirrors the
    ``service.users().messages().get(...).execute()`` call shape."""
    def __init__(self, result=None, raises=None):
        self._result = result if result is not None else {}
        self._raises = raises

    def execute(self):
        if self._raises is not None:
            raise self._raises
        return self._result


class _FakeGmailService:
    """Minimal fake of the googleapiclient Gmail service. Each verb
    (messages.list/get/modify, drafts.create/send/update, labels.list/create)
    is recorded and returns a queued/keyed ``_GmailReq``. Construct with the
    behaviours a given test needs; unspecified verbs return empty dicts.

    ``users()`` returns self; the chained accessors below build the request.
    """
    def __init__(self, **behaviour):
        self.b = behaviour
        self.calls: list = []        # (verb, kwargs) audit trail

    # users() → self ; messages()/drafts()/labels() → small dispatchers.
    def users(self):
        return self

    def messages(self):
        return _GmailMessages(self)

    def drafts(self):
        return _GmailDrafts(self)

    def labels(self):
        return _GmailLabels(self)

    def _resolve(self, key, kwargs):
        self.calls.append((key, kwargs))
        spec = self.b.get(key)
        if isinstance(spec, Exception):
            return _GmailReq(raises=spec)
        if callable(spec):
            spec = spec(kwargs)
        return _GmailReq(result=spec if spec is not None else {})


class _GmailMessages:
    def __init__(self, svc):
        self.svc = svc

    def list(self, **kw):
        return self.svc._resolve("messages.list", kw)

    def get(self, **kw):
        return self.svc._resolve("messages.get", kw)

    def modify(self, **kw):
        return self.svc._resolve("messages.modify", kw)


class _GmailDrafts:
    def __init__(self, svc):
        self.svc = svc

    def create(self, **kw):
        return self.svc._resolve("drafts.create", kw)

    def send(self, **kw):
        return self.svc._resolve("drafts.send", kw)

    def update(self, **kw):
        return self.svc._resolve("drafts.update", kw)


class _GmailLabels:
    def __init__(self, svc):
        self.svc = svc

    def list(self, **kw):
        return self.svc._resolve("labels.list", kw)

    def create(self, **kw):
        return self.svc._resolve("labels.create", kw)


def _fake_ms_graph(**attrs):
    """A stand-in ``skills.ms_graph`` module. Only the names the email_triage
    wrappers call are defined; each defaults to a MagicMock the test can pin.
    ``is_configured`` defaults False so list_unread skips Outlook unless asked.
    """
    mod = types.ModuleType("skills.ms_graph")
    defaults = {
        "is_configured": lambda: False,
        "list_unread_messages": lambda top_n=8: [],
        "get_message_thread": lambda mid: None,
        "create_draft_reply": lambda mid, body, reply_all=False: None,
        "send_draft": lambda did: False,
        "update_draft_body": lambda did, body: False,
        "archive_message": lambda mid: False,
        "apply_category": lambda mid, cat: False,
        "mark_as_read": lambda mid, val: False,
    }
    defaults.update(attrs)
    for k, v in defaults.items():
        setattr(mod, k, v)
    return mod


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
        # Reset the module-level Gmail singletons so no service handle /
        # unavailable-reason string leaks between tests.
        self.mod._gmail_service_cache[0] = None
        self.mod._gmail_unavailable_reason[0] = ""
        self.addCleanup(self._reset_gmail_singletons)

    def _reset_gmail_singletons(self):
        self.mod._gmail_service_cache[0] = None
        self.mod._gmail_unavailable_reason[0] = ""

    # -- shared helpers ----------------------------------------------------
    def _patch_ms_graph(self, mod):
        """Install a fake skills.ms_graph for the duration of one test. The
        skill late-binds it via importlib.import_module('skills.ms_graph'), so
        patching sys.modules is enough; restored on cleanup."""
        cm = mock.patch.dict(sys.modules, {"skills.ms_graph": mod})
        cm.start()
        self.addCleanup(cm.stop)
        return mod


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


class EmailImportGuardTests(EmailTriageTestBase):
    def test_path_bootstrap_inserts_project_root(self):
        # Re-exec the source with the project root absent from sys.path so the
        # `if _PROJECT_DIR not in sys.path: sys.path.insert(...)` guard runs.
        # All heavy deps are already imported (cached) so re-exec is cheap.
        path = self.mod.__file__
        proj = os.path.dirname(os.path.dirname(path))
        spec = importlib.util.spec_from_file_location("email_triage_reexec", path)
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


# ─────────────────────────────────────────────────────────────────────────
# Outlook backend wrappers (skills/email_triage.py ~147-249). Each wraps an
# ms_graph call and degrades to []/None/False when the module is missing or
# raises. We drive them through a fake skills.ms_graph injected into sys.modules.
# ─────────────────────────────────────────────────────────────────────────
class OutlookBackendTests(EmailTriageTestBase):
    def test_ms_graph_returns_none_when_both_imports_fail(self):
        # Force importlib.import_module to fail for both 'skills.ms_graph' and
        # the bare 'ms_graph' fallback → _ms_graph() returns None.
        real_import = self.mod.importlib.import_module

        def _boom(name, *a, **k):
            if name in ("skills.ms_graph", "ms_graph"):
                raise ImportError("no ms_graph")
            return real_import(name, *a, **k)

        with mock.patch.object(self.mod.importlib, "import_module",
                               side_effect=_boom):
            self.assertIsNone(self.mod._ms_graph())
            # Every wrapper degrades cleanly with no module present.
            self.assertFalse(self.mod._outlook_configured())
            self.assertEqual(self.mod._outlook_list_unread(5), [])
            self.assertIsNone(self.mod._outlook_get_thread("x"))
            self.assertIsNone(self.mod._outlook_create_draft("x", "b"))
            self.assertFalse(self.mod._outlook_send_draft("d"))
            self.assertFalse(self.mod._outlook_update_draft("d", "b"))
            self.assertFalse(self.mod._outlook_archive("x"))
            self.assertFalse(self.mod._outlook_apply_category("x", "c"))
            self.assertFalse(self.mod._outlook_mark_read("x"))

    def test_ms_graph_falls_back_to_bare_name(self):
        # 'skills.ms_graph' missing but a bare 'ms_graph' importable → used.
        bare = self._fake_bare = types.ModuleType("ms_graph")
        bare.is_configured = lambda: True
        real_import = self.mod.importlib.import_module

        def _imp(name, *a, **k):
            if name == "skills.ms_graph":
                raise ImportError("nope")
            if name == "ms_graph":
                return bare
            return real_import(name, *a, **k)

        with mock.patch.object(self.mod.importlib, "import_module",
                               side_effect=_imp):
            self.assertIs(self.mod._ms_graph(), bare)
            self.assertTrue(self.mod._outlook_configured())

    def test_configured_true(self):
        self._patch_ms_graph(_fake_ms_graph(is_configured=lambda: True))
        self.assertTrue(self.mod._outlook_configured())

    def test_list_unread_passthrough(self):
        msgs = [{"backend": "outlook", "id": "o1"}]
        self._patch_ms_graph(_fake_ms_graph(
            list_unread_messages=lambda top_n=8: msgs))
        self.assertEqual(self.mod._outlook_list_unread(3), msgs)

    def test_list_unread_swallows_exception(self):
        def _raise(top_n=8):
            raise RuntimeError("graph down")
        self._patch_ms_graph(_fake_ms_graph(list_unread_messages=_raise))
        self.assertEqual(self.mod._outlook_list_unread(3), [])

    def test_get_thread_passthrough_and_error(self):
        self._patch_ms_graph(_fake_ms_graph(
            get_message_thread=lambda mid: {"id": mid, "body_text": "b"}))
        self.assertEqual(self.mod._outlook_get_thread("o1")["id"], "o1")

        def _raise(mid):
            raise RuntimeError("x")
        self._patch_ms_graph(_fake_ms_graph(get_message_thread=_raise))
        self.assertIsNone(self.mod._outlook_get_thread("o1"))

    def test_create_draft_returns_id(self):
        self._patch_ms_graph(_fake_ms_graph(
            create_draft_reply=lambda mid, body, reply_all=False: {"id": "D9"}))
        self.assertEqual(self.mod._outlook_create_draft("o1", "hi"), "D9")

    def test_create_draft_none_when_empty_or_error(self):
        # Backend returns falsy draft → None.
        self._patch_ms_graph(_fake_ms_graph(
            create_draft_reply=lambda mid, body, reply_all=False: None))
        self.assertIsNone(self.mod._outlook_create_draft("o1", "hi"))

        # Backend returns a draft dict with no id → None.
        self._patch_ms_graph(_fake_ms_graph(
            create_draft_reply=lambda mid, body, reply_all=False: {"no": "id"}))
        self.assertIsNone(self.mod._outlook_create_draft("o1", "hi"))

        # Backend raises → None.
        def _raise(mid, body, reply_all=False):
            raise RuntimeError("x")
        self._patch_ms_graph(_fake_ms_graph(create_draft_reply=_raise))
        self.assertIsNone(self.mod._outlook_create_draft("o1", "hi"))

    def test_send_update_archive_category_mark_read_truthy(self):
        self._patch_ms_graph(_fake_ms_graph(
            send_draft=lambda did: True,
            update_draft_body=lambda did, body: True,
            archive_message=lambda mid: True,
            apply_category=lambda mid, cat: True,
            mark_as_read=lambda mid, val: True))
        self.assertTrue(self.mod._outlook_send_draft("d"))
        self.assertTrue(self.mod._outlook_update_draft("d", "b"))
        self.assertTrue(self.mod._outlook_archive("x"))
        self.assertTrue(self.mod._outlook_apply_category("x", "JARVIS/Urgent"))
        self.assertTrue(self.mod._outlook_mark_read("x"))

    def test_write_wrappers_swallow_exceptions(self):
        def _boom(*a, **k):
            raise RuntimeError("graph error")
        self._patch_ms_graph(_fake_ms_graph(
            send_draft=_boom, update_draft_body=_boom, archive_message=_boom,
            apply_category=_boom, mark_as_read=_boom))
        self.assertFalse(self.mod._outlook_send_draft("d"))
        self.assertFalse(self.mod._outlook_update_draft("d", "b"))
        self.assertFalse(self.mod._outlook_archive("x"))
        self.assertFalse(self.mod._outlook_apply_category("x", "c"))
        self.assertFalse(self.mod._outlook_mark_read("x"))


# ─────────────────────────────────────────────────────────────────────────
# Gmail dependency probe + availability + credentials + service build
# (~263-373). The real google libs are NOT a CI dep, so we inject fakes into
# sys.modules and patch the leaf importers; absence is exercised by blocking.
# ─────────────────────────────────────────────────────────────────────────
class GmailDepsAvailabilityTests(EmailTriageTestBase):
    def test_probe_deps_missing_sets_reason(self):
        # Force the internal 'from googleapiclient... import ...' to fail by
        # making __import__ raise for any google* module.
        real_import = __import__

        def _imp(name, *a, **k):
            if name.startswith("google"):
                raise ImportError(f"no {name}")
            return real_import(name, *a, **k)

        with mock.patch("builtins.__import__", side_effect=_imp):
            build, *_ = self.mod._probe_gmail_deps()
        self.assertIsNone(build)
        self.assertIn("not installed", self.mod._gmail_unavailable_reason[0])

    @unittest.skipUnless(_HAS_GOOGLE_API,
                         "google-api-python-client / google-auth-oauthlib absent")
    def test_probe_deps_success_returns_all_five(self):
        # On a host where the google libs ARE installed, the real import body
        # runs and returns the five symbols (build, Credentials, flow, Request,
        # HttpError) — exercising the success path, not just the except.
        result = self.mod._probe_gmail_deps()
        self.assertEqual(len(result), 5)
        self.assertTrue(all(r is not None for r in result))

    def test_is_gmail_available_false_without_deps(self):
        with mock.patch.object(self.mod, "_probe_gmail_deps",
                               return_value=(None, None, None, None, None)):
            self.assertFalse(self.mod.is_gmail_available())

    def test_is_gmail_available_false_without_credentials_file(self):
        # Deps "present" but the redirected creds file does not exist.
        with mock.patch.object(self.mod, "_probe_gmail_deps",
                               return_value=(object(), object(), object(),
                                             object(), object())):
            self.assertFalse(self.mod.is_gmail_available())
        self.assertIn("gmail_credentials.json missing",
                      self.mod._gmail_unavailable_reason[0])

    def test_is_gmail_available_true_with_deps_and_creds(self):
        # Create the creds file so the existence check passes.
        with open(self.mod.GMAIL_CREDENTIALS_FILE, "w", encoding="utf-8") as f:
            f.write("{}")
        with mock.patch.object(self.mod, "_probe_gmail_deps",
                               return_value=(object(), object(), object(),
                                             object(), object())):
            self.assertTrue(self.mod.is_gmail_available())


class GmailCredentialsTests(EmailTriageTestBase):
    def _creds_class(self, creds_obj):
        """A fake Credentials class whose from_authorized_user_file returns the
        given object."""
        klass = mock.MagicMock(name="Credentials")
        klass.from_authorized_user_file.return_value = creds_obj
        return klass

    def test_no_credentials_class_returns_none(self):
        with mock.patch.object(self.mod, "_probe_gmail_deps",
                               return_value=(None, None, None, None, None)):
            self.assertIsNone(self.mod._load_gmail_credentials())

    def test_no_token_file_returns_none(self):
        # Token file path (redirected) doesn't exist.
        with mock.patch.object(self.mod, "_probe_gmail_deps",
                               return_value=(object(),
                                             self._creds_class(None),
                                             object(), object(), object())):
            self.assertIsNone(self.mod._load_gmail_credentials())

    def test_valid_token_returned_directly(self):
        with open(self.mod.GMAIL_TOKEN_FILE, "w", encoding="utf-8") as f:
            f.write("{}")
        creds = types.SimpleNamespace(valid=True, expired=False,
                                      refresh_token=None)
        with mock.patch.object(self.mod, "_probe_gmail_deps",
                               return_value=(object(), self._creds_class(creds),
                                             object(), object(), object())):
            self.assertIs(self.mod._load_gmail_credentials(), creds)

    def test_unreadable_token_returns_none(self):
        with open(self.mod.GMAIL_TOKEN_FILE, "w", encoding="utf-8") as f:
            f.write("{}")
        klass = mock.MagicMock(name="Credentials")
        klass.from_authorized_user_file.side_effect = ValueError("corrupt")
        with mock.patch.object(self.mod, "_probe_gmail_deps",
                               return_value=(object(), klass, object(),
                                             object(), object())):
            self.assertIsNone(self.mod._load_gmail_credentials())

    def test_expired_token_refreshes_and_persists(self):
        with open(self.mod.GMAIL_TOKEN_FILE, "w", encoding="utf-8") as f:
            f.write("{}")
        refreshed = {"refreshed": True}
        creds = mock.MagicMock(name="creds")
        creds.valid = False
        creds.expired = True
        creds.refresh_token = "rt"
        creds.to_json.return_value = json.dumps(refreshed)
        request_cls = mock.MagicMock(name="Request")
        with mock.patch.object(self.mod, "_probe_gmail_deps",
                               return_value=(object(), self._creds_class(creds),
                                             object(), request_cls, object())):
            out = self.mod._load_gmail_credentials()
        self.assertIs(out, creds)
        creds.refresh.assert_called_once()
        # Token re-persisted to the redirected path.
        with open(self.mod.GMAIL_TOKEN_FILE, encoding="utf-8") as f:
            self.assertEqual(json.load(f), refreshed)

    def test_expired_refresh_token_save_failure_still_returns_creds(self):
        # Refresh succeeds but persisting the rotated token fails → the creds
        # are still returned (save is best-effort).
        with open(self.mod.GMAIL_TOKEN_FILE, "w", encoding="utf-8") as f:
            f.write("{}")
        creds = mock.MagicMock(name="creds")
        creds.valid = False
        creds.expired = True
        creds.refresh_token = "rt"
        creds.to_json.return_value = "{}"
        with mock.patch.object(self.mod, "_probe_gmail_deps",
                               return_value=(object(), self._creds_class(creds),
                                             object(), mock.MagicMock(),
                                             object())), \
             mock.patch.object(self.mod, "_atomic_write_json",
                               side_effect=OSError("disk full")):
            out = self.mod._load_gmail_credentials()
        self.assertIs(out, creds)

    def test_expired_refresh_failure_returns_none(self):
        with open(self.mod.GMAIL_TOKEN_FILE, "w", encoding="utf-8") as f:
            f.write("{}")
        creds = mock.MagicMock(name="creds")
        creds.valid = False
        creds.expired = True
        creds.refresh_token = "rt"
        creds.refresh.side_effect = RuntimeError("network")
        with mock.patch.object(self.mod, "_probe_gmail_deps",
                               return_value=(object(), self._creds_class(creds),
                                             object(), mock.MagicMock(),
                                             object())):
            self.assertIsNone(self.mod._load_gmail_credentials())

    def test_expired_without_refresh_token_returns_none(self):
        with open(self.mod.GMAIL_TOKEN_FILE, "w", encoding="utf-8") as f:
            f.write("{}")
        creds = types.SimpleNamespace(valid=False, expired=True,
                                      refresh_token=None)
        with mock.patch.object(self.mod, "_probe_gmail_deps",
                               return_value=(object(), self._creds_class(creds),
                                             object(), object(), object())):
            self.assertIsNone(self.mod._load_gmail_credentials())


class GmailServiceBuildTests(EmailTriageTestBase):
    def test_service_none_when_unavailable(self):
        with mock.patch.object(self.mod, "is_gmail_available",
                               return_value=False):
            self.assertIsNone(self.mod._gmail_service())

    def test_service_uses_cached_handle_when_creds_valid(self):
        sentinel = object()
        creds = types.SimpleNamespace(valid=True)
        self.mod._gmail_service_cache[0] = (sentinel, creds)
        with mock.patch.object(self.mod, "is_gmail_available",
                               return_value=True):
            self.assertIs(self.mod._gmail_service(), sentinel)

    def test_service_rebuilds_when_cache_invalid(self):
        stale = object()
        fresh = object()
        creds = types.SimpleNamespace(valid=True)
        # Cache holds an invalid-creds handle → must rebuild.
        self.mod._gmail_service_cache[0] = (stale,
                                            types.SimpleNamespace(valid=False))
        build = mock.MagicMock(return_value=fresh)
        with mock.patch.object(self.mod, "is_gmail_available",
                               return_value=True), \
             mock.patch.object(self.mod, "_probe_gmail_deps",
                               return_value=(build, None, None, None, None)), \
             mock.patch.object(self.mod, "_load_gmail_credentials",
                               return_value=creds):
            out = self.mod._gmail_service()
        self.assertIs(out, fresh)
        self.assertEqual(self.mod._gmail_service_cache[0], (fresh, creds))

    def test_service_none_when_no_credentials(self):
        with mock.patch.object(self.mod, "is_gmail_available",
                               return_value=True), \
             mock.patch.object(self.mod, "_probe_gmail_deps",
                               return_value=(mock.MagicMock(), None, None,
                                             None, None)), \
             mock.patch.object(self.mod, "_load_gmail_credentials",
                               return_value=None):
            self.assertIsNone(self.mod._gmail_service())

    def test_service_build_failure_returns_none(self):
        build = mock.MagicMock(side_effect=RuntimeError("build boom"))
        with mock.patch.object(self.mod, "is_gmail_available",
                               return_value=True), \
             mock.patch.object(self.mod, "_probe_gmail_deps",
                               return_value=(build, None, None, None, None)), \
             mock.patch.object(self.mod, "_load_gmail_credentials",
                               return_value=types.SimpleNamespace(valid=True)):
            self.assertIsNone(self.mod._gmail_service())


class AuthenticateGmailTests(EmailTriageTestBase):
    def test_auth_no_oauth_libs(self):
        with mock.patch.object(self.mod, "_probe_gmail_deps",
                               return_value=(None, None, None, None, None)):
            self.assertFalse(self.mod.authenticate_gmail())

    def test_auth_no_credentials_file(self):
        # InstalledAppFlow "present" but the creds file is absent.
        with mock.patch.object(self.mod, "_probe_gmail_deps",
                               return_value=(object(), object(), object(),
                                             object(), object())):
            self.assertFalse(self.mod.authenticate_gmail())

    def test_auth_success_writes_token(self):
        with open(self.mod.GMAIL_CREDENTIALS_FILE, "w", encoding="utf-8") as f:
            f.write("{}")
        flow = mock.MagicMock(name="flow")
        creds = mock.MagicMock(name="creds")
        creds.to_json.return_value = json.dumps({"token": "abc"})
        flow.run_local_server.return_value = creds
        flow_cls = mock.MagicMock(name="InstalledAppFlow")
        flow_cls.from_client_secrets_file.return_value = flow
        with mock.patch.object(self.mod, "_probe_gmail_deps",
                               return_value=(object(), object(), flow_cls,
                                             object(), object())):
            self.assertTrue(self.mod.authenticate_gmail())
        with open(self.mod.GMAIL_TOKEN_FILE, encoding="utf-8") as f:
            self.assertEqual(json.load(f), {"token": "abc"})


# ─────────────────────────────────────────────────────────────────────────
# Gmail body extraction edge paths (~397-431) not covered by the existing
# happy-path body tests.
# ─────────────────────────────────────────────────────────────────────────
class GmailBodyEdgeTests(EmailTriageTestBase):
    def test_decode_bad_base64_returns_empty(self):
        # Non-ASCII data → the internal .encode("ascii") raises → "" (the
        # decode helper swallows any error and degrades to empty text).
        self.assertEqual(
            self.mod._gmail_decode_part({"body": {"data": "café—not-b64"}}),
            "")

    def test_decode_none_part(self):
        self.assertEqual(self.mod._gmail_decode_part(None), "")

    def test_extract_empty_payload(self):
        self.assertEqual(self.mod._gmail_extract_body({}), ("", ""))

    def test_extract_skips_empty_body_subpart(self):
        import base64

        def enc(s):
            return base64.urlsafe_b64encode(s.encode()).decode().rstrip("=")
        # First sub-part decodes to empty (no data) → skipped; second wins.
        payload = {
            "mimeType": "multipart/mixed",
            "parts": [
                {"mimeType": "text/plain", "body": {}},               # empty
                {"mimeType": "text/plain", "body": {"data": enc("real")}},
            ],
        }
        plain, _ = self.mod._gmail_extract_body(payload)
        self.assertEqual(plain, "real")

    def test_extract_html_only_falls_back_to_strip(self):
        import base64

        def enc(s):
            return base64.urlsafe_b64encode(s.encode()).decode().rstrip("=")
        payload = {"mimeType": "text/html",
                   "body": {"data": enc("<p>only&nbsp;html</p>")}}
        # ms_graph._strip_html is consulted to derive plain text from html.
        self._patch_ms_graph(_fake_ms_graph(
            _strip_html=lambda h: "only html"))
        plain, html = self.mod._gmail_extract_body(payload)
        self.assertEqual(plain, "only html")
        self.assertIn("only", html)

    def test_extract_html_only_no_ms_graph_keeps_plain_empty(self):
        import base64

        def enc(s):
            return base64.urlsafe_b64encode(s.encode()).decode().rstrip("=")
        payload = {"mimeType": "text/html", "body": {"data": enc("<p>x</p>")}}
        # ms_graph without _strip_html → plain stays empty, html preserved.
        self._patch_ms_graph(_fake_ms_graph())   # no _strip_html attr
        plain, html = self.mod._gmail_extract_body(payload)
        self.assertEqual(plain, "")
        self.assertIn("x", html)


# ─────────────────────────────────────────────────────────────────────────
# Gmail backend operations (~470-668) driven through a fake service handle.
# We patch _gmail_service to return _FakeGmailService so no google lib / network
# is touched.
# ─────────────────────────────────────────────────────────────────────────
class GmailListUnreadTests(EmailTriageTestBase):
    def test_list_none_when_no_service(self):
        with mock.patch.object(self.mod, "_gmail_service", return_value=None):
            self.assertEqual(self.mod._gmail_list_unread(5), [])

    def test_list_returns_shaped_messages(self):
        def _get(kw):
            mid = kw["id"]
            return {"id": mid, "threadId": "t", "labelIds": ["UNREAD", "INBOX"],
                    "snippet": "hi",
                    "payload": {"headers": [
                        {"name": "From", "value": "Alice <alice@example.com>"},
                        {"name": "Subject", "value": "Hello"}]}}
        svc = _FakeGmailService(
            **{"messages.list": {"messages": [{"id": "g1"}, {"id": "g2"}]},
               "messages.get": _get})
        with mock.patch.object(self.mod, "_gmail_service", return_value=svc):
            out = self.mod._gmail_list_unread(5)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["from_addr"], "alice@example.com")
        self.assertEqual(out[0]["backend"], "gmail")

    def test_list_call_raises_returns_empty(self):
        svc = _FakeGmailService(
            **{"messages.list": RuntimeError("api 500")})
        with mock.patch.object(self.mod, "_gmail_service", return_value=svc):
            self.assertEqual(self.mod._gmail_list_unread(5), [])

    def test_list_skips_messages_whose_get_fails(self):
        svc = _FakeGmailService(
            **{"messages.list": {"messages": [{"id": "g1"}]},
               "messages.get": RuntimeError("get failed")})
        with mock.patch.object(self.mod, "_gmail_service", return_value=svc):
            self.assertEqual(self.mod._gmail_list_unread(5), [])


class GmailGetThreadTests(EmailTriageTestBase):
    def test_get_none_when_no_service(self):
        with mock.patch.object(self.mod, "_gmail_service", return_value=None):
            self.assertIsNone(self.mod._gmail_get_thread("g1"))

    def test_get_shapes_and_extracts_body(self):
        import base64
        data = base64.urlsafe_b64encode(b"Body here").decode().rstrip("=")
        full = {"id": "g1", "threadId": "t1", "labelIds": ["INBOX"],
                "snippet": "snip",
                "payload": {"mimeType": "text/plain", "body": {"data": data},
                            "headers": [
                                {"name": "From",
                                 "value": "Bob <bob@example.com>"},
                                {"name": "Subject", "value": "Re: x"}]}}
        svc = _FakeGmailService(**{"messages.get": full})
        with mock.patch.object(self.mod, "_gmail_service", return_value=svc):
            out = self.mod._gmail_get_thread("g1")
        self.assertEqual(out["body_text"], "Body here")
        self.assertEqual(out["from_addr"], "bob@example.com")

    def test_get_raises_returns_none(self):
        svc = _FakeGmailService(**{"messages.get": RuntimeError("boom")})
        with mock.patch.object(self.mod, "_gmail_service", return_value=svc):
            self.assertIsNone(self.mod._gmail_get_thread("g1"))


class GmailCreateDraftTests(EmailTriageTestBase):
    def _original(self, addr="dana@example.com", subject="Q3", thread="t1"):
        return {"backend": "gmail", "id": "g1", "from_addr": addr,
                "subject": subject, "thread_id": thread}

    def test_create_none_when_no_service(self):
        with mock.patch.object(self.mod, "_gmail_service", return_value=None):
            self.assertIsNone(self.mod._gmail_create_draft("g1", "body"))

    def test_create_none_when_original_missing(self):
        svc = _FakeGmailService()
        with mock.patch.object(self.mod, "_gmail_service", return_value=svc), \
             mock.patch.object(self.mod, "_gmail_get_thread", return_value=None):
            self.assertIsNone(self.mod._gmail_create_draft("g1", "body"))

    def test_create_none_when_no_to_addr(self):
        svc = _FakeGmailService()
        with mock.patch.object(self.mod, "_gmail_service", return_value=svc), \
             mock.patch.object(self.mod, "_gmail_get_thread",
                               return_value=self._original(addr="")):
            self.assertIsNone(self.mod._gmail_create_draft("g1", "body"))

    def test_create_success_returns_id_and_adds_re_prefix(self):
        svc = _FakeGmailService(**{"drafts.create": {"id": "draft-9"}})
        with mock.patch.object(self.mod, "_gmail_service", return_value=svc), \
             mock.patch.object(self.mod, "_gmail_get_thread",
                               return_value=self._original(subject="Status")):
            out = self.mod._gmail_create_draft("g1", "Reply body")
        self.assertEqual(out, "draft-9")
        # The raw MIME (base64url) was passed with a Re: subject.
        verb, kw = svc.calls[-1]
        self.assertEqual(verb, "drafts.create")
        import base64
        raw = base64.urlsafe_b64decode(
            kw["body"]["message"]["raw"].encode() + b"==").decode("utf-8",
                                                                   "replace")
        self.assertIn("Subject: Re: Status", raw)

    def test_create_keeps_existing_re_prefix(self):
        svc = _FakeGmailService(**{"drafts.create": {"id": "d1"}})
        with mock.patch.object(self.mod, "_gmail_service", return_value=svc), \
             mock.patch.object(self.mod, "_gmail_get_thread",
                               return_value=self._original(subject="Re: keep")):
            self.mod._gmail_create_draft("g1", "b")
        import base64
        raw = base64.urlsafe_b64decode(
            svc.calls[-1][1]["body"]["message"]["raw"].encode() + b"=="
        ).decode("utf-8", "replace")
        self.assertIn("Subject: Re: keep", raw)
        self.assertNotIn("Re: Re:", raw)

    def test_create_api_error_returns_none(self):
        svc = _FakeGmailService(**{"drafts.create": RuntimeError("quota")})
        with mock.patch.object(self.mod, "_gmail_service", return_value=svc), \
             mock.patch.object(self.mod, "_gmail_get_thread",
                               return_value=self._original()):
            self.assertIsNone(self.mod._gmail_create_draft("g1", "b"))


class GmailSendDraftTests(EmailTriageTestBase):
    def test_send_false_when_no_service(self):
        with mock.patch.object(self.mod, "_gmail_service", return_value=None):
            self.assertFalse(self.mod._gmail_send_draft("d1"))

    def test_send_success(self):
        svc = _FakeGmailService(**{"drafts.send": {"id": "d1"}})
        with mock.patch.object(self.mod, "_gmail_service", return_value=svc):
            self.assertTrue(self.mod._gmail_send_draft("d1"))

    def test_send_error_false(self):
        svc = _FakeGmailService(**{"drafts.send": RuntimeError("nope")})
        with mock.patch.object(self.mod, "_gmail_service", return_value=svc):
            self.assertFalse(self.mod._gmail_send_draft("d1"))


class GmailUpdateDraftTests(EmailTriageTestBase):
    def _original(self, addr="x@example.com", subject="S", thread="t"):
        return {"from_addr": addr, "subject": subject, "thread_id": thread}

    def test_update_false_when_no_service(self):
        with mock.patch.object(self.mod, "_gmail_service", return_value=None):
            self.assertFalse(self.mod._gmail_update_draft("d1", "b", "m1"))

    def test_update_false_when_original_missing(self):
        svc = _FakeGmailService()
        with mock.patch.object(self.mod, "_gmail_service", return_value=svc), \
             mock.patch.object(self.mod, "_gmail_get_thread", return_value=None):
            self.assertFalse(self.mod._gmail_update_draft("d1", "b", "m1"))

    def test_update_false_when_no_to_addr(self):
        svc = _FakeGmailService()
        with mock.patch.object(self.mod, "_gmail_service", return_value=svc), \
             mock.patch.object(self.mod, "_gmail_get_thread",
                               return_value=self._original(addr="")):
            self.assertFalse(self.mod._gmail_update_draft("d1", "b", "m1"))

    def test_update_success(self):
        svc = _FakeGmailService(**{"drafts.update": {"id": "d1"}})
        with mock.patch.object(self.mod, "_gmail_service", return_value=svc), \
             mock.patch.object(self.mod, "_gmail_get_thread",
                               return_value=self._original()):
            self.assertTrue(self.mod._gmail_update_draft("d1", "new", "m1"))

    def test_update_api_error_false(self):
        svc = _FakeGmailService(**{"drafts.update": RuntimeError("x")})
        with mock.patch.object(self.mod, "_gmail_service", return_value=svc), \
             mock.patch.object(self.mod, "_gmail_get_thread",
                               return_value=self._original()):
            self.assertFalse(self.mod._gmail_update_draft("d1", "new", "m1"))


class GmailArchiveMarkReadTests(EmailTriageTestBase):
    def test_archive_false_no_service(self):
        with mock.patch.object(self.mod, "_gmail_service", return_value=None):
            self.assertFalse(self.mod._gmail_archive("g1"))

    def test_archive_success_removes_inbox_label(self):
        svc = _FakeGmailService(**{"messages.modify": {"id": "g1"}})
        with mock.patch.object(self.mod, "_gmail_service", return_value=svc):
            self.assertTrue(self.mod._gmail_archive("g1"))
        self.assertEqual(svc.calls[-1][1]["body"]["removeLabelIds"], ["INBOX"])

    def test_archive_error_false(self):
        svc = _FakeGmailService(**{"messages.modify": RuntimeError("x")})
        with mock.patch.object(self.mod, "_gmail_service", return_value=svc):
            self.assertFalse(self.mod._gmail_archive("g1"))

    def test_mark_read_false_no_service(self):
        with mock.patch.object(self.mod, "_gmail_service", return_value=None):
            self.assertFalse(self.mod._gmail_mark_read("g1"))

    def test_mark_read_success_removes_unread_label(self):
        svc = _FakeGmailService(**{"messages.modify": {"id": "g1"}})
        with mock.patch.object(self.mod, "_gmail_service", return_value=svc):
            self.assertTrue(self.mod._gmail_mark_read("g1"))
        self.assertEqual(svc.calls[-1][1]["body"]["removeLabelIds"], ["UNREAD"])

    def test_mark_read_error_false(self):
        svc = _FakeGmailService(**{"messages.modify": RuntimeError("x")})
        with mock.patch.object(self.mod, "_gmail_service", return_value=svc):
            self.assertFalse(self.mod._gmail_mark_read("g1"))


class GmailApplyLabelTests(EmailTriageTestBase):
    def test_label_false_no_service(self):
        with mock.patch.object(self.mod, "_gmail_service", return_value=None):
            self.assertFalse(self.mod._gmail_apply_label("g1", "JARVIS/Urgent"))

    def test_label_list_error_false(self):
        svc = _FakeGmailService(**{"labels.list": RuntimeError("x")})
        with mock.patch.object(self.mod, "_gmail_service", return_value=svc):
            self.assertFalse(self.mod._gmail_apply_label("g1", "JARVIS/Urgent"))

    def test_label_reuses_existing(self):
        svc = _FakeGmailService(
            **{"labels.list": {"labels": [{"id": "L1",
                                           "name": "JARVIS/Urgent"}]},
               "messages.modify": {"id": "g1"}})
        with mock.patch.object(self.mod, "_gmail_service", return_value=svc):
            self.assertTrue(self.mod._gmail_apply_label("g1", "jarvis/urgent"))
        # Modify used the existing label id; no create call was made.
        self.assertEqual(svc.calls[-1][1]["body"]["addLabelIds"], ["L1"])
        self.assertNotIn("labels.create", [c[0] for c in svc.calls])

    def test_label_creates_when_absent(self):
        svc = _FakeGmailService(
            **{"labels.list": {"labels": []},
               "labels.create": {"id": "NEW"},
               "messages.modify": {"id": "g1"}})
        with mock.patch.object(self.mod, "_gmail_service", return_value=svc):
            self.assertTrue(self.mod._gmail_apply_label("g1", "JARVIS/FYI"))
        self.assertIn("labels.create", [c[0] for c in svc.calls])
        self.assertEqual(svc.calls[-1][1]["body"]["addLabelIds"], ["NEW"])

    def test_label_create_error_false(self):
        svc = _FakeGmailService(
            **{"labels.list": {"labels": []},
               "labels.create": RuntimeError("denied")})
        with mock.patch.object(self.mod, "_gmail_service", return_value=svc):
            self.assertFalse(self.mod._gmail_apply_label("g1", "JARVIS/FYI"))

    def test_label_create_returns_no_id_false(self):
        svc = _FakeGmailService(
            **{"labels.list": {"labels": []},
               "labels.create": {}})   # created label has no id
        with mock.patch.object(self.mod, "_gmail_service", return_value=svc):
            self.assertFalse(self.mod._gmail_apply_label("g1", "JARVIS/FYI"))

    def test_label_modify_error_false(self):
        svc = _FakeGmailService(
            **{"labels.list": {"labels": [{"id": "L1", "name": "JARVIS/Spam"}]},
               "messages.modify": RuntimeError("x")})
        with mock.patch.object(self.mod, "_gmail_service", return_value=svc):
            self.assertFalse(self.mod._gmail_apply_label("g1", "JARVIS/Spam"))


# ─────────────────────────────────────────────────────────────────────────
# Cross-backend dispatchers: _get_thread / _archive / _mark_read /
# _apply_category (~769-818).
# ─────────────────────────────────────────────────────────────────────────
class CrossBackendDispatchTests(EmailTriageTestBase):
    def test_get_thread_empty_id_none(self):
        self.assertIsNone(self.mod._get_thread({"backend": "gmail", "id": ""}))

    def test_get_thread_outlook(self):
        with mock.patch.object(self.mod, "_outlook_get_thread",
                               return_value={"id": "o1"}) as og:
            out = self.mod._get_thread({"backend": "outlook", "id": "o1"})
        self.assertEqual(out["id"], "o1")
        og.assert_called_once_with("o1")

    def test_get_thread_gmail(self):
        with mock.patch.object(self.mod, "_gmail_get_thread",
                               return_value={"id": "g1"}):
            out = self.mod._get_thread({"backend": "gmail", "id": "g1"})
        self.assertEqual(out["id"], "g1")

    def test_get_thread_auto_tries_outlook_then_gmail(self):
        with mock.patch.object(self.mod, "_outlook_get_thread",
                               return_value=None), \
             mock.patch.object(self.mod, "_gmail_get_thread",
                               return_value={"id": "g1"}):
            out = self.mod._get_thread({"backend": "auto", "id": "x"})
        self.assertEqual(out["id"], "g1")

    def test_archive_dispatch(self):
        self.assertFalse(self.mod._archive({"backend": "gmail", "id": ""}))
        with mock.patch.object(self.mod, "_outlook_archive", return_value=True):
            self.assertTrue(self.mod._archive({"backend": "outlook", "id": "o"}))
        with mock.patch.object(self.mod, "_gmail_archive", return_value=True):
            self.assertTrue(self.mod._archive({"backend": "gmail", "id": "g"}))
        with mock.patch.object(self.mod, "_outlook_archive",
                               return_value=False), \
             mock.patch.object(self.mod, "_gmail_archive", return_value=True):
            self.assertTrue(self.mod._archive({"backend": "auto", "id": "z"}))

    def test_mark_read_dispatch(self):
        self.assertFalse(self.mod._mark_read({"backend": "gmail", "id": ""}))
        with mock.patch.object(self.mod, "_outlook_mark_read",
                               return_value=True):
            self.assertTrue(
                self.mod._mark_read({"backend": "outlook", "id": "o"}))
        with mock.patch.object(self.mod, "_gmail_mark_read", return_value=True):
            self.assertTrue(self.mod._mark_read({"backend": "gmail", "id": "g"}))
        with mock.patch.object(self.mod, "_outlook_mark_read",
                               return_value=False), \
             mock.patch.object(self.mod, "_gmail_mark_read", return_value=True):
            self.assertTrue(self.mod._mark_read({"backend": "auto", "id": "z"}))

    def test_apply_category_rejects_unknown_verdict(self):
        self.assertFalse(self.mod._apply_category(
            {"backend": "gmail", "id": "g"}, "bogus"))

    def test_apply_category_empty_id(self):
        self.assertFalse(self.mod._apply_category(
            {"backend": "gmail", "id": ""}, "urgent"))

    def test_apply_category_outlook_maps_label(self):
        with mock.patch.object(self.mod, "_outlook_apply_category",
                               return_value=True) as oc:
            self.assertTrue(self.mod._apply_category(
                {"backend": "outlook", "id": "o"}, "urgent"))
        oc.assert_called_once_with("o", self.mod.CATEGORY_LABELS["urgent"])

    def test_apply_category_gmail_maps_label(self):
        with mock.patch.object(self.mod, "_gmail_apply_label",
                               return_value=True) as gl:
            self.assertTrue(self.mod._apply_category(
                {"backend": "gmail", "id": "g"}, "newsletter"))
        gl.assert_called_once_with("g", self.mod.CATEGORY_LABELS["newsletter"])

    def test_apply_category_auto_tries_both(self):
        with mock.patch.object(self.mod, "_outlook_apply_category",
                               return_value=False), \
             mock.patch.object(self.mod, "_gmail_apply_label",
                               return_value=True):
            self.assertTrue(self.mod._apply_category(
                {"backend": "auto", "id": "z"}, "spam"))


# ─────────────────────────────────────────────────────────────────────────
# list_unread aggregation across both backends (~673-685) + _save/_load index.
# ─────────────────────────────────────────────────────────────────────────
class ListUnreadAggregateTests(EmailTriageTestBase):
    def test_aggregates_and_sorts_then_truncates(self):
        out_msgs = [{"backend": "outlook", "id": "o1",
                     "received": "2026-06-01T08:00:00Z"}]
        gm_msgs = [{"backend": "gmail", "id": "g1",
                    "received": "2026-06-01T10:00:00Z"}]
        with mock.patch.object(self.mod, "_outlook_configured",
                               return_value=True), \
             mock.patch.object(self.mod, "_outlook_list_unread",
                               return_value=out_msgs), \
             mock.patch.object(self.mod, "is_gmail_available",
                               return_value=True), \
             mock.patch.object(self.mod, "_gmail_list_unread",
                               return_value=gm_msgs):
            out = self.mod.list_unread(top_n=5)
        # Sorted newest-first: gmail (10:00) before outlook (08:00).
        self.assertEqual([m["id"] for m in out], ["g1", "o1"])

    def test_top_n_clamped_and_only_requested_backend(self):
        gm_msgs = [{"backend": "gmail", "id": f"g{i}", "received": ""}
                   for i in range(5)]
        with mock.patch.object(self.mod, "_outlook_configured",
                               return_value=True) as oc, \
             mock.patch.object(self.mod, "is_gmail_available",
                               return_value=True), \
             mock.patch.object(self.mod, "_gmail_list_unread",
                               return_value=gm_msgs):
            out = self.mod.list_unread(top_n=2, backend="gmail")
        self.assertEqual(len(out), 2)
        oc.assert_not_called()   # outlook skipped when backend == 'gmail'

    def test_save_inbox_index_handles_write_error(self):
        # makedirs failing must be swallowed (best-effort persistence).
        with mock.patch.object(self.mod.os, "makedirs",
                               side_effect=OSError("ro fs")):
            self.mod._save_inbox_index([{"backend": "gmail", "id": "g1"}])
        # No exception propagated; nothing asserted beyond that.

    def test_load_inbox_index_missing_file_returns_empty(self):
        # Redirected path doesn't exist → early [] (no read attempted).
        self.assertEqual(self.mod._load_inbox_index(), [])

    def test_load_inbox_index_corrupt_file_returns_empty(self):
        with open(self.mod.INBOX_INDEX_FILE, "w", encoding="utf-8") as f:
            f.write("{ not json")
        self.assertEqual(self.mod._load_inbox_index(), [])

    def test_load_inbox_index_roundtrip(self):
        msgs = [{"backend": "gmail", "id": "g1", "subject": "S" * 200,
                 "from_name": "Eve"}]
        self.mod._save_inbox_index(msgs)
        loaded = self.mod._load_inbox_index()
        self.assertEqual(loaded[0]["id"], "g1")
        self.assertLessEqual(len(loaded[0]["subject"]), 120)   # truncated


# ─────────────────────────────────────────────────────────────────────────
# _resolve_handle ordinal/word branches not hit by the existing suite.
# ─────────────────────────────────────────────────────────────────────────
class ResolveHandleExtraTests(EmailTriageTestBase):
    def _index(self):
        return [{"index": i, "backend": "gmail", "id": f"G{i}"}
                for i in range(1, 6)]

    def test_word_one_and_newest_alias(self):
        with mock.patch.object(self.mod, "_load_inbox_index",
                               return_value=self._index()):
            self.assertEqual(self.mod._resolve_handle("newest")["id"], "G1")
            self.assertEqual(self.mod._resolve_handle("one")["id"], "G1")

    def test_word_fifth(self):
        with mock.patch.object(self.mod, "_load_inbox_index",
                               return_value=self._index()):
            self.assertEqual(self.mod._resolve_handle("fifth")["id"], "G5")

    def test_unknown_word_falls_through_to_raw(self):
        with mock.patch.object(self.mod, "_load_inbox_index",
                               return_value=self._index()):
            self.assertEqual(self.mod._resolve_handle("banana"),
                             {"backend": "auto", "id": "banana"})


# ─────────────────────────────────────────────────────────────────────────
# _triage_message — Haiku primary + Ollama fallback (~823-892).
# ─────────────────────────────────────────────────────────────────────────
class TriageMessageTests(EmailTriageTestBase):
    def _msg(self):
        return {"from_name": "Alice", "from_addr": "alice@example.com",
                "subject": "Invoice due", "snippet": "please pay"}

    def test_disabled_returns_none(self):
        with mock.patch.object(self.mod, "ENABLE_LLM_TRIAGE", False):
            self.assertIsNone(self.mod._triage_message(self._msg()))

    def test_anthropic_returns_valid_verdict(self):
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}), \
             mock.patch.dict(sys.modules,
                             {"anthropic": _fake_anthropic(text="urgent")}):
            self.assertEqual(self.mod._triage_message(self._msg()), "urgent")

    def test_anthropic_garbage_falls_through_to_local(self):
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}), \
             mock.patch.dict(sys.modules,
                             {"anthropic": _fake_anthropic(text="banana"),
                              "bobert_companion": _fake_bobert(
                                  local_return="spam")}):
            self.assertEqual(self.mod._triage_message(self._msg()), "spam")

    def test_anthropic_raises_falls_back_to_local(self):
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}), \
             mock.patch.dict(sys.modules,
                             {"anthropic": _fake_anthropic(
                                 raise_on_create=RuntimeError("500")),
                              "bobert_companion": _fake_bobert(
                                  local_return="fyi")}):
            self.assertEqual(self.mod._triage_message(self._msg()), "fyi")

    def test_anthropic_import_fails_uses_local(self):
        # Key set but the anthropic import raises → _claude_ok flips False.
        real_import = __import__

        def _imp(name, *a, **k):
            if name == "anthropic":
                raise ImportError("no anthropic")
            return real_import(name, *a, **k)
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}), \
             mock.patch("builtins.__import__", side_effect=_imp), \
             mock.patch.dict(sys.modules,
                             {"bobert_companion": _fake_bobert(
                                 local_return="newsletter")}):
            self.assertEqual(self.mod._triage_message(self._msg()),
                             "newsletter")

    def test_no_key_local_only(self):
        with mock.patch.dict(os.environ, {}, clear=True), \
             mock.patch.dict(sys.modules,
                             {"bobert_companion": _fake_bobert(
                                 local_return="urgent")}):
            self.assertEqual(self.mod._triage_message(self._msg()), "urgent")

    def test_local_garbage_yields_none(self):
        with mock.patch.dict(os.environ, {}, clear=True), \
             mock.patch.dict(sys.modules,
                             {"bobert_companion": _fake_bobert(
                                 local_return="???")}):
            self.assertIsNone(self.mod._triage_message(self._msg()))

    def test_local_raises_yields_none(self):
        with mock.patch.dict(os.environ, {}, clear=True), \
             mock.patch.dict(sys.modules,
                             {"bobert_companion": _fake_bobert(
                                 local_raises=RuntimeError("ollama down"))}):
            self.assertIsNone(self.mod._triage_message(self._msg()))

    def test_uses_body_text_when_no_snippet(self):
        msg = {"from_addr": "x@example.com", "subject": "S",
               "body_text": "fallback body used for preview"}
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}), \
             mock.patch.dict(sys.modules,
                             {"anthropic": _fake_anthropic(text="fyi")}):
            self.assertEqual(self.mod._triage_message(msg), "fyi")


# ─────────────────────────────────────────────────────────────────────────
# _generate_draft_reply — Haiku primary + Ollama fallback (~895-964).
# ─────────────────────────────────────────────────────────────────────────
class GenerateDraftReplyTests(EmailTriageTestBase):
    def _thread(self):
        return {"from_name": "Frank", "from_addr": "frank@example.com",
                "subject": "Proposal", "body_text": "What do you think?"}

    def test_anthropic_drafts_reply(self):
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}), \
             mock.patch.dict(sys.modules,
                             {"anthropic": _fake_anthropic(
                                 text="Sounds great. — B")}):
            out = self.mod._generate_draft_reply(self._thread())
        self.assertEqual(out, "Sounds great. — B")

    def test_instructions_included_and_anthropic_used(self):
        fake = _fake_anthropic(text="Declined politely. — B")
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}), \
             mock.patch.dict(sys.modules, {"anthropic": fake}):
            out = self.mod._generate_draft_reply(self._thread(),
                                                 "politely decline")
        self.assertIn("Declined", out)
        # The user instruction is woven into the prompt sent to the model.
        _, kwargs = fake.Anthropic.return_value.messages.create.call_args
        prompt = kwargs["messages"][0]["content"]
        self.assertIn("politely decline", prompt)

    def test_output_truncated_to_cap(self):
        long = "x" * (self.mod.MAX_DRAFT_BODY_CHARS + 50)
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}), \
             mock.patch.dict(sys.modules,
                             {"anthropic": _fake_anthropic(text=long)}):
            out = self.mod._generate_draft_reply(self._thread())
        self.assertLessEqual(len(out), self.mod.MAX_DRAFT_BODY_CHARS + 1)
        self.assertTrue(out.endswith("…"))

    def test_anthropic_empty_falls_through_to_local(self):
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}), \
             mock.patch.dict(sys.modules,
                             {"anthropic": _fake_anthropic(text="   "),
                              "bobert_companion": _fake_bobert(
                                  local_return="Local draft. — B")}):
            out = self.mod._generate_draft_reply(self._thread())
        self.assertEqual(out, "Local draft. — B")

    def test_anthropic_raises_then_local(self):
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}), \
             mock.patch.dict(sys.modules,
                             {"anthropic": _fake_anthropic(
                                 raise_on_create=RuntimeError("boom")),
                              "bobert_companion": _fake_bobert(
                                  local_return="Recovered. — B")}):
            out = self.mod._generate_draft_reply(self._thread())
        self.assertEqual(out, "Recovered. — B")

    def test_no_key_local_only(self):
        with mock.patch.dict(os.environ, {}, clear=True), \
             mock.patch.dict(sys.modules,
                             {"bobert_companion": _fake_bobert(
                                 local_return="Only local. — B")}):
            out = self.mod._generate_draft_reply(self._thread())
        self.assertEqual(out, "Only local. — B")

    def test_anthropic_import_fails_uses_local(self):
        # Key present but the anthropic import itself raises → local fallback.
        real_import = __import__

        def _imp(name, *a, **k):
            if name == "anthropic":
                raise ImportError("missing")
            return real_import(name, *a, **k)
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}), \
             mock.patch("builtins.__import__", side_effect=_imp), \
             mock.patch.dict(sys.modules,
                             {"bobert_companion": _fake_bobert(
                                 local_return="Local after import fail. — B")}):
            out = self.mod._generate_draft_reply(self._thread())
        self.assertEqual(out, "Local after import fail. — B")

    def test_all_unavailable_returns_none(self):
        with mock.patch.dict(os.environ, {}, clear=True), \
             mock.patch.dict(sys.modules,
                             {"bobert_companion": _fake_bobert(
                                 local_raises=RuntimeError("down"))}):
            self.assertIsNone(self.mod._generate_draft_reply(self._thread()))

    def test_snippet_used_when_no_body_text(self):
        thread = {"from_addr": "g@example.com", "subject": "S",
                  "snippet": "snippet body"}
        with mock.patch.dict(os.environ, {}, clear=True), \
             mock.patch.dict(sys.modules,
                             {"bobert_companion": _fake_bobert(
                                 local_return="ok. — B")}):
            self.assertEqual(self.mod._generate_draft_reply(thread),
                             "ok. — B")


# ─────────────────────────────────────────────────────────────────────────
# Pending-draft state edge paths (~977-987) + save error handling.
# ─────────────────────────────────────────────────────────────────────────
class PendingStateEdgeTests(EmailTriageTestBase):
    def test_load_pending_non_dict_json_defaults(self):
        with open(self.mod.PENDING_DRAFTS_FILE, "w", encoding="utf-8") as f:
            json.dump([1, 2, 3], f)   # a list, not the expected dict
        self.assertEqual(self.mod._load_pending(),
                         {"active": None, "history": []})

    def test_load_pending_corrupt_json_defaults(self):
        with open(self.mod.PENDING_DRAFTS_FILE, "w", encoding="utf-8") as f:
            f.write("{not json")
        self.assertEqual(self.mod._load_pending(),
                         {"active": None, "history": []})

    def test_save_pending_swallows_write_error(self):
        with mock.patch.object(self.mod.os, "makedirs",
                               side_effect=OSError("ro")):
            self.mod._save_pending({"active": None, "history": []})
        # No exception → best-effort save honoured.

    def test_history_capped_at_20(self):
        # Push 25 actives; history retains only the last 20 previous records.
        for i in range(25):
            self.mod._set_pending({"draft_id": f"d{i}"})
        state = self.mod._load_pending()
        self.assertLessEqual(len(state["history"]), 20)
        self.assertEqual(state["active"]["draft_id"], "d24")


# ─────────────────────────────────────────────────────────────────────────
# Action-layer branches the existing suite doesn't reach.
# ─────────────────────────────────────────────────────────────────────────
class ActionBranchTests(EmailTriageTestBase):
    # -- list_unread arg parsing --
    def test_list_unread_arg_selects_backend_and_count(self):
        captured = {}

        def _fake_list(top_n=8, backend="all"):
            captured["top_n"] = top_n
            captured["backend"] = backend
            return []
        with mock.patch.object(self.mod, "list_unread", side_effect=_fake_list), \
             mock.patch.object(self.mod, "_outlook_configured",
                               return_value=True), \
             mock.patch.object(self.mod, "is_gmail_available",
                               return_value=True):
            self.actions["list_unread"]("gmail 3")
        self.assertEqual(captured["backend"], "gmail")
        self.assertEqual(captured["top_n"], 3)

    # -- read_thread bad handle --
    def test_read_thread_none_handle(self):
        with mock.patch.object(self.mod, "_resolve_handle", return_value=None):
            self.assertIn("number or message id",
                          self.actions["read_thread"]("  "))

    # -- draft_reply none handle / not found --
    def test_draft_reply_none_handle(self):
        with mock.patch.object(self.mod, "_resolve_handle", return_value=None):
            self.assertIn("which message to reply",
                          self.actions["draft_reply"](""))

    def test_draft_reply_thread_not_found(self):
        with mock.patch.object(self.mod, "_resolve_handle",
                               return_value={"backend": "auto", "id": "z"}), \
             mock.patch.object(self.mod, "_get_thread", return_value=None):
            self.assertIn("not found", self.actions["draft_reply"]("z"))

    def test_draft_reply_outlook_backend_write(self):
        thread = {"backend": "outlook", "id": "o1", "from_name": "Gail",
                  "from_addr": "gail@example.com", "subject": "Plan"}
        with mock.patch.object(self.mod, "_resolve_handle",
                               return_value={"backend": "outlook", "id": "o1"}), \
             mock.patch.object(self.mod, "_get_thread", return_value=thread), \
             mock.patch.object(self.mod, "_generate_draft_reply",
                               return_value="Reply. — B"), \
             mock.patch.object(self.mod, "_outlook_create_draft",
                               return_value="OD1") as oc:
            out = self.actions["draft_reply"]("o1")
        oc.assert_called_once()
        self.assertEqual(self.mod._get_pending()["draft_id"], "OD1")
        self.assertIn("saved to your Drafts folder", out)

    def test_draft_reply_auto_backend_write_failure_held_locally(self):
        thread = {"backend": "auto", "id": "x1", "from_name": "Hank",
                  "from_addr": "hank@example.com", "subject": "Sync"}
        with mock.patch.object(self.mod, "_resolve_handle",
                               return_value={"backend": "auto", "id": "x1"}), \
             mock.patch.object(self.mod, "_get_thread", return_value=thread), \
             mock.patch.object(self.mod, "_generate_draft_reply",
                               return_value="Reply. — B"), \
             mock.patch.object(self.mod, "_outlook_create_draft",
                               return_value=None), \
             mock.patch.object(self.mod, "_gmail_create_draft",
                               return_value=None):
            out = self.actions["draft_reply"]("x1")
        self.assertIn("held locally", out)
        self.assertIsNone(self.mod._get_pending()["draft_id"])

    # -- confirm: outlook send + no-id + no-backend-id --
    def test_confirm_outlook_send(self):
        self.mod._set_pending({"backend": "outlook", "draft_id": "OD1",
                               "to": "i@example.com"})
        with mock.patch.object(self.mod, "_outlook_send_draft",
                               return_value=True):
            out = self.actions["confirm_pending_draft"]("")
        self.assertIn("Sent the reply", out)

    def test_confirm_no_backend_id(self):
        self.mod._set_pending({"backend": "gmail", "draft_id": None,
                               "to": "j@example.com"})
        out = self.actions["confirm_pending_draft"]("")
        self.assertIn("no backend id", out)
        # Pending is NOT cleared since nothing was sent.
        self.assertIsNotNone(self.mod._get_pending())

    # -- edit: no pending / outlook update / backend reject --
    def test_edit_no_pending(self):
        self.assertIn("No draft waiting",
                      self.actions["edit_pending_draft"]("new body"))

    def test_edit_outlook_backend_update(self):
        self.mod._set_pending({"backend": "outlook", "draft_id": "OD1",
                               "message_id": "o1", "body": "old"})
        with mock.patch.object(self.mod, "_outlook_update_draft",
                               return_value=True):
            out = self.actions["edit_pending_draft"]("brand new body")
        self.assertIn("Draft updated", out)

    def test_edit_backend_reject_warns(self):
        self.mod._set_pending({"backend": "gmail", "draft_id": "d1",
                               "message_id": "m1", "body": "old"})
        with mock.patch.object(self.mod, "_gmail_update_draft",
                               return_value=False):
            out = self.actions["edit_pending_draft"]("newer body")
        self.assertIn("backend rejected the", out)
        # Local copy still updated even though the PATCH failed.
        self.assertEqual(self.mod._get_pending()["body"], "newer body")

    # -- list_pending_drafts --
    def test_list_pending_none(self):
        self.assertIn("No pending drafts",
                      self.actions["list_pending_drafts"](""))

    def test_list_pending_shows_preview(self):
        self.mod._set_pending({"backend": "gmail", "to": "k@example.com",
                               "body": "short body"})
        out = self.actions["list_pending_drafts"]("")
        self.assertIn("k@example.com", out)
        self.assertIn("short body", out)

    def test_list_pending_truncates_long_body(self):
        self.mod._set_pending({"backend": "gmail", "to": "k@example.com",
                               "body": "y" * 500})
        out = self.actions["list_pending_drafts"]("")
        self.assertTrue(out.endswith("…"))

    # -- archive action --
    def test_archive_action_none_handle(self):
        with mock.patch.object(self.mod, "_resolve_handle", return_value=None):
            self.assertIn("which message to archive",
                          self.actions["archive_email"](""))

    def test_archive_action_success_and_failure(self):
        with mock.patch.object(self.mod, "_resolve_handle",
                               return_value={"backend": "gmail", "id": "g"}), \
             mock.patch.object(self.mod, "_archive", return_value=True):
            self.assertIn("Archived", self.actions["archive_email"]("1"))
        with mock.patch.object(self.mod, "_resolve_handle",
                               return_value={"backend": "gmail", "id": "g"}), \
             mock.patch.object(self.mod, "_archive", return_value=False):
            self.assertIn("Couldn't archive", self.actions["archive_email"]("1"))

    # -- categorize_inbox arg parsing --
    def test_categorize_bad_arg_defaults(self):
        with mock.patch.object(self.mod, "list_unread", return_value=[]) as lu:
            self.actions["categorize_inbox"]("not-a-number")
        # Non-numeric arg → falls back to default 20.
        self.assertEqual(lu.call_args.kwargs.get("top_n"), 20)

    # -- email_briefing: unclassified tail + empty-after-classify --
    def test_briefing_unclassified_tail(self):
        msgs = [{"backend": "gmail", "id": "g1", "from_name": "L",
                 "subject": "S"}]
        with mock.patch.object(self.mod, "list_unread", return_value=msgs), \
             mock.patch.object(self.mod, "_triage_message", return_value=None):
            out = self.actions["email_briefing"]("")
        self.assertIn("1 other", out)

    def test_briefing_fyi_count_only(self):
        msgs = [{"backend": "gmail", "id": "g1", "from_name": "L",
                 "subject": "S"}]
        with mock.patch.object(self.mod, "list_unread", return_value=msgs), \
             mock.patch.object(self.mod, "_triage_message", return_value="fyi"):
            out = self.actions["email_briefing"]("")
        self.assertIn("1 FYI", out)

    def test_briefing_urgent_without_subject(self):
        msgs = [{"backend": "gmail", "id": "g1", "from_addr": "m@example.com"}]
        with mock.patch.object(self.mod, "list_unread", return_value=msgs), \
             mock.patch.object(self.mod, "_triage_message",
                               return_value="urgent"):
            out = self.actions["email_briefing"]("")
        self.assertIn("urgent message from m@example.com", out)


# ─────────────────────────────────────────────────────────────────────────
# Status action: Gmail-available branches + registration coverage.
# ─────────────────────────────────────────────────────────────────────────
class StatusAndRegisterTests(EmailTriageTestBase):
    def test_status_gmail_token_present(self):
        with open(self.mod.GMAIL_TOKEN_FILE, "w", encoding="utf-8") as f:
            f.write("{}")
        with mock.patch.object(self.mod, "_outlook_configured",
                               return_value=False), \
             mock.patch.object(self.mod, "is_gmail_available",
                               return_value=True):
            out = self.actions["email_triage_status"]("")
        self.assertIn("Gmail: token present", out)
        self.assertIn("Outlook: not configured", out)
        self.assertIn("no pending drafts", out)

    def test_status_gmail_creds_but_token_missing(self):
        with mock.patch.object(self.mod, "_outlook_configured",
                               return_value=False), \
             mock.patch.object(self.mod, "is_gmail_available",
                               return_value=True):
            out = self.actions["email_triage_status"]("")
        self.assertIn("token missing", out)

    def test_register_binds_all_aliases(self):
        actions = {}
        self.mod.register(actions)
        for name in ("list_unread", "unread_email", "read_thread",
                     "read_email", "draft_reply", "pre_draft_reply",
                     "archive_email", "categorize_inbox", "triage_inbox",
                     "email_briefing", "confirm_pending_draft", "send_draft",
                     "scrap_pending_draft", "edit_pending_draft",
                     "list_pending_drafts", "email_triage_status",
                     "categorise_inbox"):
            self.assertIn(name, actions)
        # Aliases dispatch to the same callables.
        self.assertIs(actions["unread_email"], actions["list_unread"])
        self.assertIs(actions["send_draft"], actions["confirm_pending_draft"])


if __name__ == "__main__":
    unittest.main()
