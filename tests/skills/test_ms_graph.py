"""Logic tests for skills/ms_graph.py.

ms_graph is a pure helper module (no register()/actions) wrapping Microsoft
Graph for calendar + mail. We test the parsing/normalisation logic, the
token-absent graceful degradation (every getter → None/[] with no creds), and
the write helpers' (status → bool) mapping — all with the HTTP layer
(_graph_get / _graph_call / urlopen) mocked. No network, no token files touched.
"""
from __future__ import annotations

import datetime
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


class StripHtmlTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("ms_graph")

    def test_drops_script_and_style(self):
        html = "<style>.x{color:red}</style>Hello<script>alert(1)</script> world"
        out = self.mod._strip_html(html)
        self.assertNotIn("color:red", out)
        self.assertNotIn("alert", out)
        self.assertIn("Hello", out)
        self.assertIn("world", out)

    def test_br_and_p_become_newlines(self):
        out = self.mod._strip_html("line1<br>line2</p>")
        self.assertIn("line1", out)
        self.assertIn("line2", out)
        self.assertIn("\n", out)

    def test_unescapes_entities(self):
        out = self.mod._strip_html("Tom &amp; Jerry &lt;tag&gt; &quot;q&quot;")
        self.assertIn("Tom & Jerry", out)
        self.assertIn("<tag>", out)
        self.assertIn('"q"', out)

    def test_empty(self):
        self.assertEqual(self.mod._strip_html(""), "")


class ShapeOutlookMessageTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("ms_graph")

    def test_normalises_graph_message(self):
        raw = {
            "id": "AAMk123",
            "from": {"emailAddress": {"name": "Jane Doe", "address": "jane@x.com"}},
            "subject": "  Lunch?  ",
            "bodyPreview": "  are you free  ",
            "receivedDateTime": "2026-06-01T10:00:00Z",
            "isRead": False,
            "categories": ["Work"],
        }
        shaped = self.mod._shape_outlook_message(raw)
        self.assertEqual(shaped["backend"], "outlook")
        self.assertEqual(shaped["id"], "AAMk123")
        self.assertEqual(shaped["from_name"], "Jane Doe")
        self.assertEqual(shaped["from_addr"], "jane@x.com")
        self.assertEqual(shaped["subject"], "Lunch?")     # trimmed
        self.assertEqual(shaped["snippet"], "are you free")
        self.assertTrue(shaped["unread"])                 # isRead False → unread
        self.assertEqual(shaped["categories"], ["Work"])

    def test_read_message_not_unread(self):
        shaped = self.mod._shape_outlook_message({"id": "1", "isRead": True})
        self.assertFalse(shaped["unread"])

    def test_missing_fields_default_empty(self):
        shaped = self.mod._shape_outlook_message({})
        self.assertEqual(shaped["from_name"], "")
        self.assertEqual(shaped["subject"], "")
        # Absent isRead defaults to read → not unread.
        self.assertFalse(shaped["unread"])


class MeetingWindowTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("ms_graph")

    def test_tomorrow_window_is_full_day(self):
        start, end = self.mod._meeting_window("tomorrow")
        tomorrow = (datetime.datetime.now() + datetime.timedelta(days=1)).date()
        self.assertEqual(start.date(), tomorrow)
        self.assertEqual(start.hour, 0)
        self.assertEqual(end.hour, 23)

    def test_next_14_days_spans_two_weeks(self):
        start, end = self.mod._meeting_window("next_14_days")
        self.assertAlmostEqual((end - start).days, 14, delta=1)

    def test_default_today_to_eod(self):
        start, end = self.mod._meeting_window("today")
        self.assertEqual(end.hour, 23)
        self.assertEqual(end.minute, 59)


class ParseGraphStartTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("ms_graph")

    def test_parses_utc_to_naive_local(self):
        evt = {"start": {"dateTime": "2026-06-01T12:00:00", "timeZone": "UTC"}}
        dt = self.mod._parse_graph_start(evt)
        self.assertIsInstance(dt, datetime.datetime)
        self.assertIsNone(dt.tzinfo)   # converted to naive local

    def test_strips_fractional_seconds(self):
        evt = {"start": {"dateTime": "2026-06-01T12:00:00.123456", "timeZone": "UTC"}}
        self.assertIsNotNone(self.mod._parse_graph_start(evt))

    def test_missing_start_returns_none(self):
        self.assertIsNone(self.mod._parse_graph_start({}))

    def test_bad_datetime_returns_none(self):
        evt = {"start": {"dateTime": "not-a-date", "timeZone": "UTC"}}
        self.assertIsNone(self.mod._parse_graph_start(evt))


class TokenDegradationTests(unittest.TestCase):
    """With no MSAL app and no token file, every public getter degrades."""
    def setUp(self):
        self.mod, _ = load_skill_isolated("ms_graph")

    def test_get_access_token_none_without_auth(self):
        with mock.patch.object(self.mod, "_get_access_token_msal", return_value=None), \
             mock.patch.object(self.mod, "_load_token", return_value=None):
            self.assertIsNone(self.mod.get_access_token())

    def test_graph_get_returns_none_without_token(self):
        with mock.patch.object(self.mod, "get_access_token", return_value=None):
            self.assertIsNone(self.mod._graph_get("/me/messages"))

    def test_getters_degrade_when_graph_unavailable(self):
        # _graph_get → None (no creds) means getters return empty/None cleanly.
        with mock.patch.object(self.mod, "_graph_get", return_value=None):
            self.assertEqual(self.mod.get_upcoming_events(), [])
            self.assertIsNone(self.mod.get_unread_mail_count())
            self.assertEqual(self.mod.list_unread_messages(), [])
            self.assertIsNone(self.mod.get_teams_unread_count())

    def test_is_configured_false_without_anything(self):
        with mock.patch.object(self.mod, "_msal_app", return_value=None), \
             mock.patch.object(self.mod, "_load_token", return_value=None):
            self.assertFalse(self.mod.is_configured())

    def test_get_access_token_uses_valid_cached_token(self):
        future = self.mod.time.time() + 3600
        with mock.patch.object(self.mod, "_get_access_token_msal", return_value=None), \
             mock.patch.object(self.mod, "_load_token",
                               return_value={"access_token": "tok", "expires_at": future}):
            self.assertEqual(self.mod.get_access_token(), "tok")


class GetUpcomingEventsTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("ms_graph")

    def test_parses_event_list(self):
        body = {"value": [
            {"subject": "Standup",
             "start": {"dateTime": "2026-06-01T09:00:00", "timeZone": "UTC"},
             "organizer": {"emailAddress": {"name": "Alice"}}},
            {"subject": "Bad event with no start"},   # skipped (no start)
        ]}
        with mock.patch.object(self.mod, "_graph_get", return_value=body):
            events = self.mod.get_upcoming_events(top_n=5, when="today")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["subject"], "Standup")
        self.assertEqual(events[0]["organizer"], "Alice")
        self.assertIsInstance(events[0]["start"], datetime.datetime)


class TeamsUnreadTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("ms_graph")

    def test_counts_unread_and_top_sender(self):
        body = {"value": [
            {  # unread: last msg after last-read
                "lastMessagePreview": {"createdDateTime": "2026-06-01T12:00:00Z",
                                        "from": {"user": {"displayName": "Sam Smith"}}},
                "viewpoint": {"lastMessageReadDateTime": "2026-06-01T11:00:00Z"}},
            {  # read: last msg <= last-read → not counted
                "lastMessagePreview": {"createdDateTime": "2026-06-01T08:00:00Z",
                                        "from": {"user": {"displayName": "Bob"}}},
                "viewpoint": {"lastMessageReadDateTime": "2026-06-01T09:00:00Z"}},
            {  # system event: no human sender → skipped
                "lastMessagePreview": {"createdDateTime": "2026-06-01T13:00:00Z",
                                        "from": {}},
                "viewpoint": {}},
        ]}
        with mock.patch.object(self.mod, "_graph_get", return_value=body):
            res = self.mod.get_teams_unread_count()
        self.assertEqual(res["count"], 1)
        self.assertEqual(res["top_sender"], "Sam")   # first name of newest unread

    def test_none_when_graph_unavailable(self):
        with mock.patch.object(self.mod, "_graph_get", return_value=None):
            self.assertIsNone(self.mod.get_teams_unread_count())


class GetMessageThreadTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("ms_graph")

    def test_strips_html_body(self):
        body = {
            "id": "m1",
            "from": {"emailAddress": {"name": "X", "address": "x@y.com"}},
            "subject": "Hi",
            "bodyPreview": "preview",
            "body": {"contentType": "html",
                     "content": "<p>Hello <b>there</b></p>"},
            "toRecipients": [{"emailAddress": {"address": "me@y.com"}}],
            "conversationId": "conv1",
        }
        with mock.patch.object(self.mod, "_graph_get", return_value=body):
            thread = self.mod.get_message_thread("m1")
        self.assertEqual(thread["body_html"], "<p>Hello <b>there</b></p>")
        self.assertIn("Hello", thread["body_text"])
        self.assertNotIn("<b>", thread["body_text"])
        self.assertEqual(thread["to"], ["me@y.com"])
        self.assertEqual(thread["conversation_id"], "conv1")

    def test_empty_id_returns_none(self):
        self.assertIsNone(self.mod.get_message_thread(""))


class WriteHelperTests(unittest.TestCase):
    """The mutation helpers map a Graph (status, payload) onto a bool / id."""
    def setUp(self):
        self.mod, _ = load_skill_isolated("ms_graph")

    def test_archive_message_2xx_true(self):
        with mock.patch.object(self.mod, "_graph_call", return_value=(204, None)):
            self.assertTrue(self.mod.archive_message("m1"))

    def test_archive_message_4xx_false(self):
        with mock.patch.object(self.mod, "_graph_call", return_value=(404, None)):
            self.assertFalse(self.mod.archive_message("m1"))

    def test_archive_empty_id_false_without_call(self):
        with mock.patch.object(self.mod, "_graph_call") as call:
            self.assertFalse(self.mod.archive_message(""))
        call.assert_not_called()

    def test_mark_as_read_true(self):
        with mock.patch.object(self.mod, "_graph_call", return_value=(200, {})):
            self.assertTrue(self.mod.mark_as_read("m1"))

    def test_send_draft_true(self):
        with mock.patch.object(self.mod, "_graph_call", return_value=(202, None)):
            self.assertTrue(self.mod.send_draft("d1"))

    def test_send_draft_no_id(self):
        self.assertFalse(self.mod.send_draft(""))

    def test_create_draft_reply_returns_draft_resource(self):
        # Returns the full Graph Message resource (callers extract .id).
        with mock.patch.object(self.mod, "_graph_call",
                               return_value=(201, {"id": "draft-99"})):
            draft = self.mod.create_draft_reply("m1", "thanks")
        self.assertEqual(draft, {"id": "draft-99"})

    def test_create_draft_reply_failure_none(self):
        with mock.patch.object(self.mod, "_graph_call", return_value=(400, None)):
            self.assertIsNone(self.mod.create_draft_reply("m1", "thanks"))

    def test_apply_category_merges_existing(self):
        # First a GET for current categories, then a PATCH that must include both.
        with mock.patch.object(self.mod, "_graph_get",
                               return_value={"categories": ["Existing"]}), \
             mock.patch.object(self.mod, "_graph_call",
                               return_value=(200, {})) as call:
            ok = self.mod.apply_category("m1", "JARVIS/Urgent")
        self.assertTrue(ok)
        patched_body = call.call_args.kwargs.get("body") or call.call_args[0][-1]
        self.assertIn("Existing", patched_body["categories"])
        self.assertIn("JARVIS/Urgent", patched_body["categories"])


if __name__ == "__main__":
    unittest.main()
