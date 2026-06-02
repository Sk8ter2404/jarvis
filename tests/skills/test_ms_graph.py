"""Logic tests for skills/ms_graph.py.

ms_graph is a pure helper module (no register()/actions) wrapping Microsoft
Graph for calendar + mail. We test the parsing/normalisation logic, the
token-absent graceful degradation (every getter → None/[] with no creds), and
the write helpers' (status → bool) mapping — all with the HTTP layer
(_graph_get / _graph_call / urlopen) mocked. No network, no token files touched.

Wave-3 extension raises coverage from ~43% to >=90% by exercising the auth
stack end-to-end with everything faked:
  • MSAL — a fake ``msal`` module (PublicClientApplication + SerializableToken
    Cache) injected via mock.patch.dict(sys.modules); no real token acquisition.
  • DPAPI — a fake ``win32crypt`` module so the encrypt/decrypt round-trip runs
    identically on CI runners that lack pywin32 (mirrors test_network_deco.py).
  • urllib — ``urlopen`` patched on the module to return canned 200/JSON
    responses or raise urllib.error.HTTPError(401/429/5xx) / URLError, so the
    Graph HTTP/refresh paths, retry-status classification and error branches run
    fully offline.
  • token / cache files — _TOKEN_FILE and _MSAL_CACHE_FILE are redirected into a
    per-test tempdir (addCleanup) so the real on-disk files are never created or
    read. _config is patched per-test, never the live bobert_companion.

ISOLATION: all sys.modules writes go through mock.patch.dict (auto-restored);
tempdirs are removed in cleanup; module-level path constants are restored. No
real network, no threads with delays (the one ThreadPoolExecutor in MSAL runs a
mock that returns instantly), no personal data.
"""
from __future__ import annotations

import datetime
import json
import os
import sys
import tempfile
import types
import unittest
import urllib.error
from unittest import mock

from tests._skill_harness import load_skill_isolated


# ─── shared fakes ────────────────────────────────────────────────────────

def _fake_response(payload, status=200):
    """Context-manager stand-in for urllib.request.urlopen() (mirrors the
    helper in test_briefing_sources.py). ``payload`` may be a dict (JSON-
    encoded), bytes (sent raw), or None (empty body)."""
    if payload is None:
        body = b""
    elif isinstance(payload, (bytes, bytearray)):
        body = bytes(payload)
    else:
        body = json.dumps(payload).encode("utf-8")
    resp = mock.MagicMock()
    resp.read.return_value = body
    resp.status = status
    resp.__enter__ = mock.Mock(return_value=resp)
    resp.__exit__ = mock.Mock(return_value=False)
    return resp


def _http_error(code, body=b"err-detail"):
    """Build a urllib.error.HTTPError whose .read() yields ``body`` once.

    A real (already-closed) BytesIO is passed as ``fp`` so the interpreter's
    tempfile/GC cleanup doesn't emit a ResourceWarning about an unclosed
    HTTPError — the handler reads via the mocked .read() regardless."""
    import io
    fp = io.BytesIO(body)
    err = urllib.error.HTTPError(
        url="https://graph.microsoft.com/v1.0/x", code=code,
        msg="reason", hdrs=None, fp=fp)
    err.read = mock.Mock(return_value=body)
    err.close()
    return err


def _fake_win32crypt():
    """A fake ``win32crypt`` whose protect/unprotect are a reversible prefix
    transform — lets the DPAPI round-trip run on CI runners without pywin32."""
    fake = types.ModuleType("win32crypt")
    fake.CryptProtectData = lambda data, *a, **k: b"BLOB:" + data
    fake.CryptUnprotectData = lambda blob, *a, **k: ("desc", blob[len(b"BLOB:"):])
    return fake


class _MsGraphBase(unittest.TestCase):
    """Loads ms_graph in isolation and redirects its on-disk token + MSAL
    cache paths into a per-test tempdir so the real files are never touched.
    The path constants are module globals, restored automatically because
    load_skill_isolated re-execs a fresh module each call — but we still point
    them at a tempdir for any test that exercises the file paths."""
    def setUp(self):
        self.mod, _ = load_skill_isolated("ms_graph")
        self._tmp = tempfile.mkdtemp(prefix="msgraph_test_")
        self.addCleanup(self._cleanup_tmp)
        self.token_path = os.path.join(self._tmp, "microsoft_graph_token.json")
        self.cache_path = os.path.join(self._tmp, "ms_graph_msal_cache.json")
        self.mod._TOKEN_FILE = self.token_path
        self.mod._MSAL_CACHE_FILE = self.cache_path
        # CRITICAL ISOLATION: _config() does importlib.import_module(
        # "bobert_companion"), which boots the ~14K-line monolith (singleton
        # lock, etc.) if it isn't already cached. Default every test to a config
        # that returns the passed default, so no test ever imports the monolith.
        # Tests needing specific values override _config inside their own with-
        # block (that inner patch shadows this one). Patched as a default so
        # even helpers that reach _config indirectly (e.g. _get_access_token_
        # msal → scopes) stay hermetic regardless of test-module run order.
        self._real_config = self.mod._config   # keep a handle for ConfigTests
        cfg_patch = mock.patch.object(
            self.mod, "_config", side_effect=lambda name, default: default)
        cfg_patch.start()
        self.addCleanup(cfg_patch.stop)

    def _cleanup_tmp(self):
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)


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


# ─── _config ─────────────────────────────────────────────────────────────

class ConfigTests(_MsGraphBase):
    # These exercise the REAL _config (the base class stubs _config for every
    # other test to stay off the monolith; we call self._real_config here). A
    # fake bobert_companion is injected into sys.modules so the real
    # importlib.import_module returns the fake, never booting the live monolith.
    def test_returns_attr_from_bobert_companion(self):
        fake_bc = types.ModuleType("bobert_companion")
        fake_bc.MS_GRAPH_CLIENT_ID = "client-xyz"
        with mock.patch.dict(sys.modules, {"bobert_companion": fake_bc}):
            self.assertEqual(
                self._real_config("MS_GRAPH_CLIENT_ID", ""), "client-xyz")

    def test_default_when_attr_absent(self):
        fake_bc = types.ModuleType("bobert_companion")
        with mock.patch.dict(sys.modules, {"bobert_companion": fake_bc}):
            self.assertEqual(
                self._real_config("MISSING_ATTR", "fallback"), "fallback")

    def test_default_when_import_fails(self):
        # importlib.import_module raising → default. Patch it to raise so we
        # never touch the real monolith even on a cold sys.modules.
        with mock.patch.object(self.mod.importlib, "import_module",
                               side_effect=ImportError("no bobert")):
            self.assertEqual(self._real_config("ANY", "dft"), "dft")


# ─── _atomic_write_json ──────────────────────────────────────────────────

class AtomicWriteJsonTests(_MsGraphBase):
    def test_writes_payload(self):
        path = os.path.join(self._tmp, "out.json")
        self.mod._atomic_write_json(path, {"a": 1, "b": "two"})
        with open(path, "r", encoding="utf-8") as f:
            self.assertEqual(json.load(f), {"a": 1, "b": "two"})

    def test_unlinks_temp_and_raises_on_serialise_failure(self):
        # A non-serialisable payload makes json.dump raise after mkstemp; the
        # except branch must unlink the temp file and re-raise.
        path = os.path.join(self._tmp, "out.json")
        before = set(os.listdir(self._tmp))
        with self.assertRaises(TypeError):
            self.mod._atomic_write_json(path, {"bad": object()})
        # No leftover *.tmp file in the dir.
        after = set(os.listdir(self._tmp))
        self.assertEqual(before, after)


# ─── DPAPI encrypt/decrypt ───────────────────────────────────────────────

class DpapiTests(_MsGraphBase):
    def test_encrypt_empty_returns_none(self):
        self.assertIsNone(self.mod._dpapi_encrypt(""))

    def test_decrypt_empty_returns_none(self):
        self.assertIsNone(self.mod._dpapi_decrypt(""))

    def test_roundtrip_with_fake_win32(self):
        with mock.patch.dict(sys.modules, {"win32crypt": _fake_win32crypt()}):
            enc = self.mod._dpapi_encrypt("secret-token")
            self.assertIsInstance(enc, str)
            dec = self.mod._dpapi_decrypt(enc)
        self.assertEqual(dec, "secret-token")

    def test_encrypt_returns_none_when_win32_missing(self):
        # import win32crypt → ImportError inside _dpapi_encrypt.
        with mock.patch.dict(sys.modules, {"win32crypt": None}):
            self.assertIsNone(self.mod._dpapi_encrypt("x"))

    def test_decrypt_returns_none_on_crypt_failure(self):
        fake = types.ModuleType("win32crypt")
        def boom(*a, **k):
            raise RuntimeError("crypt fail")
        fake.CryptUnprotectData = boom
        with mock.patch.dict(sys.modules, {"win32crypt": fake}):
            self.assertIsNone(self.mod._dpapi_decrypt("QUJD"))  # valid b64


# ─── _load_token / _save_token ───────────────────────────────────────────

class LoadSaveTokenTests(_MsGraphBase):
    def test_load_missing_file_returns_none(self):
        # token_path doesn't exist yet.
        self.assertIsNone(self.mod._load_token())

    def test_load_invalid_json_returns_none(self):
        with open(self.token_path, "w", encoding="utf-8") as f:
            f.write("{not json")
        self.assertIsNone(self.mod._load_token())

    def test_load_non_dict_returns_none(self):
        with open(self.token_path, "w", encoding="utf-8") as f:
            json.dump([1, 2, 3], f)
        self.assertIsNone(self.mod._load_token())

    def test_load_encrypted_roundtrip(self):
        with mock.patch.dict(sys.modules, {"win32crypt": _fake_win32crypt()}):
            self.mod._save_token({"access_token": "enc-tok", "expires_at": 123})
            # On disk it is the {"dpapi": ...} shape, not plaintext.
            with open(self.token_path, "r", encoding="utf-8") as f:
                on_disk = json.load(f)
            self.assertIn("dpapi", on_disk)
            self.assertNotIn("access_token", on_disk)
            loaded = self.mod._load_token()
        self.assertEqual(loaded["access_token"], "enc-tok")

    def test_load_encrypted_undecryptable_returns_none(self):
        # dpapi blob present but win32crypt missing → can't decrypt → None.
        with open(self.token_path, "w", encoding="utf-8") as f:
            json.dump({"dpapi": "Z2FyYmFnZQ=="}, f)
        with mock.patch.dict(sys.modules, {"win32crypt": None}):
            self.assertIsNone(self.mod._load_token())

    def test_load_encrypted_decrypts_to_invalid_json_returns_none(self):
        # win32crypt decrypts fine, but the plaintext isn't valid JSON → None.
        fake = types.ModuleType("win32crypt")
        fake.CryptProtectData = lambda data, *a, **k: b"BLOB:" + data
        # Unprotect yields non-JSON bytes regardless of input.
        fake.CryptUnprotectData = lambda blob, *a, **k: ("d", b"{not-json")
        with open(self.token_path, "w", encoding="utf-8") as f:
            json.dump({"dpapi": "QUJD"}, f)
        with mock.patch.dict(sys.modules, {"win32crypt": fake}):
            self.assertIsNone(self.mod._load_token())

    def test_load_legacy_plaintext_migration_failure_swallowed(self):
        # If the auto-migrate _save_token raises, the plaintext dict is still
        # returned (the except just passes).
        with open(self.token_path, "w", encoding="utf-8") as f:
            json.dump({"access_token": "legacy", "expires_at": 1}, f)
        with mock.patch.object(self.mod, "_save_token",
                               side_effect=RuntimeError("save boom")):
            loaded = self.mod._load_token()
        self.assertEqual(loaded["access_token"], "legacy")

    def test_load_legacy_plaintext_auto_migrates(self):
        # A legacy plaintext dict is returned as-is AND re-saved encrypted.
        with open(self.token_path, "w", encoding="utf-8") as f:
            json.dump({"access_token": "legacy", "expires_at": 999}, f)
        with mock.patch.dict(sys.modules, {"win32crypt": _fake_win32crypt()}):
            loaded = self.mod._load_token()
            self.assertEqual(loaded["access_token"], "legacy")
            # Migration rewrote it in encrypted form.
            with open(self.token_path, "r", encoding="utf-8") as f:
                self.assertIn("dpapi", json.load(f))

    def test_save_plaintext_when_win32_missing(self):
        # No win32crypt → falls back to plaintext write (with a warning print).
        with mock.patch.dict(sys.modules, {"win32crypt": None}):
            self.mod._save_token({"access_token": "plain", "expires_at": 1})
        with open(self.token_path, "r", encoding="utf-8") as f:
            on_disk = json.load(f)
        self.assertEqual(on_disk["access_token"], "plain")

    def test_save_swallows_errors(self):
        # _atomic_write_json blowing up must not raise out of _save_token.
        with mock.patch.object(self.mod, "_dpapi_encrypt", return_value="x"), \
             mock.patch.object(self.mod, "_atomic_write_json",
                               side_effect=OSError("disk full")):
            self.mod._save_token({"access_token": "t"})  # no exception


# ─── _msal_app / _save_msal_cache ────────────────────────────────────────

def _fake_msal_module(app=None, cache=None):
    """Build a fake ``msal`` module exposing SerializableTokenCache and
    PublicClientApplication. ``app`` (if given) is returned from the PCA
    constructor; otherwise a fresh MagicMock is."""
    m = types.ModuleType("msal")

    class _Cache:
        def __init__(self):
            self.has_state_changed = False
            self._blob = ""
        def deserialize(self, s):
            self._blob = s
        def serialize(self):
            return self._blob or "SERIALIZED"
    m.SerializableTokenCache = cache or _Cache

    def _pca(client_id, authority=None, token_cache=None):
        a = app or mock.MagicMock(name="PCA")
        a._client_id = client_id
        a._authority = authority
        a._passed_cache = token_cache
        return a
    m.PublicClientApplication = _pca
    return m


class MsalAppTests(_MsGraphBase):
    def test_returns_none_without_client_id(self):
        with mock.patch.object(self.mod, "_config", return_value=""):
            self.assertIsNone(self.mod._msal_app())

    def test_returns_none_when_msal_not_installed(self):
        with mock.patch.object(self.mod, "_config",
                               side_effect=lambda n, d: "client-1" if "CLIENT" in n else d), \
             mock.patch.dict(sys.modules, {"msal": None}):
            self.assertIsNone(self.mod._msal_app())

    def test_builds_app_with_authority_and_cache(self):
        fake_app = mock.MagicMock(name="PCA")
        fake_msal = _fake_msal_module(app=fake_app)

        def cfg(name, default):
            if "CLIENT_ID" in name:
                return "client-1"
            if "TENANT_ID" in name:
                return "my-tenant"
            return default
        with mock.patch.object(self.mod, "_config", side_effect=cfg), \
             mock.patch.dict(sys.modules, {"msal": fake_msal}):
            app = self.mod._msal_app()
        self.assertIs(app, fake_app)
        self.assertEqual(app._authority,
                         "https://login.microsoftonline.com/my-tenant")
        # The cache was stashed for later persistence.
        self.assertIsNotNone(getattr(app, "_jarvis_cache", None))

    def test_deserialises_existing_cache_file(self):
        # Pre-create a cache file; _msal_app must read + deserialize it.
        with open(self.cache_path, "w", encoding="utf-8") as f:
            f.write("CACHED-STATE")
        captured = {}

        class _Cache:
            has_state_changed = False
            def deserialize(self, s):
                captured["blob"] = s
            def serialize(self):
                return ""
        fake_msal = _fake_msal_module(cache=_Cache)
        with mock.patch.object(self.mod, "_config",
                               side_effect=lambda n, d: "client-1" if "CLIENT" in n else d), \
             mock.patch.dict(sys.modules, {"msal": fake_msal}):
            self.mod._msal_app()
        self.assertEqual(captured.get("blob"), "CACHED-STATE")

    def test_save_cache_writes_when_state_changed(self):
        app = mock.MagicMock()
        cache = mock.MagicMock()
        cache.has_state_changed = True
        cache.serialize.return_value = "NEW-STATE"
        app._jarvis_cache = cache
        self.mod._save_msal_cache(app)
        with open(self.cache_path, "r", encoding="utf-8") as f:
            self.assertEqual(f.read(), "NEW-STATE")

    def test_save_cache_noop_when_unchanged(self):
        app = mock.MagicMock()
        cache = mock.MagicMock()
        cache.has_state_changed = False
        app._jarvis_cache = cache
        self.mod._save_msal_cache(app)
        self.assertFalse(os.path.exists(self.cache_path))

    def test_save_cache_noop_when_no_cache(self):
        app = mock.MagicMock()
        app._jarvis_cache = None
        self.mod._save_msal_cache(app)  # no crash, no file
        self.assertFalse(os.path.exists(self.cache_path))

    def test_save_cache_swallows_serialize_error(self):
        # cache.serialize() blowing up is caught (prints, no raise).
        app = mock.MagicMock()
        cache = mock.MagicMock()
        cache.has_state_changed = True
        cache.serialize.side_effect = RuntimeError("serialize boom")
        app._jarvis_cache = cache
        self.mod._save_msal_cache(app)  # must not raise

    def test_app_swallows_corrupt_cache_file(self):
        # A cache file whose deserialize() raises is caught; app still builds.
        with open(self.cache_path, "w", encoding="utf-8") as f:
            f.write("CORRUPT")

        class _Cache:
            has_state_changed = False
            def deserialize(self, s):
                raise ValueError("bad cache blob")
            def serialize(self):
                return ""
        fake_app = mock.MagicMock(name="PCA")
        fake_msal = _fake_msal_module(app=fake_app, cache=_Cache)
        with mock.patch.object(self.mod, "_config",
                               side_effect=lambda n, d: "client-1" if "CLIENT" in n else d), \
             mock.patch.dict(sys.modules, {"msal": fake_msal}):
            self.assertIs(self.mod._msal_app(), fake_app)


# ─── _get_access_token_msal ──────────────────────────────────────────────

class GetAccessTokenMsalTests(_MsGraphBase):
    def test_none_when_no_app(self):
        with mock.patch.object(self.mod, "_msal_app", return_value=None):
            self.assertIsNone(self.mod._get_access_token_msal())

    def test_none_when_no_accounts(self):
        app = mock.MagicMock()
        app.get_accounts.return_value = []
        with mock.patch.object(self.mod, "_msal_app", return_value=app):
            self.assertIsNone(self.mod._get_access_token_msal())

    def test_returns_access_token_on_silent_success(self):
        app = mock.MagicMock()
        app.get_accounts.return_value = [{"username": "u@x.com"}]
        app.acquire_token_silent.return_value = {"access_token": "silent-tok"}
        with mock.patch.object(self.mod, "_msal_app", return_value=app), \
             mock.patch.object(self.mod, "_save_msal_cache"), \
             mock.patch.object(self.mod, "_config", return_value=["Scope.Read"]):
            self.assertEqual(self.mod._get_access_token_msal(), "silent-tok")

    def test_none_when_silent_returns_no_token(self):
        app = mock.MagicMock()
        app.get_accounts.return_value = [{"username": "u@x.com"}]
        app.acquire_token_silent.return_value = {"error": "interaction_required"}
        with mock.patch.object(self.mod, "_msal_app", return_value=app), \
             mock.patch.object(self.mod, "_save_msal_cache"):
            self.assertIsNone(self.mod._get_access_token_msal())

    def test_none_when_get_accounts_raises(self):
        app = mock.MagicMock()
        app.get_accounts.side_effect = RuntimeError("boom")
        with mock.patch.object(self.mod, "_msal_app", return_value=app):
            self.assertIsNone(self.mod._get_access_token_msal())

    def test_timeout_falls_through_to_none(self):
        # Make the executor's .result() raise TimeoutError to exercise the
        # bounded-wait fallback branch (no real thread delay).
        app = mock.MagicMock()
        app.get_accounts.return_value = [{"username": "u@x.com"}]
        import concurrent.futures as _cf
        fake_future = mock.MagicMock()
        fake_future.result.side_effect = _cf.TimeoutError()
        fake_ex = mock.MagicMock()
        fake_ex.submit.return_value = fake_future
        with mock.patch.object(self.mod, "_msal_app", return_value=app), \
             mock.patch.object(self.mod.concurrent.futures,
                               "ThreadPoolExecutor", return_value=fake_ex):
            self.assertIsNone(self.mod._get_access_token_msal())
        fake_ex.shutdown.assert_called_once_with(wait=False)


# ─── _refresh_with_refresh_token ─────────────────────────────────────────

class RefreshTokenTests(_MsGraphBase):
    def test_none_without_refresh_token(self):
        with mock.patch.object(self.mod, "_config", return_value="client-1"):
            self.assertIsNone(self.mod._refresh_with_refresh_token({}))

    def test_none_without_client_id(self):
        with mock.patch.object(self.mod, "_config", return_value=""):
            self.assertIsNone(
                self.mod._refresh_with_refresh_token({"refresh_token": "r"}))

    def test_success_saves_and_returns_token(self):
        def cfg(name, default):
            if "CLIENT_ID" in name:
                return "client-1"
            return default
        resp = _fake_response({"access_token": "fresh", "expires_in": 3600,
                               "refresh_token": "new-refresh"})
        with mock.patch.object(self.mod, "_config", side_effect=cfg), \
             mock.patch.object(self.mod.urllib.request, "urlopen",
                               return_value=resp), \
             mock.patch.object(self.mod, "_save_token") as save:
            out = self.mod._refresh_with_refresh_token({"refresh_token": "old"})
        self.assertEqual(out["access_token"], "fresh")
        self.assertEqual(out["refresh_token"], "new-refresh")
        self.assertGreater(out["expires_at"], self.mod.time.time())
        save.assert_called_once()

    def test_keeps_old_refresh_token_when_response_omits_it(self):
        def cfg(name, default):
            return "client-1" if "CLIENT_ID" in name else default
        resp = _fake_response({"access_token": "fresh", "expires_in": 100})
        with mock.patch.object(self.mod, "_config", side_effect=cfg), \
             mock.patch.object(self.mod.urllib.request, "urlopen",
                               return_value=resp), \
             mock.patch.object(self.mod, "_save_token"):
            out = self.mod._refresh_with_refresh_token({"refresh_token": "keepme"})
        self.assertEqual(out["refresh_token"], "keepme")

    def test_none_on_http_error(self):
        def cfg(name, default):
            return "client-1" if "CLIENT_ID" in name else default
        with mock.patch.object(self.mod, "_config", side_effect=cfg), \
             mock.patch.object(self.mod.urllib.request, "urlopen",
                               side_effect=urllib.error.URLError("down")):
            self.assertIsNone(
                self.mod._refresh_with_refresh_token({"refresh_token": "r"}))

    def test_none_when_response_has_no_access_token(self):
        def cfg(name, default):
            return "client-1" if "CLIENT_ID" in name else default
        resp = _fake_response({"error": "invalid_grant"})
        with mock.patch.object(self.mod, "_config", side_effect=cfg), \
             mock.patch.object(self.mod.urllib.request, "urlopen",
                               return_value=resp):
            self.assertIsNone(
                self.mod._refresh_with_refresh_token({"refresh_token": "r"}))


# ─── get_access_token orchestration ──────────────────────────────────────

class GetAccessTokenOrchestrationTests(_MsGraphBase):
    def test_msal_token_short_circuits(self):
        with mock.patch.object(self.mod, "_get_access_token_msal",
                               return_value="msal-tok"):
            self.assertEqual(self.mod.get_access_token(), "msal-tok")

    def test_expired_token_triggers_refresh(self):
        past = self.mod.time.time() - 100
        with mock.patch.object(self.mod, "_get_access_token_msal", return_value=None), \
             mock.patch.object(self.mod, "_load_token",
                               return_value={"access_token": "old",
                                             "expires_at": past,
                                             "refresh_token": "r"}), \
             mock.patch.object(self.mod, "_refresh_with_refresh_token",
                               return_value={"access_token": "refreshed"}):
            self.assertEqual(self.mod.get_access_token(), "refreshed")

    def test_expired_token_refresh_fails_returns_none(self):
        past = self.mod.time.time() - 100
        with mock.patch.object(self.mod, "_get_access_token_msal", return_value=None), \
             mock.patch.object(self.mod, "_load_token",
                               return_value={"access_token": "old",
                                             "expires_at": past}), \
             mock.patch.object(self.mod, "_refresh_with_refresh_token",
                               return_value=None):
            self.assertIsNone(self.mod.get_access_token())


# ─── _graph_get HTTP layer ───────────────────────────────────────────────

class GraphGetHttpTests(_MsGraphBase):
    def test_success_parses_json(self):
        resp = _fake_response({"value": [{"id": "1"}]})
        with mock.patch.object(self.mod, "get_access_token", return_value="tok"), \
             mock.patch.object(self.mod.urllib.request, "urlopen",
                               return_value=resp) as up:
            out = self.mod._graph_get("/me/messages", {"$top": "5"})
        self.assertEqual(out, {"value": [{"id": "1"}]})
        # Auth header + query string were applied ($ is urlencoded to %24).
        req = up.call_args[0][0]
        self.assertIn("%24top=5", req.full_url)
        self.assertEqual(req.get_header("Authorization"), "Bearer tok")

    def test_http_error_returns_none(self):
        with mock.patch.object(self.mod, "get_access_token", return_value="tok"), \
             mock.patch.object(self.mod.urllib.request, "urlopen",
                               side_effect=_http_error(401)):
            self.assertIsNone(self.mod._graph_get("/me/messages"))

    def test_http_error_detail_read_failure_swallowed(self):
        # e.read() itself raising must not break the handler.
        err = _http_error(500)
        err.read = mock.Mock(side_effect=RuntimeError("no body"))
        with mock.patch.object(self.mod, "get_access_token", return_value="tok"), \
             mock.patch.object(self.mod.urllib.request, "urlopen",
                               side_effect=err):
            self.assertIsNone(self.mod._graph_get("/x"))

    def test_generic_exception_returns_none(self):
        with mock.patch.object(self.mod, "get_access_token", return_value="tok"), \
             mock.patch.object(self.mod.urllib.request, "urlopen",
                               side_effect=urllib.error.URLError("timeout")):
            self.assertIsNone(self.mod._graph_get("/x"))


# ─── _graph_call HTTP layer ──────────────────────────────────────────────

class GraphCallHttpTests(_MsGraphBase):
    def test_no_token_returns_status_zero(self):
        with mock.patch.object(self.mod, "get_access_token", return_value=None):
            self.assertEqual(self.mod._graph_call("POST", "/x"), (0, None))

    def test_204_no_content(self):
        resp = _fake_response(None, status=204)
        with mock.patch.object(self.mod, "get_access_token", return_value="tok"), \
             mock.patch.object(self.mod.urllib.request, "urlopen",
                               return_value=resp):
            self.assertEqual(self.mod._graph_call("DELETE", "/x"), (204, None))

    def test_200_with_json_body_and_headers(self):
        resp = _fake_response({"id": "draft-1"}, status=201)
        with mock.patch.object(self.mod, "get_access_token", return_value="tok"), \
             mock.patch.object(self.mod.urllib.request, "urlopen",
                               return_value=resp) as up:
            status, payload = self.mod._graph_call(
                "POST", "/me/messages/1/createReply",
                body={"comment": "hi"}, params={"x": "y"})
        self.assertEqual((status, payload), (201, {"id": "draft-1"}))
        req = up.call_args[0][0]
        self.assertEqual(req.get_method(), "POST")
        self.assertEqual(req.get_header("Content-type"), "application/json")
        self.assertIn("x=y", req.full_url)

    def test_200_with_unparseable_body_returns_none_payload(self):
        resp = _fake_response(b"<<not json>>", status=200)
        with mock.patch.object(self.mod, "get_access_token", return_value="tok"), \
             mock.patch.object(self.mod.urllib.request, "urlopen",
                               return_value=resp):
            self.assertEqual(self.mod._graph_call("POST", "/x"), (200, None))

    def test_http_error_returns_code(self):
        # 429 (throttle) and 4xx/5xx all surface their status code to callers.
        with mock.patch.object(self.mod, "get_access_token", return_value="tok"), \
             mock.patch.object(self.mod.urllib.request, "urlopen",
                               side_effect=_http_error(429)):
            self.assertEqual(self.mod._graph_call("PATCH", "/x"), (429, None))

    def test_http_error_detail_read_failure_swallowed(self):
        err = _http_error(503)
        err.read = mock.Mock(side_effect=RuntimeError("no body"))
        with mock.patch.object(self.mod, "get_access_token", return_value="tok"), \
             mock.patch.object(self.mod.urllib.request, "urlopen",
                               side_effect=err):
            self.assertEqual(self.mod._graph_call("POST", "/x"), (503, None))

    def test_generic_exception_returns_status_zero(self):
        with mock.patch.object(self.mod, "get_access_token", return_value="tok"), \
             mock.patch.object(self.mod.urllib.request, "urlopen",
                               side_effect=urllib.error.URLError("net down")):
            self.assertEqual(self.mod._graph_call("POST", "/x"), (0, None))


# ─── parsing + getter edge cases ─────────────────────────────────────────

class ParseGraphStartNonUtcTests(_MsGraphBase):
    def test_non_utc_timezone_returned_as_is(self):
        # A non-UTC tz string keeps the parsed naive dt unchanged (line 421).
        evt = {"start": {"dateTime": "2026-06-01T09:30:00",
                         "timeZone": "Pacific Standard Time"}}
        dt = self.mod._parse_graph_start(evt)
        self.assertEqual((dt.hour, dt.minute), (9, 30))
        self.assertIsNone(dt.tzinfo)


class GetUpcomingEventsEdgeTests(_MsGraphBase):
    def test_organizer_extraction_failure_yields_empty_string(self):
        # organizer present but malformed → caught, organizer == "".
        body = {"value": [
            {"subject": "Solo block",
             "start": {"dateTime": "2026-06-01T09:00:00", "timeZone": "UTC"},
             "organizer": "not-a-dict"},
        ]}
        with mock.patch.object(self.mod, "_graph_get", return_value=body):
            events = self.mod.get_upcoming_events()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["organizer"], "")

    def test_top_n_floored_to_one(self):
        # top_n <= 0 is clamped to 1 in the $top param; assert the call is made.
        with mock.patch.object(self.mod, "_graph_get",
                               return_value={"value": []}) as gg:
            self.mod.get_upcoming_events(top_n=0)
        params = gg.call_args[0][1]
        self.assertEqual(params["$top"], "1")


class GetFirstMeetingTests(_MsGraphBase):
    def test_returns_first_event(self):
        evt = {"start": datetime.datetime(2026, 6, 1, 9), "subject": "S",
               "organizer": ""}
        with mock.patch.object(self.mod, "get_upcoming_events", return_value=[evt]):
            self.assertEqual(self.mod.get_first_meeting(), evt)

    def test_returns_none_when_empty(self):
        with mock.patch.object(self.mod, "get_upcoming_events", return_value=[]):
            self.assertIsNone(self.mod.get_first_meeting())


class UnreadMailCountTests(_MsGraphBase):
    def test_returns_int_count(self):
        with mock.patch.object(self.mod, "_graph_get",
                               return_value={"unreadItemCount": 7}):
            self.assertEqual(self.mod.get_unread_mail_count(), 7)

    def test_none_when_field_absent(self):
        with mock.patch.object(self.mod, "_graph_get", return_value={}):
            self.assertIsNone(self.mod.get_unread_mail_count())

    def test_none_when_value_not_intlike(self):
        with mock.patch.object(self.mod, "_graph_get",
                               return_value={"unreadItemCount": "lots"}):
            self.assertIsNone(self.mod.get_unread_mail_count())


class IsConfiguredTests(_MsGraphBase):
    def test_true_when_msal_app_present(self):
        with mock.patch.object(self.mod, "_msal_app", return_value=object()):
            self.assertTrue(self.mod.is_configured())

    def test_true_when_token_file_present(self):
        with mock.patch.object(self.mod, "_msal_app", return_value=None), \
             mock.patch.object(self.mod, "_load_token",
                               return_value={"access_token": "t"}):
            self.assertTrue(self.mod.is_configured())


class TeamsUnreadEdgeTests(_MsGraphBase):
    def test_all_read_returns_zero_count_empty_sender(self):
        # No unread chats at all → {"count": 0, "top_sender": ""} (line 523).
        body = {"value": [
            {"lastMessagePreview": {"createdDateTime": "2026-06-01T08:00:00Z",
                                    "from": {"user": {"displayName": "Bob Jones"}}},
             "viewpoint": {"lastMessageReadDateTime": "2026-06-01T09:00:00Z"}},
        ]}
        with mock.patch.object(self.mod, "_graph_get", return_value=body):
            res = self.mod.get_teams_unread_count()
        self.assertEqual(res, {"count": 0, "top_sender": ""})

    def test_chat_without_preview_skipped(self):
        body = {"value": [{"viewpoint": {}}]}  # no lastMessagePreview
        with mock.patch.object(self.mod, "_graph_get", return_value=body):
            res = self.mod.get_teams_unread_count()
        self.assertEqual(res["count"], 0)

    def test_unread_with_no_prior_read_counts(self):
        # last_read_dt empty → unread regardless; sender first-name extracted.
        body = {"value": [
            {"lastMessagePreview": {"createdDateTime": "2026-06-01T12:00:00Z",
                                    "from": {"user": {"displayName": "Dana Scully"}}},
             "viewpoint": {}},
        ]}
        with mock.patch.object(self.mod, "_graph_get", return_value=body):
            res = self.mod.get_teams_unread_count()
        self.assertEqual(res, {"count": 1, "top_sender": "Dana"})


class ListUnreadMessagesTests(_MsGraphBase):
    def test_shapes_each_message(self):
        body = {"value": [
            {"id": "1", "isRead": False, "subject": "A"},
            {"id": "2", "isRead": False, "subject": "B"},
        ]}
        with mock.patch.object(self.mod, "_graph_get", return_value=body) as gg:
            out = self.mod.list_unread_messages(top_n=5)
        self.assertEqual([m["id"] for m in out], ["1", "2"])
        # top_n is clamped into [1, 50].
        self.assertEqual(gg.call_args[0][1]["$top"], "5")

    def test_top_n_clamped_to_50(self):
        with mock.patch.object(self.mod, "_graph_get",
                               return_value={"value": []}) as gg:
            self.mod.list_unread_messages(top_n=999)
        self.assertEqual(gg.call_args[0][1]["$top"], "50")

    def test_empty_value_returns_empty_list(self):
        with mock.patch.object(self.mod, "_graph_get", return_value={}):
            self.assertEqual(self.mod.list_unread_messages(), [])


class GetMessageThreadEdgeTests(_MsGraphBase):
    def test_plaintext_body_not_html_stripped(self):
        # contentType != html → body_text is the raw text, body_html is "".
        body = {
            "id": "m2",
            "body": {"contentType": "text", "content": "  plain words  "},
            "toRecipients": [],
            "conversationId": "c2",
        }
        with mock.patch.object(self.mod, "_graph_get", return_value=body):
            thread = self.mod.get_message_thread("m2")
        self.assertEqual(thread["body_text"], "plain words")
        self.assertEqual(thread["body_html"], "")

    def test_none_when_graph_returns_none(self):
        with mock.patch.object(self.mod, "_graph_get", return_value=None):
            self.assertIsNone(self.mod.get_message_thread("m1"))


# ─── write-helper guards + body construction ─────────────────────────────

class WriteHelperGuardTests(_MsGraphBase):
    def test_create_draft_reply_empty_id_none(self):
        with mock.patch.object(self.mod, "_graph_call") as call:
            self.assertIsNone(self.mod.create_draft_reply("", "body"))
        call.assert_not_called()

    def test_create_draft_reply_all_uses_createReplyAll(self):
        with mock.patch.object(self.mod, "_graph_call",
                               return_value=(201, {"id": "d"})) as call:
            self.mod.create_draft_reply("m1", "txt", reply_all=True)
        path = call.call_args[0][1]
        self.assertTrue(path.endswith("/createReplyAll"))

    def test_update_draft_body_empty_id_false(self):
        with mock.patch.object(self.mod, "_graph_call") as call:
            self.assertFalse(self.mod.update_draft_body("", "x"))
        call.assert_not_called()

    def test_update_draft_body_sends_text_content(self):
        with mock.patch.object(self.mod, "_graph_call",
                               return_value=(200, {})) as call:
            ok = self.mod.update_draft_body("d1", "new text")
        self.assertTrue(ok)
        body = call.call_args.kwargs.get("body") or call.call_args[0][2]
        self.assertEqual(body["body"]["content"], "new text")
        self.assertEqual(body["body"]["contentType"], "Text")

    def test_update_draft_body_4xx_false(self):
        with mock.patch.object(self.mod, "_graph_call", return_value=(400, None)):
            self.assertFalse(self.mod.update_draft_body("d1", "x"))

    def test_apply_category_empty_args_false(self):
        with mock.patch.object(self.mod, "_graph_call") as call:
            self.assertFalse(self.mod.apply_category("", "Cat"))
            self.assertFalse(self.mod.apply_category("m1", ""))
        call.assert_not_called()

    def test_apply_category_skips_duplicate(self):
        # Category already present → not appended twice, but still PATCHes.
        with mock.patch.object(self.mod, "_graph_get",
                               return_value={"categories": ["Urgent"]}), \
             mock.patch.object(self.mod, "_graph_call",
                               return_value=(200, {})) as call:
            ok = self.mod.apply_category("m1", "Urgent")
        self.assertTrue(ok)
        body = call.call_args.kwargs.get("body") or call.call_args[0][2]
        self.assertEqual(body["categories"], ["Urgent"])

    def test_mark_as_read_empty_id_false(self):
        with mock.patch.object(self.mod, "_graph_call") as call:
            self.assertFalse(self.mod.mark_as_read(""))
        call.assert_not_called()

    def test_mark_as_read_sends_isread_flag(self):
        with mock.patch.object(self.mod, "_graph_call",
                               return_value=(200, {})) as call:
            self.mod.mark_as_read("m1", read=False)
        body = call.call_args.kwargs.get("body") or call.call_args[0][2]
        self.assertEqual(body["isRead"], False)

    def test_send_draft_4xx_false(self):
        with mock.patch.object(self.mod, "_graph_call", return_value=(403, None)):
            self.assertFalse(self.mod.send_draft("d1"))


# ─── authenticate_device_flow ────────────────────────────────────────────

class DeviceFlowTests(_MsGraphBase):
    def test_returns_false_when_app_none(self):
        with mock.patch.object(self.mod, "_msal_app", return_value=None):
            self.assertFalse(self.mod.authenticate_device_flow())

    def test_returns_false_on_flow_init_failure(self):
        app = mock.MagicMock()
        app.initiate_device_flow.return_value = {"error": "bad_client"}  # no user_code
        with mock.patch.object(self.mod, "_msal_app", return_value=app), \
             mock.patch.object(self.mod, "_config", return_value=["Scope.Read"]):
            self.assertFalse(self.mod.authenticate_device_flow())

    def test_success_path_returns_true(self):
        app = mock.MagicMock()
        app.initiate_device_flow.return_value = {
            "user_code": "ABCD-1234",
            "message": "Go to the URL and enter ABCD-1234"}
        app.acquire_token_by_device_flow.return_value = {"access_token": "tok"}
        with mock.patch.object(self.mod, "_msal_app", return_value=app), \
             mock.patch.object(self.mod, "_save_msal_cache") as save, \
             mock.patch.object(self.mod, "_config", return_value=["Scope.Read"]):
            self.assertTrue(self.mod.authenticate_device_flow())
        save.assert_called_once_with(app)

    def test_failure_path_returns_false(self):
        app = mock.MagicMock()
        app.initiate_device_flow.return_value = {
            "user_code": "X", "message": "msg"}
        app.acquire_token_by_device_flow.return_value = {
            "error_description": "user declined"}
        with mock.patch.object(self.mod, "_msal_app", return_value=app), \
             mock.patch.object(self.mod, "_save_msal_cache"), \
             mock.patch.object(self.mod, "_config", return_value=["Scope.Read"]):
            self.assertFalse(self.mod.authenticate_device_flow())


# ─── calendar action + orchestrator sub-agent wiring ─────────────────────
#
# ms_graph now registers a string-returning calendar action so the
# orchestrator's `calendar_scanner` sub-agent (previously inert) can dispatch a
# real, registered read. These tests cover the action's formatting + graceful
# degradation (Graph fully mocked, no network) and validate the JSON spec
# parses + references a now-registered action name.

class CalendarActionTests(_MsGraphBase):
    def test_not_configured_returns_friendly_setup_line(self):
        # No creds → an honest 'not set up' line, never a crash, never events.
        with mock.patch.object(self.mod, "is_configured", return_value=False):
            out = self.mod.action_calendar_today("today")
        self.assertIn("isn't set up", out)
        self.assertIn("--auth", out)

    def test_configured_empty_window_says_nothing_scheduled(self):
        with mock.patch.object(self.mod, "is_configured", return_value=True), \
             mock.patch.object(self.mod, "_graph_get", return_value={"value": []}):
            out = self.mod.action_calendar_today("today")
        self.assertIn("Nothing on the calendar", out)

    def test_configured_with_events_formats_chronologically(self):
        body = {"value": [
            {"subject": "Standup",
             "start": {"dateTime": "2026-06-02T09:00:00", "timeZone": "UTC"},
             "organizer": {"emailAddress": {"name": "Alice"}}},
            {"subject": "Lunch",
             "start": {"dateTime": "2026-06-02T12:30:00", "timeZone": "UTC"},
             "organizer": {"emailAddress": {"name": "Bob"}}},
        ]}
        with mock.patch.object(self.mod, "is_configured", return_value=True), \
             mock.patch.object(self.mod, "_graph_get", return_value=body):
            out = self.mod.action_calendar_today("today")
        # Real subjects/organizers surface; count header present. (Times are
        # tz-converted to naive-local so we don't assert exact HH:MM.)
        self.assertIn("2 events", out)
        self.assertIn("Standup", out)
        self.assertIn("Alice", out)
        self.assertIn("Lunch", out)
        self.assertIn("Bob", out)

    def test_event_without_organizer_omits_parenthetical(self):
        body = {"value": [
            {"subject": "Focus block",
             "start": {"dateTime": "2026-06-02T14:00:00", "timeZone": "UTC"}},
        ]}
        with mock.patch.object(self.mod, "is_configured", return_value=True), \
             mock.patch.object(self.mod, "_graph_get", return_value=body):
            out = self.mod.action_calendar_today("today")
        self.assertIn("Focus block", out)
        self.assertEqual(out.count("("), 0)   # no organizer → no "(name)"

    def test_graph_exception_degrades_without_raising(self):
        # An unexpected failure inside get_upcoming_events must be swallowed.
        with mock.patch.object(self.mod, "is_configured", return_value=True), \
             mock.patch.object(self.mod, "get_upcoming_events",
                               side_effect=RuntimeError("graph boom")):
            out = self.mod.action_calendar_today("today")
        self.assertIn("Couldn't reach the calendar", out)

    def test_is_configured_exception_treated_as_unconfigured(self):
        with mock.patch.object(self.mod, "is_configured",
                               side_effect=RuntimeError("msal boom")):
            out = self.mod.action_calendar_today("today")
        self.assertIn("isn't set up", out)

    def test_tomorrow_window_passed_through(self):
        # The arg keyword reaches get_upcoming_events as the window.
        with mock.patch.object(self.mod, "is_configured", return_value=True), \
             mock.patch.object(self.mod, "get_upcoming_events",
                               return_value=[]) as gue:
            self.mod.action_calendar_today("tomorrow")
        self.assertEqual(gue.call_args.kwargs.get("when"), "tomorrow")

    def test_normalise_when_maps_free_text(self):
        n = self.mod._normalise_when
        self.assertEqual(n(""), "today")
        self.assertEqual(n("TODAY"), "today")
        self.assertEqual(n("what's on tomorrow"), "tomorrow")
        self.assertEqual(n("this week"), "next_14_days")
        self.assertEqual(n("next 14 days"), "next_14_days")
        self.assertEqual(n("upcoming"), "next_14_days")
        self.assertEqual(n("gibberish"), "today")

    def test_action_always_returns_str(self):
        # Contract for the orchestrator worker: Callable[[str], str].
        with mock.patch.object(self.mod, "is_configured", return_value=False):
            self.assertIsInstance(self.mod.action_calendar_today(""), str)


class CalendarRegistrationTests(_MsGraphBase):
    """register() must expose the action under every name the calendar_scanner
    sub-agent spec references, all pointing at the one string-returning call."""
    def test_register_populates_spec_action_names(self):
        actions = {}
        self.mod.register(actions)
        for name in ("ms_graph_calendar", "calendar_today", "calendar_next"):
            self.assertIn(name, actions)
            self.assertTrue(callable(actions[name]))
        # All three are the same underlying action.
        self.assertIs(actions["ms_graph_calendar"], actions["calendar_today"])
        self.assertIs(actions["calendar_today"], actions["calendar_next"])


class CalendarSubAgentSpecWiringTests(unittest.TestCase):
    """The orchestrator sub-agent JSON parses AND each action it references is
    now registered by a freshly-built ms_graph ACTIONS (Graph mocked, no net).

    This is the cross-module guard the task asks for: it fails if someone edits
    the spec to reference an action ms_graph doesn't register, or removes the
    registration — i.e. it would have caught the original 'inert sub-agent' bug.
    """
    def setUp(self):
        self._proj = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        self._spec_path = os.path.join(
            self._proj, "skills", "sub_agents", "calendar_scanner.json")

    def test_spec_parses_and_is_well_formed(self):
        with open(self._spec_path, "r", encoding="utf-8") as f:
            spec = json.load(f)
        self.assertEqual(spec["name"], "calendar_scanner")
        self.assertIsInstance(spec.get("allowed_actions"), list)
        self.assertTrue(spec["allowed_actions"])
        # Sub-agent model preference must be an un-dated keyword, not a dated pin.
        self.assertEqual(spec.get("model_preference"), "haiku")

    def test_spec_actions_are_registered_by_ms_graph(self):
        with open(self._spec_path, "r", encoding="utf-8") as f:
            spec = json.load(f)
        # Build the real ms_graph ACTIONS via the skill harness (register()
        # runs; no network — the action isn't *called* here, only registered).
        _mod, actions = load_skill_isolated("ms_graph")
        registered = set(actions)
        # At least one allowed_action must resolve (the worker auto-runs the
        # FIRST registered one); assert the first listed is among them so the
        # sub-agent can't silently go inert again.
        self.assertIn(spec["allowed_actions"][0], registered)
        for name in spec["allowed_actions"]:
            self.assertIn(
                name, registered,
                f"spec action {name!r} is not registered by ms_graph",
            )


if __name__ == "__main__":
    unittest.main()
