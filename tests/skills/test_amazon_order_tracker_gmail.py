"""Gmail read-only sign-in, honest failure and quiet first run for
skills/amazon_order_tracker.py (2026-10-12).

What was wrong and is pinned here:

  * the tracker could only use the mailbox access skills/email_triage holds
    (Gmail modify + compose + send scopes); it now has its OWN read-only
    credential (data/gmail_readonly_token.json, scope gmail.readonly only);
  * with no sign-in it said "no active Amazon orders" (which reads as "nothing
    is coming") and, if the flag was on, spoke a nag at every boot; now
    "I need a Google sign-in first, sir" when asked and silence otherwise, and
    the poller waits for a sign-in made later instead of never starting;
  * the first poll announced three transitions for every order of the last
    week; the first look is now silent, and a mail older than 48 h is never
    announced;
  * Amazon's subjects carry the item, not the order number, so "Shipped" and
    "Delivered" for one parcel were two unrelated "orders"; they are now one,
    named by the item;
  * a shipped mail whose preview says "will be delivered by Friday" was read as
    delivered, and "Ordered:" was not recognised at all.

No network, no Google libraries, no browser: the libraries and the Gmail client
are fakes; the token and state files live in a temp dir.
"""
from __future__ import annotations

import io
import json
import os
import sys
import types
import unittest
from unittest import mock

from tests.skills.test_amazon_order_tracker import (
    AmazonTestBase, _FakeGmailService, _fake_email_triage)
from tests._skill_harness import load_skill_isolated

READONLY = "https://www.googleapis.com/auth/gmail.readonly"


class _FakeCreds:
    def __init__(self, valid=True, expired=False, refresh_token="r",
                 refresh_raises=False):
        self.valid = valid
        self.expired = expired
        self.refresh_token = refresh_token
        self._refresh_raises = refresh_raises
        self.refreshed = False

    def refresh(self, _request):
        if self._refresh_raises:
            raise RuntimeError("invalid_grant")
        self.valid = True
        self.expired = False
        self.refreshed = True

    def to_json(self):
        return json.dumps({"token": "t", "refresh_token": self.refresh_token,
                           "scopes": [READONLY]})


class _Recorder(_FakeGmailService):
    """A Gmail double that records every call and has NO write methods."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.calls = []

    def list(self, **kw):
        self.calls.append(("list", kw))
        return super().list(**kw)

    def get(self, **kw):
        self.calls.append(("get", kw))
        return super().get(**kw)


def _google_modules(creds, service, scopes_seen, flow_log=None):
    """Fake google client libraries; Credentials records the scopes asked."""
    discovery = types.ModuleType("googleapiclient.discovery")
    discovery.build = lambda name, ver, credentials=None, cache_discovery=True: \
        service

    class Credentials:
        @staticmethod
        def from_authorized_user_file(path, scopes):
            scopes_seen.append(list(scopes))
            return creds
    oauth = types.ModuleType("google.oauth2.credentials")
    oauth.Credentials = Credentials
    req = types.ModuleType("google.auth.transport.requests")
    req.Request = lambda: object()

    class InstalledAppFlow:
        @staticmethod
        def from_client_secrets_file(path, scopes):
            if flow_log is not None:
                flow_log.append(("flow", path, list(scopes)))
            f = types.SimpleNamespace()
            f.run_local_server = lambda port=0: _FakeCreds()
            return f
    flow = types.ModuleType("google_auth_oauthlib.flow")
    flow.InstalledAppFlow = InstalledAppFlow
    pkgs = {
        "googleapiclient": types.ModuleType("googleapiclient"),
        "googleapiclient.discovery": discovery,
        "google": types.ModuleType("google"),
        "google.oauth2": types.ModuleType("google.oauth2"),
        "google.oauth2.credentials": oauth,
        "google.auth": types.ModuleType("google.auth"),
        "google.auth.transport": types.ModuleType("google.auth.transport"),
        "google.auth.transport.requests": req,
        "google_auth_oauthlib": types.ModuleType("google_auth_oauthlib"),
        "google_auth_oauthlib.flow": flow,
    }
    return pkgs


def _gmail_full(mid, subject, snippet="", addr="shipment-tracking@amazon.com",
                internal_ms=None):
    full = {"id": mid, "snippet": snippet,
            "payload": {"headers": [
                {"name": "From", "value": f"Amazon <{addr}>"},
                {"name": "Subject", "value": subject},
                {"name": "Date", "value": "Mon, 05 Oct 2026 08:00:00 +0000"}]}}
    if internal_ms is not None:
        full["internalDate"] = str(internal_ms)
    return full


class _Base(AmazonTestBase):
    def _token(self):
        with open(self.mod.GMAIL_READONLY_TOKEN_FILE, "w",
                  encoding="utf-8") as f:
            f.write("{}")

    def _signed_in(self, creds=None, service=None):
        """A read-only token on disk plus fake Google libraries."""
        self._token()
        self.scopes = []
        creds = creds or _FakeCreds()
        service = service if service is not None else _Recorder(
            list_result={"messages": []})
        mods = _google_modules(creds, service, self.scopes)
        ctx = mock.patch.dict(sys.modules, mods)
        ctx.start()
        self.addCleanup(ctx.stop)
        return creds, service


class HonestWhenNotSignedIn(_Base):
    SIGN_IN = "I need a Google sign-in first, sir"

    def test_the_problem_names_the_sign_in_and_the_command(self):
        with mock.patch.object(self.mod, "_email_triage", return_value=None), \
                mock.patch.object(self.mod, "_outlook_configured",
                                  return_value=False):
            out = self.mod._backend_problem()
        self.assertIn(self.SIGN_IN, out)
        self.assertIn("--auth-gmail", out)
        self.assertIn("read-only", out)

    def test_check_orders_says_so_instead_of_claiming_no_orders(self):
        with mock.patch.object(self.mod, "_email_triage", return_value=None), \
                mock.patch.object(self.mod, "_outlook_configured",
                                  return_value=False):
            out = self.actions["check_orders"]("")
            rec = self.actions["recent_delivery"]("")
        self.assertIn(self.SIGN_IN, out)
        self.assertIn(self.SIGN_IN, rec)
        self.assertNotIn("No active Amazon orders", out)
        self.assertNotIn("No Amazon deliveries", rec)

    def test_status_reports_not_signed_in(self):
        with mock.patch.object(self.mod, "_email_triage", return_value=None), \
                mock.patch.object(self.mod, "_outlook_configured",
                                  return_value=False):
            out = self.actions["amazon_tracking_status"]("")
        self.assertIn("Gmail: not signed in", out)
        self.assertIn(self.SIGN_IN, out)

    def test_orders_already_on_record_are_still_listed_with_the_reason(self):
        state = {"orders": {"113-1234567-7654321": {
            "status": "shipped", "last_seen_ts": 1e12}}}
        with mock.patch.object(self.mod, "_load_state", return_value=state), \
                mock.patch.object(self.mod, "_email_triage", return_value=None), \
                mock.patch.object(self.mod, "_outlook_configured",
                                  return_value=False):
            out = self.actions["check_orders"]("")
        self.assertIn("113-1234567-7654321", out)
        self.assertIn(self.SIGN_IN, out)

    def test_a_broken_sign_in_is_not_reported_as_never_signed_in(self):
        self._signed_in(creds=_FakeCreds(valid=False, expired=True,
                                         refresh_raises=True))
        with mock.patch.object(self.mod, "_email_triage", return_value=None), \
                mock.patch.object(self.mod, "_outlook_configured",
                                  return_value=False):
            self.assertIsNone(self.mod._readonly_service())
            out = self.mod._backend_problem()
        self.assertIn("needs doing again", out)
        self.assertNotIn(self.SIGN_IN, out)

    def test_missing_google_libraries_are_named(self):
        self._token()
        with mock.patch.object(self.mod, "_google_libs", return_value=None), \
                mock.patch.object(self.mod, "_email_triage", return_value=None), \
                mock.patch.object(self.mod, "_outlook_configured",
                                  return_value=False):
            out = self.mod._backend_problem()
        self.assertIn("libraries are not installed", out)

    def test_nothing_is_spoken_at_boot_when_not_signed_in(self):
        with mock.patch.object(self.mod, "_read_feature_flag", return_value=True), \
                mock.patch.object(self.mod, "_email_triage", return_value=None), \
                mock.patch.object(self.mod, "_outlook_configured",
                                  return_value=False), \
                mock.patch.object(self.mod, "_proactive_announce") as ann, \
                mock.patch.object(self.mod.threading.Thread, "start",
                                  lambda t: None):
            self.mod.register({})
        ann.assert_not_called()


class ReadOnlyGmailPath(_Base):
    def test_only_the_readonly_scope_is_ever_requested(self):
        self._signed_in()
        with mock.patch.object(self.mod, "_email_triage", return_value=None):
            self.assertIsNotNone(self.mod._readonly_service())
        self.assertEqual(self.scopes, [[READONLY]])
        for wide in ("modify", "compose", "send"):
            self.assertNotIn(wide, " ".join(self.scopes[0]))

    def test_the_fetch_only_lists_and_reads(self):
        svc = _Recorder(list_result={"messages": [{"id": "g1"}]},
                        full_by_id={"g1": _gmail_full(
                            "g1", "Shipped: “Anker Charger”")})
        self._signed_in(service=svc)
        with mock.patch.object(self.mod, "_email_triage", return_value=None):
            out = self.mod._fetch_gmail()
        self.assertEqual([m["id"] for m in out], ["g1"])
        self.assertEqual({c[0] for c in svc.calls}, {"list", "get"})
        self.assertTrue(self.mod._fetch_status["ok"])
        # metadata only: never the message bodies
        self.assertTrue(all(c[1].get("format") == "metadata"
                            for c in svc.calls if c[0] == "get"))
        self.assertIn("newer_than:7d", svc.calls[0][1]["q"])

    def test_it_works_without_email_triage_at_all(self):
        svc = _Recorder(list_result={"messages": [{"id": "g1"}]},
                        full_by_id={"g1": _gmail_full("g1", "Delivered: x")})
        self._signed_in(service=svc)
        with mock.patch.object(self.mod, "_email_triage", return_value=None):
            out = self.mod._fetch_gmail()
            self.assertTrue(self.mod._gmail_configured())
        m = out[0]
        self.assertEqual(m["from_addr"], "shipment-tracking@amazon.com")
        self.assertEqual(m["from_name"], "Amazon")
        self.assertEqual(m["subject"], "Delivered: x")

    def test_internal_date_becomes_the_received_time(self):
        svc = _Recorder(list_result={"messages": [{"id": "g1"}]},
                        full_by_id={"g1": _gmail_full(
                            "g1", "Shipped: x", internal_ms=1_790_000_000_000)})
        self._signed_in(service=svc)
        with mock.patch.object(self.mod, "_email_triage", return_value=None):
            out = self.mod._fetch_gmail()
        self.assertAlmostEqual(self.mod._parse_iso(out[0]["received"]),
                               1_790_000_000.0, delta=1)

    def test_it_is_preferred_over_the_email_triage_service(self):
        ro = _Recorder(list_result={"messages": []})
        self._signed_in(service=ro)
        wide = _Recorder(list_result={"messages": []})
        et = _fake_email_triage(gmail_service=wide)
        with mock.patch.object(self.mod, "_email_triage", return_value=et):
            self.mod._fetch_gmail()
        self.assertTrue(ro.calls)
        self.assertEqual(wide.calls, [])

    def test_without_a_token_the_email_triage_service_still_works(self):
        wide = _Recorder(list_result={"messages": [{"id": "g1"}]},
                         full_by_id={"g1": _gmail_full("g1", "Shipped: x")})
        et = _fake_email_triage(gmail_service=wide)
        with mock.patch.object(self.mod, "_email_triage", return_value=et):
            out = self.mod._fetch_gmail()
        self.assertEqual(len(out), 1)

    def test_an_expired_token_is_refreshed_and_saved(self):
        creds = _FakeCreds(valid=False, expired=True)
        self._signed_in(creds=creds)
        with mock.patch.object(self.mod, "_email_triage", return_value=None):
            svc = self.mod._readonly_service()
        self.assertIsNotNone(svc)
        self.assertTrue(creds.refreshed)
        with open(self.mod.GMAIL_READONLY_TOKEN_FILE, encoding="utf-8") as f:
            self.assertEqual(json.load(f)["scopes"], [READONLY])

    def test_a_revoked_token_gives_a_clear_problem_and_no_service(self):
        self._signed_in(creds=_FakeCreds(valid=False, expired=True,
                                         refresh_raises=True))
        with mock.patch.object(self.mod, "_email_triage", return_value=None):
            self.assertIsNone(self.mod._readonly_service())
        self.assertIn("refused", self.mod._ro_problem[0])
        with mock.patch.object(self.mod, "_email_triage", return_value=None):
            self.assertEqual(self.mod._fetch_gmail(), [])

    def test_a_failed_list_is_not_a_clean_poll(self):
        svc = _Recorder(list_raises=True)
        self._signed_in(service=svc)
        with mock.patch.object(self.mod, "_email_triage", return_value=None), \
                mock.patch.object(self.mod, "_fetch_outlook", return_value=[]):
            self.mod._poll_once()
        st = self.mod._load_state()
        self.assertEqual(st["last_poll_ts"], 0.0)
        self.assertFalse(st.get("seeded"))
        self.assertTrue(st["last_error"])

    def test_the_token_is_not_the_email_triage_one(self):
        self.assertTrue(self.mod.GMAIL_READONLY_TOKEN_FILE.endswith(
            "gmail_readonly_token.json"))
        self.assertEqual(self.mod.GMAIL_READONLY_SCOPE, READONLY)


class SignInFlow(_Base):
    def test_auth_saves_a_readonly_token(self):
        flow_log = []
        mods = _google_modules(_FakeCreds(), _Recorder(), [], flow_log)
        client = os.path.join(self._tmp.name, "gmail_credentials.json")
        with open(client, "w", encoding="utf-8") as f:
            f.write("{}")
        self.mod.GMAIL_CLIENT_FILE = client
        with mock.patch.dict(sys.modules, mods):
            ok = self.mod.authenticate_gmail_readonly()
        self.assertTrue(ok)
        self.assertEqual(flow_log, [("flow", client, [READONLY])])
        with open(self.mod.GMAIL_READONLY_TOKEN_FILE, encoding="utf-8") as f:
            self.assertEqual(json.load(f)["scopes"], [READONLY])

    def test_auth_without_the_client_file_says_what_to_do(self):
        mods = _google_modules(_FakeCreds(), _Recorder(), [])
        self.mod.GMAIL_CLIENT_FILE = os.path.join(self._tmp.name, "nope.json")
        buf = io.StringIO()
        with mock.patch.dict(sys.modules, mods), \
                mock.patch("sys.stdout", buf):
            ok = self.mod.authenticate_gmail_readonly()
        self.assertFalse(ok)
        self.assertIn("gmail_credentials.json", buf.getvalue())
        self.assertFalse(os.path.exists(self.mod.GMAIL_READONLY_TOKEN_FILE))

    def test_auth_without_the_libraries_says_so(self):
        buf = io.StringIO()
        with mock.patch.object(self.mod, "_google_libs", return_value=None), \
                mock.patch("sys.stdout", buf):
            self.assertFalse(self.mod.authenticate_gmail_readonly())
        self.assertIn("pip install", buf.getvalue())


class QuietFirstRun(_Base):
    def _poll(self, msgs, seeded_state=None):
        if seeded_state is not None:
            self.mod._save_state(seeded_state)

        def fetch():
            self.mod._fetch_status["ok"] = True
            return list(msgs)
        with mock.patch.object(self.mod, "_fetch_outlook", return_value=[]), \
                mock.patch.object(self.mod, "_fetch_gmail", side_effect=fetch), \
                mock.patch.object(self.mod, "_proactive_announce") as ann:
            self.mod._poll_once()
        return ann

    def _mail(self, mid, subject, hours_ago=1.0):
        import time as _t
        ts = _t.gmtime(_t.time() - hours_ago * 3600)
        return {"id": mid, "subject": subject, "snippet": "",
                "received": _t.strftime("%Y-%m-%dT%H:%M:%SZ", ts)}

    def test_the_first_look_announces_nothing_but_records_everything(self):
        oid = "113-1234567-7654321"
        ann = self._poll([self._mail("a", f"Shipped: {oid}", 30),
                          self._mail("b", f"Delivered: {oid}", 2)])
        ann.assert_not_called()
        st = self.mod._load_state()
        self.assertEqual(st["orders"][oid]["status"], "delivered")
        self.assertTrue(st["seeded"])

    def test_the_second_poll_announces_new_progress(self):
        oid = "113-1234567-7654321"
        self._poll([self._mail("a", f"Shipped: {oid}", 30)])
        ann = self._poll([self._mail("a", f"Shipped: {oid}", 30),
                          self._mail("b", f"Out for delivery: {oid}", 0.2)])
        self.assertEqual(ann.call_count, 1)
        self.assertIn("out for delivery", ann.call_args[0][0])

    def test_a_mail_older_than_two_days_is_recorded_but_not_spoken(self):
        oid = "113-1234567-7654321"
        self._poll([self._mail("a", f"Shipped: {oid}", 1)])
        ann = self._poll([self._mail("a", f"Shipped: {oid}", 1),
                          self._mail("old", f"Delivered: {oid}", 72)])
        ann.assert_not_called()
        self.assertEqual(self.mod._load_state()["orders"][oid]["status"],
                         "delivered")

    def test_a_state_from_before_the_flag_is_treated_as_live(self):
        oid = "113-1234567-7654321"
        legacy = {"orders": {}, "last_poll_ts": 123.0, "last_error": ""}
        ann = self._poll([self._mail("a", f"Shipped: {oid}", 1)],
                         seeded_state=legacy)
        self.assertEqual(ann.call_count, 1)


class ItemKeyedOrders(_Base):
    SHIP = "Shipped: “Anker 65W USB-C Charger”"
    OUT = "Out for delivery: “Anker 65W USB-C Charger”"
    DONE = "Delivered: “Anker 65W USB-C Charger”"

    def _run(self, subjects):
        msgs = [{"id": f"m{i}", "subject": s, "snippet": "",
                 "received": f"2026-10-0{i + 1}T08:00:00Z"}
                for i, s in enumerate(subjects)]
        said = []
        with mock.patch.object(self.mod, "_proactive_announce",
                               side_effect=said.append):
            state = self.mod._process_messages(msgs, {"orders": {}})
        return state, said

    def test_one_parcel_across_three_mails_is_one_order(self):
        state, said = self._run([self.SHIP, self.OUT, self.DONE])
        self.assertEqual(len(state["orders"]), 1)
        (key, entry), = state["orders"].items()
        self.assertTrue(key.startswith("item:"))
        self.assertEqual(entry["status"], "delivered")
        self.assertEqual(entry["item"], "Anker 65W USB-C Charger")
        self.assertEqual(len(said), 3)

    def test_it_is_named_by_the_item_not_a_synthetic_key(self):
        _state, said = self._run([self.SHIP])
        self.assertIn("Anker 65W USB-C Charger", said[0])
        self.assertNotIn("item:", said[0])
        self.assertNotIn("msg:", said[0])

    def test_two_different_items_stay_two_orders(self):
        state, _ = self._run([self.SHIP,
                              "Shipped: “Logitech Mouse”"])
        self.assertEqual(len(state["orders"]), 2)

    def test_a_long_item_is_shortened_for_speech(self):
        long_name = "Anker " + "Ultra Mega Charger " * 8
        item = self.mod._extract_item(f"Shipped: “{long_name}”")
        self.assertLessEqual(len(item), 64)
        self.assertTrue(item.endswith("..."))

    def test_an_order_number_still_wins_over_the_item(self):
        oid = "113-1234567-7654321"
        key = self.mod._order_key(f"Shipped: “X Y Z” {oid}", "", "m1")
        self.assertEqual(key, oid)

    def test_no_item_no_number_falls_back_to_the_message(self):
        self.assertTrue(self.mod._order_key("Your package", "", "abc").startswith(
            "msg:"))

    def test_the_roll_call_names_the_item(self):
        state, _ = self._run([self.SHIP])
        with mock.patch.object(self.mod, "_load_state", return_value=state), \
                mock.patch.object(self.mod, "_backend_problem", return_value=""):
            out = self.actions["check_orders"]("")
        self.assertIn("Anker 65W USB-C Charger", out)
        self.assertIn("shipped", out)


class ClassifierFixes(_Base):
    def test_the_subject_decides_before_the_preview(self):
        self.assertEqual(self.mod._classify(
            "Shipped: “X”", "Your package will be delivered by Friday"),
            "shipped")

    def test_a_preview_alone_still_classifies(self):
        self.assertEqual(self.mod._classify(
            "Your Amazon update", "Your package was delivered to the porch"),
            "delivered")

    def test_will_be_delivered_is_not_delivered(self):
        self.assertEqual(self.mod._classify(
            "Update", "it will be delivered tomorrow"), "unknown")

    def test_ordered_is_recognised(self):
        self.assertEqual(self.mod._classify("Ordered: “X”", ""),
                         "ordered")

    def test_arriving_tomorrow_is_shipped_and_today_is_out_for_delivery(self):
        self.assertEqual(self.mod._classify("Arriving tomorrow: X", ""),
                         "shipped")
        self.assertEqual(self.mod._classify("Arriving today: X", ""),
                         "out_for_delivery")


class PollerWaitsForASignIn(_Base):
    def test_no_backend_idles_and_never_polls(self):
        evt = mock.MagicMock()
        evt.is_set.return_value = False
        evt.wait.side_effect = [False, False, True]   # initial, idle, idle->stop
        with mock.patch.object(self.mod, "_any_backend", return_value=False), \
                mock.patch.object(self.mod, "_poll_once") as once:
            self.mod._poll_loop(evt)
        once.assert_not_called()
        self.assertEqual(evt.wait.call_args_list[1][0][0],
                         self.mod.NO_BACKEND_RECHECK_SECONDS)

    def test_a_sign_in_made_later_is_picked_up_without_a_restart(self):
        evt = mock.MagicMock()
        evt.is_set.return_value = False
        evt.wait.side_effect = [False, False, True]
        states = iter([False, True])
        with mock.patch.object(self.mod, "_any_backend",
                               side_effect=lambda: next(states)), \
                mock.patch.object(self.mod, "_poll_once") as once:
            self.mod._poll_loop(evt)
        once.assert_called_once()

    def test_the_wait_is_logged_once_not_every_minute(self):
        evt = mock.MagicMock()
        evt.is_set.return_value = False
        evt.wait.side_effect = [False, False, False, False, True]
        buf = io.StringIO()
        with mock.patch.object(self.mod, "_any_backend", return_value=False), \
                mock.patch.object(self.mod, "_poll_once"), \
                mock.patch("sys.stdout", buf):
            self.mod._poll_loop(evt)
        self.assertEqual(buf.getvalue().count("waiting for a Google sign-in"), 1)


class RefreshNeverHoldsTheVoiceLoop(unittest.TestCase):
    def test_a_slow_mailbox_is_answered_from_the_state_file(self):
        import threading
        import time
        mod, _a = load_skill_isolated("amazon_order_tracker", register=False)
        release = threading.Event()
        started = []
        mod._any_backend = lambda: True
        mod._poll_once = lambda: (started.append(1), release.wait(5.0))
        t0 = time.monotonic()
        mod._refresh_now(timeout=0.2)
        elapsed = time.monotonic() - t0
        release.set()
        self.assertLess(elapsed, 1.5)
        self.assertEqual(started, [1])

    def test_no_backend_means_no_refresh_thread(self):
        mod, _a = load_skill_isolated("amazon_order_tracker", register=False)
        mod._any_backend = lambda: False
        mod._poll_once = lambda: self.fail("polled with no backend")
        mod._refresh_now(timeout=0.2)


if __name__ == "__main__":
    unittest.main()
