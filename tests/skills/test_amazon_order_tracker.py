"""Logic tests for skills/amazon_order_tracker.py.

Watches Amazon shipment email (via the email_triage backends) and announces
status transitions. The skill persists state to data/amazon_order_state.json
and reads a feature flag off bobert_companion, so tests redirect _STATE_FILE to
a temp path and mock _save_state / _proactive_announce — nothing real is
written and no email backend / network is touched.

Core coverage: the subject/snippet → status classifier, delay detection, order
id + estimated-delivery parsing, the Amazon-sender heuristic, the announcement
phrasing, and the heart of the skill — _process_messages, the rank-advancing
diff engine (announce-on-forward-progress, dedup on reprocess, delay
side-channel, shipped-with-ETA promise) — plus the three voice actions.
"""
from __future__ import annotations

import os
import tempfile
import time
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


class AmazonTestBase(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("amazon_order_tracker")
        self._tmp = tempfile.TemporaryDirectory()
        self.mod._STATE_FILE = os.path.join(self._tmp.name, "amazon_state.json")
        self.addCleanup(self._tmp.cleanup)


class ClassifyTests(AmazonTestBase):
    def test_delivered(self):
        self.assertEqual(self.mod._classify("Delivered: your package", ""), "delivered")

    def test_out_for_delivery_beats_generic(self):
        self.assertEqual(self.mod._classify("Out for delivery today", ""),
                         "out_for_delivery")

    def test_arriving_today_is_out_for_delivery(self):
        self.assertEqual(self.mod._classify("Arriving today", ""), "out_for_delivery")

    def test_shipped(self):
        self.assertEqual(self.mod._classify("Your package has shipped", ""), "shipped")

    def test_ordered(self):
        self.assertEqual(self.mod._classify("Thank you for your order", ""), "ordered")

    def test_unknown(self):
        self.assertEqual(self.mod._classify("Weekly Amazon deals", ""), "unknown")


class DelayAndIdTests(AmazonTestBase):
    def test_delay_detection(self):
        self.assertTrue(self.mod._is_delayed("Your delivery is delayed", ""))
        self.assertTrue(self.mod._is_delayed("", "running late, sorry"))
        self.assertFalse(self.mod._is_delayed("On its way", ""))

    def test_extract_order_id_from_subject(self):
        self.assertEqual(
            self.mod._extract_order_id("Shipped: 113-1234567-7654321", ""),
            "113-1234567-7654321")

    def test_extract_order_id_from_snippet_fallback(self):
        self.assertEqual(
            self.mod._extract_order_id("Your package", "order 123-7654321-1234567"),
            "123-7654321-1234567")

    def test_extract_order_id_none(self):
        self.assertIsNone(self.mod._extract_order_id("no id here", "still none"))


class EstimatedDeliveryTests(AmazonTestBase):
    def test_parses_weekday_month_day(self):
        ref = time.mktime((2026, 6, 1, 12, 0, 0, 0, 0, -1))
        est = self.mod._extract_estimated_delivery("Arriving Mon, Jun 3", ref_ts=ref)
        self.assertIsNotNone(est)
        self.assertEqual(time.strftime("%Y-%m-%d", time.localtime(est)), "2026-06-03")

    def test_full_month_name(self):
        ref = time.mktime((2026, 6, 1, 12, 0, 0, 0, 0, -1))
        est = self.mod._extract_estimated_delivery("Arriving Monday, June 5", ref_ts=ref)
        self.assertEqual(time.strftime("%m-%d", time.localtime(est)), "06-05")

    def test_no_match_returns_none(self):
        self.assertIsNone(self.mod._extract_estimated_delivery("ships soon", 0))

    def test_year_rolls_forward_for_stale_month(self):
        # A 'Jan' arrival parsed against a December ref → next year.
        ref = time.mktime((2026, 12, 20, 12, 0, 0, 0, 0, -1))
        est = self.mod._extract_estimated_delivery("Arriving Wed, Jan 6", ref_ts=ref)
        self.assertEqual(time.localtime(est).tm_year, 2027)


class IsAmazonMessageTests(AmazonTestBase):
    def test_by_sender_address(self):
        self.assertTrue(self.mod._is_amazon_message(
            {"from_addr": "shipment-tracking@amazon.com"}))

    def test_by_sender_domain_variant(self):
        self.assertTrue(self.mod._is_amazon_message(
            {"from_addr": "noreply@marketplace.amazon.co.uk"}))

    def test_by_subject_with_shipping_context(self):
        self.assertTrue(self.mod._is_amazon_message(
            {"subject": "Your Amazon.com order has shipped"}))

    def test_marketing_without_context_rejected(self):
        # 'amazon' in subject but no shipping keyword → not a tracked message.
        self.assertFalse(self.mod._is_amazon_message(
            {"from_addr": "deals@nike.com", "subject": "Win an Amazon gift card"}))


class AnnounceTests(AmazonTestBase):
    def test_each_status_phrasing(self):
        self.assertIn("has been delivered",
                      self.mod._announce_message("delivered", "113-1"))
        self.assertIn("out for delivery",
                      self.mod._announce_message("out_for_delivery", "113-1"))
        self.assertIn("has shipped", self.mod._announce_message("shipped", "113-1"))
        self.assertIn("appears to be delayed",
                      self.mod._announce_message("delayed", "113-1"))

    def test_ordered_is_silent(self):
        self.assertEqual(self.mod._announce_message("ordered", "113-1"), "")

    def test_humanise_status(self):
        self.assertEqual(self.mod._humanise_status("out_for_delivery"), "out for delivery")


class ProcessMessagesTests(AmazonTestBase):
    def _process(self, msgs, state=None):
        state = state if state is not None else {"orders": {}}
        announces = []
        with mock.patch.object(self.mod, "_proactive_announce",
                               side_effect=lambda m: announces.append(m)):
            state = self.mod._process_messages(msgs, state)
        return state, announces

    def test_forward_progress_announces_each_rank(self):
        oid = "113-1234567-7654321"
        msgs = [
            {"id": "a1", "subject": f"Shipped: {oid}", "snippet": "",
             "received": "2026-06-01T08:00:00Z"},
            {"id": "a2", "subject": f"Out for delivery: {oid}", "snippet": "",
             "received": "2026-06-01T12:00:00Z"},
        ]
        state, announces = self._process(msgs)
        self.assertEqual(state["orders"][oid]["status"], "out_for_delivery")
        self.assertTrue(any("has shipped" in a for a in announces))
        self.assertTrue(any("out for delivery" in a for a in announces))

    def test_reprocess_does_not_reannounce(self):
        oid = "113-1234567-7654321"
        msgs = [{"id": "a1", "subject": f"Shipped: {oid}", "snippet": "",
                 "received": "2026-06-01T08:00:00Z"}]
        state, first = self._process(msgs)
        _state, second = self._process(msgs, state)
        self.assertTrue(first)          # announced the first time
        self.assertEqual(second, [])    # silent on reprocess (dedup)

    def test_no_backward_transition(self):
        # A late 'shipped' email after 'delivered' must not regress the status.
        oid = "113-1234567-7654321"
        state, _ = self._process([
            {"id": "d1", "subject": f"Delivered: {oid}", "snippet": "",
             "received": "2026-06-02T10:00:00Z"}])
        self.assertEqual(state["orders"][oid]["status"], "delivered")
        state, announces = self._process([
            {"id": "s_late", "subject": f"Shipped: {oid}", "snippet": "",
             "received": "2026-06-01T08:00:00Z"}], state)
        self.assertEqual(state["orders"][oid]["status"], "delivered")  # unchanged
        self.assertEqual(announces, [])

    def test_delay_side_channel_announced_once(self):
        oid = "113-1234567-7654321"
        msgs = [{"id": "x1", "subject": f"Shipped: {oid} — delayed", "snippet": "",
                 "received": "2026-06-01T08:00:00Z"}]
        state, announces = self._process(msgs)
        self.assertTrue(any("delayed" in a for a in announces))
        self.assertTrue(state["orders"][oid]["delayed_announced"])

    def test_shipped_with_eta_schedules_promise(self):
        oid = "113-1234567-7654321"
        future = time.strftime("%b %d", time.localtime(time.time() + 3 * 86400))
        msgs = [{"id": "s1",
                 "subject": f"Shipped: {oid}. Arriving {future}",
                 "snippet": "",
                 "received": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}]
        with mock.patch.object(self.mod, "_make_promise") as promise, \
             mock.patch.object(self.mod, "_proactive_announce"):
            self.mod._process_messages(msgs, {"orders": {}})
        promise.assert_called_once()
        # The promise condition is a time_at trigger.
        self.assertEqual(promise.call_args[0][1], "time_at")

    def test_message_without_order_id_uses_msg_handle(self):
        msgs = [{"id": "abc123def456", "subject": "Your package has shipped",
                 "snippet": "", "received": "2026-06-01T08:00:00Z"}]
        state, announces = self._process(msgs)
        # Falls back to a msg:-prefixed synthetic order key.
        keys = list(state["orders"].keys())
        self.assertTrue(keys[0].startswith("msg:"))
        # Spoken label is generic, not the synthetic id.
        self.assertTrue(any("an Amazon order" in a for a in announces))


class FmtDateTests(AmazonTestBase):
    def test_formats_weekday_month_day(self):
        ts = time.mktime((2026, 6, 1, 12, 0, 0, 0, 0, -1))  # Mon, Jun 1 2026
        out = self.mod._fmt_date(ts)
        self.assertIn("Jun", out)
        self.assertIn("1", out)

    def test_empty_for_zero(self):
        self.assertEqual(self.mod._fmt_date(0), "")
        self.assertEqual(self.mod._fmt_date(None), "")


class ActionTests(AmazonTestBase):
    def test_check_orders_no_state(self):
        with mock.patch.object(self.mod, "_load_state",
                               return_value={"orders": {}}):
            out = self.actions["check_orders"]("")
        self.assertIn("No active Amazon orders", out)

    def test_check_orders_nothing_in_transit(self):
        state = {"orders": {"o1": {"status": "delivered", "last_seen_ts": time.time()}}}
        with mock.patch.object(self.mod, "_load_state", return_value=state):
            out = self.actions["check_orders"]("")
        self.assertIn("Nothing currently in transit", out)

    def test_check_orders_lists_in_flight(self):
        state = {"orders": {
            "113-1234567-7654321": {"status": "out_for_delivery",
                                     "last_seen_ts": time.time(),
                                     "est_delivery_ts": time.time() + 86400},
        }}
        with mock.patch.object(self.mod, "_load_state", return_value=state):
            out = self.actions["check_orders"]("")
        self.assertIn("113-1234567-7654321", out)
        self.assertIn("out for delivery", out)

    def test_recent_delivery_none(self):
        with mock.patch.object(self.mod, "_load_state", return_value={"orders": {}}):
            out = self.actions["recent_delivery"]("")
        self.assertIn("No Amazon deliveries", out)

    def test_recent_delivery_within_window(self):
        state = {"orders": {
            "113-1234567-7654321": {"status": "delivered",
                                     "last_seen_ts": time.time() - 86400},
        }}
        with mock.patch.object(self.mod, "_load_state", return_value=state):
            out = self.actions["recent_delivery"]("")
        self.assertIn("Recently delivered", out)
        self.assertIn("113-1234567-7654321", out)

    def test_tracking_status_reports_flags(self):
        state = {"orders": {"o1": {}}, "last_poll_ts": time.time(), "last_error": ""}
        with mock.patch.object(self.mod, "_load_state", return_value=state), \
             mock.patch.object(self.mod, "_read_feature_flag", return_value=True), \
             mock.patch.object(self.mod, "_outlook_configured", return_value=True), \
             mock.patch.object(self.mod, "_gmail_configured", return_value=False):
            out = self.actions["amazon_tracking_status"]("")
        self.assertIn("enabled: True", out)
        self.assertIn("Outlook: configured", out)
        self.assertIn("Gmail: not configured", out)
        self.assertIn("orders tracked: 1", out)
        self.assertTrue(out.rstrip().endswith("sir."))


if __name__ == "__main__":
    unittest.main()
