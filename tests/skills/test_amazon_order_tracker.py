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

It also drives the I/O edges with injected fakes (NEVER real network/threads):
  • the Outlook / Gmail backend fetchers (a fake email_triage module exposing the
    same duck-typed surface the skill consults — _ms_graph / _graph_get /
    _shape_outlook_message, _gmail_service / is_gmail_available /
    _shape_gmail_message),
  • state load/save round-trips + the STATE_MAX_ORDERS cap, through a temp file,
  • the config/announce shims that lazy-import bobert_companion,
  • the poll-once / poll-loop / start- & stop-monitor lifecycle (Thread.start and
    Event.wait stubbed so nothing actually spawns, sleeps, or polls),
  • register() across its flag-off / no-backend / start paths.

ISOLATION: every fake module is installed only for the duration of a with-block
via ``inject_modules`` (save/restore of sys.modules AND the parent-package
attribute), so ``bobert_companion`` / ``skills.email_triage`` are never left
behind. The skill's own globals (_poll_thread, _stop_evt) live on the per-test
freshly-exec'd module the harness builds, and the poller is never really started.
"""
from __future__ import annotations

import contextlib
import io
import importlib.util
import json
import logging
import os
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


_SENTINEL = object()


@contextlib.contextmanager
def inject_modules(**mods):
    """Temporarily install fake modules into sys.modules. For dotted names
    (``skills.email_triage``) the leaf is ALSO set as an attribute on its already
    -imported parent package, because ``import a.b`` / ``from a import b`` resolve
    the leaf via ``getattr(parent, leaf)`` when the parent is a real package.
    Restores the previous state — including absence — on exit so the next test
    sees exactly what it injects. ``obj=None`` removes the module for the block.
    (Same contract as tests/skills/test_self_diagnostic.py:inject_modules.)
    """
    import sys
    saved_mod: dict[str, object] = {}
    missing: set[str] = set()
    saved_attr: list = []
    for name, obj in mods.items():
        saved_mod[name] = sys.modules.get(name, _SENTINEL)
        if saved_mod[name] is _SENTINEL:
            missing.add(name)
        if obj is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = obj
            if "." in name:
                parent_name, _, leaf = name.rpartition(".")
                parent = sys.modules.get(parent_name)
                if parent is not None:
                    saved_attr.append(
                        (parent, leaf, getattr(parent, leaf, _SENTINEL)))
                    setattr(parent, leaf, obj)
    try:
        yield
    finally:
        for parent, leaf, prev in reversed(saved_attr):
            if prev is _SENTINEL:
                try:
                    delattr(parent, leaf)
                except AttributeError:
                    pass
            else:
                setattr(parent, leaf, prev)
        for name in mods:
            prev = saved_mod.get(name, _SENTINEL)
            if name in missing:
                sys.modules.pop(name, None)
            elif prev is not _SENTINEL:
                sys.modules[name] = prev


def _fake_bc(**attrs):
    """A stand-in ``bobert_companion`` module exposing only the attributes the
    skill reads off it (AMAZON_TRACKING_ENABLED, proactive_announce)."""
    bc = types.ModuleType("bobert_companion")
    for k, v in attrs.items():
        setattr(bc, k, v)
    return bc


def _fake_ms_graph(*, configured=True, messages=None, get_raises=False,
                   shape=None, drop_graph_get=False, drop_shape=False):
    """Fake of the ms_graph module the Outlook fetcher pulls off email_triage.

    ``messages`` is the list returned under the Graph ``value`` key. ``shape``
    overrides the per-message normaliser (default mirrors the real
    _shape_outlook_message just enough for the fetcher)."""
    mg = types.SimpleNamespace()
    mg.is_configured = lambda: configured

    def _graph_get(path, params=None):
        if get_raises:
            raise RuntimeError("graph offline")
        return {"value": list(messages or [])}

    def _default_shape(m):
        sender = (m.get("from") or {}).get("emailAddress") or {}
        return {
            "id":        m.get("id") or "",
            "from_addr": (sender.get("address") or "").strip(),
            "from_name": (sender.get("name") or "").strip(),
            "subject":   (m.get("subject") or "").strip(),
            "snippet":   (m.get("bodyPreview") or "").strip(),
            "received":  m.get("receivedDateTime") or "",
        }

    if not drop_graph_get:
        mg._graph_get = _graph_get
    if not drop_shape:
        mg._shape_outlook_message = shape or _default_shape
    return mg


class _FakeGmailService:
    """Minimal Gmail discovery-client double: ``.users().messages().list(...)
    .execute()`` and ``.get(...).execute()`` chains the fetcher walks."""
    def __init__(self, list_result=None, full_by_id=None,
                 list_raises=False, get_raises_ids=()):
        self._list_result = list_result if list_result is not None else {}
        self._full_by_id = full_by_id or {}
        self._list_raises = list_raises
        self._get_raises_ids = set(get_raises_ids)

    def users(self):
        return self

    def messages(self):
        return self

    def list(self, **kwargs):
        svc = self

        class _Exec:
            def execute(self_inner):
                if svc._list_raises:
                    raise RuntimeError("gmail list 500")
                return svc._list_result
        return _Exec()

    def get(self, **kwargs):
        svc = self
        mid = kwargs.get("id")

        class _Exec:
            def execute(self_inner):
                if mid in svc._get_raises_ids:
                    raise RuntimeError("gmail get 500")
                return svc._full_by_id.get(mid, {"id": mid})
        return _Exec()


def _fake_email_triage(*, outlook_ms=None, outlook_raises=False,
                       gmail_available=True, gmail_service=_SENTINEL,
                       gmail_service_raises=False, gmail_shape=None,
                       drop_gmail_shape=False):
    """A fake ``skills.email_triage`` exposing exactly the surface the Amazon
    fetchers consult. Nothing here touches a real backend or the network."""
    et = types.ModuleType("skills.email_triage")

    def _ms_graph():
        if outlook_raises:
            raise RuntimeError("ms_graph import boom")
        return outlook_ms

    et._ms_graph = _ms_graph
    et._outlook_configured = lambda: outlook_ms is not None and outlook_ms.is_configured()
    et.is_gmail_available = lambda: gmail_available

    def _gmail_service():
        if gmail_service_raises:
            raise RuntimeError("gmail auth boom")
        return None if gmail_service is _SENTINEL else gmail_service

    et._gmail_service = _gmail_service

    def _default_gmail_shape(full):
        headers = ((full.get("payload") or {}).get("headers")) or []
        hmap = {h["name"]: h["value"] for h in headers}
        return {
            "id":        full.get("id") or "",
            "from_addr": hmap.get("From", ""),
            "from_name": "",
            "subject":   hmap.get("Subject", ""),
            "snippet":   (full.get("snippet") or "").strip(),
            "received":  hmap.get("Date", ""),
        }

    if not drop_gmail_shape:
        et._shape_gmail_message = gmail_shape or _default_gmail_shape
    return et


class AmazonTestBase(unittest.TestCase):
    def setUp(self):
        # CRITICAL: the harness's register() call runs _read_feature_flag(),
        # which does importlib.import_module("bobert_companion"). Without a fake
        # in place that would import the real ~14K-line monolith (forbidden, and
        # it spawns threads). Install a neutral fake BEFORE loading the skill and
        # restore the prior sys.modules state (including absence) on teardown, so
        # nothing leaks between tests. Individual tests still inject their own
        # bobert_companion via inject_modules() when they need specific flags.
        import sys as _sys
        self._saved_bc = _sys.modules.get("bobert_companion", _SENTINEL)
        _sys.modules["bobert_companion"] = _fake_bc(AMAZON_TRACKING_ENABLED=False)
        self.addCleanup(self._restore_bc)

        self.mod, self.actions = load_skill_isolated("amazon_order_tracker")
        self._tmp = tempfile.TemporaryDirectory()
        self.mod._STATE_FILE = os.path.join(self._tmp.name, "amazon_state.json")
        self.addCleanup(self._tmp.cleanup)
        # A handful of code paths (register / start_monitor) print status banners
        # straight to stdout. The harness only captures during import, not the
        # test body, so silence them here to keep suite output clean. (Tests
        # assert on return values, not prints.)
        _quiet = contextlib.redirect_stdout(io.StringIO())
        _quiet.__enter__()
        self.addCleanup(_quiet.__exit__, None, None, None)
        # The crash-path tests deliberately drive _log.exception / logging
        # .exception (poll-loop & fetch failure handling). Mute logging for the
        # test body so those expected tracebacks don't spam stderr; restore the
        # prior threshold afterward.
        logging.disable(logging.CRITICAL)
        self.addCleanup(logging.disable, logging.NOTSET)
        # The poller must never actually run. Stop it and null the handle on
        # teardown in case any test populated _poll_thread.
        self.addCleanup(self._stop_poller)

    def _restore_bc(self):
        import sys as _sys
        if self._saved_bc is _SENTINEL:
            _sys.modules.pop("bobert_companion", None)
        else:
            _sys.modules["bobert_companion"] = self._saved_bc

    def _stop_poller(self):
        try:
            self.mod._stop_evt.set()
            self.mod._poll_thread[0] = None
        except Exception:
            pass


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


# ─── Config / announce shims (lazy-import bobert_companion + email_triage) ──
class ConfigShimTests(AmazonTestBase):
    def test_read_feature_flag_true(self):
        with inject_modules(bobert_companion=_fake_bc(AMAZON_TRACKING_ENABLED=True)):
            self.assertTrue(self.mod._read_feature_flag())

    def test_read_feature_flag_default_false_when_attr_missing(self):
        with inject_modules(bobert_companion=_fake_bc()):
            self.assertFalse(self.mod._read_feature_flag())

    def test_read_feature_flag_import_failure_is_false(self):
        # No bobert_companion importable → swallowed → False.
        with inject_modules(bobert_companion=None):
            with mock.patch.object(self.mod.importlib, "import_module",
                                   side_effect=ImportError("no bc")):
                self.assertFalse(self.mod._read_feature_flag())

    def test_email_triage_resolves_first_name(self):
        et = _fake_email_triage()
        with inject_modules(**{"skills.email_triage": et}):
            self.assertIs(self.mod._email_triage(), et)

    def test_email_triage_none_when_all_imports_fail(self):
        with mock.patch.object(self.mod.importlib, "import_module",
                               side_effect=ImportError("x")):
            self.assertIsNone(self.mod._email_triage())

    def test_proactive_announce_routes_to_bc(self):
        seen = []
        bc = _fake_bc(proactive_announce=lambda msg, source=None: seen.append((msg, source)))
        with inject_modules(bobert_companion=bc):
            self.mod._proactive_announce("hello sir")
        self.assertEqual(seen, [("hello sir", "amazon")])

    def test_proactive_announce_fallback_prints_when_no_announcer(self):
        # bc present but without a callable proactive_announce → print fallback.
        with inject_modules(bobert_companion=_fake_bc()):
            with mock.patch("builtins.print") as pr:
                self.mod._proactive_announce("fallback line")
        self.assertTrue(any("fallback line" in str(c) for c in pr.call_args_list))

    def test_proactive_announce_fallback_when_import_raises(self):
        # Enter the print patch FIRST (mock.patch resolves "builtins.print" via
        # importlib internally — doing it before stubbing the module's
        # import_module avoids tripping that resolution).
        with mock.patch("builtins.print") as pr:
            with mock.patch.object(self.mod.importlib, "import_module",
                                   side_effect=RuntimeError("boom")):
                self.mod._proactive_announce("via except")
        self.assertTrue(any("via except" in str(c) for c in pr.call_args_list))

    # ── _outlook_configured / _gmail_configured ──────────────────────────
    def test_outlook_configured_true(self):
        et = _fake_email_triage(outlook_ms=_fake_ms_graph(configured=True))
        with mock.patch.object(self.mod, "_email_triage", return_value=et):
            self.assertTrue(self.mod._outlook_configured())

    def test_outlook_configured_false_when_no_triage(self):
        with mock.patch.object(self.mod, "_email_triage", return_value=None):
            self.assertFalse(self.mod._outlook_configured())

    def test_outlook_configured_swallows_exception(self):
        et = types.SimpleNamespace(_outlook_configured=mock.MagicMock(
            side_effect=RuntimeError("boom")))
        with mock.patch.object(self.mod, "_email_triage", return_value=et):
            self.assertFalse(self.mod._outlook_configured())

    def test_gmail_configured_true(self):
        et = _fake_email_triage(gmail_available=True)
        with mock.patch.object(self.mod, "_email_triage", return_value=et):
            self.assertTrue(self.mod._gmail_configured())

    def test_gmail_configured_false_when_no_triage(self):
        with mock.patch.object(self.mod, "_email_triage", return_value=None):
            self.assertFalse(self.mod._gmail_configured())

    def test_gmail_configured_swallows_exception(self):
        et = types.SimpleNamespace(is_gmail_available=mock.MagicMock(
            side_effect=RuntimeError("boom")))
        with mock.patch.object(self.mod, "_email_triage", return_value=et):
            self.assertFalse(self.mod._gmail_configured())


# ─── State persistence (real temp-file round-trips) ─────────────────────────
class StatePersistenceTests(AmazonTestBase):
    def test_empty_state_shape(self):
        s = self.mod._empty_state()
        self.assertEqual(s, {"orders": {}, "last_poll_ts": 0.0, "last_error": ""})

    def test_load_state_missing_file_returns_empty(self):
        # _STATE_FILE points at a not-yet-created temp path.
        self.assertEqual(self.mod._load_state(), self.mod._empty_state())

    def test_save_then_load_round_trip(self):
        state = {"orders": {"113-1234567-7654321": {"status": "shipped",
                                                     "last_seen_ts": 5.0}},
                 "last_poll_ts": 123.0, "last_error": ""}
        self.mod._save_state(state)
        self.assertTrue(os.path.exists(self.mod._STATE_FILE))
        loaded = self.mod._load_state()
        self.assertEqual(loaded["orders"]["113-1234567-7654321"]["status"], "shipped")
        self.assertEqual(loaded["last_poll_ts"], 123.0)

    def test_load_state_fills_missing_keys(self):
        with open(self.mod._STATE_FILE, "w", encoding="utf-8") as f:
            json.dump({"orders": {"o1": {"status": "ordered"}}}, f)
        loaded = self.mod._load_state()
        self.assertIn("last_poll_ts", loaded)
        self.assertIn("last_error", loaded)

    def test_load_state_non_dict_payload_resets(self):
        with open(self.mod._STATE_FILE, "w", encoding="utf-8") as f:
            json.dump([1, 2, 3], f)
        self.assertEqual(self.mod._load_state(), self.mod._empty_state())

    def test_load_state_orders_not_dict_coerced(self):
        with open(self.mod._STATE_FILE, "w", encoding="utf-8") as f:
            json.dump({"orders": ["not", "a", "dict"]}, f)
        self.assertEqual(self.mod._load_state()["orders"], {})

    def test_load_state_corrupt_json_records_error(self):
        with open(self.mod._STATE_FILE, "w", encoding="utf-8") as f:
            f.write("{ this is not json")
        loaded = self.mod._load_state()
        self.assertEqual(loaded["orders"], {})
        self.assertTrue(loaded["last_error"])   # captured the parse error

    def test_save_state_caps_oldest_orders(self):
        cap = self.mod.STATE_MAX_ORDERS
        orders = {f"o{i:04d}": {"last_seen_ts": float(i)} for i in range(cap + 10)}
        state = {"orders": orders, "last_poll_ts": 0.0, "last_error": ""}
        self.mod._save_state(state)
        loaded = self.mod._load_state()
        self.assertEqual(len(loaded["orders"]), cap)
        # The newest (highest last_seen_ts) survive; the oldest are dropped.
        self.assertIn(f"o{cap + 9:04d}", loaded["orders"])
        self.assertNotIn("o0000", loaded["orders"])

    def test_save_state_swallows_write_failure(self):
        # _atomic_write_json raising must not propagate out of _save_state.
        # (Mute the module logger so the expected warning doesn't print.)
        with mock.patch.object(self.mod, "_atomic_write_json",
                               side_effect=OSError("disk full")), \
             mock.patch.object(self.mod._log, "warning"):
            self.mod._save_state(self.mod._empty_state())   # no raise


# ─── _parse_iso fallback paths ──────────────────────────────────────────────
class ParseIsoTests(AmazonTestBase):
    def test_empty_is_zero(self):
        self.assertEqual(self.mod._parse_iso(""), 0.0)

    def test_iso_with_zulu(self):
        self.assertGreater(self.mod._parse_iso("2026-06-01T08:00:00Z"), 0.0)

    def test_rfc2822_fallback(self):
        # Not ISO — exercises the email.utils.parsedate_to_datetime branch.
        ts = self.mod._parse_iso("Mon, 01 Jun 2026 08:00:00 +0000")
        self.assertGreater(ts, 0.0)

    def test_unparseable_returns_zero(self):
        self.assertEqual(self.mod._parse_iso("not a date at all"), 0.0)


# ─── _extract_estimated_delivery edge paths ─────────────────────────────────
class EstimatedDeliveryEdgeTests(AmazonTestBase):
    def test_empty_text_returns_none(self):
        self.assertIsNone(self.mod._extract_estimated_delivery("", 0))

    def test_invalid_day_returns_none(self):
        # 'Arriving Jun 40' — day out of 1..31 range → rejected.
        self.assertIsNone(self.mod._extract_estimated_delivery("Arriving Jun 40", 0))

    def test_uses_now_when_no_ref_ts(self):
        # ref_ts=0 → falls back to time.localtime() for the base year.
        cur_year = time.localtime().tm_year
        est = self.mod._extract_estimated_delivery("Arriving Jun 15", 0)
        self.assertIsNotNone(est)
        self.assertIn(time.localtime(est).tm_year, (cur_year, cur_year + 1))

    def test_mktime_failure_returns_none(self):
        with mock.patch.object(self.mod.time, "mktime",
                               side_effect=OverflowError("bad")):
            self.assertIsNone(
                self.mod._extract_estimated_delivery("Arriving Jun 3", 0))

    def test_group_parse_exception_returns_none(self):
        # Defensive except: force the month-lookup to blow up mid-parse. The
        # regex contract makes this unreachable in practice, but the guard
        # exists — lock it in.
        boom = types.SimpleNamespace(get=mock.MagicMock(side_effect=RuntimeError("x")))
        with mock.patch.object(self.mod, "_MONTH_INDEX", boom):
            self.assertIsNone(
                self.mod._extract_estimated_delivery("Arriving Jun 3", 0))


# ─── _is_amazon_message branch coverage ─────────────────────────────────────
class IsAmazonMessageBranchTests(AmazonTestBase):
    def test_by_from_name_startswith(self):
        self.assertTrue(self.mod._is_amazon_message(
            {"from_addr": "noreply@shipping.example.com",
             "from_name": "Amazon Shipment"}))

    def test_by_from_name_contains_amazon_com(self):
        self.assertTrue(self.mod._is_amazon_message(
            {"from_addr": "x@carrier.example", "from_name": "via amazon.com"}))

    def test_unrelated_message_rejected(self):
        self.assertFalse(self.mod._is_amazon_message(
            {"from_addr": "team@example.com", "from_name": "Example",
             "subject": "Lunch?"}))


# ─── Outlook fetcher ────────────────────────────────────────────────────────
class FetchOutlookTests(AmazonTestBase):
    def _patch_triage(self, et):
        return mock.patch.object(self.mod, "_email_triage", return_value=et)

    def test_no_triage_returns_empty(self):
        with self._patch_triage(None):
            self.assertEqual(self.mod._fetch_outlook(), [])

    def test_not_configured_returns_empty(self):
        et = _fake_email_triage(outlook_ms=_fake_ms_graph(configured=False))
        with self._patch_triage(et):
            self.assertEqual(self.mod._fetch_outlook(), [])

    def test_ms_graph_none_returns_empty(self):
        et = _fake_email_triage(outlook_ms=None)
        with self._patch_triage(et):
            self.assertEqual(self.mod._fetch_outlook(), [])

    def test_missing_graph_get_or_shape_returns_empty(self):
        et = _fake_email_triage(outlook_ms=_fake_ms_graph(drop_graph_get=True))
        with self._patch_triage(et):
            self.assertEqual(self.mod._fetch_outlook(), [])

    def test_happy_path_filters_to_amazon(self):
        recent = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        msgs = [
            {"id": "m1", "from": {"emailAddress": {"address": "ship@amazon.com"}},
             "subject": "Shipped: 113-1234567-7654321", "bodyPreview": "",
             "receivedDateTime": recent},
            {"id": "m2", "from": {"emailAddress": {"address": "deals@nike.com"}},
             "subject": "Sale", "bodyPreview": "", "receivedDateTime": recent},
        ]
        et = _fake_email_triage(outlook_ms=_fake_ms_graph(messages=msgs))
        with self._patch_triage(et):
            out = self.mod._fetch_outlook()
        self.assertEqual([m["id"] for m in out], ["m1"])

    def test_drops_messages_older_than_lookback(self):
        old = time.strftime("%Y-%m-%dT%H:%M:%SZ",
                            time.gmtime(time.time() - (self.mod.LOOKBACK_DAYS + 3) * 86400))
        msgs = [{"id": "old1", "from": {"emailAddress": {"address": "ship@amazon.com"}},
                 "subject": "Shipped: 113-1234567-7654321", "bodyPreview": "",
                 "receivedDateTime": old}]
        et = _fake_email_triage(outlook_ms=_fake_ms_graph(messages=msgs))
        with self._patch_triage(et):
            self.assertEqual(self.mod._fetch_outlook(), [])

    def test_graph_get_raises_returns_empty(self):
        et = _fake_email_triage(outlook_ms=_fake_ms_graph(get_raises=True))
        with self._patch_triage(et):
            self.assertEqual(self.mod._fetch_outlook(), [])

    def test_empty_body_returns_empty(self):
        ms = _fake_ms_graph()
        ms._graph_get = lambda path, params=None: None
        et = _fake_email_triage(outlook_ms=ms)
        with self._patch_triage(et):
            self.assertEqual(self.mod._fetch_outlook(), [])

    def test_shape_raising_skips_message(self):
        recent = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        msgs = [{"id": "boom", "from": {"emailAddress": {"address": "ship@amazon.com"}},
                 "subject": "Shipped", "bodyPreview": "", "receivedDateTime": recent}]
        def _bad_shape(_m):
            raise ValueError("malformed")
        et = _fake_email_triage(outlook_ms=_fake_ms_graph(messages=msgs, shape=_bad_shape))
        with self._patch_triage(et):
            self.assertEqual(self.mod._fetch_outlook(), [])


# ─── Gmail fetcher ──────────────────────────────────────────────────────────
class FetchGmailTests(AmazonTestBase):
    def _patch_triage(self, et):
        return mock.patch.object(self.mod, "_email_triage", return_value=et)

    def _gmail_full(self, mid, addr="ship@amazon.com", subject="Shipped: 113-1234567-7654321"):
        return {"id": mid, "snippet": "",
                "payload": {"headers": [{"name": "From", "value": addr},
                                        {"name": "Subject", "value": subject},
                                        {"name": "Date", "value": "Mon, 01 Jun 2026 08:00:00 +0000"}]}}

    def test_no_triage_returns_empty(self):
        with self._patch_triage(None):
            self.assertEqual(self.mod._fetch_gmail(), [])

    def test_not_available_returns_empty(self):
        et = _fake_email_triage(gmail_available=False)
        with self._patch_triage(et):
            self.assertEqual(self.mod._fetch_gmail(), [])

    def test_service_raises_returns_empty(self):
        et = _fake_email_triage(gmail_service_raises=True)
        with self._patch_triage(et):
            self.assertEqual(self.mod._fetch_gmail(), [])

    def test_service_none_returns_empty(self):
        et = _fake_email_triage(gmail_available=True, gmail_service=None)
        with self._patch_triage(et):
            self.assertEqual(self.mod._fetch_gmail(), [])

    def test_missing_shape_returns_empty(self):
        svc = _FakeGmailService(list_result={"messages": [{"id": "g1"}]})
        et = _fake_email_triage(gmail_service=svc, drop_gmail_shape=True)
        with self._patch_triage(et):
            self.assertEqual(self.mod._fetch_gmail(), [])

    def test_list_raises_returns_empty(self):
        svc = _FakeGmailService(list_raises=True)
        et = _fake_email_triage(gmail_service=svc)
        with self._patch_triage(et):
            self.assertEqual(self.mod._fetch_gmail(), [])

    def test_happy_path_returns_amazon_messages(self):
        svc = _FakeGmailService(
            list_result={"messages": [{"id": "g1"}, {"id": "g2"}]},
            full_by_id={"g1": self._gmail_full("g1"),
                        "g2": self._gmail_full("g2", addr="deals@nike.com",
                                               subject="Sale")})
        et = _fake_email_triage(gmail_service=svc)
        with self._patch_triage(et):
            out = self.mod._fetch_gmail()
        self.assertEqual([m["id"] for m in out], ["g1"])  # g2 filtered out

    def test_get_raises_skips_message(self):
        svc = _FakeGmailService(
            list_result={"messages": [{"id": "g1"}]},
            get_raises_ids=("g1",))
        et = _fake_email_triage(gmail_service=svc)
        with self._patch_triage(et):
            self.assertEqual(self.mod._fetch_gmail(), [])

    def test_shape_raises_skips_message(self):
        svc = _FakeGmailService(
            list_result={"messages": [{"id": "g1"}]},
            full_by_id={"g1": self._gmail_full("g1")})
        def _bad_shape(_full):
            raise ValueError("bad")
        et = _fake_email_triage(gmail_service=svc, gmail_shape=_bad_shape)
        with self._patch_triage(et):
            self.assertEqual(self.mod._fetch_gmail(), [])

    def test_empty_list_result_returns_empty(self):
        svc = _FakeGmailService(list_result={})
        et = _fake_email_triage(gmail_service=svc)
        with self._patch_triage(et):
            self.assertEqual(self.mod._fetch_gmail(), [])


# ─── _make_promise ──────────────────────────────────────────────────────────
class MakePromiseTests(AmazonTestBase):
    def test_no_skill_utils_is_noop(self):
        self.mod.skill_utils = None
        self.mod._make_promise("m", "time_at", {"epoch": 1.0})  # no raise

    def test_skill_utils_not_dict_is_noop(self):
        self.mod.skill_utils = object()
        self.mod._make_promise("m", "time_at", {"epoch": 1.0})

    def test_make_promise_absent_key_is_noop(self):
        self.mod.skill_utils = {}
        self.mod._make_promise("m", "time_at", {"epoch": 1.0})

    def test_make_promise_invoked_with_source(self):
        calls = []
        self.mod.skill_utils = {"make_promise":
                                lambda msg, cond, params=None, source=None:
                                calls.append((msg, cond, params, source))}
        self.mod._make_promise("hi", "time_at", {"epoch": 9.0})
        self.assertEqual(calls, [("hi", "time_at", {"epoch": 9.0}, "amazon")])

    def test_make_promise_swallows_callee_exception(self):
        def _boom(*a, **k):
            raise RuntimeError("promise blew up")
        self.mod.skill_utils = {"make_promise": _boom}
        self.mod._make_promise("hi", "time_at", {"epoch": 9.0})  # no raise

    def test_make_promise_non_callable_value_is_noop(self):
        self.mod.skill_utils = {"make_promise": "not callable"}
        self.mod._make_promise("hi", "time_at", {"epoch": 9.0})


# ─── _process_messages remaining branches ──────────────────────────────────
class ProcessMessagesEdgeTests(AmazonTestBase):
    def test_message_without_id_is_skipped(self):
        state = {"orders": {}}
        with mock.patch.object(self.mod, "_proactive_announce"):
            out = self.mod._process_messages(
                [{"id": "", "subject": "Shipped: 113-1234567-7654321"}], state)
        self.assertEqual(out["orders"], {})   # blank id → skipped entirely

    def test_announce_message_unknown_status_fallthrough(self):
        # A status not in the explicit map → generic humanised phrasing.
        msg = self.mod._announce_message("ordered_partial", "113-1")
        self.assertIn("113-1", msg)
        self.assertIn("now", msg)

    def test_same_poll_shipped_then_delivered_climbs(self):
        oid = "113-1234567-7654321"
        msgs = [
            {"id": "s", "subject": f"Shipped: {oid}", "snippet": "",
             "received": "2026-06-01T08:00:00Z"},
            {"id": "d", "subject": f"Delivered: {oid}", "snippet": "",
             "received": "2026-06-01T20:00:00Z"},
        ]
        announces = []
        with mock.patch.object(self.mod, "_proactive_announce",
                               side_effect=lambda m: announces.append(m)):
            state = self.mod._process_messages(msgs, {"orders": {}})
        self.assertEqual(state["orders"][oid]["status"], "delivered")
        self.assertTrue(any("has shipped" in a for a in announces))
        self.assertTrue(any("has been delivered" in a for a in announces))

    def test_shipped_with_past_eta_does_not_schedule(self):
        oid = "113-1234567-7654321"
        past = time.strftime("%b %d", time.localtime(time.time() - 5 * 86400))
        msgs = [{"id": "s1", "subject": f"Shipped: {oid}. Arriving {past}",
                 "snippet": "",
                 "received": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}]
        with mock.patch.object(self.mod, "_make_promise") as promise, \
             mock.patch.object(self.mod, "_proactive_announce"):
            self.mod._process_messages(msgs, {"orders": {}})
        promise.assert_not_called()   # ETA already in the past → no reminder

    def test_delay_without_order_id_uses_generic_label(self):
        msgs = [{"id": "longmsgid000111", "subject": "Your order is delayed",
                 "snippet": "running late", "received": "2026-06-01T08:00:00Z"}]
        announces = []
        with mock.patch.object(self.mod, "_proactive_announce",
                               side_effect=lambda m: announces.append(m)):
            self.mod._process_messages(msgs, {"orders": {}})
        self.assertTrue(any("an Amazon order" in a and "delayed" in a
                            for a in announces))


# ─── _fmt_date exception path ───────────────────────────────────────────────
class FmtDateEdgeTests(AmazonTestBase):
    def test_fmt_date_bad_value_returns_empty(self):
        with mock.patch.object(self.mod.time, "localtime",
                               side_effect=ValueError("bad ts")):
            self.assertEqual(self.mod._fmt_date(123.0), "")


# ─── Action edge branches not yet covered ───────────────────────────────────
class ActionEdgeTests(AmazonTestBase):
    def test_check_orders_msg_label_and_delayed_no_eta(self):
        state = {"orders": {
            "msg:abc": {"status": "shipped", "last_seen_ts": time.time(),
                        "delayed_announced": True},
        }}
        with mock.patch.object(self.mod, "_load_state", return_value=state):
            out = self.actions["check_orders"]("")
        self.assertIn("an order", out)        # synthetic id hidden behind label
        self.assertIn("(delayed)", out)

    def test_recent_delivery_msg_label_without_date(self):
        state = {"orders": {
            "msg:zzz": {"status": "delivered", "last_seen_ts": time.time() - 3600},
        }}
        with mock.patch.object(self.mod, "_load_state", return_value=state), \
             mock.patch.object(self.mod, "_fmt_date", return_value=""):
            out = self.actions["recent_delivery"]("")
        self.assertIn("an order", out)

    def test_recent_delivery_outside_window_excluded(self):
        state = {"orders": {
            "113-1234567-7654321": {"status": "delivered",
                                     "last_seen_ts": time.time() - 10 * 86400},
        }}
        with mock.patch.object(self.mod, "_load_state", return_value=state):
            out = self.actions["recent_delivery"]("")
        self.assertIn("No Amazon deliveries", out)

    def test_tracking_status_never_polled_and_error(self):
        state = {"orders": {}, "last_poll_ts": 0.0, "last_error": "graph 500"}
        with mock.patch.object(self.mod, "_load_state", return_value=state), \
             mock.patch.object(self.mod, "_read_feature_flag", return_value=False), \
             mock.patch.object(self.mod, "_outlook_configured", return_value=False), \
             mock.patch.object(self.mod, "_gmail_configured", return_value=False):
            out = self.actions["amazon_tracking_status"]("")
        self.assertIn("last poll: never", out)
        self.assertIn("last error: graph 500", out)
        self.assertIn("poller: stopped", out)


# ─── Poll lifecycle: _poll_once / _poll_loop ────────────────────────────────
class PollOnceTests(AmazonTestBase):
    def test_poll_once_no_messages_updates_ts(self):
        with mock.patch.object(self.mod, "_fetch_outlook", return_value=[]), \
             mock.patch.object(self.mod, "_fetch_gmail", return_value=[]):
            n = self.mod._poll_once()
        self.assertEqual(n, 0)
        self.assertGreater(self.mod._load_state()["last_poll_ts"], 0.0)

    def test_poll_once_processes_fetched_messages(self):
        oid = "113-1234567-7654321"
        outlook_msg = {"id": "m1", "subject": f"Shipped: {oid}", "snippet": "",
                       "received": "2026-06-01T08:00:00Z"}
        with mock.patch.object(self.mod, "_fetch_outlook", return_value=[outlook_msg]), \
             mock.patch.object(self.mod, "_fetch_gmail", return_value=[]), \
             mock.patch.object(self.mod, "_proactive_announce"):
            n = self.mod._poll_once()
        self.assertEqual(n, 1)
        self.assertEqual(self.mod._load_state()["orders"][oid]["status"], "shipped")

    def test_poll_once_survives_outlook_fetch_crash(self):
        with mock.patch.object(self.mod, "_fetch_outlook",
                               side_effect=RuntimeError("boom")), \
             mock.patch.object(self.mod, "_fetch_gmail", return_value=[]):
            n = self.mod._poll_once()
        self.assertEqual(n, 0)   # crash swallowed; gmail still contributes 0

    def test_poll_once_survives_gmail_fetch_crash(self):
        with mock.patch.object(self.mod, "_fetch_outlook", return_value=[]), \
             mock.patch.object(self.mod, "_fetch_gmail",
                               side_effect=RuntimeError("boom")):
            n = self.mod._poll_once()
        self.assertEqual(n, 0)


class PollLoopTests(AmazonTestBase):
    def test_loop_exits_immediately_on_initial_wait(self):
        # stop_evt.wait(INITIAL_DELAY) returning True → loop returns at once.
        evt = mock.MagicMock()
        evt.wait.return_value = True
        with mock.patch.object(self.mod, "_poll_once") as once:
            self.mod._poll_loop(evt)
        once.assert_not_called()

    def test_loop_runs_one_iteration_then_stops(self):
        evt = mock.MagicMock()
        # First wait (initial delay) False → enter loop; is_set False once so the
        # body runs; the post-iteration wait returns True → exit.
        evt.wait.side_effect = [False, True]
        evt.is_set.return_value = False
        with mock.patch.object(self.mod, "_poll_once") as once:
            self.mod._poll_loop(evt)
        once.assert_called_once()

    def test_loop_iteration_crash_is_swallowed_and_records_error(self):
        evt = mock.MagicMock()
        evt.wait.side_effect = [False, True]   # initial delay, then exit
        evt.is_set.return_value = False
        with mock.patch.object(self.mod, "_poll_once",
                               side_effect=RuntimeError("iter boom")):
            self.mod._poll_loop(evt)
        self.assertEqual(self.mod._load_state()["last_error"],
                         "poll crashed (see log)")

    def test_loop_crash_recovery_save_also_failing_is_swallowed(self):
        # Both the poll AND the error-recording _load_state blow up → the inner
        # except: pass must keep the loop from propagating.
        evt = mock.MagicMock()
        evt.wait.side_effect = [False, True]
        evt.is_set.return_value = False
        with mock.patch.object(self.mod, "_poll_once",
                               side_effect=RuntimeError("iter boom")), \
             mock.patch.object(self.mod, "_load_state",
                               side_effect=RuntimeError("state boom")):
            self.mod._poll_loop(evt)   # no raise escapes


# ─── start_monitor / stop_monitor ───────────────────────────────────────────
class MonitorLifecycleTests(AmazonTestBase):
    def test_start_monitor_returns_false_when_flag_off(self):
        with mock.patch.object(self.mod, "_read_feature_flag", return_value=False):
            self.assertFalse(self.mod.start_monitor())

    def test_start_monitor_true_when_already_running(self):
        alive = mock.MagicMock()
        alive.is_alive.return_value = True
        self.mod._poll_thread[0] = alive
        with mock.patch.object(self.mod, "_read_feature_flag", return_value=True):
            self.assertTrue(self.mod.start_monitor())

    def test_start_monitor_false_when_no_backend(self):
        with mock.patch.object(self.mod, "_read_feature_flag", return_value=True), \
             mock.patch.object(self.mod, "_outlook_configured", return_value=False), \
             mock.patch.object(self.mod, "_gmail_configured", return_value=False):
            self.assertFalse(self.mod.start_monitor())

    def test_start_monitor_spawns_thread_when_configured(self):
        # Thread.start is neutered (patched to a no-op) so nothing really runs;
        # we assert the handle is stored and a thread object was constructed.
        started = []
        with mock.patch.object(self.mod, "_read_feature_flag", return_value=True), \
             mock.patch.object(self.mod, "_outlook_configured", return_value=True), \
             mock.patch.object(self.mod, "_gmail_configured", return_value=False), \
             mock.patch.object(self.mod.threading.Thread, "start",
                               lambda self_t: started.append(self_t)):
            ok = self.mod.start_monitor()
        self.assertTrue(ok)
        self.assertIsNotNone(self.mod._poll_thread[0])
        self.assertEqual(self.mod._poll_thread[0].name, "amazon-order-tracker")
        self.assertEqual(len(started), 1)

    def test_stop_monitor_idempotent_with_no_thread(self):
        self.mod._poll_thread[0] = None
        self.mod.stop_monitor()   # no raise
        self.assertIsNone(self.mod._poll_thread[0])

    def test_stop_monitor_joins_live_thread(self):
        fake = mock.MagicMock()
        fake.is_alive.return_value = True
        self.mod._poll_thread[0] = fake
        self.mod.stop_monitor()
        fake.join.assert_called_once()
        self.assertIsNone(self.mod._poll_thread[0])
        self.assertTrue(self.mod._stop_evt.is_set())


# ─── register() ─────────────────────────────────────────────────────────────
class RegisterTests(AmazonTestBase):
    def test_register_maps_all_actions(self):
        actions = {}
        with mock.patch.object(self.mod, "_read_feature_flag", return_value=False):
            self.mod.register(actions)
        for name in ("check_orders", "check_amazon_orders", "amazon_orders",
                     "recent_delivery", "recent_deliveries", "amazon_tracking_status"):
            self.assertIn(name, actions)
        self.assertIs(actions["check_orders"], self.mod.action_check_orders)
        self.assertIs(actions["recent_deliveries"], self.mod.action_recent_delivery)

    def test_register_flag_off_does_not_start(self):
        with mock.patch.object(self.mod, "_read_feature_flag", return_value=False), \
             mock.patch.object(self.mod, "start_monitor") as start:
            self.mod.register({})
        start.assert_not_called()

    def test_register_no_backend_announces_and_skips_start(self):
        with mock.patch.object(self.mod, "_read_feature_flag", return_value=True), \
             mock.patch.object(self.mod, "_outlook_configured", return_value=False), \
             mock.patch.object(self.mod, "_gmail_configured", return_value=False), \
             mock.patch.object(self.mod, "_proactive_announce") as ann, \
             mock.patch.object(self.mod, "start_monitor") as start:
            self.mod.register({})
        ann.assert_called_once()
        start.assert_not_called()

    def test_register_starts_monitor_when_enabled_and_configured(self):
        with mock.patch.object(self.mod, "_read_feature_flag", return_value=True), \
             mock.patch.object(self.mod, "_outlook_configured", return_value=True), \
             mock.patch.object(self.mod, "_gmail_configured", return_value=False), \
             mock.patch.object(self.mod, "start_monitor") as start:
            self.mod.register({})
        start.assert_called_once()


class AmazonImportGuardTests(unittest.TestCase):
    def test_path_bootstrap_inserts_project_root(self):
        # Re-exec the source with the project root removed from sys.path so the
        # `if _PROJECT_DIR not in sys.path: sys.path.insert(...)` guard runs.
        mod, _ = load_skill_isolated("amazon_order_tracker")
        path = mod.__file__
        proj = os.path.dirname(os.path.dirname(path))
        spec = importlib.util.spec_from_file_location("amazon_reexec", path)
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
