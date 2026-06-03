"""Tests for core.message_bus — the in-process pub/sub + request/response bus.

CI-safe: stdlib only; no network, no monolith.
"""
from __future__ import annotations

import unittest
from unittest import mock

import core.message_bus as mb


def _raiser(_payload=None):
    raise RuntimeError("boom")


class PubSubTests(unittest.TestCase):
    def setUp(self):
        self.bus = mb.MessageBus()

    def test_subscribe_publish(self):
        got = []
        self.bus.subscribe("t", got.append)
        self.assertEqual(self.bus.publish("t", 42), 1)
        self.assertEqual(got, [42])

    def test_multiple_subscribers(self):
        a, b = [], []
        self.bus.subscribe("t", a.append)
        self.bus.subscribe("t", b.append)
        self.assertEqual(self.bus.publish("t", "x"), 2)
        self.assertEqual((a, b), (["x"], ["x"]))

    def test_publish_no_subscribers(self):
        self.assertEqual(self.bus.publish("nobody", 1), 0)

    def test_unsubscribe_last_removes_topic(self):
        got = []
        unsub = self.bus.subscribe("t", got.append)
        self.bus.publish("t", 1)
        unsub()
        self.bus.publish("t", 2)
        self.assertEqual(got, [1])
        self.assertNotIn("t", self.bus.topics())

    def test_unsubscribe_one_of_many(self):
        a, b = [], []
        un_a = self.bus.subscribe("t", a.append)
        self.bus.subscribe("t", b.append)
        un_a()
        self.bus.publish("t", 9)
        self.assertEqual((a, b), ([], [9]))
        self.assertIn("t", self.bus.topics())

    def test_unsubscribe_idempotent(self):
        unsub = self.bus.subscribe("t", lambda _p: None)
        unsub()
        unsub()   # topic already gone -> early return, no error

    def test_subscriber_exception_isolated(self):
        seen, errs = [], []
        bus = mb.MessageBus(on_error=lambda what, e: errs.append((what, e)))
        bus.subscribe("t", _raiser)
        bus.subscribe("t", seen.append)
        n = bus.publish("t", 5)
        self.assertEqual(n, 1)          # only the good subscriber counted
        self.assertEqual(seen, [5])     # good subscriber still delivered
        self.assertEqual(len(errs), 1)
        self.assertIn("subscriber for topic", errs[0][0])

    def test_subscriber_exception_default_reporter_prints(self):
        bus = mb.MessageBus()           # no on_error -> console path
        bus.subscribe("t", _raiser)
        with mock.patch("builtins.print") as p:
            bus.publish("t")
        p.assert_called()

    def test_on_error_that_raises_is_swallowed(self):
        def bad_reporter(_w, _e):
            raise RuntimeError("reporter broke")
        bus = mb.MessageBus(on_error=bad_reporter)
        bus.subscribe("t", _raiser)
        bus.publish("t")                # must not raise


class RequestResponseTests(unittest.TestCase):
    def setUp(self):
        self.bus = mb.MessageBus()

    def test_register_and_request(self):
        self.bus.register_handler("add", lambda p: p["a"] + p["b"])
        self.assertEqual(self.bus.request("add", {"a": 2, "b": 3}), 5)

    def test_request_no_handler(self):
        with self.assertRaises(mb.BusError):
            self.bus.request("missing")

    def test_request_handler_raises_wraps(self):
        errs = []
        bus = mb.MessageBus(on_error=lambda w, e: errs.append(w))
        bus.register_handler("boom", _raiser)
        with self.assertRaises(mb.BusError):
            bus.request("boom")
        self.assertTrue(errs)

    def test_register_replaces_prior(self):
        self.bus.register_handler("m", lambda _p: 1)
        self.bus.register_handler("m", lambda _p: 2)
        self.assertEqual(self.bus.request("m"), 2)

    def test_unregister_idempotent(self):
        self.bus.register_handler("m", lambda _p: 1)
        self.bus.unregister_handler("m")
        self.bus.unregister_handler("m")    # idempotent
        with self.assertRaises(mb.BusError):
            self.bus.request("m")

    def test_topics_and_methods(self):
        self.bus.register_handler("m", lambda _p: 1)
        self.bus.subscribe("t", lambda _p: None)
        self.assertEqual(self.bus.methods(), ["m"])
        self.assertEqual(self.bus.topics(), ["t"])


class DefaultBusTests(unittest.TestCase):
    def test_singleton(self):
        with mock.patch.object(mb, "_DEFAULT", None):
            b1 = mb.default_bus()
            b2 = mb.default_bus()
        self.assertIs(b1, b2)
        self.assertIsInstance(b1, mb.MessageBus)


if __name__ == "__main__":
    unittest.main()
