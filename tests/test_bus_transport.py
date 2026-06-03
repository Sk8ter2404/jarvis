"""Tests for core.bus_transport — the cross-process bus wire codec + transport.

CI-safe: stdlib only; socket tests use a loopback socket.socketpair().
"""
from __future__ import annotations

import socket
import unittest

import core.bus_transport as bt


class CodecTests(unittest.TestCase):
    def test_encode_decode_roundtrip(self):
        frame = bt.encode_frame("event", "wake", {"conf": 0.9})
        msgs, rest = bt.decode_frames(frame)
        self.assertEqual(rest, b"")
        self.assertEqual(msgs, [{"kind": "event", "name": "wake",
                                 "payload": {"conf": 0.9}}])

    def test_two_frames_back_to_back(self):
        buf = bt.encode_frame("event", "a") + bt.encode_frame("request", "b", 1)
        msgs, rest = bt.decode_frames(buf)
        self.assertEqual([m["name"] for m in msgs], ["a", "b"])
        self.assertEqual(rest, b"")

    def test_incomplete_payload_is_leftover(self):
        frame = bt.encode_frame("event", "x", [1, 2, 3])
        msgs, rest = bt.decode_frames(frame[:-2])
        self.assertEqual(msgs, [])
        self.assertEqual(rest, frame[:-2])

    def test_header_only_is_leftover(self):
        msgs, rest = bt.decode_frames(b"\x00\x00")  # fewer than 4 header bytes
        self.assertEqual(msgs, [])
        self.assertEqual(rest, b"\x00\x00")

    def test_corrupt_json_skipped_but_stream_continues(self):
        bad = bt._HEADER.pack(5) + b"{not!"
        good = bt.encode_frame("event", "ok")
        msgs, rest = bt.decode_frames(bad + good)
        self.assertEqual([m["name"] for m in msgs], ["ok"])
        self.assertEqual(rest, b"")

    def test_non_object_json_skipped(self):
        body = b"[1,2]"
        frame = bt._HEADER.pack(len(body)) + body
        msgs, rest = bt.decode_frames(frame)
        self.assertEqual(msgs, [])
        self.assertEqual(rest, b"")


class FrameReaderTests(unittest.TestCase):
    def test_streaming_reassembly(self):
        r = bt.FrameReader()
        frame = bt.encode_frame("event", "hi", "yo")
        self.assertEqual(r.feed(frame[:3]), [])      # partial
        self.assertEqual(r.pending, 3)
        msgs = r.feed(frame[3:])
        self.assertEqual([m["name"] for m in msgs], ["hi"])
        self.assertEqual(r.pending, 0)


class SocketTests(unittest.TestCase):
    def _pair(self):
        a, b = socket.socketpair()
        self.addCleanup(a.close)
        self.addCleanup(b.close)
        return a, b

    def test_send_and_recv(self):
        a, b = self._pair()
        self.assertTrue(bt.send_frame(a, "event", "ping", {"n": 1}))
        msgs = bt.recv_into(b, bt.FrameReader())
        self.assertEqual(msgs, [{"kind": "event", "name": "ping", "payload": {"n": 1}}])

    def test_send_on_closed_socket_returns_false(self):
        a, b = socket.socketpair()
        b.close()
        a.close()
        self.assertFalse(bt.send_frame(a, "event", "x"))

    def test_recv_peer_closed_returns_none(self):
        a, b = socket.socketpair()
        self.addCleanup(b.close)
        a.close()  # peer closed cleanly -> empty read
        self.assertIsNone(bt.recv_into(b, bt.FrameReader()))

    def test_recv_error_returns_none(self):
        a, b = socket.socketpair()
        a.close()
        b.close()  # recv on a closed socket -> OSError
        self.assertIsNone(bt.recv_into(b, bt.FrameReader()))


if __name__ == "__main__":
    unittest.main()
