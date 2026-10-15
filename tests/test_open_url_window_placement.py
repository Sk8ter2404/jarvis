"""open_url / web_search / youtube place the browser window they open (todo
2026-07-06: a new window sometimes opened off the right edge of the right
monitor). They called webbrowser.open bare; now the monolith's placement
runs on any window handle that is NEW since the pre-launch snapshot."""
from __future__ import annotations

import unittest
from unittest import mock

import core.actions as A


def _fake_bc(snapshot=frozenset({101, 102}), guard=False):
    bc = mock.Mock()
    bc._window_handles_snapshot.return_value = set(snapshot)
    bc._focus_steal_guard_active.return_value = guard
    bc._VIDEO_QUERY_HINTS = ()
    return bc


class WebbrowserOpenPlacementTests(unittest.TestCase):
    def test_snapshot_is_taken_before_the_launch_and_placement_follows(self):
        bc = _fake_bc()
        order = []
        bc._window_handles_snapshot.side_effect = lambda: (
            order.append("snapshot") or {101, 102})
        bc._place_new_browser_window_async.side_effect = (
            lambda *a, **k: order.append("place"))
        with mock.patch.object(A, "_loaded_bc", return_value=bc), \
                mock.patch.object(A.webbrowser, "open",
                                  side_effect=lambda u: order.append("open") or True):
            self.assertTrue(A._webbrowser_open("https://example.com"))
        self.assertEqual(order, ["snapshot", "open", "place"])
        args, kwargs = bc._place_new_browser_window_async.call_args
        self.assertEqual(args[0], {101, 102})
        self.assertTrue(kwargs["activate"])

    def test_no_focus_steal_while_a_game_holds_the_screen(self):
        bc = _fake_bc(guard=True)
        with mock.patch.object(A, "_loaded_bc", return_value=bc), \
                mock.patch.object(A.webbrowser, "open", return_value=True):
            A._webbrowser_open("https://example.com")
        self.assertFalse(bc._place_new_browser_window_async.call_args.kwargs["activate"])

    def test_without_a_monolith_it_is_plain_webbrowser_open(self):
        with mock.patch.object(A, "_loaded_bc", return_value=None), \
                mock.patch.object(A.webbrowser, "open", return_value=True) as mopen:
            self.assertTrue(A._webbrowser_open("https://example.com"))
        mopen.assert_called_once_with("https://example.com")

    def test_placement_failure_never_breaks_the_open(self):
        bc = _fake_bc()
        bc._place_new_browser_window_async.side_effect = RuntimeError("boom")
        with mock.patch.object(A, "_loaded_bc", return_value=bc), \
                mock.patch.object(A.webbrowser, "open", return_value=True):
            self.assertTrue(A._webbrowser_open("https://example.com"))
        bc2 = _fake_bc()
        bc2._window_handles_snapshot.side_effect = OSError("no windows")
        with mock.patch.object(A, "_loaded_bc", return_value=bc2), \
                mock.patch.object(A.webbrowser, "open", return_value=True):
            self.assertTrue(A._webbrowser_open("https://example.com"))
        bc2._place_new_browser_window_async.assert_not_called()

    def test_the_actions_route_through_the_placing_helper(self):
        bc = _fake_bc()
        with mock.patch.object(A, "_loaded_bc", return_value=bc), \
                mock.patch.object(A, "_bc", return_value=bc), \
                mock.patch.object(A, "_await_opened_page", return_value=None), \
                mock.patch.object(A, "_note_browser_tab_open"), \
                mock.patch.object(A, "_streaming_url_fix",
                                  side_effect=lambda u, *a, **k: (u, "")), \
                mock.patch.object(A.webbrowser, "open", return_value=True) as mopen:
            A._act_open_url("example.com")
            A._act_web_search("weather today")
            A._act_youtube("cats")
        self.assertEqual(mopen.call_count, 3)
        self.assertEqual(bc._place_new_browser_window_async.call_count, 3)


if __name__ == "__main__":
    unittest.main()
