"""Focused unit tests for ``skills.notification_triage``.

These target the 2026-07-08 fix (finding #35): the manual IAsyncOperation poll
fallback in ``_request_access`` / ``_get_notifications`` must be bounded by a
deadline so a wedged WinRT op can't spin the triage daemon forever. Stdlib
``unittest`` only — no winsdk, no real notifications, no long sleeps.
"""
from __future__ import annotations

import unittest
from unittest import mock

from skills import notification_triage as nt


class _NeverCompletesOp:
    """An IAsyncOperation-like whose ``.get()`` fails (forcing the manual poll)
    and which never flips ``.completed`` — i.e. the wedged-op scenario."""
    completed = False

    def get(self):
        raise RuntimeError("no synchronous get()")

    def get_results(self):  # pragma: no cover - never reached
        raise AssertionError("should time out before get_results")


class _Listener:
    def __init__(self, op):
        self._op = op

    def request_access_async(self):
        return self._op

    def get_notifications_async(self, _kind):
        return self._op


class _Kinds:
    TOAST = object()


class AsyncPollDeadlineTests(unittest.TestCase):
    def setUp(self):
        # Shrink the ceiling so the test is fast; restored on tearDown.
        self._saved_timeout = nt.ASYNC_OP_TIMEOUT_SECONDS
        nt.ASYNC_OP_TIMEOUT_SECONDS = 0.2
        self._saved_kinds = nt._winsdk_modules.get("NotificationKinds")
        nt._winsdk_modules["NotificationKinds"] = _Kinds

    def tearDown(self):
        nt.ASYNC_OP_TIMEOUT_SECONDS = self._saved_timeout
        if self._saved_kinds is None:
            nt._winsdk_modules.pop("NotificationKinds", None)
        else:
            nt._winsdk_modules["NotificationKinds"] = self._saved_kinds

    def test_request_access_times_out_instead_of_spinning(self):
        listener = _Listener(_NeverCompletesOp())
        with mock.patch.object(nt.time, "sleep", return_value=None):
            with self.assertRaises(RuntimeError):
                nt._request_access(listener)

    def test_get_notifications_times_out_instead_of_spinning(self):
        listener = _Listener(_NeverCompletesOp())
        with mock.patch.object(nt.time, "sleep", return_value=None):
            with self.assertRaises(RuntimeError):
                nt._get_notifications(listener)

    def test_timeout_constant_is_bounded(self):
        self.assertTrue(0 < self._saved_timeout <= 30)


if __name__ == "__main__":
    unittest.main()
