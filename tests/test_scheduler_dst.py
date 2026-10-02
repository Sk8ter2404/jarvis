"""Audit A93: "tomorrow at 9am" / "today at 9am" must land on 9 am local wall
clock even when a daylight-saving change falls in between.

parse_when took ``now = datetime.now().astimezone()``. On CPython that tzinfo is
a FIXED-offset ``datetime.timezone`` (e.g. UTC-05:00 "Central Daylight Time"),
not a zone with rules, so ``(now + timedelta(days=1)).replace(hour=9)`` kept
TODAY's offset for TOMORROW's date. Spoken on 2026-10-31 (CDT, -05:00), the
reminder was stamped 09:00-05:00 = 08:00 CST on 2026-11-01: an hour early. The
spring change made it an hour late.

The host zone is pinned explicitly: ``core.scheduler.datetime`` is swapped for
a datetime subclass whose clock is frozen and whose ``astimezone()`` behaves
exactly like CPython's on a host set to America/Chicago -- a naive value is
read as Chicago wall time, and the result carries a FIXED offset (the shape the
bug rides on), never the zoneinfo object itself. So the test is independent of
the CI runner's zone (UTC) and of this machine's. Skipped only if the tz
database has no America/Chicago.
"""
from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

import core.scheduler as sched

try:
    from zoneinfo import ZoneInfo
    _CHICAGO = ZoneInfo("America/Chicago")
except Exception:  # pragma: no cover - no tz database on this host
    _CHICAGO = None

UTC = timezone.utc
CDT = timedelta(hours=-5)
CST = timedelta(hours=-6)


class _ChicagoHostDatetime(datetime):
    """``datetime`` as CPython behaves on a host whose local zone is
    America/Chicago, with ``now()`` frozen at ``_frozen`` (naive wall time)."""

    _frozen: datetime | None = None

    @classmethod
    def now(cls, tz=None):
        if tz is not None:
            return cls._frozen.replace(tzinfo=_CHICAGO).astimezone(tz)
        return cls._frozen

    def astimezone(self, tz=None):
        aware = self if self.tzinfo is not None else self.replace(tzinfo=_CHICAGO)
        if tz is not None:
            return datetime.astimezone(aware, tz)
        local = datetime.astimezone(aware, _CHICAGO)
        return local.replace(tzinfo=timezone(local.utcoffset(), local.tzname()))


def _frozen_at(*wall):
    """A _ChicagoHostDatetime subclass frozen at the given Chicago wall time."""
    return type("_Frozen", (_ChicagoHostDatetime,),
                {"_frozen": _ChicagoHostDatetime(*wall)})


@unittest.skipIf(_CHICAGO is None, "tz database has no America/Chicago")
class ParseWhenAcrossDstTests(unittest.TestCase):
    def _parse(self, text, *wall):
        with mock.patch.object(sched, "datetime", _frozen_at(*wall)):
            return sched.parse_when(text)

    def _assert_instant(self, result, utc_wall, offset):
        self.assertIsNotNone(result)
        self.assertEqual(result.astimezone(UTC),
                         datetime(*utc_wall, tzinfo=UTC))
        self.assertEqual(result.utcoffset(), offset)
        local = result.astimezone(_CHICAGO)
        self.assertEqual((local.hour, local.minute), (9, 0))

    def test_harness_reproduces_the_fixed_offset_shape(self):
        # Blindness guard: the fake must hand back a FIXED offset, as CPython
        # does, or the old code would pass by accident.
        now = _frozen_at(2026, 10, 31, 12, 0).now().astimezone()
        self.assertIsInstance(now.tzinfo, timezone)
        self.assertEqual(now.utcoffset(), CDT)

    def test_tomorrow_across_fall_back(self):
        # Spoken at noon CDT on Sat 2026-10-31; clocks fall back 2026-11-01.
        result = self._parse("tomorrow at 9am", 2026, 10, 31, 12, 0)
        self._assert_instant(result, (2026, 11, 1, 15, 0), CST)

    def test_tomorrow_across_spring_forward(self):
        # Spoken at noon CST on Sat 2027-03-13; clocks spring forward 03-14.
        result = self._parse("tomorrow 9am", 2027, 3, 13, 12, 0)
        self._assert_instant(result, (2027, 3, 14, 14, 0), CDT)

    def test_today_past_time_rolled_to_tomorrow_across_fall_back(self):
        # 9 am has passed at noon on 10-31, so it rolls to 11-01 (CST).
        result = self._parse("today at 9am", 2026, 10, 31, 12, 0)
        self._assert_instant(result, (2026, 11, 1, 15, 0), CST)

    def test_today_spoken_before_the_change_on_change_day(self):
        # 00:30 CDT on 2026-11-01: "today at 9am" is after the 2 am change.
        result = self._parse("today 9am", 2026, 11, 1, 0, 30)
        self._assert_instant(result, (2026, 11, 1, 15, 0), CST)

    def test_no_change_in_between_is_unaffected(self):
        result = self._parse("tomorrow at 9am", 2026, 7, 15, 12, 0)
        self._assert_instant(result, (2026, 7, 16, 14, 0), CDT)
        result = self._parse("today at 3pm", 2026, 7, 15, 12, 0)
        self.assertEqual(result.astimezone(UTC),
                         datetime(2026, 7, 15, 20, 0, tzinfo=UTC))

    def test_relative_offsets_stay_exact(self):
        # "in N hours" is an exact duration across the change, not wall time.
        result = self._parse("in 2 hours", 2026, 11, 1, 0, 30)
        self.assertEqual(result.astimezone(UTC),
                         datetime(2026, 11, 1, 7, 30, tzinfo=UTC))


if __name__ == "__main__":
    unittest.main()
