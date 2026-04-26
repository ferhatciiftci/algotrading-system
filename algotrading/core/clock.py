"""
core/clock.py
─────────────
Market clock and trading-calendar utilities.

Rules:
- All times are UTC-aware.
- The clock is PASSIVE in backtests (driven by bar timestamps).
- In live mode it wraps datetime.now(UTC).
- Never call datetime.now() directly outside this module.
"""

from __future__ import annotations

import calendar
from datetime import date, datetime, time, timedelta, timezone
from typing import Iterator, Optional, Set

UTC = timezone.utc


# ─── Helpers ──────────────────────────────────────────────────────────────────

def utcnow() -> datetime:
    return datetime.now(UTC)


def to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


# ─── Simple US equity trading calendar ────────────────────────────────────────
# A minimal hard-coded set of NYSE holidays for 2020-2030.
# In production, replace with pandas_market_calendars or a database.

_NYSE_FIXED_HOLIDAYS: Set[tuple[int, int]] = {
    (1, 1),   # New Year's Day
    (7, 4),   # Independence Day
    (12, 25), # Christmas
    (12, 26), # Christmas observed (if 25 is Sunday)
}


def _is_weekend(d: date) -> bool:
    return d.weekday() >= 5  # Sat=5, Sun=6


def _is_us_market_holiday(d: date) -> bool:
    """Very conservative check – does not cover all holidays."""
    if (d.month, d.day) in _NYSE_FIXED_HOLIDAYS:
        return True
    # MLK Day: 3rd Monday in January
    if d.month == 1 and d.weekday() == 0:
        mondays = sum(1 for day in range(1, d.day + 1)
                      if date(d.year, 1, day).weekday() == 0)
        if mondays == 3:
            return True
    # Memorial Day: last Monday in May
    if d.month == 5 and d.weekday() == 0:
        next_monday = d + timedelta(days=7)
        if next_monday.month != 5:
            return True
    # Labor Day: 1st Monday in September
    if d.month == 9 and d.weekday() == 0:
        first_monday = d - timedelta(days=(d.day - 1) % 7)
        if d == first_monday:
            return True
    # Thanksgiving: 4th Thursday in November
    if d.month == 11 and d.weekday() == 3:
        thursdays = sum(1 for day in range(1, d.day + 1)
                        if date(d.year, 11, day).weekday() == 3)
        if thursdays == 4:
            return True
    return False


# ─── Market session definition ────────────────────────────────────────────────

_NYSE_OPEN  = time(14, 30, tzinfo=UTC)   # 09:30 ET = 14:30 UTC (standard time)
_NYSE_CLOSE = time(21, 0,  tzinfo=UTC)   # 16:00 ET = 21:00 UTC


def is_trading_day(d: date) -> bool:
    return not _is_weekend(d) and not _is_us_market_holiday(d)


def market_open_utc(d: date) -> datetime:
    return datetime.combine(d, _NYSE_OPEN)


def market_close_utc(d: date) -> datetime:
    return datetime.combine(d, _NYSE_CLOSE)


def is_within_session(dt: datetime) -> bool:
    d = dt.date()
    if not is_trading_day(d):
        return False
    return market_open_utc(d) <= dt <= market_close_utc(d)


def trading_days_between(start: date, end: date) -> list[date]:
    """Inclusive range of trading days."""
    days = []
    current = start
    while current <= end:
        if is_trading_day(current):
            days.append(current)
        current += timedelta(days=1)
    return days


# ─── Clock ────────────────────────────────────────────────────────────────────

class Clock:
    """
    A simple clock abstraction.

    Backtest mode: advance() is called with each bar timestamp.
    Live mode    : now() reads the system clock.
    """

    def __init__(self, live: bool = False) -> None:
        self._live = live
        self._current: Optional[datetime] = None

    def advance(self, ts: datetime) -> None:
        """Called by the backtest engine with each new bar timestamp."""
        if self._live:
            raise RuntimeError("Cannot manually advance a live clock.")
        ts = to_utc(ts)
        if self._current is not None and ts < self._current:
            raise ValueError(
                f"Clock went backwards: {ts} < {self._current} — "
                "check data ordering."
            )
        self._current = ts

    def now(self) -> datetime:
        if self._live:
            return utcnow()
        if self._current is None:
            raise RuntimeError("Clock has not been advanced yet.")
        return self._current

    @property
    def is_live(self) -> bool:
        return self._live

    def reset(self) -> None:
        if not self._live:
            self._current = None
