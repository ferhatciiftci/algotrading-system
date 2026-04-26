"""
monitoring/circuit_breaker.py
──────────────────────────────
System-level circuit breaker.

Distinct from the KillSwitch (which is trade-level):
the CircuitBreaker monitors SYSTEM health — data delays, fill anomalies,
PnL velocity, and error rates.  It fires the EventBus HALT event.

Triggers:
  1. Data staleness: no new bar in > max_data_delay_s seconds
  2. Fill error rate: > max_error_rate fraction of orders rejected/errored
  3. PnL velocity: equity moves > max_pnl_velocity_pct in one bar
  4. Consecutive error events: unhandled exceptions in handlers
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from algotrading.core.events import EventBus

logger = logging.getLogger(__name__)

UTC = timezone.utc


class CircuitBreaker:

    def __init__(
        self,
        bus                   : EventBus,
        max_data_delay_s      : float = 300.0,    # 5 minutes
        max_pnl_velocity_pct  : float = 0.05,     # 5% single-bar equity move
        max_error_rate        : float = 0.20,     # 20% fill error rate
        max_consecutive_errors: int   = 3,
    ) -> None:
        self._bus                    = bus
        self.max_data_delay_s        = max_data_delay_s
        self.max_pnl_velocity_pct    = max_pnl_velocity_pct
        self.max_error_rate          = max_error_rate
        self.max_consecutive_errors  = max_consecutive_errors

        self._last_bar_ts        : Optional[datetime] = None
        self._last_equity        : Optional[float]    = None
        self._consecutive_errors : int   = 0
        self._total_orders       : int   = 0
        self._errored_orders     : int   = 0
        self._triggered          : bool  = False

    # ── Check methods (called by monitoring loop) ─────────────────────────────

    def on_bar(self, timestamp: datetime, equity: float) -> None:
        if self._triggered:
            return

        # Data staleness check (live mode only — skip in backtest)
        if timestamp.tzinfo:
            now = datetime.now(UTC)
            delay = (now - timestamp).total_seconds()
            if delay > self.max_data_delay_s and delay < 86400:  # ignore very old backtest bars
                self._fire(f"Data stale: last bar {delay:.0f}s ago (limit {self.max_data_delay_s}s)",
                           timestamp)
                return

        # PnL velocity check
        if self._last_equity is not None and self._last_equity > 0:
            move = abs(equity - self._last_equity) / self._last_equity
            if move > self.max_pnl_velocity_pct:
                self._fire(
                    f"PnL velocity {move:.2%} in one bar exceeds limit {self.max_pnl_velocity_pct:.2%}",
                    timestamp
                )
                return

        self._last_equity = equity
        self._last_bar_ts = timestamp

    def on_fill_error(self, timestamp: datetime) -> None:
        self._errored_orders += 1
        self._check_error_rate(timestamp)

    def on_order(self) -> None:
        self._total_orders += 1

    def on_exception(self, timestamp: datetime) -> None:
        self._consecutive_errors += 1
        if self._consecutive_errors >= self.max_consecutive_errors:
            self._fire(
                f"Consecutive handler errors: {self._consecutive_errors}",
                timestamp
            )

    def on_success(self) -> None:
        self._consecutive_errors = 0

    # ── Internals ─────────────────────────────────────────────────────────────

    def _check_error_rate(self, timestamp: datetime) -> None:
        if self._total_orders == 0:
            return
        rate = self._errored_orders / self._total_orders
        if rate > self.max_error_rate:
            self._fire(
                f"Fill error rate {rate:.1%} > limit {self.max_error_rate:.1%}",
                timestamp
            )

    def _fire(self, reason: str, timestamp: datetime) -> None:
        if not self._triggered:
            self._triggered = True
            logger.critical("CIRCUIT BREAKER FIRED @ %s | %s", timestamp, reason)
            self._bus.halt(reason, timestamp)

    @property
    def is_triggered(self) -> bool:
        return self._triggered
