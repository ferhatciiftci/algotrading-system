"""
monitoring/pnl_tracker.py
──────────────────────────
Real-time P&L tracker.  Works in both backtest and live modes.

Tracks:
- Running equity curve
- Daily P&L
- Per-symbol P&L
- Realised vs unrealised split
- Anomalous equity moves
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import date, datetime, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

UTC = timezone.utc


class PnLTracker:

    def __init__(
        self,
        initial_equity       : float,
        anomaly_threshold_pct: float = 0.03,  # flag single-bar moves > 3%
    ) -> None:
        self._initial_equity       = initial_equity
        self._anomaly_threshold    = anomaly_threshold_pct

        self._equity_history       : List[Tuple[datetime, float]] = []
        self._daily_pnl            : Dict[date, float] = {}
        self._symbol_pnl           : Dict[str, float]  = defaultdict(float)
        self._last_equity          : float = initial_equity
        self._day_start_equity     : Dict[date, float] = {}
        self._anomalies            : List[dict] = []

    # ── Update ────────────────────────────────────────────────────────────────

    def update(
        self,
        timestamp : datetime,
        equity    : float,
        symbol    : Optional[str] = None,
        bar_pnl   : Optional[float] = None,
    ) -> None:
        today = timestamp.date()

        # Initialise day
        if today not in self._day_start_equity:
            self._day_start_equity[today] = self._last_equity

        # Anomaly check
        if self._last_equity > 0:
            move = (equity - self._last_equity) / self._last_equity
            if abs(move) > self._anomaly_threshold:
                anomaly = {
                    "timestamp"  : timestamp,
                    "equity_from": self._last_equity,
                    "equity_to"  : equity,
                    "move_pct"   : round(move * 100, 2),
                }
                self._anomalies.append(anomaly)
                logger.warning(
                    "PnL ANOMALY @ %s: equity %.0f → %.0f (%.2f%%)",
                    timestamp, self._last_equity, equity, move * 100
                )

        self._equity_history.append((timestamp, equity))
        self._daily_pnl[today] = equity - self._day_start_equity[today]
        self._last_equity = equity

        if symbol and bar_pnl is not None:
            self._symbol_pnl[symbol] += bar_pnl

    # ── Queries ───────────────────────────────────────────────────────────────

    @property
    def current_equity(self) -> float:
        return self._last_equity

    @property
    def total_pnl(self) -> float:
        return self._last_equity - self._initial_equity

    @property
    def total_return_pct(self) -> float:
        return (self._last_equity / self._initial_equity - 1.0) * 100

    def daily_pnl(self, d: date) -> float:
        return self._daily_pnl.get(d, 0.0)

    def equity_curve(self) -> pd.DataFrame:
        if not self._equity_history:
            return pd.DataFrame(columns=["timestamp", "equity"])
        df = pd.DataFrame(self._equity_history, columns=["timestamp", "equity"])
        df["returns"]   = df["equity"].pct_change()
        df["drawdown"]  = df["equity"] / df["equity"].cummax() - 1.0
        return df

    def anomalies(self) -> List[dict]:
        return list(self._anomalies)

    def symbol_pnl_summary(self) -> dict:
        return dict(self._symbol_pnl)

    def daily_pnl_summary(self) -> pd.DataFrame:
        if not self._daily_pnl:
            return pd.DataFrame()
        return pd.DataFrame(
            list(self._daily_pnl.items()),
            columns=["date", "daily_pnl"]
        ).sort_values("date")

    def snapshot(self) -> dict:
        return {
            "equity"           : round(self._last_equity, 2),
            "total_pnl"        : round(self.total_pnl, 2),
            "total_return_pct" : round(self.total_return_pct, 3),
            "anomalies"        : len(self._anomalies),
            "data_points"      : len(self._equity_history),
        }
