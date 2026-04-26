"""
backtest/portfolio.py
─────────────────────
Tracks portfolio state during a backtest: cash, positions, equity curve.

Rules:
- All fills are applied immediately (no settlement lag in MVP).
- Equity = cash + sum(market_values of open positions).
- No leverage in MVP (can be added via risk engine).
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from algotrading.core.types import Direction, Fill, Position

logger = logging.getLogger(__name__)


class Portfolio:
    """
    Mutable portfolio state for one backtest run.
    Thread-safety is NOT guaranteed — single-threaded backtest only.
    """

    def __init__(self, initial_capital: float = 100_000.0) -> None:
        if initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        self.initial_capital : float = initial_capital
        self.cash            : float = initial_capital
        self.positions       : Dict[str, Position] = {}
        # Equity curve: list of (timestamp, equity) tuples
        self._equity_curve   : List[tuple[datetime, float]] = []
        self._fills          : List[Fill] = []

    # ── Core operations ───────────────────────────────────────────────────────

    def apply_fill(self, fill: Fill) -> None:
        """
        Update cash and position after a fill.
        Cash is reduced for buys, increased for sells.
        Commission always reduces cash.
        """
        sym = fill.symbol
        if sym not in self.positions:
            self.positions[sym] = Position(symbol=sym)

        pos = self.positions[sym]
        pos.apply_fill(fill)

        # Update cash
        notional = fill.fill_price * fill.quantity
        if fill.direction == Direction.LONG:
            self.cash -= notional + fill.commission
        else:
            self.cash += notional - fill.commission

        # Remove flat positions to keep dict clean
        if abs(pos.quantity) < 1e-9:
            del self.positions[sym]

        self._fills.append(fill)
        logger.debug(
            "Fill applied: %s %s %.0f @ %.4f | cash=%.2f",
            fill.direction.value, sym, fill.quantity, fill.fill_price, self.cash
        )

    def mark_to_market(self, prices: Dict[str, float], timestamp: datetime) -> float:
        """
        Update last prices for all open positions and record equity snapshot.
        Returns current equity.
        """
        for sym, pos in self.positions.items():
            if sym in prices:
                pos.update_price(prices[sym])
        equity = self.equity
        self._equity_curve.append((timestamp, equity))
        return equity

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def equity(self) -> float:
        return self.cash + sum(p.market_value for p in self.positions.values())

    @property
    def unrealised_pnl(self) -> float:
        return sum(p.unrealised_pnl for p in self.positions.values())

    @property
    def realised_pnl(self) -> float:
        return sum(p.realised_pnl for p in self.positions.values())

    @property
    def total_pnl(self) -> float:
        return self.equity - self.initial_capital

    @property
    def return_pct(self) -> float:
        return (self.equity / self.initial_capital - 1.0) * 100

    @property
    def equity_curve(self) -> pd.DataFrame:
        if not self._equity_curve:
            return pd.DataFrame(columns=["timestamp", "equity"])
        df = pd.DataFrame(self._equity_curve, columns=["timestamp", "equity"])
        df["drawdown"] = df["equity"] / df["equity"].cummax() - 1.0
        df["returns"]  = df["equity"].pct_change()
        return df

    @property
    def fills(self) -> List[Fill]:
        return list(self._fills)

    def position_for(self, symbol: str) -> Optional[Position]:
        return self.positions.get(symbol.upper())

    def gross_exposure(self) -> float:
        return sum(abs(p.market_value) for p in self.positions.values())

    def net_exposure(self) -> float:
        return sum(p.market_value for p in self.positions.values())

    # ── Reporting ─────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        return {
            "initial_capital" : self.initial_capital,
            "final_equity"    : round(self.equity, 2),
            "cash"            : round(self.cash, 2),
            "total_pnl"       : round(self.total_pnl, 2),
            "return_pct"      : round(self.return_pct, 3),
            "open_positions"  : len(self.positions),
            "total_fills"     : len(self._fills),
        }

    def reset(self) -> None:
        self.cash = self.initial_capital
        self.positions.clear()
        self._equity_curve.clear()
        self._fills.clear()
