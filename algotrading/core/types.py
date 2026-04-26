"""
core/types.py
─────────────
Immutable dataclasses for every domain object that crosses module boundaries.
All timestamps are UTC-aware.  No mutable state lives here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum, auto
from typing import Optional


# ─── Enumerations ─────────────────────────────────────────────────────────────

class Direction(Enum):
    LONG  = "LONG"
    SHORT = "SHORT"
    FLAT  = "FLAT"


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT  = "LIMIT"


class OrderStatus(Enum):
    PENDING   = "PENDING"
    SUBMITTED = "SUBMITTED"
    FILLED    = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED  = "REJECTED"


class SignalStrength(Enum):
    STRONG = "STRONG"
    WEAK   = "WEAK"


class DecisionAction(Enum):
    TRADE    = "TRADE"     # proceed with the signal
    NO_TRADE = "NO_TRADE"  # blocked (risk, no signal, or conflict)
    REDUCE   = "REDUCE"    # partial risk reduction required


# ─── Market data ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Bar:
    """A single OHLCV bar.  Timestamps are the bar-close time (UTC)."""
    symbol    : str
    timestamp : datetime        # bar-close, UTC-aware
    open      : float
    high      : float
    low       : float
    close     : float
    volume    : float
    adjusted  : bool = False    # True if close is already adj-close

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None:
            raise ValueError(f"Bar timestamp must be UTC-aware: {self.timestamp}")
        if not (self.low <= self.open <= self.high):
            raise ValueError(f"OHLC sanity failed for {self.symbol} @ {self.timestamp}")
        if not (self.low <= self.close <= self.high):
            raise ValueError(f"OHLC sanity failed for {self.symbol} @ {self.timestamp}")


# ─── Signals ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Signal:
    """Raw directional signal produced by a strategy."""
    strategy_id : str
    symbol      : str
    timestamp   : datetime
    direction   : Direction
    strength    : SignalStrength
    reason      : str           # human-readable explanation (mandatory)
    confidence  : float = 1.0   # [0, 1] – for multi-strategy weighting

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Signal confidence must be in [0, 1]")
        if not self.reason:
            raise ValueError("Signal must carry a reason string")


# ─── Orders & fills ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Order:
    """An order sent to the execution layer."""
    order_id    : str
    symbol      : str
    direction   : Direction
    order_type  : OrderType
    quantity    : float         # shares / contracts (always positive)
    limit_price : Optional[float]
    timestamp   : datetime      # when the order was created
    strategy_id : str
    reason      : str           # propagated from Signal or risk action

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError(f"Order quantity must be positive, got {self.quantity}")


@dataclass(frozen=True)
class Fill:
    """Confirmation of an executed order."""
    fill_id       : str
    order_id      : str
    symbol        : str
    direction     : Direction
    quantity      : float
    fill_price    : float
    commission    : float
    slippage      : float       # signed: positive = worse fill for us
    timestamp     : datetime
    exchange      : str = "SIM"

    @property
    def total_cost(self) -> float:
        """Signed notional including commission and slippage."""
        sign = 1.0 if self.direction == Direction.LONG else -1.0
        return sign * self.quantity * (self.fill_price + self.slippage) + self.commission


# ─── Positions ────────────────────────────────────────────────────────────────

@dataclass
class Position:
    """Mutable live position for one symbol."""
    symbol        : str
    quantity      : float = 0.0     # + long, - short
    avg_cost      : float = 0.0
    realised_pnl  : float = 0.0
    last_price    : float = 0.0

    @property
    def direction(self) -> Direction:
        if self.quantity > 0:
            return Direction.LONG
        if self.quantity < 0:
            return Direction.SHORT
        return Direction.FLAT

    @property
    def unrealised_pnl(self) -> float:
        return (self.last_price - self.avg_cost) * self.quantity

    @property
    def market_value(self) -> float:
        return self.last_price * self.quantity

    def update_price(self, price: float) -> None:
        self.last_price = price

    def apply_fill(self, fill: Fill) -> None:
        """Update position after a fill.  Handles partial closes."""
        if fill.direction == Direction.LONG:
            new_qty = self.quantity + fill.quantity
            if self.quantity >= 0:
                # adding to long
                total_cost = self.avg_cost * self.quantity + fill.fill_price * fill.quantity
                self.avg_cost = total_cost / new_qty if new_qty != 0 else 0.0
            else:
                # covering short
                closed = min(fill.quantity, abs(self.quantity))
                self.realised_pnl += closed * (self.avg_cost - fill.fill_price)
                if new_qty > 0:
                    self.avg_cost = fill.fill_price
            self.quantity = new_qty
        else:  # SHORT fill
            new_qty = self.quantity - fill.quantity
            if self.quantity <= 0:
                # adding to short
                total_cost = self.avg_cost * abs(self.quantity) + fill.fill_price * fill.quantity
                self.avg_cost = total_cost / abs(new_qty) if new_qty != 0 else 0.0
            else:
                # closing long
                closed = min(fill.quantity, self.quantity)
                self.realised_pnl += closed * (fill.fill_price - self.avg_cost)
                if new_qty < 0:
                    self.avg_cost = fill.fill_price
            self.quantity = new_qty
        self.last_price = fill.fill_price


# ─── Decision ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Decision:
    """
    The final trade/no-trade verdict produced by the Orchestrator.
    Every field is mandatory so the audit trail is always complete.
    """
    timestamp       : datetime
    symbol          : str
    action          : DecisionAction
    direction       : Direction
    target_quantity : float         # 0 for NO_TRADE
    strategy_id     : str
    reason          : str           # plain-English explanation
    risk_approved   : bool
    signal_strength : SignalStrength
    override_by     : Optional[str] = None   # e.g. "RISK_ENGINE/kill_switch"

    def is_actionable(self) -> bool:
        return self.action == DecisionAction.TRADE and self.risk_approved
