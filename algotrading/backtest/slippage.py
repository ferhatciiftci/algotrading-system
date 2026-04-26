"""
backtest/slippage.py
─────────────────────
Slippage models.

Slippage is the difference between the decision price (usually close) and
the actual fill price.  It is always adverse: you buy higher / sell lower.

Available models:
  - ZeroSlippage         : unrealistic, for baseline only
  - FixedBps             : constant n basis points
  - VolatilitySlippage   : scales with ATR (realistic for medium-freq)
  - MarketImpact         : square-root model for larger orders
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from algotrading.core.types import Direction


class SlippageModel(ABC):
    @abstractmethod
    def calculate(
        self,
        price     : float,
        quantity  : float,
        direction : Direction,
        atr       : float = 0.0,
        adv       : float = 0.0,   # average daily volume
    ) -> float:
        """
        Return signed slippage in price units.
        Positive = worse fill for the initiator of the trade.
        """
        ...


class ZeroSlippage(SlippageModel):
    def calculate(self, price, quantity, direction, atr=0.0, adv=0.0) -> float:
        return 0.0


@dataclass
class FixedBps(SlippageModel):
    """
    Fixed slippage in basis points.
    Default: 5 bps (0.05%) — conservative for liquid large-caps.
    """
    bps : float = 5.0

    def calculate(self, price, quantity, direction, atr=0.0, adv=0.0) -> float:
        return price * (self.bps / 10_000)


@dataclass
class VolatilitySlippage(SlippageModel):
    """
    Slippage proportional to ATR (average true range).
    Models the spread + market microstructure cost.

    fill_price = close ± (atr_multiple * ATR)
    Typical: 0.1–0.3x daily ATR for a medium-frequency strategy.
    """
    atr_multiple : float = 0.1
    min_bps      : float = 2.0    # floor in basis points

    def calculate(self, price, quantity, direction, atr=0.0, adv=0.0) -> float:
        floor = price * (self.min_bps / 10_000)
        if atr <= 0:
            return floor
        return max(self.atr_multiple * atr, floor)


@dataclass
class MarketImpact(SlippageModel):
    """
    Square-root market impact model.
    Appropriate when order size is a non-trivial fraction of ADV.

    impact = sigma * sqrt(order_qty / ADV) * price

    sigma: volatility coefficient (typically 0.1–0.3)
    """
    sigma        : float = 0.1
    atr_multiple : float = 0.05   # minimum slippage

    def calculate(self, price, quantity, direction, atr=0.0, adv=0.0) -> float:
        minimum = price * self.atr_multiple * (atr / price) if atr > 0 else 0.0
        if adv <= 0:
            return minimum
        participation = quantity / adv
        impact = self.sigma * math.sqrt(participation) * price
        return max(impact, minimum)
