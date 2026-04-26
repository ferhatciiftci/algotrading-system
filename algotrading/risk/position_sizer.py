"""
risk/position_sizer.py
───────────────────────
Position sizing methods.

Design principle: size positions based on risk, not on arbitrary dollar amounts.
The risk engine controls HOW MUCH we risk per trade, never more.

Available sizers:
  - FixedFractional  : risk X% of equity per trade (Kelly-inspired, conservative)
  - ATRSizer         : size so that 1 ATR move = X% of equity (volatility parity)
  - FixedShares      : constant shares (for testing only)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


class PositionSizer(ABC):

    @abstractmethod
    def size(
        self,
        equity         : float,
        price          : float,
        atr            : Optional[float] = None,
        signal_confidence: float = 1.0,
    ) -> float:
        """Return number of shares/units to trade (always positive)."""
        ...


@dataclass
class FixedFractional(PositionSizer):
    """
    Risk a fixed fraction of current equity per trade.
    Uses a stop-loss distance (1x ATR) to back-calculate position size.

    max_risk_pct : fraction of equity to risk per trade (e.g. 0.01 = 1%)
    atr_stop_mult: how many ATRs away the implied stop is
    max_position_pct: hard cap — position notional / equity
    """
    max_risk_pct     : float = 0.01     # 1% of equity per trade
    atr_stop_mult    : float = 2.0      # stop at 2x ATR
    max_position_pct : float = 0.10     # never exceed 10% of equity in one name

    def size(self, equity, price, atr=None, signal_confidence=1.0) -> float:
        if price <= 0 or equity <= 0:
            return 0.0

        dollar_risk = equity * self.max_risk_pct * signal_confidence

        if atr and atr > 0:
            stop_distance = atr * self.atr_stop_mult
            shares = dollar_risk / stop_distance
        else:
            # Fallback: assume 2% stop
            stop_distance = price * 0.02
            shares = dollar_risk / stop_distance

        # Apply notional cap
        max_notional = equity * self.max_position_pct
        shares = min(shares, max_notional / price)

        shares = max(0.0, shares)
        logger.debug(
            "FixedFractional: equity=%.0f risk_$=%.0f stop=%.4f → %.1f shares",
            equity, dollar_risk, stop_distance, shares
        )
        return shares


@dataclass
class ATRSizer(PositionSizer):
    """
    Volatility-parity sizing: each position contributes equal ATR-based risk.

    target_vol_pct : target daily portfolio volatility per position
    """
    target_vol_pct   : float = 0.005    # 0.5% daily vol contribution per trade
    max_position_pct : float = 0.15

    def size(self, equity, price, atr=None, signal_confidence=1.0) -> float:
        if price <= 0 or equity <= 0:
            return 0.0
        if not atr or atr <= 0:
            logger.warning("ATRSizer called without ATR — using 0 size")
            return 0.0

        target_dollar_vol = equity * self.target_vol_pct * signal_confidence
        shares = target_dollar_vol / atr
        shares = min(shares, (equity * self.max_position_pct) / price)
        return max(0.0, shares)


@dataclass
class FixedShares(PositionSizer):
    """Fixed share count.  For testing only."""
    shares: float = 100.0

    def size(self, equity, price, atr=None, signal_confidence=1.0) -> float:
        return self.shares
