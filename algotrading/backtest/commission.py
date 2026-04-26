"""
backtest/commission.py
──────────────────────
Commission models.  All models are deterministic and configurable.

Available models:
  - ZeroCommission       : for academic comparison only
  - FixedPerShare        : e.g. IB tiered ~$0.005/share
  - PercentageOfNotional : e.g. 0.05% of trade value
  - TieredPerShare       : bracket-based (simulates broker tiers)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


class CommissionModel(ABC):
    @abstractmethod
    def calculate(self, price: float, quantity: float) -> float:
        """Return commission in dollars (always positive)."""
        ...


class ZeroCommission(CommissionModel):
    def calculate(self, price: float, quantity: float) -> float:
        return 0.0


@dataclass
class FixedPerShare(CommissionModel):
    """
    Flat rate per share with optional minimum per order.
    IB US equities: ~$0.005/share, min $1.00.
    """
    rate_per_share : float = 0.005
    min_per_order  : float = 1.00

    def calculate(self, price: float, quantity: float) -> float:
        return max(self.rate_per_share * quantity, self.min_per_order)


@dataclass
class PercentageOfNotional(CommissionModel):
    """
    Commission as a percentage of the notional value.
    Typical for non-US brokers: 0.05–0.1%.
    """
    rate : float = 0.0005   # 0.05% default

    def calculate(self, price: float, quantity: float) -> float:
        return price * quantity * self.rate


@dataclass
class TieredPerShare(CommissionModel):
    """
    Tiered pricing: lower rate for higher monthly share volume.
    Tiers: (volume_threshold, rate_per_share).
    Applied per-order based on order quantity alone (conservative approximation).
    """
    tiers: list[tuple[float, float]] = None

    def __post_init__(self) -> None:
        if self.tiers is None:
            # IB Pro tiers (simplified): monthly volume -> rate
            self.tiers = [
                (300_000, 0.0035),
                (3_000_000, 0.0020),
                (20_000_000, 0.0015),
                (float("inf"), 0.0010),
            ]

    def calculate(self, price: float, quantity: float) -> float:
        for threshold, rate in self.tiers:
            if quantity <= threshold:
                return max(rate * quantity, 1.0)
        return max(self.tiers[-1][1] * quantity, 1.0)
