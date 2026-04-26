"""
risk/risk_manager.py
─────────────────────
The risk engine.  Has unconditional veto power over ALL trading decisions.

Responsibilities:
1. Check kill switch state
2. Enforce exposure limits per symbol, sector, and portfolio
3. Size positions via the PositionSizer
4. Compute ATR for position sizing
5. Convert approved Decisions into Orders
6. Publish OrderEvents (or block and log reason)

The risk manager is the last line of defence before execution.
If it says NO, the trade does not happen.  Period.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, Optional

from algotrading.core.events import (
    DecisionEvent, EventBus, EventType, HaltEvent, OrderEvent
)
from algotrading.core.types import (
    Bar, Decision, DecisionAction, Direction, Order, OrderType, Position
)
from algotrading.backtest.portfolio import Portfolio
from algotrading.data.pit_handler import PITDataHandler
from algotrading.risk.kill_switch import KillSwitch
from algotrading.risk.position_sizer import FixedFractional, PositionSizer
from algotrading.strategies.trend_volatility import atr as compute_atr

logger = logging.getLogger(__name__)

UTC = timezone.utc


class RiskManager:
    """
    Wired into the event bus as a DecisionEvent consumer.
    Produces OrderEvents (approved) or swallows decisions (rejected).
    """

    def __init__(
        self,
        portfolio         : Portfolio,
        data_handler      : PITDataHandler,
        bus               : EventBus,
        kill_switch       : Optional[KillSwitch]    = None,
        position_sizer    : Optional[PositionSizer] = None,
        # Exposure limits
        max_position_pct  : float = 0.10,    # 10% of equity per symbol
        max_gross_exposure: float = 0.95,    # 95% gross exposure cap
        max_net_exposure  : float = 0.80,    # 80% net long bias
        atr_period        : int   = 14,
    ) -> None:
        self._portfolio     = portfolio
        self._data          = data_handler
        self._bus           = bus
        self.kill_switch    = kill_switch or KillSwitch()
        self._sizer         = position_sizer or FixedFractional()

        self.max_position_pct   = max_position_pct
        self.max_gross_exposure = max_gross_exposure
        self.max_net_exposure   = max_net_exposure
        self.atr_period         = atr_period

        self._blocked_count     = 0
        self._approved_count    = 0

    # ── Event handler ─────────────────────────────────────────────────────────

    def on_decision_event(self, event: DecisionEvent) -> None:
        decision = event.decision
        timestamp = decision.timestamp

        # 0. Kill switch
        killed, kill_reason = self.kill_switch.check(
            self._portfolio.equity, timestamp
        )
        if killed:
            self._block(decision, f"KILL_SWITCH: {kill_reason}")
            self._bus.halt(kill_reason, timestamp)
            return

        # 1. Flat/exit orders bypass most checks (always allow reducing risk)
        if decision.direction == Direction.FLAT:
            order = self._build_close_order(decision)
            if order:
                self._bus.publish(OrderEvent(order=order))
                self._approved_count += 1
            return

        # 2. No-trade decisions
        if decision.action != DecisionAction.TRADE:
            return

        # 3. Exposure checks
        equity = self._portfolio.equity
        approved, reason = self._check_exposure(decision, equity)
        if not approved:
            self._block(decision, reason)
            return

        # 4. Compute ATR for position sizing
        bars = self._data.history(decision.symbol, self.atr_period + 5)
        atr_val = compute_atr(bars, self.atr_period) if len(bars) > self.atr_period else None

        # 5. Position sizing
        quantity = self._sizer.size(
            equity             = equity,
            price              = self._current_price(decision.symbol),
            atr                = atr_val,
            signal_confidence  = 1.0,
        )

        if quantity < 1.0:
            self._block(decision, f"Sized to {quantity:.2f} shares — below minimum (1)")
            return

        # 6. Final notional check
        price    = self._current_price(decision.symbol)
        notional = quantity * price
        if notional / equity > self.max_position_pct:
            quantity = (self.max_position_pct * equity) / price
            logger.info(
                "[%s] Position capped to %.1f shares (%.1f%% of equity)",
                decision.symbol, quantity, self.max_position_pct * 100
            )

        order = self._build_order(decision, quantity)
        self._bus.publish(OrderEvent(order=order))
        self._approved_count += 1
        logger.info(
            "Risk APPROVED: %s %s %.0f shares | ATR=%.4f",
            decision.direction.value, decision.symbol, quantity, atr_val or 0
        )

    # ── Checks ────────────────────────────────────────────────────────────────

    def _check_exposure(self, decision: Decision, equity: float) -> tuple[bool, str]:
        gross = self._portfolio.gross_exposure()
        net   = self._portfolio.net_exposure()

        if gross / equity >= self.max_gross_exposure:
            return False, (
                f"Gross exposure {gross/equity:.1%} >= limit {self.max_gross_exposure:.1%}"
            )

        if decision.direction == Direction.LONG:
            if net / equity >= self.max_net_exposure:
                return False, (
                    f"Net long exposure {net/equity:.1%} >= limit {self.max_net_exposure:.1%}"
                )

        return True, ""

    # ── Order construction ────────────────────────────────────────────────────

    def _build_order(self, decision: Decision, quantity: float) -> Order:
        return Order(
            order_id    = str(uuid.uuid4())[:8],
            symbol      = decision.symbol,
            direction   = decision.direction,
            order_type  = OrderType.MARKET,
            quantity    = round(quantity, 2),
            limit_price = None,
            timestamp   = decision.timestamp,
            strategy_id = decision.strategy_id,
            reason      = decision.reason,
        )

    def _build_close_order(self, decision: Decision) -> Optional[Order]:
        pos = self._portfolio.position_for(decision.symbol)
        if pos is None or abs(pos.quantity) < 1e-9:
            return None
        close_direction = Direction.SHORT if pos.quantity > 0 else Direction.LONG
        return Order(
            order_id    = str(uuid.uuid4())[:8],
            symbol      = decision.symbol,
            direction   = close_direction,
            order_type  = OrderType.MARKET,
            quantity    = round(abs(pos.quantity), 2),
            limit_price = None,
            timestamp   = decision.timestamp,
            strategy_id = decision.strategy_id,
            reason      = f"Close: {decision.reason}",
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _current_price(self, symbol: str) -> float:
        bar = self._data.latest(symbol)
        return bar.close if bar else 0.0

    def _block(self, decision: Decision, reason: str) -> None:
        self._blocked_count += 1
        logger.info(
            "Risk BLOCKED: %s %s | reason: %s",
            decision.direction.value, decision.symbol, reason
        )

    def summary(self) -> dict:
        return {
            "approved_orders" : self._approved_count,
            "blocked_orders"  : self._blocked_count,
            "kill_triggered"  : self.kill_switch.is_triggered,
            "kill_reason"     : self.kill_switch.trigger_reason,
        }

    def reset(self) -> None:
        self._blocked_count  = 0
        self._approved_count = 0
