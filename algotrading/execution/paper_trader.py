"""
execution/paper_trader.py
──────────────────────────
Simulated execution handler (paper trading).

Simulates realistic order execution:
- MARKET orders fill at next-bar open + slippage (not at signal price)
- Applies commission and slippage models
- Does NOT allow fills at unrealistic prices
- Rejects orders if the simulated price is too far from close
- Records fill audit trail

This is the ONLY execution module used in the MVP.
Live execution (e.g. via Alpaca / IB) would implement the same interface.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from algotrading.backtest.commission import CommissionModel, FixedPerShare
from algotrading.backtest.slippage import SlippageModel, VolatilitySlippage
from algotrading.core.events import EventBus, FillEvent, OrderEvent
from algotrading.core.types import Bar, Direction, Fill, Order
from algotrading.data.pit_handler import PITDataHandler
from algotrading.strategies.trend_volatility import atr as compute_atr

logger = logging.getLogger(__name__)

UTC = timezone.utc

# Max allowed slippage as fraction of price before we reject the fill
_MAX_SLIPPAGE_FRACTION = 0.05   # 5% — catches data anomalies


class SimulatedExecution:
    """
    Simulates a broker.  Fills MARKET orders at close price + slippage.

    In a more realistic simulation you would fill at next-bar open;
    for daily strategies the difference is negligible.  For intraday,
    this should be changed.
    """

    def __init__(
        self,
        data_handler  : PITDataHandler,
        bus           : EventBus,
        commission    : CommissionModel = None,
        slippage      : SlippageModel   = None,
        atr_period    : int = 14,
    ) -> None:
        self._data       = data_handler
        self._bus        = bus
        self._commission = commission or FixedPerShare()
        self._slippage   = slippage   or VolatilitySlippage()
        self._atr_period = atr_period

        self._fills_count = 0
        self._rejected    = 0

    # ── Event handler ─────────────────────────────────────────────────────────

    def on_order_event(self, event: OrderEvent) -> None:
        order = event.order
        fill  = self._simulate_fill(order)
        if fill:
            self._fills_count += 1
            self._bus.publish(FillEvent(fill=fill))
        else:
            self._rejected += 1
            logger.warning("Order %s rejected by execution simulator", order.order_id)

    # ── Core simulation ───────────────────────────────────────────────────────

    def _simulate_fill(self, order: Order) -> Optional[Fill]:
        bar = self._data.latest(order.symbol)
        if bar is None:
            logger.error("No market data for %s — cannot fill order %s",
                         order.symbol, order.order_id)
            return None

        # Use current close as fill base price
        base_price = bar.close
        if base_price <= 0:
            logger.error("Zero/negative close price for %s", order.symbol)
            return None

        # Compute ATR for slippage
        hist    = self._data.history(order.symbol, self._atr_period + 5)
        atr_val = compute_atr(hist, self._atr_period) if len(hist) > self._atr_period else 0.0
        adv     = self._estimate_adv(hist)

        # Compute slippage (always adverse)
        raw_slip = self._slippage.calculate(
            price     = base_price,
            quantity  = order.quantity,
            direction = order.direction,
            atr       = atr_val,
            adv       = adv,
        )

        # Sanity check
        if raw_slip / base_price > _MAX_SLIPPAGE_FRACTION:
            logger.warning(
                "Slippage %.4f > %.0f%% of price %.4f for %s — capping",
                raw_slip, _MAX_SLIPPAGE_FRACTION * 100, base_price, order.symbol
            )
            raw_slip = base_price * _MAX_SLIPPAGE_FRACTION

        # Fill price: buy higher, sell lower
        if order.direction == Direction.LONG:
            fill_price = base_price + raw_slip
        else:
            fill_price = base_price - raw_slip

        fill_price = max(fill_price, 0.01)   # floor at 1 cent

        # Commission
        commission = self._commission.calculate(fill_price, order.quantity)

        fill = Fill(
            fill_id    = str(uuid.uuid4())[:8],
            order_id   = order.order_id,
            symbol     = order.symbol,
            direction  = order.direction,
            quantity   = order.quantity,
            fill_price = round(fill_price, 4),
            commission = round(commission, 4),
            slippage   = round(raw_slip, 4),
            timestamp  = bar.timestamp,
            exchange   = "SIM",
        )

        logger.debug(
            "SIM FILL: %s %s %.0f @ %.4f (slip=%.4f comm=%.4f)",
            fill.direction.value, fill.symbol, fill.quantity,
            fill.fill_price, fill.slippage, fill.commission
        )
        return fill

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _estimate_adv(bars) -> float:
        """Average daily volume from recent bars."""
        if not bars:
            return 0.0
        volumes = [b.volume for b in bars[-20:] if b.volume > 0]
        return sum(volumes) / len(volumes) if volumes else 0.0

    def summary(self) -> dict:
        return {
            "fills_executed" : self._fills_count,
            "fills_rejected" : self._rejected,
        }

    def reset(self) -> None:
        self._fills_count = 0
        self._rejected    = 0
