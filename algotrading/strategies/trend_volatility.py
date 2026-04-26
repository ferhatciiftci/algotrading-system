"""
strategies/trend_volatility.py
───────────────────────────────
Strategy: EMA Trend-Following with ATR Volatility Filter

Logic (fully explainable, medium frequency):
──────────────────────────────────────────────
SIGNAL GENERATION:
  Long  entry:  EMA_fast > EMA_slow  AND  price > EMA_slow  AND vol_regime == LOW
  Short entry:  EMA_fast < EMA_slow  AND  price < EMA_slow  AND vol_regime == LOW
  Exit  long :  EMA_fast < EMA_slow  OR   vol_regime == HIGH
  Exit  short:  EMA_fast > EMA_slow  OR   vol_regime == HIGH

VOLATILITY FILTER:
  ATR(atr_period) is computed each bar.
  vol_regime = HIGH  if  ATR / close > vol_threshold  (normalised ATR)
  High volatility → no new entries (avoid choppy / stressed markets)

WARMUP:
  Requires max(ema_slow, atr_period) + 5 bars before generating signals.

PARAMETERS (all overridable via config):
  ema_fast      : 20   bars
  ema_slow      : 50   bars
  atr_period    : 14   bars
  vol_threshold : 0.025 (2.5% normalised ATR)  ← main regime filter

SIGNAL CONFIDENCE:
  confidence = (EMA_fast - EMA_slow) / EMA_slow
  Clipped to [0, 1] after normalisation.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Deque, List, Optional

import numpy as np

from algotrading.core.events import EventBus
from algotrading.core.types import Bar, Direction, Signal, SignalStrength
from algotrading.data.pit_handler import PITDataHandler
from algotrading.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


# ─── Indicator helpers (pure functions, easily unit-tested) ───────────────────

def ema(prices: List[float], period: int) -> float:
    """Exponential moving average of the last `period` values."""
    if len(prices) < period:
        return float("nan")
    k = 2.0 / (period + 1)
    result = prices[-period]          # seed with first value in window
    for p in prices[-period + 1:]:
        result = p * k + result * (1 - k)
    return result


def atr(bars: List[Bar], period: int) -> float:
    """Average True Range over `period` bars."""
    if len(bars) < period + 1:
        return float("nan")
    trs = []
    for i in range(1, len(bars)):
        prev_close = bars[i - 1].close
        tr = max(
            bars[i].high - bars[i].low,
            abs(bars[i].high - prev_close),
            abs(bars[i].low  - prev_close),
        )
        trs.append(tr)
    # Wilder's smoothed ATR (RMA)
    result = sum(trs[-period:]) / period
    return result


# ─── Strategy ─────────────────────────────────────────────────────────────────

class TrendVolatilityStrategy(BaseStrategy):

    STRATEGY_ID = "trend_volatility_v1"

    def __init__(
        self,
        data_handler   : PITDataHandler,
        bus            : EventBus,
        symbol         : str,
        ema_fast       : int   = 20,
        ema_slow       : int   = 50,
        atr_period     : int   = 14,
        vol_threshold  : float = 0.025,    # normalised ATR filter
    ) -> None:
        super().__init__(data_handler, bus)
        self.symbol        = symbol.upper()
        self.ema_fast      = ema_fast
        self.ema_slow      = ema_slow
        self.atr_period    = atr_period
        self.vol_threshold = vol_threshold

        self._bars_seen    = 0
        self._last_signal  : Optional[Direction] = None  # track current state

    def warmup_period(self) -> int:
        return max(self.ema_slow, self.atr_period) + 5

    # ── Core logic ────────────────────────────────────────────────────────────

    def on_bar(self, bar: Bar) -> Optional[Signal]:
        if bar.symbol != self.symbol:
            return None

        self._bars_seen += 1
        if self._bars_seen < self.warmup_period():
            return None

        # Retrieve history (PIT-safe: includes current bar)
        lookback = self.ema_slow + self.atr_period + 10
        hist     = self._data.history(self.symbol, lookback)
        if len(hist) < self.warmup_period():
            return None

        closes = [b.close for b in hist]

        # ── Compute indicators ────────────────────────────────────────────────
        ema_f = ema(closes, self.ema_fast)
        ema_s = ema(closes, self.ema_slow)
        atr_v = atr(hist, self.atr_period)

        if any(np.isnan(x) for x in (ema_f, ema_s, atr_v)):
            return None

        price     = bar.close
        norm_atr  = atr_v / price          # normalised ATR (relative volatility)
        high_vol  = norm_atr > self.vol_threshold

        trend_up   = ema_f > ema_s and price > ema_s
        trend_down = ema_f < ema_s and price < ema_s

        # ── Signal logic ──────────────────────────────────────────────────────

        direction  : Optional[Direction] = None
        reason     : str = ""
        strength   : SignalStrength = SignalStrength.WEAK
        confidence : float = 0.0

        if self._last_signal == Direction.LONG:
            # Manage open long
            if ema_f < ema_s or high_vol:
                direction = Direction.FLAT
                reason    = (
                    f"Exit long: EMA_fast({ema_f:.2f}) {'<' if ema_f < ema_s else '>='} "
                    f"EMA_slow({ema_s:.2f})"
                    + (" | high vol" if high_vol else "")
                )
                strength = SignalStrength.STRONG

        elif self._last_signal == Direction.SHORT:
            # Manage open short
            if ema_f > ema_s or high_vol:
                direction = Direction.FLAT
                reason    = (
                    f"Exit short: EMA_fast({ema_f:.2f}) {'>' if ema_f > ema_s else '<='} "
                    f"EMA_slow({ema_s:.2f})"
                    + (" | high vol" if high_vol else "")
                )
                strength = SignalStrength.STRONG

        else:
            # No position — look for entry
            if not high_vol:
                if trend_up:
                    direction  = Direction.LONG
                    separation = (ema_f - ema_s) / ema_s
                    confidence = min(separation * 20, 1.0)   # scale to [0,1]
                    strength   = SignalStrength.STRONG if confidence > 0.5 else SignalStrength.WEAK
                    reason     = (
                        f"Long entry: EMA_fast({ema_f:.2f}) > EMA_slow({ema_s:.2f}) "
                        f"price({price:.2f}) > EMA_slow | norm_ATR={norm_atr:.3f}"
                    )

                elif trend_down:
                    direction  = Direction.SHORT
                    separation = (ema_s - ema_f) / ema_s
                    confidence = min(separation * 20, 1.0)
                    strength   = SignalStrength.STRONG if confidence > 0.5 else SignalStrength.WEAK
                    reason     = (
                        f"Short entry: EMA_fast({ema_f:.2f}) < EMA_slow({ema_s:.2f}) "
                        f"price({price:.2f}) < EMA_slow | norm_ATR={norm_atr:.3f}"
                    )
            else:
                logger.debug(
                    "[%s] No entry: high volatility regime | norm_ATR=%.3f > threshold=%.3f",
                    self.symbol, norm_atr, self.vol_threshold
                )

        if direction is None:
            return None

        self._last_signal = direction

        return Signal(
            strategy_id = self.STRATEGY_ID,
            symbol      = self.symbol,
            timestamp   = bar.timestamp,
            direction   = direction,
            strength    = strength,
            reason      = reason,
            confidence  = confidence if direction != Direction.FLAT else 1.0,
        )

    def reset(self) -> None:
        self._bars_seen   = 0
        self._last_signal = None
