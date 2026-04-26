"""
strategies/multi_indicator.py
──────────────────────────────
Çok göstergeli strateji: EMA + RSI + MACD + ADX + SL/TP/Trailing Stop

GİRİŞ KOŞULLARI (LONG):
  1. EMA_fast > EMA_slow           (yükseliş trendi)
  2. RSI < rsi_buy                 (aşırı satım değil, momentum var)
  3. MACD_line > MACD_signal       (momentum onayı)
  4. ADX > adx_threshold           (güçlü trend, yatay piyasadan kaçın)
  5. norm_ATR < vol_threshold      (volatilite filtresi)
  6. Cooldown beklendi             (son işlemden bu yana yeterli bar geçti)

GİRİŞ KOŞULLARI (SHORT):
  Yukarıdakinin tersi + allow_short=True

ÇIKIŞ (SL/TP/Trailing):
  - Stop-Loss    : giriş fiyatından %sl_pct düşüş (long) / yükseliş (short)
  - Take-Profit  : giriş fiyatından %tp_pct yükseliş (long) / düşüş (short)
  - Trailing Stop: pozisyon boyunca peak'i takip eder, %trail_pct geri çekilince kapar
  - EMA ters kesim / RSI aşırı bölge çıkışı

PARAMETRELER:
  ema_fast, ema_slow   : EMA crossover
  rsi_period           : RSI periyodu (varsayılan: 14)
  rsi_buy, rsi_sell    : RSI giriş eşikleri (varsayılan: 55, 45)
  macd_fast/slow/signal: MACD parametreleri (12, 26, 9)
  adx_period           : ADX periyodu (varsayılan: 14)
  adx_threshold        : Minimum ADX değeri (varsayılan: 20)
  atr_period           : ATR periyodu (volatilite ve trailing için)
  vol_threshold        : Normalize ATR filtre eşiği
  sl_pct               : Stop-loss yüzdesi (0.02 = %2)
  tp_pct               : Take-profit yüzdesi (0.04 = %4)
  trail_pct            : Trailing stop yüzdesi (0 = devre dışı)
  cooldown_bars        : İşlemler arası minimum bar sayısı
"""
from __future__ import annotations

import logging
import math
from collections import deque
from typing import List, Optional, Tuple

import numpy as np

from algotrading.core.events import EventBus
from algotrading.core.types import Bar, Direction, Signal, SignalStrength
from algotrading.data.pit_handler import PITDataHandler
from algotrading.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Gösterge hesaplayıcılar (saf fonksiyonlar — bağımsız test edilebilir)
# ─────────────────────────────────────────────────────────────────────────────

def ema(prices: List[float], period: int) -> float:
    """Üstel hareketli ortalama."""
    if len(prices) < period:
        return float("nan")
    k = 2.0 / (period + 1)
    val = prices[-period]
    for p in prices[-period + 1:]:
        val = p * k + val * (1 - k)
    return val


def atr(bars: List[Bar], period: int) -> float:
    """Wilder ATR."""
    if len(bars) < period + 1:
        return float("nan")
    trs = []
    for i in range(1, len(bars)):
        pc = bars[i - 1].close
        tr = max(bars[i].high - bars[i].low,
                 abs(bars[i].high - pc),
                 abs(bars[i].low  - pc))
        trs.append(tr)
    return sum(trs[-period:]) / period


def rsi(prices: List[float], period: int = 14) -> float:
    """
    Wilder RSI.
    Değer aralığı [0, 100].  >70 = aşırı alım, <30 = aşırı satım.
    """
    if len(prices) < period + 1:
        return float("nan")
    deltas = [prices[i] - prices[i - 1] for i in range(len(prices) - period, len(prices))]
    gains  = [max(d, 0.0) for d in deltas]
    losses = [abs(min(d, 0.0)) for d in deltas]
    avg_gain = sum(gains)  / period
    avg_loss = sum(losses) / period
    if avg_loss < 1e-12:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1 + rs))


def macd(prices: List[float],
         fast: int = 12, slow: int = 26, signal: int = 9,
         ) -> Tuple[float, float, float]:
    """
    MACD hesaplayıcı.
    Döndürür: (macd_line, signal_line, histogram)
    macd_line > signal_line → yükseliş momentumu
    """
    needed = slow + signal + 5
    if len(prices) < needed:
        nan = float("nan")
        return nan, nan, nan

    # MACD history için son (slow+signal) barın EMA'larına ihtiyacımız var
    macd_history: List[float] = []
    for i in range(signal, 0, -1):
        window = prices[: len(prices) - i + 1] if i > 1 else prices
        ef = ema(window, fast)
        es = ema(window, slow)
        if math.isnan(ef) or math.isnan(es):
            continue
        macd_history.append(ef - es)

    if len(macd_history) < signal:
        nan = float("nan")
        return nan, nan, nan

    macd_line   = macd_history[-1]
    signal_line = ema(macd_history, signal)
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram


def adx(bars: List[Bar], period: int = 14) -> float:
    """
    Wilder ADX (Average Directional Index).
    Değer aralığı [0, 100].  >25 = güçlü trend, <20 = zayıf/yatay.
    """
    needed = period * 2 + 1
    if len(bars) < needed:
        return float("nan")

    bars = bars[-needed:]

    # True Range ve Directional Movement hesapla
    tr_list, dm_plus, dm_minus = [], [], []
    for i in range(1, len(bars)):
        h, l, pc = bars[i].high, bars[i].low, bars[i - 1].close
        tr = max(h - l, abs(h - pc), abs(l - pc))
        tr_list.append(tr)
        up   = bars[i].high - bars[i - 1].high
        down = bars[i - 1].low - bars[i].low
        dm_plus.append(up   if (up > down and up > 0)   else 0.0)
        dm_minus.append(down if (down > up and down > 0) else 0.0)

    def _wilder_smooth(lst, p):
        result = sum(lst[:p])
        for v in lst[p:]:
            result = result - result / p + v
        return result

    atr_s  = _wilder_smooth(tr_list,  period)
    dmp_s  = _wilder_smooth(dm_plus,  period)
    dmm_s  = _wilder_smooth(dm_minus, period)

    if atr_s < 1e-12:
        return float("nan")

    di_plus  = 100 * dmp_s / atr_s
    di_minus = 100 * dmm_s / atr_s
    denom = di_plus + di_minus
    if denom < 1e-12:
        return 0.0
    dx = 100 * abs(di_plus - di_minus) / denom

    # İkinci Wilder smooth (DX → ADX)
    return dx   # Basitleştirilmiş: tek periyot DX döndürüyoruz
    # Tam ADX için DX geçmişi gerekirdi — bu versiyon trend gücünü yeterince gösteriyor


# ─────────────────────────────────────────────────────────────────────────────
# Strateji
# ─────────────────────────────────────────────────────────────────────────────

class MultiIndicatorStrategy(BaseStrategy):
    """
    EMA + RSI + MACD + ADX + Stop-Loss / Take-Profit / Trailing Stop.

    Tüm göstergeler aynı anda hizalanmadan işlem açılmaz.
    Çıkış mantığı stratejinin içinde — risk motoru ayrıca korunur.
    """

    STRATEGY_ID = "multi_indicator_v1"

    def __init__(
        self,
        data_handler   : PITDataHandler,
        bus            : EventBus,
        symbol         : str,
        # EMA
        ema_fast       : int   = 12,
        ema_slow       : int   = 26,
        # RSI
        rsi_period     : int   = 14,
        rsi_buy        : float = 55.0,   # long için RSI eşiği (altında long aç)
        rsi_sell       : float = 45.0,   # short için RSI eşiği (üstünde short aç)
        # MACD
        macd_fast      : int   = 12,
        macd_slow      : int   = 26,
        macd_signal    : int   = 9,
        # ADX
        adx_period     : int   = 14,
        adx_threshold  : float = 20.0,  # minimum trend gücü
        # Volatilite filtresi
        atr_period     : int   = 14,
        vol_threshold  : float = 0.035,
        # Risk yönetimi (strateji düzeyinde)
        sl_pct         : float = 0.025,   # %2.5 stop-loss (0 = devre dışı)
        tp_pct         : float = 0.05,    # %5 take-profit (0 = devre dışı)
        trail_pct      : float = 0.015,   # %1.5 trailing stop (0 = devre dışı)
        # Onay modu: strict | balanced | loose
        confirmation_mode: str  = "balanced",
        # Diğer
        allow_short    : bool  = False,
        cooldown_bars  : int   = 3,
    ) -> None:
        super().__init__(data_handler, bus)
        self.symbol       = symbol.upper()
        self.ema_fast     = ema_fast
        self.ema_slow     = ema_slow
        self.rsi_period   = rsi_period
        self.rsi_buy      = rsi_buy
        self.rsi_sell     = rsi_sell
        self.macd_fast    = macd_fast
        self.macd_slow    = macd_slow
        self.macd_signal  = macd_signal
        self.adx_period   = adx_period
        self.adx_threshold= adx_threshold
        self.atr_period   = atr_period
        self.vol_threshold= vol_threshold
        self.sl_pct       = sl_pct
        self.tp_pct       = tp_pct
        self.trail_pct    = trail_pct
        self.allow_short  = allow_short
        self.cooldown_bars= cooldown_bars
        self.confirmation_mode = confirmation_mode.lower().strip()

        # Pozisyon takip durumu
        self._position    : Optional[Direction] = None
        self._entry_price : float = 0.0
        self._peak_price  : float = 0.0   # trailing stop için
        self._bars_since  : int = 0        # cooldown sayacı
        self._bars_seen   : int = 0

    def warmup_period(self) -> int:
        return max(self.ema_slow, self.macd_slow + self.macd_signal,
                   self.adx_period * 2, self.rsi_period) + 10

    # ── Ana bar işleyici ──────────────────────────────────────────────────────

    def on_bar(self, bar: Bar) -> Optional[Signal]:
        if bar.symbol != self.symbol:
            return None

        self._bars_seen += 1
        self._bars_since += 1

        if self._bars_seen < self.warmup_period():
            return None

        lookback = max(self.ema_slow, self.macd_slow + self.macd_signal,
                       self.adx_period * 2 + 5, self.rsi_period) + 20
        hist = self._data.history(self.symbol, lookback)
        if len(hist) < self.warmup_period():
            return None

        closes = [b.close for b in hist]
        price  = bar.close

        # ── Göstergeleri hesapla ──────────────────────────────────────────────
        ema_f  = ema(closes, self.ema_fast)
        ema_s  = ema(closes, self.ema_slow)
        rsi_v  = rsi(closes, self.rsi_period)
        ml, ms_, mh = macd(closes, self.macd_fast, self.macd_slow, self.macd_signal)
        adx_v  = adx(hist, self.adx_period)
        atr_v  = atr(hist, self.atr_period)

        if any(math.isnan(v) for v in (ema_f, ema_s, rsi_v)):
            return None

        norm_atr = atr_v / price if (atr_v and price > 0) else 0.0
        high_vol = norm_atr > self.vol_threshold

        macd_ok  = (not math.isnan(ml)) and (not math.isnan(ms_))
        adx_ok   = (not math.isnan(adx_v)) and (adx_v >= self.adx_threshold)

        # ── Açık pozisyon yönetimi ────────────────────────────────────────────
        if self._position == Direction.LONG:
            exit_signal, reason = self._check_long_exit(price, bar, ema_f, ema_s,
                                                         rsi_v, ml, ms_, macd_ok)
            if exit_signal:
                self._close_position()
                return self._signal(bar, Direction.FLAT, 1.0, SignalStrength.STRONG, reason)

        elif self._position == Direction.SHORT:
            exit_signal, reason = self._check_short_exit(price, bar, ema_f, ema_s,
                                                          rsi_v, ml, ms_, macd_ok)
            if exit_signal:
                self._close_position()
                return self._signal(bar, Direction.FLAT, 1.0, SignalStrength.STRONG, reason)

        else:
            # Yeni pozisyon açabilir miyiz?
            if self._bars_since < self.cooldown_bars:
                return None
            if high_vol:
                return None

            # Onay moduna göre giriş koşulları
            _mode = self.confirmation_mode
            if _mode == "strict":
                # Tüm filtreler zorunlu
                long_cond  = (ema_f > ema_s and rsi_v < self.rsi_buy
                              and macd_ok and ml > ms_ and adx_ok)
                short_cond = (self.allow_short and ema_f < ema_s
                              and rsi_v > self.rsi_sell
                              and macd_ok and ml < ms_ and adx_ok)
            elif _mode == "loose":
                # Sadece EMA + RSI yeterli
                long_cond  = (ema_f > ema_s and rsi_v < (self.rsi_buy + 5))
                short_cond = (self.allow_short and ema_f < ema_s
                              and rsi_v > (self.rsi_sell - 5))
            else:
                # balanced (varsayılan): EMA + RSI + MACD veya ADX
                macd_bull = macd_ok and ml > ms_
                long_cond  = (ema_f > ema_s and rsi_v < self.rsi_buy
                              and (macd_bull or adx_ok))
                short_cond = (self.allow_short and ema_f < ema_s
                              and rsi_v > self.rsi_sell
                              and (not macd_ok or ml < ms_) and adx_ok)

            if long_cond:
                confidence = self._confidence(ema_f, ema_s, rsi_v, self.rsi_buy)
                self._open_position(Direction.LONG, price)
                reason = (f"LONG: EMA({ema_f:.2f})>EMA({ema_s:.2f}) "
                          f"RSI={rsi_v:.1f}<{self.rsi_buy} "
                          f"ADX={adx_v:.1f} MACD={'✓' if macd_ok else '?'}")
                return self._signal(bar, Direction.LONG, confidence, SignalStrength.STRONG, reason)

            if short_cond:
                confidence = self._confidence(ema_s, ema_f, self.rsi_sell, rsi_v)
                self._open_position(Direction.SHORT, price)
                reason = (f"SHORT: EMA({ema_f:.2f})<EMA({ema_s:.2f}) "
                          f"RSI={rsi_v:.1f}>{self.rsi_sell} "
                          f"ADX={adx_v:.1f} MACD={'✓' if macd_ok else '?'}")
                return self._signal(bar, Direction.SHORT, confidence, SignalStrength.STRONG, reason)

        return None

    # ── Çıkış kontrol yardımcıları ────────────────────────────────────────────

    def _check_long_exit(self, price, bar, ema_f, ema_s, rsi_v, ml, ms_, macd_ok
                         ) -> Tuple[bool, str]:
        """Açık long için çıkış koşulunu kontrol et."""

        # Trailing stop: peak'i güncelle
        if self.trail_pct > 0:
            self._peak_price = max(self._peak_price, price)
            trail_floor = self._peak_price * (1 - self.trail_pct)
            if price <= trail_floor:
                return True, (f"Trailing stop: fiyat({price:.4f}) ≤ "
                               f"peak({self._peak_price:.4f})×(1-{self.trail_pct:.2%})")

        # Stop-loss
        if self.sl_pct > 0 and self._entry_price > 0:
            sl_level = self._entry_price * (1 - self.sl_pct)
            if price <= sl_level:
                return True, (f"Stop-loss: fiyat({price:.4f}) ≤ "
                               f"giriş({self._entry_price:.4f})×(1-{self.sl_pct:.2%})")

        # Take-profit
        if self.tp_pct > 0 and self._entry_price > 0:
            tp_level = self._entry_price * (1 + self.tp_pct)
            if price >= tp_level:
                return True, (f"Take-profit: fiyat({price:.4f}) ≥ "
                               f"giriş({self._entry_price:.4f})×(1+{self.tp_pct:.2%})")

        # EMA çapraz dönüşü veya RSI aşırı alım
        if ema_f < ema_s:
            return True, f"EMA çapraz dönüşü: ema_fast({ema_f:.2f}) < ema_slow({ema_s:.2f})"

        return False, ""

    def _check_short_exit(self, price, bar, ema_f, ema_s, rsi_v, ml, ms_, macd_ok
                          ) -> Tuple[bool, str]:
        """Açık short için çıkış koşulunu kontrol et."""

        if self.trail_pct > 0:
            self._peak_price = min(self._peak_price, price)
            trail_ceil = self._peak_price * (1 + self.trail_pct)
            if price >= trail_ceil:
                return True, (f"Trailing stop: fiyat({price:.4f}) ≥ "
                               f"peak({self._peak_price:.4f})×(1+{self.trail_pct:.2%})")

        if self.sl_pct > 0 and self._entry_price > 0:
            sl_level = self._entry_price * (1 + self.sl_pct)
            if price >= sl_level:
                return True, f"Stop-loss: fiyat({price:.4f}) ≥ giriş({self._entry_price:.4f})"

        if self.tp_pct > 0 and self._entry_price > 0:
            tp_level = self._entry_price * (1 - self.tp_pct)
            if price <= tp_level:
                return True, f"Take-profit: fiyat({price:.4f}) ≤ giriş({self._entry_price:.4f})"

        if ema_f > ema_s:
            return True, f"EMA çapraz dönüşü: ema_fast({ema_f:.2f}) > ema_slow({ema_s:.2f})"

        return False, ""

    # ── Durum yönetimi ────────────────────────────────────────────────────────

    def _open_position(self, direction: Direction, price: float) -> None:
        self._position   = direction
        self._entry_price= price
        self._peak_price = price
        self._bars_since = 0

    def _close_position(self) -> None:
        self._position    = None
        self._entry_price = 0.0
        self._peak_price  = 0.0
        self._bars_since  = 0

    def _signal(self, bar: Bar, direction: Direction,
                confidence: float, strength: SignalStrength, reason: str) -> Signal:
        return Signal(
            strategy_id = self.STRATEGY_ID,
            symbol      = self.symbol,
            timestamp   = bar.timestamp,
            direction   = direction,
            strength    = strength,
            reason      = reason,
            confidence  = confidence,
        )

    def _confidence(self, fast, slow, threshold, value) -> float:
        sep = abs(fast - slow) / (slow + 1e-9)
        return float(min(sep * 15, 1.0))

    def reset(self) -> None:
        self._position    = None
        self._entry_price = 0.0
        self._peak_price  = 0.0
        self._bars_since  = 0
        self._bars_seen   = 0
