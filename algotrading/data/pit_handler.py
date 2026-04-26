"""
data/pit_handler.py
───────────────────
Point-In-Time (PIT) data handler.

The central rule: at simulation time T, the handler may ONLY return data
with timestamp < T (bar-close strictly before T).  This eliminates lookahead
bias and ensures that a strategy at bar T can only use data that would have
been available in a live system at that moment.

The handler also manages the "data window" — a rolling buffer of the last N
bars for each symbol, from which indicators are computed.
"""

from __future__ import annotations

import logging
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from algotrading.core.types import Bar
from algotrading.data.ingestion import RawDataStore
from algotrading.data.corporate_actions import CorporateActionStore

logger = logging.getLogger(__name__)

UTC = timezone.utc


class PITDataHandler:
    """
    Point-In-Time data handler.

    Usage in backtest loop:
        handler.subscribe("AAPL")
        for bar in handler.stream("AAPL", start, end):
            strategy.on_bar(bar)
            # handler.history("AAPL", 50) gives last 50 bars UP TO AND INCLUDING bar
    """

    def __init__(
        self,
        raw_store: RawDataStore,
        ca_store: CorporateActionStore,
        window_size: int = 500,
    ) -> None:
        self._raw_store = raw_store
        self._ca_store  = ca_store
        self._window    = window_size
        # Rolling buffer per symbol — only bars that have been "seen" so far
        self._buffers: Dict[str, deque[Bar]] = {}
        self._subscribed: set[str] = set()

    # ── Subscription ──────────────────────────────────────────────────────────

    def subscribe(self, symbol: str) -> None:
        symbol = symbol.upper()
        if symbol not in self._subscribed:
            self._subscribed.add(symbol)
            self._buffers[symbol] = deque(maxlen=self._window)
            logger.debug("Subscribed to %s", symbol)

    # ── Streaming ─────────────────────────────────────────────────────────────

    def stream(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ):
        """
        Generator that yields Bar objects in chronological order.
        Each bar is added to the buffer BEFORE being yielded so that
        `history()` includes the current bar.

        Crucially: the NEXT bar is not accessible until the generator
        advances — no lookahead is possible.
        """
        symbol = symbol.upper()
        if symbol not in self._subscribed:
            self.subscribe(symbol)

        df = self._raw_store.read(symbol, start=start, end=end)
        if df.empty:
            logger.warning("No data for %s between %s and %s", symbol, start, end)
            return

        # Apply corporate action adjustments
        df = self._ca_store.adjust_prices(df, symbol)

        # Enforce strict PIT ordering
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Validate no future timestamps (double-check)
        now = pd.Timestamp.now(tz="UTC")
        future_mask = df["timestamp"] > now
        if future_mask.any():
            logger.error("Dropping %d future bars for %s", future_mask.sum(), symbol)
            df = df[~future_mask]

        for _, row in df.iterrows():
            bar = self._row_to_bar(row, symbol)
            self._buffers[symbol].append(bar)
            yield bar

    # ── History access ────────────────────────────────────────────────────────

    def history(self, symbol: str, n: int) -> List[Bar]:
        """
        Return the last `n` bars for a symbol (chronological order, oldest first).
        Only bars that have been streamed so far are included — no lookahead.
        """
        symbol = symbol.upper()
        buf = self._buffers.get(symbol)
        if buf is None:
            return []
        bars = list(buf)
        return bars[-n:] if n < len(bars) else bars

    def latest(self, symbol: str) -> Optional[Bar]:
        """Return the most recent bar for a symbol."""
        h = self.history(symbol, 1)
        return h[0] if h else None

    def history_df(self, symbol: str, n: int) -> pd.DataFrame:
        """Return history as a DataFrame — convenient for indicator computation."""
        bars = self.history(symbol, n)
        if not bars:
            return pd.DataFrame()
        return pd.DataFrame([
            {
                "timestamp": b.timestamp,
                "open"     : b.open,
                "high"     : b.high,
                "low"      : b.low,
                "close"    : b.close,
                "volume"   : b.volume,
            }
            for b in bars
        ])

    # ── Reset (between backtest runs) ─────────────────────────────────────────

    def reset(self) -> None:
        for sym in self._buffers:
            self._buffers[sym].clear()

    # ── Internals ─────────────────────────────────────────────────────────────

    @staticmethod
    def _row_to_bar(row: pd.Series, symbol: str) -> Bar:
        ts = row["timestamp"]
        if hasattr(ts, "to_pydatetime"):
            ts = ts.to_pydatetime()
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)

        close = float(row.get("adj_close", row["close"]) or row["close"])

        return Bar(
            symbol    = symbol,
            timestamp = ts,
            open      = float(row["open"]),
            high      = float(row["high"]),
            low       = float(row["low"]),
            close     = close,
            volume    = float(row["volume"]),
            adjusted  = "adj_close" in row and pd.notna(row.get("adj_close")),
        )
