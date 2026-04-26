"""
data/connectors/binance_connector.py
──────────────────────────────────────
Downloads OHLCV data from Binance public REST API.

No API key required for historical kline data.

Install: pip install requests
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)
UTC = timezone.utc

# Binance kline column names (positional)
_KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "num_trades",
    "taker_buy_base_volume", "taker_buy_quote_volume", "ignore",
]

_BINANCE_BASE = "https://api.binance.com"
_MAX_LIMIT    = 1000   # Binance max per request


class BinanceConnector:
    """
    Fetches historical OHLCV kline data from Binance (no API key needed).

    Example
    -------
    >>> conn = BinanceConnector(interval="1d")
    >>> df = conn.download("BTCUSDT", "2022-01-01", "2024-01-01")
    >>> print(df.head())
    """

    SOURCE = "binance"

    def __init__(self, interval: str = "1d") -> None:
        """
        Parameters
        ----------
        interval : str
            Binance kline interval — "1m", "5m", "1h", "4h", "1d", etc.
        """
        self.interval = interval

    # ── Public API ─────────────────────────────────────────────────────────────

    def download(
        self,
        symbol: str,
        start : str | datetime,
        end   : str | datetime,
    ) -> pd.DataFrame:
        """
        Download historical klines and return a normalised DataFrame.

        Automatically paginates to cover the full date range.

        Returns
        -------
        pd.DataFrame with columns:
            symbol, timestamp (UTC-aware), open, high, low, close, volume
        Raises
        ------
        ImportError   if requests is not installed
        RuntimeError  if Binance returns an error response
        ValueError    if no data for the given range
        """
        try:
            import requests
        except ImportError:
            raise ImportError(
                "requests is required for BinanceConnector.\n"
                "Install it with: pip install requests"
            )

        start_ms = self._to_ms(start)
        end_ms   = self._to_ms(end)

        all_klines: list[list] = []
        current_ms = start_ms

        while current_ms < end_ms:
            params = {
                "symbol"   : symbol.upper(),
                "interval" : self.interval,
                "startTime": current_ms,
                "endTime"  : end_ms,
                "limit"    : _MAX_LIMIT,
            }
            resp = requests.get(f"{_BINANCE_BASE}/api/v3/klines", params=params, timeout=30)
            if resp.status_code != 200:
                raise RuntimeError(
                    f"Binance API error {resp.status_code}: {resp.text[:200]}"
                )
            batch = resp.json()
            if not batch:
                break
            all_klines.extend(batch)
            # Advance past last returned close_time
            current_ms = batch[-1][6] + 1   # close_time + 1 ms
            if len(batch) < _MAX_LIMIT:
                break
            time.sleep(0.1)   # polite pause

        if not all_klines:
            raise ValueError(
                f"Binance returned no klines for {symbol} ({start} → {end})"
            )

        df = self._normalise(all_klines, symbol)
        logger.info(
            "[binance] Downloaded %d bars for %s (%s → %s)",
            len(df), symbol, df["timestamp"].min().date(), df["timestamp"].max().date()
        )
        return df

    def fetch_and_store(
        self,
        symbol   : str,
        start    : str | datetime,
        end      : str | datetime,
        raw_store,
    ) -> str:
        """Download and persist to raw_store. Returns SHA-256 hash."""
        df = self.download(symbol, start, end)
        return raw_store.write(df, symbol, source=self.SOURCE)

    # ── Internals ──────────────────────────────────────────────────────────────

    @staticmethod
    def _to_ms(dt: str | datetime) -> int:
        if isinstance(dt, str):
            dt = datetime.fromisoformat(dt)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return int(dt.timestamp() * 1000)

    @staticmethod
    def _normalise(klines: list, symbol: str) -> pd.DataFrame:
        df = pd.DataFrame(klines, columns=_KLINE_COLS)
        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = df[col].astype(float)
        df["symbol"]    = symbol.upper()
        df["adj_close"] = float("nan")
        df["vwap"]      = float("nan")
        keep = ["symbol", "timestamp", "open", "high", "low", "close", "volume", "adj_close", "vwap"]
        return df[keep].reset_index(drop=True)
