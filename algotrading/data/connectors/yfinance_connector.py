"""
data/connectors/yfinance_connector.py
──────────────────────────────────────
Downloads OHLCV data from Yahoo Finance via the yfinance library.

Install: pip install yfinance
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)
UTC = timezone.utc


class YFinanceConnector:
    """
    Fetches historical OHLCV data from Yahoo Finance.

    Example
    -------
    >>> conn = YFinanceConnector()
    >>> df = conn.download("SPY", "2020-01-01", "2024-01-01")
    >>> print(df.head())
    """

    SOURCE = "yfinance"

    def __init__(self, interval: str = "1d") -> None:
        """
        Parameters
        ----------
        interval : str
            Bar interval — "1d", "1h", "30m", etc.  (yfinance notation)
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
        Download OHLCV data and return a normalised DataFrame.

        Returns
        -------
        pd.DataFrame with columns:
            symbol, timestamp (UTC-aware), open, high, low, close, volume, adj_close
        Raises
        ------
        ImportError   if yfinance is not installed
        ValueError    if no data is returned for the symbol/range
        """
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError(
                "yfinance is required for this connector.\n"
                "Install it with: pip install yfinance"
            )

        ticker = yf.Ticker(symbol)
        raw = ticker.history(
            start    = start if isinstance(start, str) else start.strftime("%Y-%m-%d"),
            end      = end   if isinstance(end,   str) else end.strftime("%Y-%m-%d"),
            interval = self.interval,
            auto_adjust = False,   # keep unadjusted + Adj Close separately
        )

        if raw.empty:
            raise ValueError(f"yfinance returned no data for {symbol} ({start} → {end})")

        df = self._normalise(raw, symbol)
        logger.info(
            "[yfinance] Downloaded %d bars for %s (%s → %s)",
            len(df), symbol, df["timestamp"].min().date(), df["timestamp"].max().date()
        )
        return df

    def fetch_and_store(
        self,
        symbol   : str,
        start    : str | datetime,
        end      : str | datetime,
        raw_store,           # RawDataStore — avoid circular import
    ) -> str:
        """
        Download and immediately persist to raw_store.
        Returns the SHA-256 hash of the written Parquet file.
        """
        df = self.download(symbol, start, end)
        return raw_store.write(df, symbol, source=self.SOURCE)

    # ── Internals ──────────────────────────────────────────────────────────────

    @staticmethod
    def _normalise(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Convert yfinance DataFrame columns to our standard schema."""
        df = raw.copy()
        df.index.name = "timestamp"
        df = df.reset_index()

        rename = {
            "Open"     : "open",
            "High"     : "high",
            "Low"      : "low",
            "Close"    : "close",
            "Volume"   : "volume",
            "Adj Close": "adj_close",
        }
        df = df.rename(columns=rename)

        # Ensure UTC-aware timestamps
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        df["symbol"] = symbol.upper()
        df["vwap"]   = float("nan")   # yfinance daily bars have no VWAP

        keep = ["symbol", "timestamp", "open", "high", "low", "close", "volume", "adj_close", "vwap"]
        existing = [c for c in keep if c in df.columns]
        for c in keep:
            if c not in df.columns:
                df[c] = float("nan")
        return df[keep].reset_index(drop=True)
