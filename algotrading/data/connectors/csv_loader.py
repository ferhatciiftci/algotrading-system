"""
data/connectors/csv_loader.py
──────────────────────────────
Load OHLCV data from a local CSV or TSV file.

Supported column names (case-insensitive):
    timestamp / date / datetime / time
    open / o
    high / h
    low  / l
    close / c
    volume / vol / v
    adj_close / adjusted_close / adj close (optional)
    vwap (optional)

Example CSV:
    date,open,high,low,close,volume
    2020-01-02,3244.67,3258.39,3235.53,3257.85,3300000

Usage:
    loader = CSVLoader()
    df = loader.load("data/SPY.csv", symbol="SPY")
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)
UTC = timezone.utc

# Candidate column name mappings (source → canonical)
_TIMESTAMP_COLS = {"timestamp", "date", "datetime", "time", "Date", "Datetime", "Time"}
_OPEN_COLS      = {"open", "Open", "o", "O"}
_HIGH_COLS      = {"high", "High", "h", "H"}
_LOW_COLS       = {"low",  "Low",  "l", "L"}
_CLOSE_COLS     = {"close", "Close", "c", "C"}
_VOLUME_COLS    = {"volume", "Volume", "vol", "Vol", "v", "V"}
_ADJ_CLOSE_COLS = {"adj_close", "adj close", "Adj Close", "adjusted_close", "AdjClose"}
_VWAP_COLS      = {"vwap", "VWAP", "Vwap"}


class CSVLoader:
    """
    Load OHLCV data from a local CSV/TSV file.

    The loader is deliberately flexible — it will try to auto-detect column
    names and timestamp formats.  A clear ValueError is raised if required
    columns are missing.
    """

    SOURCE = "csv"

    def __init__(self, sep: str = ",") -> None:
        self.sep = sep

    # ── Public API ─────────────────────────────────────────────────────────────

    def load(
        self,
        path  : str | Path,
        symbol: str,
    ) -> pd.DataFrame:
        """
        Load and normalise a CSV file.

        Parameters
        ----------
        path   : path to the CSV file
        symbol : canonical symbol to assign to all rows

        Returns
        -------
        pd.DataFrame with columns:
            symbol, timestamp (UTC-aware), open, high, low, close, volume,
            adj_close, vwap
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {path}")

        raw = pd.read_csv(path, sep=self.sep)
        raw.columns = [str(c).strip() for c in raw.columns]

        df = self._normalise(raw, symbol)
        logger.info(
            "[csv] Loaded %d bars for %s from %s",
            len(df), symbol, path.name
        )
        return df

    def load_and_store(
        self,
        path     : str | Path,
        symbol   : str,
        raw_store,
    ) -> str:
        """Load CSV and persist to raw_store. Returns SHA-256 hash."""
        df = self.load(path, symbol)
        return raw_store.write(df, symbol, source=self.SOURCE)

    # ── Internals ──────────────────────────────────────────────────────────────

    def _normalise(self, raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
        col_map = self._detect_columns(raw.columns.tolist())

        df = pd.DataFrame()
        df["timestamp"] = pd.to_datetime(raw[col_map["timestamp"]], utc=True)
        df["open"]      = raw[col_map["open"]].astype(float)
        df["high"]      = raw[col_map["high"]].astype(float)
        df["low"]       = raw[col_map["low"]].astype(float)
        df["close"]     = raw[col_map["close"]].astype(float)
        df["volume"]    = raw[col_map["volume"]].astype(float)
        df["adj_close"] = raw[col_map["adj_close"]].astype(float) if col_map.get("adj_close") else float("nan")
        df["vwap"]      = raw[col_map["vwap"]].astype(float)     if col_map.get("vwap")      else float("nan")
        df["symbol"]    = symbol.upper()

        return df[["symbol", "timestamp", "open", "high", "low", "close",
                   "volume", "adj_close", "vwap"]].reset_index(drop=True)

    @staticmethod
    def _detect_columns(columns: list[str]) -> dict:
        """Map canonical names to actual CSV column names."""
        col_set = set(columns)

        def find(candidates: set) -> str | None:
            matched = col_set & candidates
            # Also try case-insensitive
            if not matched:
                lower_map = {c.lower(): c for c in col_set}
                for cand in candidates:
                    if cand.lower() in lower_map:
                        return lower_map[cand.lower()]
                return None
            return next(iter(matched))

        result: dict[str, str | None] = {}
        result["timestamp"] = find(_TIMESTAMP_COLS)
        result["open"]      = find(_OPEN_COLS)
        result["high"]      = find(_HIGH_COLS)
        result["low"]       = find(_LOW_COLS)
        result["close"]     = find(_CLOSE_COLS)
        result["volume"]    = find(_VOLUME_COLS)
        result["adj_close"] = find(_ADJ_CLOSE_COLS)   # optional
        result["vwap"]      = find(_VWAP_COLS)         # optional

        required = ["timestamp", "open", "high", "low", "close", "volume"]
        missing  = [k for k in required if result.get(k) is None]
        if missing:
            raise ValueError(
                f"CSV is missing required columns: {missing}. "
                f"Found columns: {columns}"
            )
        return result
