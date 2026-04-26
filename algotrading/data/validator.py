"""
data/validator.py
─────────────────
Data quality checks run at ingestion and before any strategy consumption.

Checks performed:
  1. Timestamp ordering (strictly monotonic)
  2. No future timestamps
  3. OHLC consistency (high >= low, open/close in [low, high])
  4. No negative prices or volumes
  5. Suspicious price spikes (> configurable threshold)
  6. Missing value counts
  7. Gap detection (missing sessions)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

UTC = timezone.utc

# ─── Result container ─────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    passed   : bool
    errors   : List[str] = field(default_factory=list)
    warnings : List[str] = field(default_factory=list)
    stats    : dict       = field(default_factory=dict)

    def fail(self, msg: str) -> None:
        self.passed = False
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)


# ─── Validator ────────────────────────────────────────────────────────────────

class DataValidator:

    # Price spike: bar-over-bar return exceeding this multiple is suspicious
    MAX_BAR_RETURN: float = 0.25        # 25%
    # Max allowed fraction of NaN values in any column
    MAX_NAN_FRACTION: float = 0.02
    # Max gap between consecutive bars (in minutes, for daily data: 1440*2)
    MAX_GAP_MINUTES: int = 1440 * 4    # 4 trading days

    def validate_raw(self, df: pd.DataFrame) -> ValidationResult:
        result = ValidationResult(passed=True)

        if df.empty:
            result.fail("DataFrame is empty")
            return result

        self._check_required_columns(df, result)
        if not result.passed:
            return result

        self._check_timestamps(df, result)
        self._check_no_future_timestamps(df, result)
        self._check_ohlc(df, result)
        self._check_no_negatives(df, result)
        self._check_nans(df, result)
        self._check_price_spikes(df, result)
        self._check_gaps(df, result)

        result.stats = {
            "rows"         : len(df),
            "symbols"      : df["symbol"].nunique() if "symbol" in df.columns else 1,
            "date_range"   : (str(df["timestamp"].min()), str(df["timestamp"].max())),
            "nan_counts"   : df.isnull().sum().to_dict(),
        }
        return result

    # ── Individual checks ─────────────────────────────────────────────────────

    def _check_required_columns(self, df: pd.DataFrame, r: ValidationResult) -> None:
        required = {"symbol", "timestamp", "open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            r.fail(f"Missing required columns: {missing}")

    def _check_timestamps(self, df: pd.DataFrame, r: ValidationResult) -> None:
        ts = df["timestamp"]
        if not ts.is_monotonic_increasing:
            n_violations = (ts.diff().dropna() < pd.Timedelta(0)).sum()
            r.fail(f"Timestamps not monotonically increasing ({n_violations} violations)")

        dupes = df.duplicated(subset=["symbol", "timestamp"]).sum()
        if dupes > 0:
            r.fail(f"Found {dupes} duplicate (symbol, timestamp) rows")

    def _check_no_future_timestamps(self, df: pd.DataFrame, r: ValidationResult) -> None:
        now = pd.Timestamp.now(tz="UTC")
        future = (df["timestamp"] > now).sum()
        if future > 0:
            r.fail(f"{future} bars have future timestamps — possible lookahead bias")

    def _check_ohlc(self, df: pd.DataFrame, r: ValidationResult) -> None:
        bad_hl = (df["high"] < df["low"]).sum()
        bad_o  = ((df["open"] < df["low"]) | (df["open"] > df["high"])).sum()
        bad_c  = ((df["close"] < df["low"]) | (df["close"] > df["high"])).sum()
        if bad_hl:
            r.fail(f"high < low in {bad_hl} rows")
        if bad_o:
            r.warn(f"open outside [low, high] in {bad_o} rows")
        if bad_c:
            r.warn(f"close outside [low, high] in {bad_c} rows")

    def _check_no_negatives(self, df: pd.DataFrame, r: ValidationResult) -> None:
        for col in ("open", "high", "low", "close", "volume"):
            if col in df.columns:
                n = (df[col] < 0).sum()
                if n:
                    r.fail(f"Negative values in '{col}': {n} rows")

    def _check_nans(self, df: pd.DataFrame, r: ValidationResult) -> None:
        for col in ("open", "high", "low", "close"):
            if col in df.columns:
                frac = df[col].isna().mean()
                if frac > self.MAX_NAN_FRACTION:
                    r.fail(f"Column '{col}' has {frac:.1%} NaN values (limit {self.MAX_NAN_FRACTION:.1%})")
                elif frac > 0:
                    r.warn(f"Column '{col}' has {df[col].isna().sum()} NaN values")

    def _check_price_spikes(self, df: pd.DataFrame, r: ValidationResult) -> None:
        returns = df["close"].pct_change().abs()
        spikes  = (returns > self.MAX_BAR_RETURN).sum()
        if spikes > 0:
            r.warn(
                f"{spikes} bars with |return| > {self.MAX_BAR_RETURN:.0%} "
                f"— check for data errors or corporate actions"
            )

    def _check_gaps(self, df: pd.DataFrame, r: ValidationResult) -> None:
        deltas = df["timestamp"].diff().dropna()
        max_gap = deltas.max()
        if pd.notna(max_gap) and max_gap > pd.Timedelta(minutes=self.MAX_GAP_MINUTES):
            r.warn(
                f"Largest gap between bars: {max_gap} "
                f"(threshold: {self.MAX_GAP_MINUTES} min)"
            )
