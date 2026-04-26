"""
data/corporate_actions.py
--------------------------
Applies corporate actions (splits, dividends) to a price series.

Point-in-time principle: the adjustment factor for a given bar is derived
only from corporate actions with ex_date <= bar timestamp.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    _HAS_PARQUET = True
except ImportError:
    pa = None; pq = None; _HAS_PARQUET = False  # type: ignore

from algotrading.data.schema import CorporateAction, SymbolMapping

logger = logging.getLogger(__name__)


class CorporateActionStore:
    """Stores and applies corporate actions."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._ca_dir = self.root / "corporate_actions"
        self._ca_dir.mkdir(parents=True, exist_ok=True)
        self._symbol_map_dir = self.root / "symbol_maps"
        self._symbol_map_dir.mkdir(parents=True, exist_ok=True)

    def save_action(self, action: CorporateAction) -> None:
        path = self._ca_dir / f"{action.symbol.upper()}.parquet"
        row = pd.DataFrame([action.model_dump()])
        row["ex_date"] = pd.to_datetime(row["ex_date"], utc=True)
        if path.exists():
            existing = pd.read_parquet(path)
            row = pd.concat([existing, row]).drop_duplicates(
                subset=["symbol", "ex_date", "action_type"]
            )
        row.to_parquet(path, index=False)

    def load_actions(self, symbol: str) -> List[CorporateAction]:
        path = self._ca_dir / f"{symbol.upper()}.parquet"
        if not path.exists():
            return []
        df = pd.read_parquet(path)
        return [CorporateAction(**row) for _, row in df.iterrows()]

    def compute_adjustment_factors(
        self,
        symbol: str,
        timestamps: pd.Series,
    ) -> pd.Series:
        """
        Returns cumulative price-adjustment factors (PIT-safe).
        Only actions whose ex_date is BEFORE the bar's timestamp are applied.
        """
        actions = self.load_actions(symbol)
        if not actions:
            return pd.Series(1.0, index=timestamps.index)

        split_actions = [a for a in actions if a.action_type == "split"]
        if not split_actions:
            return pd.Series(1.0, index=timestamps.index)

        action_df = pd.DataFrame(
            [{"ex_date": a.ex_date, "factor": 1.0 / a.factor} for a in split_actions]
        )
        action_df["ex_date"] = pd.to_datetime(action_df["ex_date"], utc=True)
        action_df = action_df.sort_values("ex_date")

        result = pd.Series(1.0, index=timestamps.index)
        for _, row in action_df.iterrows():
            mask = timestamps < row["ex_date"]
            result[mask] *= row["factor"]

        return result

    def adjust_prices(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Apply split adjustments. Preserves original OHLC; adds adj_close."""
        df = df.copy()
        factors = self.compute_adjustment_factors(symbol, df["timestamp"])
        df["adj_factor"] = factors.values
        df["adj_close"]  = df["close"] * factors.values
        for col in ("open", "high", "low"):
            df[f"adj_{col}"] = df[col] * factors.values
        return df

    def save_symbol_mapping(self, mapping: SymbolMapping) -> None:
        path = self._symbol_map_dir / "mappings.parquet"
        row  = pd.DataFrame([mapping.model_dump()])
        row["valid_from"] = pd.to_datetime(row["valid_from"], utc=True)
        row["valid_to"]   = pd.to_datetime(row["valid_to"],   utc=True)
        if path.exists():
            existing = pd.read_parquet(path)
            row = pd.concat([existing, row]).drop_duplicates(
                subset=["vendor_symbol", "valid_from"]
            )
        row.to_parquet(path, index=False)

    def resolve_symbol(self, vendor_symbol: str, at: datetime) -> Optional[str]:
        """Resolve vendor symbol to canonical at a point in time."""
        path = self._symbol_map_dir / "mappings.parquet"
        if not path.exists():
            return vendor_symbol

        df = pd.read_parquet(path)
        df = df[df["vendor_symbol"].str.upper() == vendor_symbol.upper()]
        at_ts = pd.Timestamp(at, tz="UTC")
        df = df[df["valid_from"] <= at_ts]
        df = df[df["valid_to"].isna() | (df["valid_to"] >= at_ts)]
        if df.empty:
            logger.warning("No canonical mapping for %s at %s", vendor_symbol, at)
            return None
        return df.sort_values("valid_from").iloc[-1]["canonical_symbol"]
