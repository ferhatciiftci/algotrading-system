"""
data/ingestion.py
-----------------
Writes raw market data to an immutable Parquet store.

Design principles:
- Raw data is NEVER overwritten -- append-only per symbol/date partition.
- Each write is logged with a hash for reproducibility.
- Validation runs before any data is persisted.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# Parquet support -- requires pyarrow or fastparquet.
# Imported lazily so the module loads even when pyarrow is absent.
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    _HAS_PARQUET = True
    _RAW_SCHEMA = pa.schema([
        pa.field("symbol",    pa.string()),
        pa.field("timestamp", pa.timestamp("us", tz="UTC")),
        pa.field("open",      pa.float64()),
        pa.field("high",      pa.float64()),
        pa.field("low",       pa.float64()),
        pa.field("close",     pa.float64()),
        pa.field("volume",    pa.float64()),
        pa.field("adj_close", pa.float64()),
        pa.field("vwap",      pa.float64()),
    ])
    _RAW_COLS = list(_RAW_SCHEMA.names)
except ImportError:
    pa = None          # type: ignore
    pq = None          # type: ignore
    _HAS_PARQUET = False
    _RAW_SCHEMA  = None
    _RAW_COLS    = ["symbol","timestamp","open","high","low","close","volume","adj_close","vwap"]

from algotrading.data.schema import RawBar
from algotrading.data.validator import DataValidator, ValidationResult

logger = logging.getLogger(__name__)
UTC = timezone.utc


class RawDataStore:
    """
    Append-only Parquet store for raw OHLCV data.

    Directory layout:
        <root>/raw/<symbol>/year=YYYY/month=MM/<hash>.parquet
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.raw_dir = self.root / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self._validator = DataValidator()

    # ---- Public API ----------------------------------------------------------

    def write(self, df: pd.DataFrame, symbol: str, source: str = "unknown") -> str:
        """
        Validate and persist a DataFrame of raw OHLCV bars.
        Returns the content hash of the written file.
        Raises ValueError if validation fails.
        """
        if not _HAS_PARQUET:
            raise RuntimeError(
                "pyarrow is required for Parquet I/O. "
                "Install it with: pip install pyarrow"
            )
        df = self._normalise(df, symbol)
        result: ValidationResult = self._validator.validate_raw(df)
        if not result.passed:
            raise ValueError(
                f"Raw data validation failed for {symbol}: {result.errors}"
            )
        for w in result.warnings:
            logger.warning("[%s] %s", symbol, w)

        path = self._partition_path(symbol, df)
        if path.exists():
            logger.info("Skipping write -- partition already exists: %s", path)
            return RawDataStore._file_hash(path)

        path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(df, schema=_RAW_SCHEMA, preserve_index=False)
        pq.write_table(table, path, compression="snappy")
        content_hash = RawDataStore._file_hash(path)
        logger.info(
            "Wrote %d rows for %s -> %s (hash=%s)",
            len(df), symbol, path, content_hash[:8],
        )
        return content_hash

    def read(
        self,
        symbol: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        """
        Read raw bars for a symbol, optionally filtered by UTC timestamp range.
        Returns an empty DataFrame if no data exists.
        """
        if not _HAS_PARQUET:
            raise RuntimeError("pyarrow is required for Parquet I/O.")

        sym_dir = self.raw_dir / symbol.upper()
        if not sym_dir.exists():
            logger.warning("No raw data found for %s", symbol)
            return pd.DataFrame()

        dfs = []
        for parquet_file in sorted(sym_dir.rglob("*.parquet")):
            dfs.append(pd.read_parquet(parquet_file))

        if not dfs:
            return pd.DataFrame()

        df = pd.concat(dfs, ignore_index=True)
        df = df.sort_values("timestamp").drop_duplicates(subset=["symbol", "timestamp"])

        if start:
            ts_start = pd.Timestamp(start)
            if ts_start.tzinfo is None:
                ts_start = ts_start.tz_localize("UTC")
            df = df[df["timestamp"] >= ts_start]
        if end:
            ts_end = pd.Timestamp(end)
            if ts_end.tzinfo is None:
                ts_end = ts_end.tz_localize("UTC")
            df = df[df["timestamp"] <= ts_end]

        return df.reset_index(drop=True)

    # ---- Internals -----------------------------------------------------------

    def _normalise(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        df = df.copy()
        df.columns = [c.lower().strip() for c in df.columns]
        df["symbol"] = symbol.upper()

        if "timestamp" not in df.columns:
            raise ValueError("DataFrame must have a 'timestamp' column")
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        for col in ("adj_close", "vwap"):
            if col not in df.columns:
                df[col] = float("nan")

        return df[_RAW_COLS]

    def _partition_path(self, symbol: str, df: pd.DataFrame) -> Path:
        """Derive a deterministic partition path from symbol and date range."""
        first_ts = df["timestamp"].min()
        year  = first_ts.year
        month = first_ts.month
        key = f"{symbol}-{first_ts.date()}-{df['timestamp'].max().date()}"
        h   = hashlib.md5(key.encode()).hexdigest()[:12]
        return (
            self.raw_dir
            / symbol.upper()
            / f"year={year}"
            / f"month={month:02d}"
            / f"{h}.parquet"
        )

    @staticmethod
    def _file_hash(path: Path) -> str:
        """SHA-256 hash of the file at path."""
        with open(path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()
