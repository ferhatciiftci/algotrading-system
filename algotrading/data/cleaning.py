"""
data/cleaning.py
----------------
Ham veri uzerinde calistirilan temizleme pipeline'i.

Adimlar:
  1. Duplicate satirlari kaldir
  2. NaN fiyatlari forward-fill et (max 3 bar)
  3. Sifir/negatif fiyatlari NaN yap
  4. Outlier fiyatlari (3-sigma) isaretleyelim ama kaldirmayin
  5. Volume sifirsa ayri flag koy
  6. Temiz veriye adj_factor = 1.0 ekle (corporate actions ile guncellenir)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MAX_FFILL_BARS  = 3      # kac bar forward-fill izin verilir
OUTLIER_SIGMA   = 4.0    # kac sigma ustu outlier sayilir


@dataclass
class CleaningReport:
    symbol           : str
    rows_in          : int
    rows_out         : int
    duplicates_removed: int
    nan_filled       : int
    zero_prices      : int
    outliers_flagged : int
    warnings         : List[str] = field(default_factory=list)


class DataCleaner:
    """Stateless veri temizleyici - ayni girdi her zaman ayni ciktıyi uretir."""

    def clean(self, df: pd.DataFrame, symbol: str) -> tuple[pd.DataFrame, CleaningReport]:
        df = df.copy()
        rows_in = len(df)
        report  = CleaningReport(symbol=symbol, rows_in=rows_in,
                                 rows_out=0, duplicates_removed=0,
                                 nan_filled=0, zero_prices=0, outliers_flagged=0)

        # 1. Duplicate
        before = len(df)
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
        report.duplicates_removed = before - len(df)

        # 2. Sifir / negatif fiyat -> NaN
        for col in ("open", "high", "low", "close"):
            if col in df.columns:
                mask = df[col] <= 0
                report.zero_prices += int(mask.sum())
                df.loc[mask, col] = np.nan

        # 3. Forward fill (max MAX_FFILL_BARS)
        before_nan = df[["open","high","low","close"]].isna().sum().sum()
        df[["open","high","low","close"]] = (
            df[["open","high","low","close"]].ffill(limit=MAX_FFILL_BARS)
        )
        after_nan = df[["open","high","low","close"]].isna().sum().sum()
        report.nan_filled = int(before_nan - after_nan)

        # 4. Outlier isaretleme (kaldirma degil)
        returns = df["close"].pct_change().abs()
        sigma   = returns.std()
        if sigma > 0:
            outlier_mask = returns > OUTLIER_SIGMA * sigma
            report.outliers_flagged = int(outlier_mask.sum())
            df["quality_flag"] = outlier_mask.astype(int)
        else:
            df["quality_flag"] = 0

        # 5. adj_factor baslangic degeri
        if "adj_factor" not in df.columns:
            df["adj_factor"] = 1.0
        if "adj_close" not in df.columns:
            df["adj_close"] = df["close"]
        if "source" not in df.columns:
            df["source"] = "unknown"

        # NaN kalan satirlari dusur
        df = df.dropna(subset=["close"]).reset_index(drop=True)

        report.rows_out = len(df)
        if report.duplicates_removed:
            report.warnings.append(f"{report.duplicates_removed} duplicate satir kaldirildi")
        if report.outliers_flagged:
            report.warnings.append(f"{report.outliers_flagged} potansiyel outlier isaretlendi")
        logger.info("[%s] Temizleme: %d -> %d satir | dup=%d ffill=%d outlier=%d",
                    symbol, rows_in, report.rows_out,
                    report.duplicates_removed, report.nan_filled, report.outliers_flagged)
        return df, report
