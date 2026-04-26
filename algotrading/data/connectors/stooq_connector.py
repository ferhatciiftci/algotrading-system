"""
data/connectors/stooq_connector.py
────────────────────────────────────
Stooq.com üzerinden ücretsiz günlük OHLCV verisi indirir.

API anahtarı gerekmez.
Desteklenen semboller:
  Hisseler (ABD) : spy.us, aapl.us, msft.us
  Endeksler      : ^spx, ^dji, ^ndx
  Döviz          : eurusd, gbpusd
  Kripto         : btc.v, eth.v

URL formatı:
  https://stooq.com/q/d/l/?s={symbol}&d1={YYYYMMDD}&d2={YYYYMMDD}&i=d
"""

from __future__ import annotations

import io
import logging
from datetime import datetime, timezone
from typing import Union

import pandas as pd

logger = logging.getLogger(__name__)
UTC = timezone.utc

_STOOQ_BASE = "https://stooq.com/q/d/l/"
_TIMEOUT_S  = 30


class StooqConnector:
    """
    Stooq.com'dan günlük OHLCV verisi indirir.

    Örnek
    -----
    >>> conn = StooqConnector()
    >>> df = conn.download("spy.us", "2020-01-01", "2024-01-01")
    >>> print(df.head())
    """

    SOURCE = "stooq"

    def download(
        self,
        symbol: str,
        start : Union[str, datetime],
        end   : Union[str, datetime],
    ) -> pd.DataFrame:
        """
        Stooq'dan günlük veri indir ve standart formata dönüştür.

        Döndürür
        --------
        pd.DataFrame with columns:
            symbol, timestamp (UTC-aware), open, high, low, close, volume
        """
        try:
            import requests
        except ImportError:
            raise ImportError("requests gerekli: pip install requests")

        d1 = self._fmt_date(start)
        d2 = self._fmt_date(end)

        params = {"s": symbol.lower(), "d1": d1, "d2": d2, "i": "d"}
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }

        resp = requests.get(_STOOQ_BASE, params=params,
                            headers=headers, timeout=_TIMEOUT_S)
        if resp.status_code != 200:
            raise RuntimeError(
                f"Stooq yanıt hatası {resp.status_code}: {resp.text[:200]}"
            )

        text = resp.text.strip()
        if not text or "No data" in text or len(text) < 30:
            raise ValueError(
                f"Stooq '{symbol}' için veri döndürmedi. "
                f"Sembol formatını kontrol edin (örn. spy.us, aapl.us)."
            )

        df = pd.read_csv(io.StringIO(text))
        if df.empty:
            raise ValueError(f"Stooq '{symbol}' için boş CSV döndürdü.")

        df = self._normalise(df, symbol)
        logger.info(
            "[stooq] %d bar indirildi: %s (%s → %s)",
            len(df), symbol,
            df["timestamp"].min().date(),
            df["timestamp"].max().date(),
        )
        return df

    def fetch_and_store(
        self,
        symbol   : str,
        start    : Union[str, datetime],
        end      : Union[str, datetime],
        raw_store,
    ) -> str:
        """İndir ve raw_store'a kaydet. SHA-256 hash döndürür."""
        df = self.download(symbol, start, end)
        return raw_store.write(df, symbol, source=self.SOURCE)

    # ── Yardımcı ──────────────────────────────────────────────────────────────

    @staticmethod
    def _fmt_date(dt: Union[str, datetime]) -> str:
        if isinstance(dt, str):
            return dt.replace("-", "")
        return dt.strftime("%Y%m%d")

    @staticmethod
    def _normalise(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        df = df.copy()
        df.columns = [c.strip().lower() for c in df.columns]

        rename = {
            "date"  : "timestamp",
            "open"  : "open",
            "high"  : "high",
            "low"   : "low",
            "close" : "close",
            "volume": "volume",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        for col in ("open", "high", "low", "close"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

        if "volume" not in df.columns:
            df["volume"] = float("nan")
        else:
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)

        df["symbol"]    = symbol.upper()
        df["adj_close"] = float("nan")
        df["vwap"]      = float("nan")

        df = df.dropna(subset=["open", "high", "low", "close"])
        keep = ["symbol", "timestamp", "open", "high", "low", "close",
                "volume", "adj_close", "vwap"]
        return df[keep].sort_values("timestamp").reset_index(drop=True)
