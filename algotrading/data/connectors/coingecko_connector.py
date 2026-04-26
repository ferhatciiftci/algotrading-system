"""
data/connectors/coingecko_connector.py
────────────────────────────────────────
CoinGecko ücretsiz API ile kripto para tarihsel OHLCV verisi.

API anahtarı gerekmez (demo/public endpoint).
Hız sınırı: ~50 istek/dakika (ücretsiz katman).

Desteklenen coin kimlikleri:
  bitcoin, ethereum, solana, ripple, cardano,
  dogecoin, avalanche-2, chainlink, polkadot, litecoin

NOT:
  CoinGecko ücretsiz API'de gerçek OHLCV verisi için /ohlc endpoint'i
  kullanılır. İstek başına maks. 365 gün döner; daha uzun dönemler için
  ardışık istekler atılır. Rate limit durumunda Türkçe hata mesajı gösterilir.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)
UTC = timezone.utc

_CG_BASE    = "https://api.coingecko.com/api/v3"
_TIMEOUT_S  = 20
_RETRY_WAIT = 65   # rate limit sonrası bekleme (saniye)

# Tanınmış sembol → CoinGecko coin id eşlemesi
SYMBOL_TO_ID: dict[str, str] = {
    "bitcoin"    : "bitcoin",
    "btc"        : "bitcoin",
    "ethereum"   : "ethereum",
    "eth"        : "ethereum",
    "solana"     : "solana",
    "sol"        : "solana",
    "ripple"     : "ripple",
    "xrp"        : "ripple",
    "cardano"    : "cardano",
    "ada"        : "cardano",
    "dogecoin"   : "dogecoin",
    "doge"       : "dogecoin",
    "avalanche-2": "avalanche-2",
    "avax"       : "avalanche-2",
    "chainlink"  : "chainlink",
    "link"       : "chainlink",
    "polkadot"   : "polkadot",
    "dot"        : "polkadot",
    "litecoin"   : "litecoin",
    "ltc"        : "litecoin",
}

SUPPORTED_COINS = [
    "bitcoin", "ethereum", "solana", "ripple", "cardano",
    "dogecoin", "avalanche-2", "chainlink", "polkadot", "litecoin",
]


class CoinGeckoConnector:
    """
    CoinGecko ücretsiz API üzerinden kripto OHLCV verisi indirir.

    Örnek
    -----
    >>> conn = CoinGeckoConnector()
    >>> df = conn.download("bitcoin", "2021-01-01", "2024-01-01")
    """

    SOURCE = "coingecko"

    def __init__(self, vs_currency: str = "usd") -> None:
        self.vs_currency = vs_currency

    # ── Public ──────────────────────────────────────────────────────────────

    def download(
        self,
        coin_id  : str,
        start    : Union[str, datetime],
        end      : Union[str, datetime],
    ) -> pd.DataFrame:
        """
        Belirtilen coin için günlük OHLCV verisi indirir.

        coin_id, SYMBOL_TO_ID eşlemesi üzerinden çözülür
        (örn. "btc" → "bitcoin", "eth" → "ethereum").

        Uzun dönemler için ardışık 365 günlük istekler atılır.
        CoinGecko rate limitine takılırsa Türkçe hata fırlatır.

        Döndürür
        --------
        pd.DataFrame: symbol, timestamp (UTC), open, high, low, close, volume
        """
        try:
            import requests
        except ImportError:
            raise ImportError("requests gerekli: pip install requests")

        coin_id = self._resolve_id(coin_id)
        start_dt = self._parse_dt(start)
        end_dt   = self._parse_dt(end)

        all_frames: list[pd.DataFrame] = []
        cursor     = start_dt

        while cursor < end_dt:
            chunk_end = min(cursor + timedelta(days=364), end_dt)
            days_needed = (chunk_end - cursor).days + 1

            try:
                chunk_df = self._fetch_ohlc(coin_id, days_needed, cursor, chunk_end)
                if chunk_df is not None and not chunk_df.empty:
                    all_frames.append(chunk_df)
            except _RateLimitError:
                raise RuntimeError(
                    f"CoinGecko hız sınırına ulaşıldı. "
                    f"Birkaç dakika bekleyip tekrar deneyin. "
                    f"(coin: {coin_id})"
                )
            except Exception as e:
                logger.warning("[coingecko] OHLC başarısız, market_chart denenecek: %s", e)
                try:
                    chunk_df = self._fetch_market_chart(coin_id, cursor, chunk_end)
                    if chunk_df is not None and not chunk_df.empty:
                        all_frames.append(chunk_df)
                        logger.info(
                            "[coingecko] market_chart kullanıldı — gerçek H/L yok, "
                            "yaklaşık OHLC türetildi."
                        )
                except _RateLimitError:
                    raise RuntimeError(
                        "CoinGecko hız sınırına ulaşıldı. "
                        "Birkaç dakika bekleyip tekrar deneyin."
                    )

            cursor = chunk_end + timedelta(days=1)
            time.sleep(0.5)   #礼貌性 bekleme

        if not all_frames:
            raise ValueError(
                f"CoinGecko '{coin_id}' için {start} → {end} aralığında "
                f"veri döndürmedi. Coin kimliğini veya tarih aralığını kontrol edin."
            )

        df = (pd.concat(all_frames, ignore_index=True)
                .sort_values("timestamp")
                .drop_duplicates(subset="timestamp")
                .reset_index(drop=True))

        # Tarih filtresi
        ts_start = pd.Timestamp(start_dt, tz="UTC")
        ts_end   = pd.Timestamp(end_dt,   tz="UTC")
        df = df[(df["timestamp"] >= ts_start) & (df["timestamp"] <= ts_end)]

        if df.empty:
            raise ValueError(
                f"CoinGecko filtre sonrası boş tablo: "
                f"'{coin_id}' {start} → {end}"
            )

        logger.info(
            "[coingecko] %d bar indirildi: %s (%s → %s)",
            len(df), coin_id,
            df["timestamp"].min().date(),
            df["timestamp"].max().date(),
        )
        return df.reset_index(drop=True)

    def fetch_and_store(
        self,
        coin_id  : str,
        start    : Union[str, datetime],
        end      : Union[str, datetime],
        raw_store,
    ) -> str:
        df = self.download(coin_id, start, end)
        return raw_store.write(df, coin_id.upper(), source=self.SOURCE)

    # ── Internals ────────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_id(coin_id: str) -> str:
        return SYMBOL_TO_ID.get(coin_id.lower().strip(), coin_id.lower().strip())

    @staticmethod
    def _parse_dt(dt: Union[str, datetime]) -> datetime:
        if isinstance(dt, str):
            dt = datetime.fromisoformat(dt)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt

    def _fetch_ohlc(
        self,
        coin_id  : str,
        days     : int,
        start_dt : datetime,
        end_dt   : datetime,
    ) -> Optional[pd.DataFrame]:
        """
        /coins/{id}/ohlc endpoint — gerçek OHLCV döner.
        CoinGecko ücretsiz katmanda günlük çözünürlük için min 30 gün önerilir.
        """
        import requests

        params = {
            "vs_currency": self.vs_currency,
            "days"       : max(days, 30),   # <30 gün = saatlik granülasyon
        }
        url  = f"{_CG_BASE}/coins/{coin_id}/ohlc"
        resp = requests.get(url, params=params, timeout=_TIMEOUT_S)

        if resp.status_code == 429:
            raise _RateLimitError("Rate limit")
        if resp.status_code == 404:
            raise ValueError(f"CoinGecko'da coin bulunamadı: '{coin_id}'")
        if resp.status_code != 200:
            raise RuntimeError(
                f"CoinGecko /ohlc hatası {resp.status_code}: {resp.text[:200]}"
            )

        data = resp.json()
        if not data:
            return None

        df = pd.DataFrame(data, columns=["ts_ms", "open", "high", "low", "close"])
        df["timestamp"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
        df["volume"]    = float("nan")   # OHLC endpoint'inde volume yok
        df["symbol"]    = coin_id.upper()
        df["adj_close"] = float("nan")
        df["vwap"]      = float("nan")
        return df[["symbol","timestamp","open","high","low","close",
                   "volume","adj_close","vwap"]]

    def _fetch_market_chart(
        self,
        coin_id  : str,
        start_dt : datetime,
        end_dt   : datetime,
    ) -> Optional[pd.DataFrame]:
        """
        /coins/{id}/market_chart/range — close + volume, gerçek H/L yok.
        Open/High/Low, close fiyatından yaklaşık türetilir.
        """
        import requests

        params = {
            "vs_currency": self.vs_currency,
            "from"       : int(start_dt.timestamp()),
            "to"         : int(end_dt.timestamp()),
        }
        url  = f"{_CG_BASE}/coins/{coin_id}/market_chart/range"
        resp = requests.get(url, params=params, timeout=_TIMEOUT_S)

        if resp.status_code == 429:
            raise _RateLimitError("Rate limit")
        if resp.status_code != 200:
            raise RuntimeError(
                f"CoinGecko /market_chart/range hatası "
                f"{resp.status_code}: {resp.text[:200]}"
            )

        raw     = resp.json()
        prices  = raw.get("prices", [])
        volumes = raw.get("total_volumes", [])

        if not prices:
            return None

        df_p = pd.DataFrame(prices,  columns=["ts_ms", "close"])
        df_v = pd.DataFrame(volumes, columns=["ts_ms", "volume"])

        df = df_p.merge(df_v, on="ts_ms", how="left")
        df["timestamp"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
        df["timestamp"] = df["timestamp"].dt.floor("D")   # Günlük hizala

        # Yaklaşık OHLC (close-only veriden)
        df["open"]      = df["close"].shift(1).fillna(df["close"])
        df["high"]      = df[["open","close"]].max(axis=1)
        df["low"]       = df[["open","close"]].min(axis=1)
        df["symbol"]    = coin_id.upper()
        df["adj_close"] = float("nan")
        df["vwap"]      = float("nan")

        return df[["symbol","timestamp","open","high","low","close",
                   "volume","adj_close","vwap"]]


class _RateLimitError(Exception):
    pass
