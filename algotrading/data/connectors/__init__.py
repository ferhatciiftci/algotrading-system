"""
data/connectors/
────────────────
Pluggable market-data connectors.

Each connector fetches OHLCV data from an external source and writes it
to the RawDataStore using the standard ingestion pipeline (validation
included).

Available connectors:
  - YFinanceConnector   — Yahoo Finance via yfinance (equities, ETFs)
  - BinanceConnector    — Binance public REST API (crypto, no key needed)
  - CSVLoader           — Load from a local CSV file

All connectors share the same interface:

    connector.download(symbol, start, end)  → pd.DataFrame
    connector.fetch_and_store(symbol, start, end, raw_store)  → str (hash)
"""

from algotrading.data.connectors.yfinance_connector import YFinanceConnector
from algotrading.data.connectors.binance_connector  import BinanceConnector
from algotrading.data.connectors.csv_loader         import CSVLoader

__all__ = ["YFinanceConnector", "BinanceConnector", "CSVLoader"]
