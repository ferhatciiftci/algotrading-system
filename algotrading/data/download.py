"""
data/download.py
----------------
Desteklenen kaynaklardan piyasa verisi indirme CLI.

Kullanim:
    python -m algotrading.data.download \
        --source yfinance --symbol SPY \
        --start 2020-01-01 --end 2024-01-01

    python -m algotrading.data.download \
        --source csv --symbol AAPL --file /yol/AAPL.csv

Veri nereye kaydedilir?
    PROJECT_ROOT/data/raw/<SEMBOL>/...
    Burada PROJECT_ROOT, algotrading/ paketinin bir ust dizinidir.
    Bu, python run_backtest.py ile her zaman AYNI klasoru kullanir.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level    = logging.INFO,
    format   = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers = [logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("algotrading.download")


def _project_root() -> Path:
    """
    Proje kokunu dondurur.
    Bu dosya (download.py) her zaman PROJECT_ROOT/algotrading/data/download.py
    konumundadir, dolayisiyla proje koku = bu dosyanin UC ust dizinidir.
    """
    return Path(__file__).resolve().parent.parent.parent


def _resolve_data_dir(args: argparse.Namespace) -> Path:
    """
    Veri dizinini belirler.

    Oncelik sirasi:
      1. --data-dir argumani (verilmisse)
      2. --config dosyasindaki data.root_dir (PROJECT_ROOT'a gore)
      3. Otomatik bulunan config'deki data.root_dir
      4. Fallback: PROJECT_ROOT/data
    """
    if args.data_dir:
        p = Path(args.data_dir)
        return p if p.is_absolute() else (_project_root() / p).resolve()

    # Config dosyasini bul
    cfg_path: Path | None = None
    if args.config:
        p = Path(args.config)
        if not p.is_absolute():
            p = (_project_root() / p).resolve()
        if p.exists():
            cfg_path = p

    if cfg_path is None:
        # Otomatik ara
        for candidate in [
            _project_root() / "algotrading" / "config" / "default.yaml",
            _project_root() / "config" / "default.yaml",
        ]:
            if candidate.exists():
                cfg_path = candidate
                break

    if cfg_path is not None:
        try:
            import yaml
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            raw = cfg["data"]["root_dir"]
            p   = Path(raw)
            resolved = p if p.is_absolute() else (_project_root() / raw).resolve()
            return resolved
        except Exception as e:
            logger.warning("Config okunamadi (%s), varsayilan kullaniliyor", e)

    fallback = (_project_root() / "data").resolve()
    return fallback


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AlgoTrading MVP -- Piyasa Verisi Indirici",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--source",   required=True,
                        choices=["yfinance", "binance", "csv"])
    parser.add_argument("--symbol",   required=True)
    parser.add_argument("--start",    default=None)
    parser.add_argument("--end",      default=None)
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--file",     default=None)
    parser.add_argument("--data-dir", default=None,
                        dest="data_dir")
    parser.add_argument("--config",   default=None)
    args = parser.parse_args()

    data_dir = _resolve_data_dir(args)
    data_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Veri deposu : %s", data_dir)

    from algotrading.data.ingestion import RawDataStore
    raw_store = RawDataStore(data_dir)
    symbol    = args.symbol.upper()

    if args.source == "yfinance":
        if not args.start or not args.end:
            parser.error("--start ve --end zorunludur")
        from algotrading.data.connectors.yfinance_connector import YFinanceConnector
        conn = YFinanceConnector(interval=args.interval)
        logger.info("Yahoo Finance'dan %s indiriliyor (%s - %s) ...",
                    symbol, args.start, args.end)
        h = conn.fetch_and_store(symbol, args.start, args.end, raw_store)
        logger.info("Tamamlandi. Hash: %s", h[:16])
        logger.info("Dosya konumu: %s", data_dir / "raw" / symbol)

    elif args.source == "binance":
        if not args.start or not args.end:
            parser.error("--start ve --end zorunludur")
        from algotrading.data.connectors.binance_connector import BinanceConnector
        conn = BinanceConnector(interval=args.interval)
        logger.info("Binance'dan %s indiriliyor (%s - %s) ...",
                    symbol, args.start, args.end)
        h = conn.fetch_and_store(symbol, args.start, args.end, raw_store)
        logger.info("Tamamlandi. Hash: %s", h[:16])
        logger.info("Dosya konumu: %s", data_dir / "raw" / symbol)

    elif args.source == "csv":
        if not args.file:
            parser.error("--file zorunludur")
        from algotrading.data.connectors.csv_loader import CSVLoader
        loader = CSVLoader()
        logger.info("CSV'den %s yukleniyor: %s ...", symbol, args.file)
        h = loader.load_and_store(args.file, symbol, raw_store)
        logger.info("Tamamlandi. Hash: %s", h[:16])
        logger.info("Dosya konumu: %s", data_dir / "raw" / symbol)


if __name__ == "__main__":
    main()
