#!/usr/bin/env python3
"""
run_backtest.py
---------------
Proje kokunden backtest calistirmak icin giris noktasi.

Kullanim:
    python run_backtest.py
    python run_backtest.py --config algotrading/config/default.yaml
    python run_backtest.py --mode backtest

Veri yoksa once indirin:
    python -m algotrading.data.download --source yfinance --symbol SPY \
        --start 2020-01-01 --end 2024-01-01
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algotrading.main import main  # noqa: E402

if __name__ == "__main__":
    if "--config" not in sys.argv:
        default_cfg = PROJECT_ROOT / "algotrading" / "config" / "default.yaml"
        if default_cfg.exists():
            sys.argv += ["--config", str(default_cfg)]
    main()
