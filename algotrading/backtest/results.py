"""
backtest/results.py
-------------------
Backtest sonuclarini JSON / CSV olarak kaydeder ve ozetler.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def save_results(result: Any, output_dir: Path, name: str = "backtest") -> Path:
    """
    BacktestResult nesnesini output_dir altina kaydeder.
    Returns: kaydedilen dizin yolu.
    """
    output_dir = Path(output_dir)
    ts   = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    dest = output_dir / f"{name}_{ts}"
    dest.mkdir(parents=True, exist_ok=True)

    # Equity curve -> CSV
    if hasattr(result, "equity_curve") and not result.equity_curve.empty:
        result.equity_curve.to_csv(dest / "equity_curve.csv", index=False)

    # Fills -> CSV
    if hasattr(result, "fills") and result.fills:
        fills_data = []
        for f in result.fills:
            fills_data.append({
                "fill_id"   : f.fill_id,
                "symbol"    : f.symbol,
                "direction" : f.direction.value,
                "quantity"  : f.quantity,
                "fill_price": f.fill_price,
                "commission": f.commission,
                "slippage"  : f.slippage,
                "timestamp" : str(f.timestamp),
            })
        pd.DataFrame(fills_data).to_csv(dest / "fills.csv", index=False)

    # Summary -> JSON
    summary = {
        "config_fingerprint": getattr(result, "config_fingerprint", ""),
        "start_time"        : str(getattr(result, "start_time", "")),
        "end_time"          : str(getattr(result, "end_time", "")),
        "run_duration_s"    : getattr(result, "run_duration_s", 0),
        "events_processed"  : getattr(result, "events_processed", 0),
        "portfolio"         : getattr(result, "portfolio_summary", {}),
    }
    with open(dest / "summary.json", "w") as fp:
        json.dump(summary, fp, indent=2, default=str)

    logger.info("Sonuclar kaydedildi: %s", dest)
    return dest
