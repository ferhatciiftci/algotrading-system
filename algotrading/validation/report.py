"""
validation/report.py
--------------------
Walk-forward ve overfitting sonuclarini okunabilir rapor olarak ciktilar.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def print_validation_report(wfa_result, overfit_report=None) -> None:
    """Konsola okunabilir validasyon raporu yazdirir."""
    print("\n" + "=" * 60)
    print("VALIDASYON RAPORU")
    print("=" * 60)
    print(f"  Fold sayisi          : {len(wfa_result.windows)}")
    print(f"  Karli fold orani     : {wfa_result.pct_windows_profitable:.0f}%")
    print(f"  Ort. Verimlilik oran : {wfa_result.avg_efficiency_ratio:.3f}  (>0.5 iyi)")

    if wfa_result.combined_oos:
        oos = wfa_result.combined_oos
        print(f"\n  --- Birlestirilis OOS Metrikleri ---")
        print(f"  Toplam Getiri        : {oos.total_return_pct:.2f}%")
        print(f"  Yillik Getiri        : {oos.annualised_return:.2f}%")
        print(f"  Sharpe               : {oos.sharpe_ratio:.3f}")
        print(f"  Sortino              : {oos.sortino_ratio:.3f}")
        print(f"  Max Drawdown         : {oos.max_drawdown_pct:.2f}%")
        print(f"  Stabilite R2         : {oos.stability_r2:.4f}")
        print(f"  Deflated Sharpe      : {oos.deflated_sharpe:.4f}")

    print(f"\n  --- Fold Detaylari ---")
    for w in wfa_result.windows:
        is_sr  = f"{w.is_report.sharpe_ratio:.2f}"  if w.is_report  else "N/A"
        oos_sr = f"{w.oos_report.sharpe_ratio:.2f}" if w.oos_report else "N/A"
        eff    = f"{w.efficiency_ratio:.2f}"
        print(f"  Fold {w.fold_id:2d} | IS Sharpe={is_sr:6} | OOS Sharpe={oos_sr:6} | Verimlilik={eff}")

    if overfit_report:
        print(f"\n  --- Overfitting Kontrol ---")
        status = "GECTI" if overfit_report.passed else "KALDI"
        print(f"  Sonuc                : {status}")
        print(f"  Verimlilik Orani     : {overfit_report.efficiency_ratio:.3f}")
        print(f"  PSR (min bar gerek.) : {overfit_report.psr_min_bars} (mevcut: {overfit_report.psr_current_bars})")
        print(f"  Monte Carlo p-degeri : {overfit_report.mc_p_value:.4f}  (<0.05 iyi)")
        for w in overfit_report.warnings:
            print(f"  [UYARI] {w}")

    print("=" * 60 + "\n")


def save_validation_report(wfa_result, overfit_report=None,
                            output_dir: Path = Path("."),
                            name: str = "validation") -> Path:
    """Sonuclari JSON dosyasina kaydeder."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts   = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"{name}_{ts}.json"

    data: dict = {
        "timestamp"            : ts,
        "num_folds"            : len(wfa_result.windows),
        "pct_profitable_folds" : wfa_result.pct_windows_profitable,
        "avg_efficiency_ratio" : wfa_result.avg_efficiency_ratio,
        "combined_oos"         : None,
        "overfitting"          : None,
    }
    if wfa_result.combined_oos:
        oos = wfa_result.combined_oos
        data["combined_oos"] = {
            "total_return_pct"  : oos.total_return_pct,
            "annualised_return" : oos.annualised_return,
            "sharpe_ratio"      : oos.sharpe_ratio,
            "max_drawdown_pct"  : oos.max_drawdown_pct,
            "stability_r2"      : oos.stability_r2,
        }
    if overfit_report:
        data["overfitting"] = {
            "passed"            : overfit_report.passed,
            "efficiency_ratio"  : overfit_report.efficiency_ratio,
            "mc_p_value"        : overfit_report.mc_p_value,
            "warnings"          : overfit_report.warnings,
        }
    with open(path, "w") as fp:
        json.dump(data, fp, indent=2)
    logger.info("Validasyon raporu kaydedildi: %s", path)
    return path
