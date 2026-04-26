"""
research/memory.py
───────────────────
Araştırma Hafızası — Backtest sonuçlarını saklar ve analiz eder.

ÖNEMLİ: Bu bir makine öğrenmesi sistemi değildir.
         Geçmiş backtest sonuçlarını kaydeder ve özetler.
         Streamlit Cloud'da oturum sona erdiğinde JSON dosyası kaybolabilir;
         bu nedenle sonuçlar aynı zamanda st.session_state'de tutulur.

Saklananlar:
  - Zaman damgası
  - Veri kaynağı
  - Sembol
  - Strateji adı
  - Parametreler (özet)
  - Metrikler (getiri, drawdown, Sharpe, işlem sayısı, kazanma oranı)
  - Bileşik skor
  - Verdict (Güçlü / Orta / Zayıf / Yetersiz Veri)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)
UTC = timezone.utc

_DEFAULT_FILE = "research_results.json"


@dataclass
class ResearchResult:
    timestamp       : str
    source          : str
    symbol          : str
    strategy        : str
    confirmation_mode: str
    params_summary  : str   # kısa özet, örn. "ema12/26 rsi14"
    total_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio    : float
    num_trades      : int
    win_rate        : float
    score           : float
    verdict         : str   # Güçlü | Orta | Zayıf | Yetersiz Veri
    start_date      : str
    end_date        : str


def _verdict(num_trades: int, total_return_pct: float,
             sharpe: float, score: float) -> str:
    if num_trades < 5:
        return "Yetersiz Veri"
    if score >= 0.60 and total_return_pct > 5 and sharpe > 0.5:
        return "Güçlü"
    if score >= 0.40 or (total_return_pct > 0 and num_trades >= 10):
        return "Orta"
    return "Zayıf"


def build_result(
    symbol          : str,
    source          : str,
    strategy        : str,
    confirmation_mode: str,
    cfg_params      : dict,
    metrics         ,   # algotrading.validation.metrics sonucu
    trade_metrics   : dict,
    score           : float,
    start_date      : str,
    end_date        : str,
) -> ResearchResult:
    """Backtest sonuçlarından ResearchResult oluştur."""
    sp = cfg_params
    summary_parts = []
    if "ema_fast" in sp and "ema_slow" in sp:
        summary_parts.append(f"ema{sp['ema_fast']}/{sp['ema_slow']}")
    if "rsi_period" in sp:
        summary_parts.append(f"rsi{sp['rsi_period']}")
    if "adx_threshold" in sp:
        summary_parts.append(f"adx{sp['adx_threshold']:.0f}")
    params_summary = " ".join(summary_parts) or "default"

    tr  = round(float(getattr(metrics, "total_return_pct", 0.0)), 2)
    dd  = round(float(getattr(metrics, "max_drawdown_pct",  0.0)), 2)
    sh  = round(float(getattr(metrics, "sharpe_ratio",       0.0)), 3)
    nt  = int(trade_metrics.get("num_trades", 0))
    wr  = round(float(trade_metrics.get("win_rate", 0.0)), 1)
    sc  = round(float(score), 4)

    return ResearchResult(
        timestamp        = datetime.now(UTC).strftime("%Y-%m-%d %H:%M"),
        source           = source,
        symbol           = symbol,
        strategy         = strategy,
        confirmation_mode= confirmation_mode,
        params_summary   = params_summary,
        total_return_pct = tr,
        max_drawdown_pct = dd,
        sharpe_ratio     = sh,
        num_trades       = nt,
        win_rate         = wr,
        score            = sc,
        verdict          = _verdict(nt, tr, sh, sc),
        start_date       = start_date,
        end_date         = end_date,
    )


class ResearchMemory:
    """
    Araştırma sonuçlarını hem bellekte (session) hem JSON dosyasında tutar.

    Streamlit'te:
        mem = ResearchMemory(data_dir)
        mem.add(result)
        df  = mem.to_dataframe()
    """

    def __init__(self, data_dir: Optional[Path] = None) -> None:
        self._data_dir = Path(data_dir) if data_dir else Path(".")
        self._file     = self._data_dir / _DEFAULT_FILE
        self._results  : List[ResearchResult] = []
        self._load()

    # ── Public ───────────────────────────────────────────────────────────────

    def add(self, result: ResearchResult) -> None:
        """Sonuç ekle ve dosyaya kaydet."""
        self._results.append(result)
        self._save()

    def all_results(self) -> List[ResearchResult]:
        return list(self._results)

    def to_dataframe(self):
        import pandas as pd
        if not self._results:
            return pd.DataFrame()
        rows = [asdict(r) for r in self._results]
        df = pd.DataFrame(rows)
        # Sütun yeniden adlandırma (Türkçe)
        tr_cols = {
            "timestamp"       : "Zaman",
            "source"          : "Kaynak",
            "symbol"          : "Sembol",
            "strategy"        : "Strateji",
            "confirmation_mode":"Onay Modu",
            "params_summary"  : "Parametreler",
            "total_return_pct": "Getiri %",
            "max_drawdown_pct": "Maks. DD %",
            "sharpe_ratio"    : "Sharpe",
            "num_trades"      : "İşlem",
            "win_rate"        : "Kazanma %",
            "score"           : "Skor",
            "verdict"         : "Sonuç",
            "start_date"      : "Başlangıç",
            "end_date"        : "Bitiş",
        }
        df = df.rename(columns={k: v for k, v in tr_cols.items() if k in df.columns})
        return df

    def best(self, n: int = 5) -> List[ResearchResult]:
        return sorted(self._results, key=lambda r: r.score, reverse=True)[:n]

    def worst(self, n: int = 5) -> List[ResearchResult]:
        valid = [r for r in self._results if r.num_trades >= 5]
        return sorted(valid, key=lambda r: r.score)[:n]

    def insufficient_data(self) -> List[ResearchResult]:
        return [r for r in self._results if r.num_trades < 5]

    def clear(self) -> None:
        self._results.clear()
        try:
            self._file.unlink(missing_ok=True)
        except Exception:
            pass

    # ── Internals ─────────────────────────────────────────────────────────────

    def _save(self) -> None:
        try:
            self._data_dir.mkdir(parents=True, exist_ok=True)
            data = [asdict(r) for r in self._results]
            with open(self._file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning("Araştırma hafızası kaydedilemedi: %s", e)

    def _load(self) -> None:
        try:
            if self._file.exists():
                with open(self._file, encoding="utf-8") as f:
                    data = json.load(f)
                for row in data:
                    try:
                        self._results.append(ResearchResult(**row))
                    except Exception:
                        pass
                logger.info("Araştırma hafızası yüklendi: %d kayıt", len(self._results))
        except Exception as e:
            logger.warning("Araştırma hafızası yüklenemedi: %s", e)
