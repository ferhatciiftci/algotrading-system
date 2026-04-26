"""
reporting/performance.py
─────────────────────────
Backtest sonuçlarını analiz eder, Türkçe yorumlar üretir.
Dashboard ve terminal çıktısı için ortak katman.

Hiçbir trading mantığı içermez — sadece raporlama.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Veri yapıları
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FullReport:
    """Tek backtest koşusu için tam rapor."""
    # Temel
    initial_capital  : float
    final_equity     : float
    total_return_pct : float
    ann_return_pct   : float
    ann_vol_pct      : float

    # Risk
    max_drawdown_pct  : float
    avg_drawdown_pct  : float
    max_dd_duration   : int       # bars

    # Risk ayarlı getiri
    sharpe_ratio      : float
    sortino_ratio     : float
    calmar_ratio      : float
    omega_ratio       : float

    # İşlem istatistikleri
    num_trades        : int
    win_rate          : float     # 0–100
    avg_win_pct       : float
    avg_loss_pct      : float
    profit_factor     : float
    best_trade_pct    : float
    worst_trade_pct   : float
    max_consec_losses : int

    # Buy & Hold kıyaslama
    bh_return_pct     : float     # NaN ise kıyaslama yok
    bh_ann_return_pct : float

    # Overfitting göstergesi
    trade_count_ok    : bool      # >= 30 işlem → istatistiksel anlamlılık
    param_count       : int       # optimize edilen parametre sayısı

    # Ham veriler (grafik için)
    equity_curve      : pd.DataFrame   # timestamp, equity, drawdown, returns
    fills             : List[Any]


@dataclass
class CommentaryItem:
    category : str    # 'karlilik' | 'risk' | 'sharpe' | 'strateji' | 'overfitting' | 'benchmark'
    icon     : str    # '✅' | '⚠️' | '❌'
    title    : str
    body     : str


# ─────────────────────────────────────────────────────────────────────────────
# Performans hesaplama
# ─────────────────────────────────────────────────────────────────────────────

def compute_full_report(
    result,
    cfg              : dict,
    param_count      : int = 4,
) -> FullReport:
    """
    BacktestResult + config → FullReport.
    """
    from algotrading.validation.metrics import compute_metrics, compute_trade_metrics

    eq_curve = result.equity_curve
    fills    = result.fills
    initial  = float(cfg["backtest"]["initial_capital"])
    bpy      = int(cfg["data"].get("bars_per_year", 252))

    if len(eq_curve) < 5:
        raise ValueError("Yeterli equity verisi yok (en az 5 bar gerekli)")

    m  = compute_metrics(eq_curve["equity"], bars_per_year=bpy)
    tm = compute_trade_metrics(fills)

    # En iyi / en kötü işlem
    best_t = worst_t = 0.0
    max_cl = 0
    if fills:
        pnls = _fill_pnls(fills)
        if pnls:
            best_t  = max(pnls)
            worst_t = min(pnls)
        # Maksimum ardışık kayıp
        max_cl = _max_consec_losses(fills)

    # Buy & Hold
    bh_ret = bh_ann = float("nan")
    eq_df = eq_curve.copy()
    if "timestamp" in eq_df.columns and len(eq_df) >= 2:
        try:
            first_eq = float(eq_df["equity"].iloc[0])
            last_eq  = float(eq_df["equity"].iloc[-1])
            # B&H: varsayım ilk günün close fiyatı sabit hisse miktarı
            # Burada portföyün tümü B&H'de tutulduğunu varsayıyoruz
            # Gerçek B&H için fiyat verisi gerekirdi — yaklaşım olarak
            # equity curve'ün ilk değerini B&H başlangıcı say
            # (gerçek fiyat verisi olmadan varsayılan = initial_capital → ilk bar equity)
            bh_ret = float("nan")  # Fiyat verisi olmadan hesaplanamaz — dashboard halleder
        except Exception:
            pass

    return FullReport(
        initial_capital  = initial,
        final_equity     = float(eq_df["equity"].iloc[-1]),
        total_return_pct = m.total_return_pct,
        ann_return_pct   = m.annualised_return,
        ann_vol_pct      = m.annualised_vol,
        max_drawdown_pct = m.max_drawdown_pct,
        avg_drawdown_pct = m.avg_drawdown_pct,
        max_dd_duration  = m.max_drawdown_duration,
        sharpe_ratio     = m.sharpe_ratio,
        sortino_ratio    = m.sortino_ratio,
        calmar_ratio     = m.calmar_ratio,
        omega_ratio      = m.omega_ratio,
        num_trades       = tm.get("num_trades", 0),
        win_rate         = tm.get("win_rate", 0.0),
        avg_win_pct      = tm.get("avg_win_pct", 0.0),
        avg_loss_pct     = tm.get("avg_loss_pct", 0.0),
        profit_factor    = tm.get("profit_factor", 0.0),
        best_trade_pct   = best_t,
        worst_trade_pct  = worst_t,
        max_consec_losses= max_cl,
        bh_return_pct    = bh_ret,
        bh_ann_return_pct= bh_ann,
        trade_count_ok   = tm.get("num_trades", 0) >= 20,
        param_count      = param_count,
        equity_curve     = eq_df,
        fills            = fills,
    )


def attach_benchmark(report: FullReport, bh_return_pct: float, bh_ann_pct: float) -> None:
    """Buy & Hold benchmark değerlerini sonradan ekle."""
    report.bh_return_pct     = bh_return_pct
    report.bh_ann_return_pct = bh_ann_pct


# ─────────────────────────────────────────────────────────────────────────────
# Türkçe yorum üretici
# ─────────────────────────────────────────────────────────────────────────────

def generate_commentary(r: FullReport) -> List[CommentaryItem]:
    items: List[CommentaryItem] = []

    # ── 1. Kârlılık ──────────────────────────────────────────────────────────
    if r.total_return_pct > 30:
        items.append(CommentaryItem("karlilik", "✅",
            "Sistem kârlı",
            f"Toplam getiri +{r.total_return_pct:.1f}%, yıllık +{r.ann_return_pct:.1f}%. "
            "Güçlü bir performans, ancak geçmiş başarı gelecek getiriyi garanti etmez."))
    elif r.total_return_pct > 5:
        items.append(CommentaryItem("karlilik", "🟡",
            "Sistem hafif kârlı",
            f"Toplam getiri +{r.total_return_pct:.1f}%. Komisyon ve enflasyon düşüldüğünde "
            "gerçek getiri daha düşük olabilir."))
    elif r.total_return_pct >= 0:
        items.append(CommentaryItem("karlilik", "🟠",
            "Sistem neredeyse başabaş",
            f"Toplam getiri {r.total_return_pct:+.1f}%. Risk alınmasına oranla getiri düşük."))
    else:
        items.append(CommentaryItem("karlilik", "❌",
            "Sistem zararlı",
            f"Toplam getiri {r.total_return_pct:.1f}%. Strateji veya parametreler "
            "gözden geçirilmeli."))

    # ── 2. Buy & Hold kıyaslaması ─────────────────────────────────────────────
    if not math.isnan(r.bh_return_pct):
        diff = r.total_return_pct - r.bh_return_pct
        if diff > 5:
            items.append(CommentaryItem("benchmark", "✅",
                "Buy & Hold'dan iyi",
                f"Strateji, basit al-tut'tan {diff:+.1f}% daha iyi performans gösterdi "
                f"({r.total_return_pct:+.1f}% vs {r.bh_return_pct:+.1f}%)."))
        elif diff > -3:
            items.append(CommentaryItem("benchmark", "🟡",
                "Buy & Hold ile benzer",
                f"Strateji ile basit al-tut arasındaki fark {diff:+.1f}%. "
                "Aktif strateji ekstra işlem maliyeti yaratıyor, bu fark önemli."))
        else:
            items.append(CommentaryItem("benchmark", "❌",
                "Buy & Hold'dan kötü",
                f"Basit al-tut {-diff:.1f}% daha iyi performans gösterdi. "
                "Bu durumda aktif strateji değer katmıyor."))

    # ── 3. Risk seviyesi ──────────────────────────────────────────────────────
    dd_abs = abs(r.max_drawdown_pct)
    if dd_abs < 8:
        items.append(CommentaryItem("risk", "✅",
            "Düşük risk",
            f"Maksimum düşüş {dd_abs:.1f}%. Kill switch ve risk motorunun etkin çalıştığını gösteriyor."))
    elif dd_abs < 18:
        items.append(CommentaryItem("risk", "🟡",
            "Orta risk",
            f"Maksimum düşüş {dd_abs:.1f}%. Kabul edilebilir, ancak psikolojik olarak "
            "zor bir süreç olabilir."))
    elif dd_abs < 30:
        items.append(CommentaryItem("risk", "🟠",
            "Yüksek risk",
            f"Maksimum düşüş {dd_abs:.1f}%. Kill switch eşikleri sıkılaştırılabilir. "
            "Pozisyon boyutu küçültülmeli."))
    else:
        items.append(CommentaryItem("risk", "❌",
            "Çok yüksek risk",
            f"Maksimum düşüş {dd_abs:.1f}%. Bu düşüş gerçek portfolyoda dayanılması "
            "zor bir kayıp anlamına gelir."))

    # ── 4. Sharpe ────────────────────────────────────────────────────────────
    if r.sharpe_ratio >= 1.5:
        items.append(CommentaryItem("sharpe", "✅",
            f"Sharpe oranı güçlü ({r.sharpe_ratio:.2f})",
            "Risk başına alınan getiri tatminkâr. Kurumsal standart genellikle 1.0 üzeridir."))
    elif r.sharpe_ratio >= 0.8:
        items.append(CommentaryItem("sharpe", "🟡",
            f"Sharpe oranı kabul edilebilir ({r.sharpe_ratio:.2f})",
            "Orta seviye risk-getiri dengesi. İyileştirme alanı var."))
    elif r.sharpe_ratio >= 0.3:
        items.append(CommentaryItem("sharpe", "🟠",
            f"Sharpe oranı düşük ({r.sharpe_ratio:.2f})",
            "Strateji risksiz getiriyi zor geçiyor. Parametreler revize edilmeli."))
    else:
        items.append(CommentaryItem("sharpe", "❌",
            f"Sharpe oranı negatif/çok düşük ({r.sharpe_ratio:.2f})",
            "Strateji risk almaya değer üretmiyor."))

    # ── 5. Strateji gücü ─────────────────────────────────────────────────────
    if r.profit_factor >= 1.8 and r.win_rate >= 50:
        items.append(CommentaryItem("strateji", "✅",
            "Strateji güçlü",
            f"Kâr faktörü {r.profit_factor:.2f}, kazanma oranı %{r.win_rate:.0f}. "
            "Sistem hem sıklık hem büyüklük açısından iyi."))
    elif r.profit_factor >= 1.2:
        items.append(CommentaryItem("strateji", "🟡",
            "Strateji orta güçte",
            f"Kâr faktörü {r.profit_factor:.2f}. Kazançlar kayıpları karşılıyor "
            "ancak kenar (edge) dar."))
    elif r.profit_factor >= 1.0:
        items.append(CommentaryItem("strateji", "🟠",
            "Strateji zayıf",
            f"Kâr faktörü {r.profit_factor:.2f}. Sistem neredeyse başabaş çalışıyor."))
    else:
        items.append(CommentaryItem("strateji", "❌",
            "Strateji negatif kenar",
            f"Kâr faktörü {r.profit_factor:.2f} < 1.0. Kayıplar kazançları geçiyor."))

    # ── 6. Overfitting uyarısı ───────────────────────────────────────────────
    of_issues = []
    if not r.trade_count_ok:
        of_issues.append(f"az işlem ({r.num_trades} adet — istatistiksel güven için en az 20 gerekli)")
    if r.param_count >= 5:
        of_issues.append(f"çok parametre ({r.param_count} adet — artık her parametre aşırı uyum riskini artırır)")

    if of_issues:
        items.append(CommentaryItem("overfitting", "⚠️",
            "Aşırı uyum (Overfitting) riski var",
            "Dikkat: " + ", ".join(of_issues) + ". "
            "Bu parametreler geçmiş veriye fazla uyarlanmış olabilir. "
            "Farklı dönem ve sembollerde test edin."))
    else:
        items.append(CommentaryItem("overfitting", "✅",
            "Overfitting riski düşük",
            f"{r.num_trades} işlem ve {r.param_count} parametre ile sonuçlar görece güvenilir. "
            "Yine de farklı dönemlerle walk-forward testi önerilir."))

    return items


# ─────────────────────────────────────────────────────────────────────────────
# Yardımcı fonksiyonlar
# ─────────────────────────────────────────────────────────────────────────────

def _fill_pnls(fills) -> List[float]:
    """Fill listesinden yaklaşık % PnL hesaplar (eşleştirilmiş long-short)."""
    pnls = []
    open_fills = {}
    for f in fills:
        sym = f.symbol
        dir_val = f.direction.value.upper()
        if dir_val in ("LONG", "BUY"):
            open_fills[sym] = f.fill_price
        elif dir_val in ("SHORT", "SELL", "FLAT") and sym in open_fills:
            entry = open_fills.pop(sym)
            if entry > 0:
                pnl_pct = (f.fill_price - entry) / entry * 100
                pnls.append(pnl_pct)
    return pnls


def _max_consec_losses(fills) -> int:
    pnls = _fill_pnls(fills)
    max_cl = cur = 0
    for p in pnls:
        if p < 0:
            cur += 1
            max_cl = max(max_cl, cur)
        else:
            cur = 0
    return max_cl


def compute_bh_return(
    equity_curve : pd.DataFrame,
    initial_capital: float,
    price_series  : Optional[pd.Series] = None,
) -> Tuple[float, float, pd.Series]:
    """
    Buy & Hold getirisi hesaplar.
    price_series verilirse gerçek B&H; yoksa equity curve'ün ilk close'u kullanır.
    Döndürür: (toplam_getiri_pct, yillik_getiri_pct, bh_equity_series)
    """
    eq = equity_curve.copy()
    eq["timestamp"] = pd.to_datetime(eq["timestamp"])
    eq = eq.set_index("timestamp").sort_index()

    n_bars = len(eq)
    if n_bars < 2:
        return float("nan"), float("nan"), pd.Series(dtype=float)

    if price_series is not None and len(price_series) >= 2:
        p = price_series.values
        bh_equity = pd.Series(
            initial_capital * (p / p[0]),
            index=price_series.index,
        )
    else:
        # Fiyat verisi yok — equity curve'ün ilk değerini normalize et
        eq_vals = eq["equity"].values
        bh_equity = pd.Series(
            initial_capital * (eq_vals / eq_vals[0]),
            index=eq.index,
        )

    total_ret = (bh_equity.iloc[-1] / initial_capital - 1.0) * 100
    bars_per_year = 252
    ann_ret = ((1 + total_ret / 100) ** (bars_per_year / n_bars) - 1.0) * 100

    return total_ret, ann_ret, bh_equity
