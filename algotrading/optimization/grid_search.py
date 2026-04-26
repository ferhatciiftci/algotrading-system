"""
optimization/grid_search.py
────────────────────────────
Çok amaçlı parametre optimizasyonu (güvenli, overfitting karşıtı).

TASARIM İLKELERİ:
- Sadece en yüksek getiri için optimize ETMEZ.
- Skor = ağırlıklı bileşik: Sharpe, getiri, drawdown, işlem sayısı, istikrar.
- Her kombinasyon için aynı backtest motoru kullanılır.
- Sonuçlar şeffaf biçimde döndürülür — nihai karar kullanıcıya aittir.
- Overfitting riski açıkça raporlanır.

UYARI: Parametreler geçmiş veriye göre optimize edilir.
       Gelecek performansı garanti etmez.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)
UTC = timezone.utc


@dataclass
class OptResult:
    """Tek parametre kombinasyonu sonucu."""
    params           : Dict[str, Any]
    total_return_pct : float
    ann_return_pct   : float
    max_drawdown_pct : float
    sharpe_ratio     : float
    sortino_ratio    : float
    calmar_ratio     : float
    profit_factor    : float
    num_trades       : int
    win_rate         : float
    score            : float = 0.0   # bileşik skor (yüksek = iyi)
    error            : Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Skor fonksiyonu
# ─────────────────────────────────────────────────────────────────────────────

def composite_score(
    total_return_pct : float,
    max_drawdown_pct : float,
    sharpe_ratio     : float,
    num_trades       : int,
    profit_factor    : float,
    # Ağırlıklar (toplamı = 1.0)
    w_sharpe   : float = 0.35,
    w_return   : float = 0.25,
    w_drawdown : float = 0.25,
    w_trades   : float = 0.10,
    w_pf       : float = 0.05,
) -> float:
    """
    Çok amaçlı bileşik skor. Her bileşen [0, 1] aralığına normalize edilir.
    Yüksek skor = daha iyi risk-getiri dengesi.
    """
    # Sharpe: [-1, 3] → [0, 1]
    s_sharpe   = _clip01((sharpe_ratio + 1.0) / 4.0)

    # Getiri: [-20, 60] → [0, 1]
    s_return   = _clip01((total_return_pct + 20) / 80.0)

    # Drawdown: [-50, 0] → [1, 0]  (düşük drawdown = iyi)
    s_drawdown = _clip01(1.0 - abs(max_drawdown_pct) / 50.0)

    # İşlem sayısı: [0, 50] → [0, 1]  (çok az işlem = güvenilmez)
    s_trades   = _clip01(num_trades / 50.0)

    # Kâr faktörü: [0, 3] → [0, 1]
    s_pf       = _clip01((profit_factor - 0.5) / 2.5)

    score = (
        w_sharpe   * s_sharpe   +
        w_return   * s_return   +
        w_drawdown * s_drawdown +
        w_trades   * s_trades   +
        w_pf       * s_pf
    )
    return round(float(score), 4)


def _clip01(v: float) -> float:
    return max(0.0, min(1.0, v))


# ─────────────────────────────────────────────────────────────────────────────
# Grid arama
# ─────────────────────────────────────────────────────────────────────────────

def run_grid_search(
    param_grid   : Dict[str, List[Any]],
    cfg_base     : dict,
    data_handler ,     # InMemoryDataHandler veya PITDataHandler uyumlu
    max_combos   : int = 50,
    top_n        : int = 10,
    progress_cb  = None,    # Optional[Callable[[int, int], None]]
) -> List[OptResult]:
    """
    param_grid: {'ema_fast': [10,20], 'ema_slow': [30,50], ...}
    cfg_base  : temel config dict (kopyalanır, değiştirilmez)
    data_handler: InMemoryDataHandler veya PITDataHandler

    Döndürür: puan sıralı OptResult listesi (en iyi başta)
    """
    import copy

    keys   = list(param_grid.keys())
    values = list(param_grid.values())
    all_combos = list(itertools.product(*values))

    if len(all_combos) > max_combos:
        # Rastgele örnekle
        import random
        rng = random.Random(42)
        all_combos = rng.sample(all_combos, max_combos)
        logger.info("Grid %d kombinasyondan %d tanesi seçildi (örnekleme)", len(all_combos), max_combos)

    total   = len(all_combos)
    results : List[OptResult] = []

    for idx, combo in enumerate(all_combos):
        params = dict(zip(keys, combo))

        if progress_cb:
            try:
                progress_cb(idx + 1, total)
            except Exception:
                pass

        try:
            r = _run_single(params, cfg_base, data_handler)
            results.append(r)
        except Exception as e:
            results.append(OptResult(
                params=params,
                total_return_pct=0, ann_return_pct=0,
                max_drawdown_pct=0, sharpe_ratio=0,
                sortino_ratio=0, calmar_ratio=0,
                profit_factor=0, num_trades=0,
                win_rate=0, score=0,
                error=str(e),
            ))

    # Hata olmayanları sırala
    valid = [r for r in results if r.error is None]
    valid.sort(key=lambda r: r.score, reverse=True)
    return valid[:top_n]


def _run_single(
    params      : Dict[str, Any],
    cfg_base    : dict,
    data_handler,
) -> OptResult:
    """Tek parametre seti için backtest çalıştır."""
    import copy
    from datetime import timezone
    UTC = timezone.utc

    from algotrading.backtest.commission  import FixedPerShare
    from algotrading.backtest.engine      import BacktestConfig, BacktestEngine
    from algotrading.backtest.portfolio   import Portfolio
    from algotrading.backtest.slippage    import VolatilitySlippage
    from algotrading.core.events          import EventBus
    from algotrading.execution.paper_trader import SimulatedExecution
    from algotrading.orchestrator.orchestrator import Orchestrator
    from algotrading.risk.kill_switch     import KillSwitch
    from algotrading.risk.position_sizer  import FixedFractional
    from algotrading.risk.risk_manager    import RiskManager
    from algotrading.strategies.trend_volatility import TrendVolatilityStrategy
    from algotrading.validation.metrics   import compute_metrics, compute_trade_metrics

    cfg = copy.deepcopy(cfg_base)
    bt  = cfg["backtest"]
    sp  = cfg["strategy"]["params"]
    rk  = cfg["risk"]
    cm  = cfg["commission"]
    sl  = cfg["slippage"]
    ks  = cfg["kill_switch"]
    oc  = cfg["orchestrator"]

    # Parametre enjeksiyonu
    for k, v in params.items():
        if k in ("ema_fast","ema_slow","atr_period","vol_threshold"):
            sp[k] = v
        elif k in ("max_risk_pct","max_position_pct","atr_stop_mult"):
            rk[k] = v

    symbol  = bt["symbols"][0]
    start   = datetime.fromisoformat(bt["start"]).replace(tzinfo=UTC)
    end     = datetime.fromisoformat(bt["end"]).replace(tzinfo=UTC)
    capital = float(bt["initial_capital"])

    bus       = EventBus()
    portfolio = Portfolio(initial_capital=capital)

    strategy = TrendVolatilityStrategy(
        data_handler  = data_handler,
        bus           = bus,
        symbol        = symbol,
        ema_fast      = int(sp["ema_fast"]),
        ema_slow      = int(sp["ema_slow"]),
        atr_period    = int(sp["atr_period"]),
        vol_threshold = float(sp["vol_threshold"]),
    )
    orchestrator = Orchestrator(
        bus           = bus,
        cooldown_bars = int(oc.get("cooldown_bars", 3)),
        min_confidence= float(oc.get("min_confidence", 0.0)),
        allow_short   = bool(oc.get("allow_short", False)),
    )
    kill_switch = KillSwitch(
        max_daily_loss_pct     = float(ks["max_daily_loss_pct"]),
        max_drawdown_pct       = float(ks["max_drawdown_pct"]),
        max_consecutive_losses = int(ks["max_consecutive_losses"]),
    )
    kill_switch.initialise(capital, at=start)

    risk_manager = RiskManager(
        portfolio          = portfolio,
        data_handler       = data_handler,
        bus                = bus,
        kill_switch        = kill_switch,
        position_sizer     = FixedFractional(
            max_risk_pct     = float(rk.get("max_risk_pct", 0.01)),
            atr_stop_mult    = float(rk.get("atr_stop_mult", 2.0)),
            max_position_pct = float(rk.get("max_position_pct", 0.10)),
        ),
        max_position_pct   = float(rk.get("max_position_pct", 0.10)),
        max_gross_exposure = float(rk.get("max_gross_exposure", 0.95)),
        max_net_exposure   = float(rk.get("max_net_exposure", 0.80)),
    )
    commission     = FixedPerShare(
        rate_per_share = float(cm["rate_per_share"]),
        min_per_order  = float(cm["min_per_order"]),
    )
    slippage_model = VolatilitySlippage(
        atr_multiple = float(sl["atr_multiple"]),
        min_bps      = float(sl["min_bps"]),
    )
    execution = SimulatedExecution(
        data_handler = data_handler,
        bus          = bus,
        commission   = commission,
        slippage     = slippage_model,
    )
    bc = BacktestConfig(
        name            = "opt_run",
        symbols         = [symbol],
        start           = start,
        end             = end,
        initial_capital = capital,
        random_seed     = int(cfg["system"].get("random_seed", 42)),
    )
    engine = BacktestEngine(
        config=bc, data_handler=data_handler,
        strategy=strategy, orchestrator=orchestrator,
        risk_manager=risk_manager, execution=execution,
        commission=commission, slippage=slippage_model,
        bus=bus, portfolio=portfolio,
    )
    result = engine.run()

    eq = result.equity_curve
    if len(eq) < 5:
        raise ValueError("Yeterli veri yok")

    m  = compute_metrics(eq["equity"], bars_per_year=int(cfg["data"].get("bars_per_year", 252)))
    tm = compute_trade_metrics(result.fills)

    score = composite_score(
        total_return_pct = m.total_return_pct,
        max_drawdown_pct = m.max_drawdown_pct,
        sharpe_ratio     = m.sharpe_ratio,
        num_trades       = tm.get("num_trades", 0),
        profit_factor    = tm.get("profit_factor", 1.0),
    )

    return OptResult(
        params           = params,
        total_return_pct = round(m.total_return_pct, 2),
        ann_return_pct   = round(m.annualised_return, 2),
        max_drawdown_pct = round(m.max_drawdown_pct, 2),
        sharpe_ratio     = round(m.sharpe_ratio, 3),
        sortino_ratio    = round(m.sortino_ratio, 3),
        calmar_ratio     = round(m.calmar_ratio, 3),
        profit_factor    = round(tm.get("profit_factor", 0.0), 3),
        num_trades       = tm.get("num_trades", 0),
        win_rate         = round(tm.get("win_rate", 0.0), 1),
        score            = score,
    )
