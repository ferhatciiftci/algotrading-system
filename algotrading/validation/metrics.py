"""
validation/metrics.py
──────────────────────
Performance and risk metrics computed from an equity curve.

All metrics are computed from a daily (or bar-frequency) returns series.
No in-place modification of input data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class PerformanceReport:
    # Return metrics
    total_return_pct    : float
    annualised_return   : float
    # Risk metrics
    annualised_vol      : float
    sharpe_ratio        : float
    sortino_ratio       : float
    calmar_ratio        : float
    max_drawdown_pct    : float
    avg_drawdown_pct    : float
    max_drawdown_duration: int    # bars
    # Trade metrics
    num_trades          : int
    win_rate            : float
    avg_win_pct         : float
    avg_loss_pct        : float
    profit_factor       : float
    # Stability
    stability_r2        : float   # R² of log-equity vs time (1 = perfectly smooth)
    skewness            : float
    kurtosis            : float
    # Overfitting proxies
    deflated_sharpe     : float   # Sharpe adjusted for multiple trials
    omega_ratio         : float


def compute_metrics(
    equity_curve       : pd.Series,     # indexed by timestamp or int
    returns            : Optional[pd.Series] = None,
    bars_per_year      : int = 252,
    risk_free_rate_ann : float = 0.04,
    num_trials         : int = 1,       # for deflated Sharpe
) -> PerformanceReport:
    """
    Compute a full performance report from an equity curve.

    equity_curve: Series of equity values (e.g. portfolio value over time)
    """
    if len(equity_curve) < 5:
        raise ValueError("Need at least 5 data points for metrics")

    eq = equity_curve.dropna().reset_index(drop=True)
    if returns is None:
        returns = eq.pct_change().dropna()
    else:
        returns = returns.dropna()

    n   = len(returns)
    rfr = risk_free_rate_ann / bars_per_year   # per-bar risk-free rate

    # ── Return ────────────────────────────────────────────────────────────────
    total_return  = (eq.iloc[-1] / eq.iloc[0]) - 1.0
    ann_return    = (1 + total_return) ** (bars_per_year / len(eq)) - 1.0

    # ── Volatility ────────────────────────────────────────────────────────────
    ann_vol = returns.std() * math.sqrt(bars_per_year)

    # ── Sharpe ────────────────────────────────────────────────────────────────
    excess = returns - rfr
    sharpe = (excess.mean() / excess.std() * math.sqrt(bars_per_year)
              if excess.std() > 1e-12 else 0.0)

    # ── Sortino ───────────────────────────────────────────────────────────────
    downside = returns[returns < rfr]
    downside_vol = downside.std() * math.sqrt(bars_per_year) if len(downside) > 1 else 1e-9
    sortino = (ann_return - risk_free_rate_ann) / downside_vol if downside_vol > 0 else 0.0

    # ── Drawdown ──────────────────────────────────────────────────────────────
    running_max  = eq.cummax()
    drawdown     = (eq - running_max) / running_max
    max_dd       = drawdown.min()
    avg_dd       = drawdown[drawdown < 0].mean() if (drawdown < 0).any() else 0.0

    # Drawdown duration
    in_dd          = drawdown < 0
    dd_duration    = 0
    max_dd_dur     = 0
    for v in in_dd:
        if v:
            dd_duration += 1
            max_dd_dur   = max(max_dd_dur, dd_duration)
        else:
            dd_duration  = 0

    # ── Calmar ────────────────────────────────────────────────────────────────
    calmar = ann_return / abs(max_dd) if abs(max_dd) > 1e-9 else 0.0

    # ── Omega ─────────────────────────────────────────────────────────────────
    gains  = returns[returns > rfr] - rfr
    losses = rfr - returns[returns <= rfr]
    omega  = gains.sum() / losses.sum() if losses.sum() > 1e-9 else float("inf")

    # ── Stability (R² of log-equity) ──────────────────────────────────────────
    log_eq = np.log(eq.clip(lower=1e-9))
    x      = np.arange(len(log_eq))
    corr   = np.corrcoef(x, log_eq)[0, 1]
    r2     = corr ** 2

    # ── Skew / Kurt ───────────────────────────────────────────────────────────
    skew = float(returns.skew())
    kurt = float(returns.kurtosis())

    # ── Deflated Sharpe (Bailey & López de Prado 2014) ────────────────────────
    # Adjusts for the probability of selecting a lucky Sharpe from num_trials
    if num_trials > 1 and n > 4:
        import scipy.stats as stats
        # Expected maximum Sharpe under null (all trials independent)
        e_max_sr = _expected_max_sharpe(num_trials, n)
        var_sr   = (1 - skew * sharpe + (kurt / 4) * sharpe**2) / (n - 1)
        deflated = (sharpe - e_max_sr) / math.sqrt(var_sr) if var_sr > 0 else sharpe
        deflated_sharpe = float(stats.norm.cdf(deflated))
    else:
        deflated_sharpe = 1.0

    return PerformanceReport(
        total_return_pct      = round(total_return * 100, 3),
        annualised_return     = round(ann_return * 100, 3),
        annualised_vol        = round(ann_vol * 100, 3),
        sharpe_ratio          = round(sharpe, 3),
        sortino_ratio         = round(sortino, 3),
        calmar_ratio          = round(calmar, 3),
        max_drawdown_pct      = round(max_dd * 100, 3),
        avg_drawdown_pct      = round(avg_dd * 100, 3),
        max_drawdown_duration = max_dd_dur,
        num_trades            = 0,        # populated by caller if available
        win_rate              = 0.0,      # populated by caller
        avg_win_pct           = 0.0,
        avg_loss_pct          = 0.0,
        profit_factor         = 0.0,
        stability_r2          = round(r2, 4),
        skewness              = round(skew, 4),
        kurtosis              = round(kurt, 4),
        deflated_sharpe       = round(deflated_sharpe, 4),
        omega_ratio           = round(omega, 3),
    )


def compute_trade_metrics(fills) -> dict:
    """Compute win rate, profit factor, and per-trade stats from fill list."""
    if not fills:
        return {"num_trades": 0, "win_rate": 0.0, "profit_factor": 0.0}

    pnls = []
    for i in range(0, len(fills) - 1, 2):
        try:
            entry = fills[i]
            exit_ = fills[i + 1]
            if entry.direction.value == "LONG":
                pnl = (exit_.fill_price - entry.fill_price) * entry.quantity
            else:
                pnl = (entry.fill_price - exit_.fill_price) * entry.quantity
            pnl -= entry.commission + exit_.commission
            pnls.append(pnl)
        except (IndexError, AttributeError):
            continue

    if not pnls:
        return {"num_trades": 0, "win_rate": 0.0, "profit_factor": 0.0}

    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    pf     = sum(wins) / abs(sum(losses)) if losses else float("inf")

    return {
        "num_trades"  : len(pnls),
        "win_rate"    : round(len(wins) / len(pnls) * 100, 1),
        "avg_win_pct" : round(sum(wins) / len(wins) if wins else 0, 2),
        "avg_loss_pct": round(sum(losses) / len(losses) if losses else 0, 2),
        "profit_factor": round(pf, 3),
    }


def _expected_max_sharpe(n_trials: int, n_obs: int) -> float:
    """E[max Sharpe | n_trials iid trials, each with n_obs observations]."""
    import scipy.stats as stats
    e = (1 - 0.5772156649) / math.log(n_trials) + 0.5772156649 / math.log(n_trials)
    # Bailey-de Prado approximation
    z = stats.norm.ppf(1 - 1.0 / n_trials)
    return z
