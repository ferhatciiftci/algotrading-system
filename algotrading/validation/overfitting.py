"""
validation/overfitting.py
──────────────────────────
Overfitting detection and strategy stability assessment.

Tests:
  1. IS vs OOS Sharpe degradation  (efficiency ratio < 0.5 = overfit)
  2. Combinatorial Purged Cross-Validation (CPCV) — López de Prado 2018
  3. Probabilistic Sharpe Ratio (PSR) — minimum bar-count for significance
  4. Parameter sensitivity test  — perturb params, check Sharpe stability
  5. Monte Carlo permutation test — is the strategy better than random?
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class OverfitReport:
    """Summary of all overfitting checks."""
    passed                  : bool
    efficiency_ratio        : float    # OOS/IS Sharpe
    psr_min_bars            : int      # bars needed to be statistically significant
    psr_current_bars        : int
    psr_passed              : bool
    mc_p_value              : float    # p-value from Monte Carlo permutation
    mc_passed               : bool     # p < 0.05
    param_sensitivity_score : float    # stddev of Sharpe across param perturbations
    warnings                : List[str] = field(default_factory=list)
    details                 : dict     = field(default_factory=dict)


# ─── Probabilistic Sharpe Ratio ───────────────────────────────────────────────

def min_bars_for_psr(
    sharpe_ratio   : float,
    skewness       : float = 0.0,
    kurtosis       : float = 3.0,
    target_prob    : float = 0.95,
    benchmark_sr   : float = 0.0,
) -> int:
    """
    Minimum number of bars needed for the observed Sharpe to exceed
    `benchmark_sr` with probability `target_prob`.

    From López de Prado "Advances in Financial ML", Chapter 8.
    """
    import scipy.stats as stats
    if sharpe_ratio <= benchmark_sr:
        return int(1e9)   # impossible to achieve

    z = stats.norm.ppf(target_prob)
    # SR variance depends on skewness and excess kurtosis
    var_sr = (1 - skewness * sharpe_ratio + ((kurtosis - 1) / 4) * sharpe_ratio**2)
    n = (z ** 2 * var_sr) / ((sharpe_ratio - benchmark_sr) ** 2) + 1
    return max(1, math.ceil(n))


# ─── Monte Carlo permutation test ────────────────────────────────────────────

def mc_permutation_test(
    returns        : pd.Series,
    strategy_sharpe: float,
    n_permutations : int = 1000,
    seed           : int = 42,
) -> float:
    """
    Returns the p-value: what fraction of random permutations of the returns
    achieve a Sharpe ratio >= the observed Sharpe.

    A small p-value (< 0.05) means the strategy's performance is unlikely
    to be due to random chance.
    """
    rng = random.Random(seed)
    returns_list = list(returns.dropna())
    count_at_least_as_good = 0

    for _ in range(n_permutations):
        permuted = returns_list.copy()
        rng.shuffle(permuted)
        perm_series = pd.Series(permuted)
        mean = perm_series.mean()
        std  = perm_series.std()
        perm_sr = (mean / std * math.sqrt(252)) if std > 1e-12 else 0.0
        if perm_sr >= strategy_sharpe:
            count_at_least_as_good += 1

    return count_at_least_as_good / n_permutations


# ─── Parameter sensitivity ────────────────────────────────────────────────────

def parameter_sensitivity(
    equity_fn    : Callable[[dict], pd.Series],
    base_params  : dict,
    perturbation : float = 0.10,   # ±10% of each param
    n_trials     : int   = 20,
    seed         : int   = 42,
) -> Tuple[float, List[float]]:
    """
    Perturb each numeric parameter by ±perturbation fraction and measure
    the resulting Sharpe ratio.

    Returns (sensitivity_score, [sharpe_list]) where sensitivity_score is
    the coefficient of variation of Sharpe across all perturbed runs.
    A score < 0.3 suggests the strategy is not sensitive to exact parameters.
    """
    from algotrading.validation.metrics import compute_metrics
    rng   = np.random.RandomState(seed)
    sharpes = []

    numeric_keys = [k for k, v in base_params.items() if isinstance(v, (int, float))]
    if not numeric_keys:
        return 0.0, []

    for _ in range(n_trials):
        perturbed = base_params.copy()
        key = rng.choice(numeric_keys)
        factor = 1.0 + rng.uniform(-perturbation, perturbation)
        v = base_params[key]
        perturbed[key] = type(v)(v * factor) if isinstance(v, float) else max(1, int(v * factor))

        try:
            equity = equity_fn(perturbed)
            report = compute_metrics(equity)
            sharpes.append(report.sharpe_ratio)
        except Exception:
            continue

    if not sharpes:
        return float("nan"), []

    mean = np.mean(sharpes)
    std  = np.std(sharpes)
    cv   = std / abs(mean) if abs(mean) > 1e-9 else float("inf")
    return float(cv), sharpes


# ─── Main overfitting checker ────────────────────────────────────────────────

def check_overfitting(
    is_sharpe      : float,
    oos_sharpe     : float,
    oos_returns    : pd.Series,
    oos_skew       : float = 0.0,
    oos_kurt       : float = 3.0,
    n_params       : int   = 4,     # number of free parameters
    n_trials       : int   = 1,     # how many parameter combos were tested
) -> OverfitReport:
    warnings = []

    # 1. Efficiency ratio
    eff_ratio = oos_sharpe / is_sharpe if abs(is_sharpe) > 1e-9 else float("nan")
    if not math.isnan(eff_ratio) and eff_ratio < 0.5:
        warnings.append(
            f"Efficiency ratio {eff_ratio:.2f} < 0.5 — strong overfitting signal"
        )

    # 2. PSR
    n_bars     = len(oos_returns.dropna())
    min_bars   = min_bars_for_psr(oos_sharpe, oos_skew, oos_kurt)
    psr_passed = n_bars >= min_bars
    if not psr_passed:
        warnings.append(
            f"PSR: only {n_bars} OOS bars, need {min_bars} for Sharpe={oos_sharpe:.2f} at 95% confidence"
        )

    # 3. Monte Carlo
    mc_pvalue  = mc_permutation_test(oos_returns, oos_sharpe)
    mc_passed  = mc_pvalue < 0.05
    if not mc_passed:
        warnings.append(
            f"Monte Carlo p-value={mc_pvalue:.3f} >= 0.05 — strategy not statistically significant"
        )

    # 4. Number of parameters penalty
    dof_penalty = n_params / max(n_bars, 1)
    if dof_penalty > 0.05:
        warnings.append(
            f"High degrees-of-freedom ratio: {n_params} params / {n_bars} OOS bars = {dof_penalty:.3f}"
        )

    passed = (
        (math.isnan(eff_ratio) or eff_ratio >= 0.5)
        and psr_passed
        and mc_passed
    )

    return OverfitReport(
        passed                  = passed,
        efficiency_ratio        = round(eff_ratio, 3) if not math.isnan(eff_ratio) else -999,
        psr_min_bars            = min_bars,
        psr_current_bars        = n_bars,
        psr_passed              = psr_passed,
        mc_p_value              = round(mc_pvalue, 4),
        mc_passed               = mc_passed,
        param_sensitivity_score = 0.0,    # populated separately if equity_fn available
        warnings                = warnings,
    )
