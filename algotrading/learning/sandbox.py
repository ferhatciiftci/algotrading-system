"""
learning/sandbox.py
────────────────────
Safe Learning Loop — sandbox retraining environment.

PRINCIPLE: No model is EVER retrained while live capital is at risk.
The loop is:

  1. Copy live data to sandbox (read-only snapshot)
  2. Retrain / re-optimise strategy parameters in isolation
  3. Run full validation (WFA + overfitting checks) on sandbox
  4. If validation passes → produce candidate config
  5. Human review + sign-off required before canary deployment
  6. Canary deployment (small allocation) → monitor for N days
  7. If canary passes → promote to full deployment

This module implements steps 1–4.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from algotrading.validation.metrics import PerformanceReport, compute_metrics
from algotrading.validation.walk_forward import WalkForwardValidator, WalkForwardResult
from algotrading.validation.overfitting import check_overfitting, OverfitReport

logger = logging.getLogger(__name__)

UTC = timezone.utc


@dataclass
class CandidateConfig:
    """
    A proposed new strategy configuration produced by the sandbox.
    Must pass human review before promotion.
    """
    strategy_id     : str
    params          : dict
    created_at      : datetime
    wfa_result      : WalkForwardResult
    overfit_report  : OverfitReport
    fingerprint     : str   = ""      # hash of params + data slice
    approved        : bool  = False   # set by human operator
    approved_by     : str   = ""
    approved_at     : Optional[datetime] = None

    def __post_init__(self) -> None:
        if not self.fingerprint:
            raw = json.dumps(self.params, sort_keys=True, default=str)
            self.fingerprint = hashlib.sha256(raw.encode()).hexdigest()[:16]

    @property
    def is_promotable(self) -> bool:
        """Config can only be promoted after human approval AND validation pass."""
        return (
            self.approved
            and self.overfit_report.passed
            and self.wfa_result.combined_oos is not None
            and self.wfa_result.combined_oos.sharpe_ratio > 0.5
        )


class Sandbox:
    """
    Isolated environment for strategy research and retraining.

    Never touches live positions or the live data store directly.
    Operates on a frozen snapshot of data.
    """

    def __init__(
        self,
        output_dir     : Path,
        equity_fn      : Callable[[pd.DataFrame, dict], pd.Series],
        strategy_id    : str,
        wfa_train_bars : int = 252,
        wfa_test_bars  : int = 63,
    ) -> None:
        self.output_dir  = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._equity_fn  = equity_fn
        self.strategy_id = strategy_id

        self._validator = WalkForwardValidator(
            train_bars = wfa_train_bars,
            test_bars  = wfa_test_bars,
        )

    def evaluate(
        self,
        data        : pd.DataFrame,
        params      : dict,
        data_label  : str = "default",
    ) -> CandidateConfig:
        """
        Run full validation on a candidate parameter set.
        Returns a CandidateConfig — NOT yet promotable without human approval.
        """
        logger.info(
            "Sandbox evaluating strategy=%s params=%s on %d bars",
            self.strategy_id, params, len(data)
        )

        # Walk-forward validation
        wfa_result = self._validator.run(
            equity_fn = self._equity_fn,
            data      = data,
            params    = params,
        )

        # Overfitting check on combined OOS
        if wfa_result.combined_oos and wfa_result.windows:
            # Splice OOS returns from all windows
            oos_returns = self._splice_oos_returns(data, wfa_result)
            last_window = wfa_result.windows[-1]
            is_sharpe   = last_window.is_report.sharpe_ratio if last_window.is_report else 0.0
            oos_sharpe  = wfa_result.combined_oos.sharpe_ratio

            overfit = check_overfitting(
                is_sharpe   = is_sharpe,
                oos_sharpe  = oos_sharpe,
                oos_returns = oos_returns,
                n_params    = len(params),
                n_trials    = 1,
            )
        else:
            from algotrading.validation.overfitting import OverfitReport
            overfit = OverfitReport(
                passed=False, efficiency_ratio=0.0,
                psr_min_bars=9999, psr_current_bars=0, psr_passed=False,
                mc_p_value=1.0, mc_passed=False, param_sensitivity_score=0.0,
                warnings=["Insufficient OOS data for overfitting checks"],
            )

        candidate = CandidateConfig(
            strategy_id  = self.strategy_id,
            params       = copy.deepcopy(params),
            created_at   = datetime.now(UTC),
            wfa_result   = wfa_result,
            overfit_report = overfit,
        )

        self._persist(candidate)
        return candidate

    def approve(self, candidate: CandidateConfig, approved_by: str) -> CandidateConfig:
        """
        Human approval gate.  Sets approved=True and records operator identity.
        In production this would require a 2FA token or audit-signed approval.
        """
        candidate.approved    = True
        candidate.approved_by = approved_by
        candidate.approved_at = datetime.now(UTC)
        logger.warning(
            "Candidate %s APPROVED by %s — eligible for canary deployment",
            candidate.fingerprint, approved_by
        )
        return candidate

    # ── Internals ─────────────────────────────────────────────────────────────

    def _splice_oos_returns(
        self,
        data       : pd.DataFrame,
        wfa_result : WalkForwardResult,
    ) -> pd.Series:
        """Combine OOS returns from all walk-forward windows."""
        all_parts = []
        splits = self._validator.split(len(data))
        for (_, test_sl), window in zip(splits, wfa_result.windows):
            test_data = data.iloc[test_sl]
            try:
                eq = self._equity_fn(test_data, window.params_used)
                all_parts.append(eq.pct_change().dropna())
            except Exception:
                pass
        if not all_parts:
            return pd.Series(dtype=float)
        return pd.concat(all_parts, ignore_index=True)

    def _persist(self, candidate: CandidateConfig) -> None:
        path = self.output_dir / f"candidate_{candidate.fingerprint}.json"
        data = {
            "strategy_id"   : candidate.strategy_id,
            "params"        : candidate.params,
            "created_at"    : candidate.created_at.isoformat(),
            "fingerprint"   : candidate.fingerprint,
            "overfit_passed": candidate.overfit_report.passed,
            "oos_sharpe"    : (candidate.wfa_result.combined_oos.sharpe_ratio
                               if candidate.wfa_result.combined_oos else None),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Candidate saved to %s", path)
