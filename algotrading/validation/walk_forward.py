"""
validation/walk_forward.py
───────────────────────────
Walk-Forward Analysis (WFA) and Nested (anchored) Walk-Forward.

Walk-forward validation simulates realistic deployment:
  - Train on in-sample window
  - Test on out-of-sample window (no overlap)
  - Roll forward
  - Report OOS performance only

This detects overfitting by comparing IS vs OOS Sharpe ratios.
A healthy strategy has OOS_Sharpe / IS_Sharpe > 0.5 ("efficiency ratio").

Nested WFA (double cross-validation):
  - Outer loop: WFA for final OOS results
  - Inner loop: CV within each IS window for parameter selection
  - Prevents optimisation bias in parameter selection
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from algotrading.validation.metrics import PerformanceReport, compute_metrics

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardWindow:
    fold_id       : int
    train_start   : datetime
    train_end     : datetime
    test_start    : datetime
    test_end      : datetime
    is_report     : Optional[PerformanceReport] = None
    oos_report    : Optional[PerformanceReport] = None
    params_used   : dict = field(default_factory=dict)

    @property
    def efficiency_ratio(self) -> float:
        """OOS Sharpe / IS Sharpe.  < 0.5 suggests overfitting."""
        if self.is_report is None or self.oos_report is None:
            return float("nan")
        if abs(self.is_report.sharpe_ratio) < 1e-9:
            return float("nan")
        return self.oos_report.sharpe_ratio / self.is_report.sharpe_ratio


@dataclass
class WalkForwardResult:
    windows        : List[WalkForwardWindow]
    combined_oos   : Optional[PerformanceReport] = None  # metrics on spliced OOS

    @property
    def avg_efficiency_ratio(self) -> float:
        ratios = [w.efficiency_ratio for w in self.windows
                  if not np.isnan(w.efficiency_ratio)]
        return float(np.mean(ratios)) if ratios else float("nan")

    @property
    def pct_windows_profitable(self) -> float:
        profitable = sum(
            1 for w in self.windows
            if w.oos_report and w.oos_report.total_return_pct > 0
        )
        return profitable / len(self.windows) * 100 if self.windows else 0.0


class WalkForwardValidator:
    """
    Runs walk-forward validation.

    RunFn signature:
        run_fn(train_data, test_data, params) -> (train_equity, test_equity)
        where equity is a pd.Series of portfolio values (indexed by bar).
    """

    def __init__(
        self,
        train_bars    : int = 252,    # IS window length in bars
        test_bars     : int = 63,     # OOS window length in bars
        step_bars     : int = None,   # roll step (defaults to test_bars)
        min_train     : int = 100,    # skip window if IS too short
        anchored      : bool = False, # if True: expanding IS window
    ) -> None:
        self.train_bars = train_bars
        self.test_bars  = test_bars
        self.step_bars  = step_bars or test_bars
        self.min_train  = min_train
        self.anchored   = anchored

    def split(self, n_bars: int) -> List[Tuple[slice, slice]]:
        """
        Generate (train_slice, test_slice) index pairs.
        """
        splits = []
        test_start = self.train_bars

        while test_start + self.test_bars <= n_bars:
            if self.anchored:
                train_slice = slice(0, test_start)
            else:
                train_slice = slice(max(0, test_start - self.train_bars), test_start)

            test_slice = slice(test_start, test_start + self.test_bars)
            splits.append((train_slice, test_slice))
            test_start += self.step_bars

        return splits

    def run(
        self,
        equity_fn   : Callable[[pd.DataFrame, dict], pd.Series],
        data        : pd.DataFrame,
        params      : dict,
        timestamps  : Optional[pd.Series] = None,
    ) -> WalkForwardResult:
        """
        equity_fn(slice_data, params) → pd.Series of equity values.

        data: DataFrame with at least a 'close' column.
        """
        splits  = self.split(len(data))
        windows = []

        all_oos_equity = []

        for fold_id, (train_sl, test_sl) in enumerate(splits):
            train_data = data.iloc[train_sl]
            test_data  = data.iloc[test_sl]

            if len(train_data) < self.min_train:
                logger.warning("Fold %d: IS too short (%d bars), skipping", fold_id, len(train_data))
                continue

            # Run strategy on IS
            try:
                is_equity = equity_fn(train_data, params)
                oos_equity = equity_fn(test_data, params)
            except Exception:
                logger.exception("Fold %d: equity_fn raised", fold_id)
                continue

            is_report  = compute_metrics(is_equity)  if len(is_equity)  > 5 else None
            oos_report = compute_metrics(oos_equity) if len(oos_equity) > 5 else None

            # Derive timestamps for window metadata
            ts_col = timestamps if timestamps is not None else data.index
            window = WalkForwardWindow(
                fold_id     = fold_id,
                train_start = _ts_at(ts_col, train_sl.start),
                train_end   = _ts_at(ts_col, train_sl.stop - 1),
                test_start  = _ts_at(ts_col, test_sl.start),
                test_end    = _ts_at(ts_col, test_sl.stop - 1),
                is_report   = is_report,
                oos_report  = oos_report,
                params_used = params,
            )
            windows.append(window)
            all_oos_equity.append(oos_equity)

            logger.info(
                "Fold %d | IS Sharpe=%.2f | OOS Sharpe=%.2f | Efficiency=%.2f",
                fold_id,
                is_report.sharpe_ratio  if is_report  else float("nan"),
                oos_report.sharpe_ratio if oos_report else float("nan"),
                window.efficiency_ratio,
            )

        # Splice OOS equity curves
        combined_oos = None
        if all_oos_equity:
            spliced = pd.concat(all_oos_equity, ignore_index=True)
            # Re-base to 1.0
            spliced = spliced / spliced.iloc[0]
            combined_oos = compute_metrics(spliced, num_trials=len(windows))

        result = WalkForwardResult(windows=windows, combined_oos=combined_oos)
        logger.info(
            "WFA complete | %d folds | avg_efficiency=%.2f | %% profitable_folds=%.0f%%",
            len(windows), result.avg_efficiency_ratio, result.pct_windows_profitable
        )
        return result


def _ts_at(ts_col, idx: int):
    try:
        return ts_col.iloc[idx]
    except Exception:
        return idx
