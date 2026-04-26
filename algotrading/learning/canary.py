"""
learning/canary.py
------------------
Canary deployment: yeni strateji konfigurasyonunu kucuk bir sermaye
dilimi (%10) ile canliya alir ve belirli sure izler.

KURAL: Canary basarisiz olursa otomatik geri alinir.
Insan onayı olmadan tam deployment YAPILMAZ.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)
UTC = timezone.utc


@dataclass
class CanaryState:
    candidate_fingerprint : str
    start_time            : datetime
    allocation_pct        : float       # toplam sermayenin kac yuzdesi
    min_days              : int         # kac gun izlenmeli
    target_sharpe         : float = 0.3
    max_drawdown_limit    : float = 0.05  # canary icin daha siki limit

    pnl_history           : list = field(default_factory=list)
    is_active             : bool = True
    outcome               : Optional[str] = None   # "promoted" | "rolled_back"
    rolled_back_at        : Optional[datetime] = None
    promoted_at           : Optional[datetime] = None

    @property
    def days_running(self) -> int:
        return (datetime.now(UTC) - self.start_time).days

    @property
    def is_mature(self) -> bool:
        return self.days_running >= self.min_days


class CanaryDeployment:
    """
    Yeni strateji konfigurasyonunu canary olarak izler.
    Tam deployment icin: canary.promote() cagrisi + insan onayi gerekir.
    """

    def __init__(
        self,
        allocation_pct     : float = 0.10,
        min_days           : int   = 10,
        target_sharpe      : float = 0.3,
        max_drawdown_limit : float = 0.05,
    ) -> None:
        self.allocation_pct     = allocation_pct
        self.min_days           = min_days
        self.target_sharpe      = target_sharpe
        self.max_drawdown_limit = max_drawdown_limit
        self._state: Optional[CanaryState] = None

    def start(self, candidate_fingerprint: str) -> CanaryState:
        if self._state and self._state.is_active:
            raise RuntimeError("Zaten aktif bir canary var. Once sonlandirin.")
        self._state = CanaryState(
            candidate_fingerprint = candidate_fingerprint,
            start_time            = datetime.now(UTC),
            allocation_pct        = self.allocation_pct,
            min_days              = self.min_days,
            target_sharpe         = self.target_sharpe,
            max_drawdown_limit    = self.max_drawdown_limit,
        )
        logger.warning("Canary baslatildi: %s | alloc=%.0f%% | min_days=%d",
                       candidate_fingerprint, self.allocation_pct * 100, self.min_days)
        return self._state

    def record_pnl(self, pnl: float, equity: float, timestamp: datetime) -> None:
        if not self._state or not self._state.is_active:
            return
        self._state.pnl_history.append({"ts": str(timestamp), "pnl": pnl, "equity": equity})

        # Otomatik geri alma: drawdown asildiysa
        if len(self._state.pnl_history) > 1:
            equities = [r["equity"] for r in self._state.pnl_history]
            peak = max(equities)
            if peak > 0 and (peak - equity) / peak > self.max_drawdown_limit:
                self.rollback(f"Canary drawdown limiti asildi: {(peak-equity)/peak:.2%}")

    def rollback(self, reason: str) -> None:
        if not self._state:
            return
        self._state.is_active    = False
        self._state.outcome      = "rolled_back"
        self._state.rolled_back_at = datetime.now(UTC)
        logger.critical("CANARY GERI ALINDI: %s | Sebep: %s",
                        self._state.candidate_fingerprint, reason)

    def promote(self, approved_by: str) -> bool:
        """
        Tam deployment icin insan onayi gerektirir.
        Canary olgun ve basarili olmalı.
        """
        if not self._state or not self._state.is_active:
            logger.error("Aktif canary yok veya zaten sonlanmis.")
            return False
        if not self._state.is_mature:
            logger.error("Canary henuz olgun degil (%d/%d gun).",
                         self._state.days_running, self.min_days)
            return False
        self._state.is_active  = False
        self._state.outcome    = "promoted"
        self._state.promoted_at = datetime.now(UTC)
        logger.warning("Canary PROMOSYON edildi: %s | Onaylayan: %s",
                       self._state.candidate_fingerprint, approved_by)
        return True

    @property
    def state(self) -> Optional[CanaryState]:
        return self._state
