"""
risk/kill_switch.py
────────────────────
The kill switch is the system's emergency stop.

It triggers when any of these conditions are met:
  1. Daily loss exceeds max_daily_loss_pct of starting equity
  2. Intraday drawdown from peak exceeds max_drawdown_pct
  3. Consecutive losing trades exceed max_consecutive_losses
  4. External manual trigger (for live operation)

Once triggered, the kill switch is LATCHING — it cannot be reset without
an explicit human action (call reset_manual() with a confirmation token).
This prevents accidental re-enabling.
"""

from __future__ import annotations

import hashlib
import logging
import secrets
from datetime import date, datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

UTC = timezone.utc


class KillSwitch:

    def __init__(
        self,
        max_daily_loss_pct     : float = 0.03,   # 3% daily loss limit
        max_drawdown_pct       : float = 0.10,   # 10% peak-to-trough
        max_consecutive_losses : int   = 5,
    ) -> None:
        self.max_daily_loss_pct     = max_daily_loss_pct
        self.max_drawdown_pct       = max_drawdown_pct
        self.max_consecutive_losses = max_consecutive_losses

        self._triggered         : bool              = False
        self._trigger_reason    : Optional[str]     = None
        self._trigger_time      : Optional[datetime] = None
        self._reset_token       : Optional[str]     = None  # must be supplied to reset

        # Tracking state
        self._peak_equity       : float = 0.0
        self._day_start_equity  : float = 0.0
        self._current_day       : Optional[date] = None
        self._consecutive_losses: int = 0

    # ── Main check (called after every mark-to-market) ────────────────────────

    def check(
        self,
        equity    : float,
        timestamp : datetime,
    ) -> tuple[bool, str]:
        """
        Returns (triggered, reason).  If triggered for the first time,
        latches and returns True with a reason string.
        """
        if self._triggered:
            return True, self._trigger_reason

        # Initialise day tracking
        today = timestamp.date()
        if self._current_day != today:
            self._current_day      = today
            self._day_start_equity = equity

        # 1. Daily loss check
        if self._day_start_equity > 0:
            daily_loss_pct = (self._day_start_equity - equity) / self._day_start_equity
            if daily_loss_pct > self.max_daily_loss_pct:
                return self._fire(
                    f"Daily loss {daily_loss_pct:.2%} exceeds limit {self.max_daily_loss_pct:.2%}",
                    timestamp,
                )

        # 2. Drawdown from peak
        self._peak_equity = max(self._peak_equity, equity)
        if self._peak_equity > 0:
            dd = (self._peak_equity - equity) / self._peak_equity
            if dd > self.max_drawdown_pct:
                return self._fire(
                    f"Drawdown {dd:.2%} exceeds limit {self.max_drawdown_pct:.2%} "
                    f"(peak={self._peak_equity:.0f}, current={equity:.0f})",
                    timestamp,
                )

        # 3. Consecutive losses (updated by record_trade_result)
        if self._consecutive_losses >= self.max_consecutive_losses:
            return self._fire(
                f"Consecutive losing trades: {self._consecutive_losses} "
                f">= limit {self.max_consecutive_losses}",
                timestamp,
            )

        return False, ""

    def record_trade_result(self, pnl: float) -> None:
        """Call after each closed trade to track consecutive losses."""
        if pnl < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

    def trigger_manual(self, reason: str, timestamp: Optional[datetime] = None) -> str:
        """
        Manually trigger the kill switch (operator use).
        Returns the reset token that must be supplied to re-enable.
        """
        ts    = timestamp or datetime.now(UTC)
        token = self._fire(f"MANUAL: {reason}", ts)
        return self._reset_token or ""

    def reset_manual(self, token: str, initial_equity: float) -> bool:
        """
        Re-enable the kill switch.  Requires the token issued at trigger time.
        Returns True if reset was successful.
        """
        if not self._triggered:
            return True
        if not self._reset_token:
            return False
        expected = hashlib.sha256(self._reset_token.encode()).hexdigest()
        supplied = hashlib.sha256(token.encode()).hexdigest()
        if secrets.compare_digest(expected, supplied):
            logger.warning("Kill switch RESET by operator")
            self._triggered          = False
            self._trigger_reason     = None
            self._trigger_time       = None
            self._reset_token        = None
            self._consecutive_losses = 0
            self._peak_equity        = initial_equity
            self._day_start_equity   = initial_equity
            return True
        logger.error("Kill switch reset attempt with INVALID token")
        return False

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_triggered(self) -> bool:
        return self._triggered

    @property
    def trigger_reason(self) -> Optional[str]:
        return self._trigger_reason

    def initialise(self, equity: float, at: "datetime | None" = None) -> None:
        """Call at strategy start / after reset with current equity."""
        self._peak_equity      = equity
        self._day_start_equity = equity
        # Prime the current day so the first check() does not overwrite day_start
        if at is not None:
            self._current_day = at.date()

    # ── Internals ─────────────────────────────────────────────────────────────

    def _fire(self, reason: str, timestamp: datetime) -> tuple[bool, str]:
        if not self._triggered:
            self._triggered      = True
            self._trigger_reason = reason
            self._trigger_time   = timestamp
            self._reset_token    = secrets.token_hex(16)
            logger.critical(
                "KILL SWITCH TRIGGERED @ %s | reason: %s | reset_token: %s",
                timestamp, reason, self._reset_token
            )
        return True, self._trigger_reason
