"""
orchestrator/orchestrator.py
─────────────────────────────
Decision Orchestrator — the system's brain.

Responsibilities:
1. Receive raw signals from one or more strategies
2. Resolve conflicts when multiple signals disagree (currently: single strategy)
3. Apply pre-trade filters (market regime, time-of-day, cooldown)
4. Produce a deterministic, fully-audited Decision
5. Forward Decision to the risk engine via the event bus

Design rules:
- DETERMINISTIC: same inputs always produce the same decision
- EXPLAINABLE: every decision carries a human-readable reason chain
- CONSERVATIVE: when in doubt, emit NO_TRADE
- NO SIDE EFFECTS: the orchestrator does not modify portfolio or orders directly
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from algotrading.core.events import (
    DecisionEvent, EventBus, SignalEvent
)
from algotrading.core.types import (
    Decision, DecisionAction, Direction, Signal, SignalStrength
)

logger = logging.getLogger(__name__)

UTC = timezone.utc


class Orchestrator:
    """
    Single-strategy orchestrator (MVP).
    Easily extensible to multi-strategy consensus.
    """

    def __init__(
        self,
        bus                : EventBus,
        cooldown_bars      : int   = 3,     # bars to wait after a trade
        min_confidence     : float = 0.0,   # minimum signal confidence to act
        allow_short        : bool  = True,
    ) -> None:
        self._bus           = bus
        self.cooldown_bars  = cooldown_bars
        self.min_confidence = min_confidence
        self.allow_short    = allow_short

        # State
        self._last_decision_ts  : Optional[datetime] = None
        self._bars_since_trade  : int = 0
        self._decision_log      : List[Decision] = []

    # ── Event handler (wired by engine) ───────────────────────────────────────

    def on_signal_event(self, event: SignalEvent) -> None:
        signal = event.signal
        decision = self._evaluate(signal)
        self._decision_log.append(decision)

        if decision.is_actionable():
            logger.info(
                "Orchestrator TRADE: %s %s | %s",
                decision.direction.value, decision.symbol, decision.reason
            )
            self._bars_since_trade = 0
        else:
            logger.debug(
                "Orchestrator NO_TRADE: %s | %s",
                decision.symbol, decision.reason
            )

        self._bus.publish(DecisionEvent(decision=decision))

    # ── Core evaluation logic ─────────────────────────────────────────────────

    def _evaluate(self, signal: Signal) -> Decision:
        """
        Produce a Decision from a Signal.
        Every branch must produce a complete Decision with a full reason.
        """
        ts = signal.timestamp

        # ── Gate 1: Cooldown ──────────────────────────────────────────────────
        if self._bars_since_trade < self.cooldown_bars and self._last_decision_ts is not None:
            return self._no_trade(
                signal,
                f"Cooldown: {self.cooldown_bars - self._bars_since_trade} bars remaining "
                f"since last trade @ {self._last_decision_ts}"
            )

        # ── Gate 2: Confidence floor ──────────────────────────────────────────
        if signal.confidence < self.min_confidence:
            return self._no_trade(
                signal,
                f"Confidence {signal.confidence:.2f} < threshold {self.min_confidence:.2f}"
            )

        # ── Gate 3: Short filter ───────────────────────────────────────────────
        if signal.direction == Direction.SHORT and not self.allow_short:
            return self._no_trade(signal, "Short selling disabled by config")

        # ── Gate 4: Flat/exit signals always pass ─────────────────────────────
        if signal.direction == Direction.FLAT:
            self._last_decision_ts = ts
            self._bars_since_trade = 0
            return Decision(
                timestamp       = ts,
                symbol          = signal.symbol,
                action          = DecisionAction.TRADE,
                direction       = Direction.FLAT,
                target_quantity = 0.0,
                strategy_id     = signal.strategy_id,
                reason          = f"Exit signal: {signal.reason}",
                risk_approved   = False,   # risk manager will set final
                signal_strength = signal.strength,
            )

        # ── Gate 5: Weak signal filter ────────────────────────────────────────
        if signal.strength == SignalStrength.WEAK and self.min_confidence > 0:
            return self._no_trade(signal, "Weak signal filtered out")

        # ── All gates passed → TRADE ──────────────────────────────────────────
        self._last_decision_ts  = ts
        self._bars_since_trade  = 0

        return Decision(
            timestamp       = ts,
            symbol          = signal.symbol,
            action          = DecisionAction.TRADE,
            direction       = signal.direction,
            target_quantity = 0.0,          # risk manager determines quantity
            strategy_id     = signal.strategy_id,
            reason          = (
                f"[{signal.strategy_id}] {signal.reason} "
                f"| confidence={signal.confidence:.2f} strength={signal.strength.value}"
            ),
            risk_approved   = False,        # pending risk manager review
            signal_strength = signal.strength,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _no_trade(self, signal: Signal, reason: str) -> Decision:
        self._bars_since_trade += 1
        return Decision(
            timestamp       = signal.timestamp,
            symbol          = signal.symbol,
            action          = DecisionAction.NO_TRADE,
            direction       = signal.direction,
            target_quantity = 0.0,
            strategy_id     = signal.strategy_id,
            reason          = reason,
            risk_approved   = False,
            signal_strength = signal.strength,
        )

    # ── Audit ─────────────────────────────────────────────────────────────────

    def decision_log(self) -> List[Decision]:
        return list(self._decision_log)

    def summary(self) -> dict:
        trades   = sum(1 for d in self._decision_log if d.action == DecisionAction.TRADE)
        no_trade = sum(1 for d in self._decision_log if d.action == DecisionAction.NO_TRADE)
        return {
            "total_decisions"  : len(self._decision_log),
            "trade_decisions"  : trades,
            "no_trade_decisions": no_trade,
            "trade_rate_pct"   : round(trades / max(len(self._decision_log), 1) * 100, 1),
        }

    def reset(self) -> None:
        self._last_decision_ts = None
        self._bars_since_trade = 0
        self._decision_log.clear()
