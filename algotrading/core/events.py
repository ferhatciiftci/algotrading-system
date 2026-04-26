"""
core/events.py
──────────────
Event types and a simple synchronous event bus.

Event flow (backtest & live):

  DataFeed  ──MarketEvent──►  Strategy  ──SignalEvent──►  Orchestrator
  Orchestrator ──OrderEvent──►  RiskEngine  ──OrderEvent──►  Execution
  Execution ──FillEvent──►  Portfolio / Monitoring
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Type

from algotrading.core.types import Bar, Decision, Fill, Order, Signal

logger = logging.getLogger(__name__)


# ─── Event types ──────────────────────────────────────────────────────────────

class EventType(Enum):
    MARKET   = "MARKET"    # new bar arrived
    SIGNAL   = "SIGNAL"    # strategy produced a signal
    DECISION = "DECISION"  # orchestrator made a trade/no-trade decision
    ORDER    = "ORDER"     # risk-approved order ready for execution
    FILL     = "FILL"      # execution confirmed a fill
    HALT     = "HALT"      # kill switch / circuit breaker fired


@dataclass(frozen=True)
class MarketEvent:
    type : EventType = field(default=EventType.MARKET, init=False)
    bar  : Bar


@dataclass(frozen=True)
class SignalEvent:
    type   : EventType = field(default=EventType.SIGNAL, init=False)
    signal : Signal


@dataclass(frozen=True)
class DecisionEvent:
    type     : EventType = field(default=EventType.DECISION, init=False)
    decision : Decision


@dataclass(frozen=True)
class OrderEvent:
    type  : EventType = field(default=EventType.ORDER, init=False)
    order : Order


@dataclass(frozen=True)
class FillEvent:
    type : EventType = field(default=EventType.FILL, init=False)
    fill : Fill


@dataclass(frozen=True)
class HaltEvent:
    type   : EventType = field(default=EventType.HALT, init=False)
    reason : str
    timestamp : datetime


# Union alias for type hints
AnyEvent = (
    MarketEvent | SignalEvent | DecisionEvent | OrderEvent | FillEvent | HaltEvent
)

# ─── Event bus ────────────────────────────────────────────────────────────────

Handler = Callable[[AnyEvent], None]


class EventBus:
    """
    Simple synchronous pub/sub bus.
    Handlers are called in registration order.
    Exceptions in handlers are logged but do NOT stop other handlers.
    """

    def __init__(self) -> None:
        self._handlers: Dict[EventType, List[Handler]] = {et: [] for et in EventType}
        self._halted: bool = False

    def subscribe(self, event_type: EventType, handler: Handler) -> None:
        self._handlers[event_type].append(handler)

    def publish(self, event: AnyEvent) -> None:
        if self._halted and event.type != EventType.HALT:
            logger.warning("Bus halted – dropping %s", event.type)
            return
        handlers = self._handlers.get(event.type, [])
        for h in handlers:
            try:
                h(event)
            except Exception:
                logger.exception("Handler %s raised for event %s", h, event.type)

    def halt(self, reason: str, timestamp: datetime) -> None:
        """Publish a HALT event and freeze the bus."""
        self._halted = True
        halt_ev = HaltEvent(reason=reason, timestamp=timestamp)
        for h in self._handlers[EventType.HALT]:
            try:
                h(halt_ev)
            except Exception:
                logger.exception("HALT handler raised")

    @property
    def is_halted(self) -> bool:
        return self._halted

    def reset(self) -> None:
        """Used between backtest runs — clears halt state."""
        self._halted = False
