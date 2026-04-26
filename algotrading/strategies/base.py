"""
strategies/base.py
──────────────────
Abstract base class for all strategies.

Every concrete strategy MUST:
1. Implement on_bar() — pure signal logic, no I/O
2. Provide a STRATEGY_ID class constant
3. Return a Signal (or None) — no side effects
4. Be stateless between on_bar calls except for its own indicator buffers

Strategies do NOT:
- Know about orders, fills, or portfolio state
- Communicate directly with the execution layer
- Hold mutable external state
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from algotrading.core.events import EventBus, MarketEvent, SignalEvent
from algotrading.core.types import Bar, Signal
from algotrading.data.pit_handler import PITDataHandler


class BaseStrategy(ABC):

    # Must be overridden in each subclass
    STRATEGY_ID: str = "base"

    def __init__(self, data_handler: PITDataHandler, bus: EventBus) -> None:
        self._data   = data_handler
        self._bus    = bus
        self._active = True

    # ── Event handler (wired by the engine) ───────────────────────────────────

    def on_market_event(self, event: MarketEvent) -> None:
        if not self._active:
            return
        bar = event.bar
        signal = self.on_bar(bar)
        if signal is not None:
            self._bus.publish(SignalEvent(signal=signal))

    # ── Abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    def on_bar(self, bar: Bar) -> Optional[Signal]:
        """
        Called once per bar.  Must:
        - Be deterministic (same input → same output)
        - Only use data available via self._data.history()
        - Return None if no signal, or a fully populated Signal
        - Never raise (catch all exceptions internally)
        """
        ...

    @abstractmethod
    def warmup_period(self) -> int:
        """Minimum number of bars required before generating signals."""
        ...

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def deactivate(self) -> None:
        self._active = False

    def activate(self) -> None:
        self._active = True

    def reset(self) -> None:
        """Called between backtest runs.  Override to clear indicator state."""
        pass
