"""
backtest/engine.py
──────────────────
Event-driven backtest engine.

Architecture:
  1. DataFeed emits MarketEvents (one bar at a time, in order)
  2. Strategy consumes MarketEvents → emits SignalEvents
  3. Orchestrator consumes SignalEvents → emits DecisionEvents
  4. RiskManager consumes DecisionEvents → emits OrderEvents (or blocks)
  5. SimulatedExecution consumes OrderEvents → emits FillEvents
  6. Portfolio + Monitoring consume FillEvents + MarketEvents

Reproducibility guarantees:
  - Fixed random seed propagated through all components
  - Deterministic ordering: events processed strictly in timestamp order
  - No randomness in fill logic (slippage is deterministic given price + ATR)
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from algotrading.core.clock import Clock
from algotrading.core.events import EventBus, EventType, FillEvent, MarketEvent
from algotrading.core.types import Bar
from algotrading.backtest.portfolio import Portfolio
from algotrading.backtest.commission import CommissionModel, FixedPerShare
from algotrading.backtest.slippage import SlippageModel, VolatilitySlippage
from algotrading.data.pit_handler import PITDataHandler

logger = logging.getLogger(__name__)

UTC = timezone.utc


@dataclass
class BacktestConfig:
    """All parameters that define a backtest run."""
    name            : str
    symbols         : List[str]
    start           : datetime
    end             : datetime
    initial_capital : float  = 100_000.0
    random_seed     : int    = 42

    def fingerprint(self) -> str:
        """SHA-256 of the config — used to detect result drift."""
        d = {
            "name"            : self.name,
            "symbols"         : sorted(self.symbols),
            "start"           : self.start.isoformat(),
            "end"             : self.end.isoformat(),
            "initial_capital" : self.initial_capital,
            "random_seed"     : self.random_seed,
        }
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()


@dataclass
class BacktestResult:
    config_fingerprint : str
    start_time         : datetime      # wall-clock when run started
    end_time           : datetime
    equity_curve       : pd.DataFrame
    fills              : List[Any]     # List[Fill]
    portfolio_summary  : dict
    events_processed   : int
    run_duration_s     : float

    @property
    def total_return_pct(self) -> float:
        return self.portfolio_summary.get("return_pct", 0.0)


class BacktestEngine:
    """
    Wires together all components and runs the event loop.

    The engine owns:
    - Clock
    - EventBus
    - Portfolio
    - PITDataHandler (injected)

    Strategy, Orchestrator, RiskManager, and Execution are injected
    at construction time.  They register themselves as event handlers.
    """

    def __init__(
        self,
        config         : BacktestConfig,
        data_handler   : PITDataHandler,
        strategy       ,  # BaseStrategy
        orchestrator   ,  # Orchestrator
        risk_manager   ,  # RiskManager
        execution      ,  # SimulatedExecution
        commission     : CommissionModel = None,
        slippage       : SlippageModel   = None,
        bus            : "EventBus | None"  = None,
        portfolio      : "Portfolio | None" = None,
    ) -> None:
        self.config        = config
        self.data_handler  = data_handler
        self.strategy      = strategy
        self.orchestrator  = orchestrator
        self.risk_manager  = risk_manager
        self.execution     = execution

        self.commission = commission or FixedPerShare()
        self.slippage   = slippage   or VolatilitySlippage()

        self.clock = Clock(live=False)
        # Accept injected bus/portfolio so all components share the same objects.
        # If not provided, create fresh ones (useful for isolated unit tests).
        self.bus       = bus       or EventBus()
        self.portfolio = portfolio or Portfolio(initial_capital=config.initial_capital)

        self._events_processed = 0
        self._setup_handlers()

    # ── Wiring ────────────────────────────────────────────────────────────────

    def _setup_handlers(self) -> None:
        bus = self.bus

        # Market → Strategy
        bus.subscribe(EventType.MARKET,   self.strategy.on_market_event)
        # Market → Portfolio mark-to-market
        bus.subscribe(EventType.MARKET,   self._on_market_mtm)

        # Signal → Orchestrator
        bus.subscribe(EventType.SIGNAL,   self.orchestrator.on_signal_event)

        # Decision → Risk Manager
        bus.subscribe(EventType.DECISION, self.risk_manager.on_decision_event)

        # Order → Execution
        bus.subscribe(EventType.ORDER,    self.execution.on_order_event)

        # Fill → Portfolio
        bus.subscribe(EventType.FILL,     self._on_fill)

        # Halt → log and freeze
        bus.subscribe(EventType.HALT,     self._on_halt)

    # ── Main run ──────────────────────────────────────────────────────────────

    def run(self) -> BacktestResult:
        logger.info(
            "Starting backtest '%s' | %s → %s | capital=%.0f",
            self.config.name, self.config.start.date(), self.config.end.date(),
            self.config.initial_capital
        )
        wall_start = time.monotonic()
        run_start  = datetime.now(UTC)

        self._reset_all()

        # Subscribe data handler for all symbols
        for sym in self.config.symbols:
            self.data_handler.subscribe(sym)

        # Build a merged, time-ordered stream of bars across all symbols
        for bar in self._merged_stream():
            if self.bus.is_halted:
                logger.warning("Bus halted — stopping backtest loop.")
                break

            self.clock.advance(bar.timestamp)
            event = MarketEvent(bar=bar)
            self.bus.publish(event)
            self._events_processed += 1

        wall_end = time.monotonic()

        result = BacktestResult(
            config_fingerprint = self.config.fingerprint(),
            start_time         = run_start,
            end_time           = datetime.now(UTC),
            equity_curve       = self.portfolio.equity_curve,
            fills              = self.portfolio.fills,
            portfolio_summary  = self.portfolio.summary(),
            events_processed   = self._events_processed,
            run_duration_s     = round(wall_end - wall_start, 3),
        )

        logger.info(
            "Backtest complete | return=%.2f%% | fills=%d | duration=%.1fs",
            result.total_return_pct,
            len(result.fills),
            result.run_duration_s,
        )
        return result

    # ── Event handlers ────────────────────────────────────────────────────────

    def _on_market_mtm(self, event: MarketEvent) -> None:
        bar = event.bar
        self.portfolio.mark_to_market(
            {bar.symbol: bar.close},
            bar.timestamp
        )

    def _on_fill(self, event: FillEvent) -> None:
        self.portfolio.apply_fill(event.fill)

    def _on_halt(self, event) -> None:
        logger.critical("SYSTEM HALT: %s @ %s", event.reason, event.timestamp)

    # ── Data merging ──────────────────────────────────────────────────────────

    def _merged_stream(self):
        """
        Merge bar streams from multiple symbols into a single sorted stream.
        Uses a heap for O(n log k) merging.
        """
        import heapq

        iterators = {
            sym: self.data_handler.stream(sym, self.config.start, self.config.end)
            for sym in self.config.symbols
        }

        heap: list[tuple] = []
        for sym, it in iterators.items():
            try:
                bar = next(it)
                heapq.heappush(heap, (bar.timestamp, sym, bar, it))
            except StopIteration:
                pass

        while heap:
            ts, sym, bar, it = heapq.heappop(heap)
            yield bar
            try:
                next_bar = next(it)
                heapq.heappush(heap, (next_bar.timestamp, sym, next_bar, it))
            except StopIteration:
                pass

    # ── Reset ─────────────────────────────────────────────────────────────────

    def _reset_all(self) -> None:
        self.clock.reset()
        self.bus.reset()
        self.portfolio.reset()
        self.data_handler.reset()
        self._events_processed = 0
  