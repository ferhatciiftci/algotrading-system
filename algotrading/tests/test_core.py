"""
tests/test_core.py
──────────────────
Unit tests for core types, events, and clock.
"""

import pytest
from datetime import datetime, timezone

UTC = timezone.utc


def ts(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=UTC)


# ─── Bar ──────────────────────────────────────────────────────────────────────

def test_bar_requires_tz_aware_timestamp():
    from algotrading.core.types import Bar
    with pytest.raises(ValueError, match="UTC-aware"):
        Bar(symbol="AAPL", timestamp=datetime(2023, 1, 1),
            open=100, high=105, low=99, close=103, volume=1e6)


def test_bar_ohlc_consistency():
    from algotrading.core.types import Bar
    with pytest.raises(ValueError, match="OHLC"):
        Bar(symbol="AAPL", timestamp=ts("2023-01-01"),
            open=110, high=105, low=99, close=103, volume=1e6)


def test_bar_valid():
    from algotrading.core.types import Bar
    bar = Bar(symbol="aapl", timestamp=ts("2023-01-01"),
              open=100, high=105, low=99, close=103, volume=1e6)
    assert bar.symbol == "aapl"   # symbol not normalised in Bar (normalised upstream)


# ─── Signal ───────────────────────────────────────────────────────────────────

def test_signal_requires_reason():
    from algotrading.core.types import Direction, Signal, SignalStrength
    with pytest.raises(ValueError, match="reason"):
        Signal(strategy_id="s1", symbol="AAPL",
               timestamp=ts("2023-01-01"), direction=Direction.LONG,
               strength=SignalStrength.STRONG, reason="")


def test_signal_confidence_range():
    from algotrading.core.types import Direction, Signal, SignalStrength
    with pytest.raises(ValueError, match="confidence"):
        Signal(strategy_id="s1", symbol="AAPL",
               timestamp=ts("2023-01-01"), direction=Direction.LONG,
               strength=SignalStrength.STRONG, reason="test", confidence=1.5)


# ─── Position ─────────────────────────────────────────────────────────────────

def test_position_apply_long_fill():
    from algotrading.core.types import Direction, Fill, Position
    pos  = Position(symbol="AAPL")
    fill = Fill(fill_id="f1", order_id="o1", symbol="AAPL",
                direction=Direction.LONG, quantity=100, fill_price=150.0,
                commission=0.5, slippage=0.1, timestamp=ts("2023-01-01"))
    pos.apply_fill(fill)
    assert pos.quantity == 100
    assert abs(pos.avg_cost - 150.0) < 1e-6
    assert pos.direction.value == "LONG"


def test_position_close_long():
    from algotrading.core.types import Direction, Fill, Position
    pos = Position(symbol="AAPL")
    pos.apply_fill(Fill("f1","o1","AAPL",Direction.LONG,100,150.0,0.5,0.1,ts("2023-01-01")))
    pos.apply_fill(Fill("f2","o2","AAPL",Direction.SHORT,100,160.0,0.5,0.1,ts("2023-01-02")))
    assert abs(pos.quantity) < 1e-9
    assert pos.realised_pnl == pytest.approx(1000.0)   # 10 * 100


# ─── Clock ────────────────────────────────────────────────────────────────────

def test_clock_advances_monotonically():
    from algotrading.core.clock import Clock
    clk = Clock()
    clk.advance(ts("2023-01-01"))
    clk.advance(ts("2023-01-02"))
    assert clk.now() == ts("2023-01-02")


def test_clock_rejects_backwards():
    from algotrading.core.clock import Clock
    clk = Clock()
    clk.advance(ts("2023-01-02"))
    with pytest.raises(ValueError, match="backwards"):
        clk.advance(ts("2023-01-01"))


# ─── EventBus ─────────────────────────────────────────────────────────────────

def test_event_bus_publishes_to_subscribers():
    from algotrading.core.events import EventBus, EventType, MarketEvent
    from algotrading.core.types import Bar
    bus = EventBus()
    received = []
    bus.subscribe(EventType.MARKET, lambda e: received.append(e))
    bar = Bar("AAPL", ts("2023-01-01"), 100, 105, 99, 103, 1e6)
    bus.publish(MarketEvent(bar=bar))
    assert len(received) == 1


def test_event_bus_halts():
    from algotrading.core.events import EventBus, EventType, MarketEvent
    from algotrading.core.types import Bar
    bus = EventBus()
    received = []
    bus.subscribe(EventType.MARKET, lambda e: received.append(e))
    bus.halt("test halt", ts("2023-01-01"))
    bar = Bar("AAPL", ts("2023-01-01"), 100, 105, 99, 103, 1e6)
    bus.publish(MarketEvent(bar=bar))
    assert len(received) == 0   # bus halted, nothing received


# ─── Strategy indicators ──────────────────────────────────────────────────────

def test_ema_computation():
    from algotrading.strategies.trend_volatility import ema
    prices = [100.0] * 20 + [110.0] * 5
    result = ema(prices, 20)
    assert result > 100.0   # should trend toward 110


def test_atr_computation():
    from algotrading.strategies.trend_volatility import atr
    from algotrading.core.types import Bar
    bars = [
        Bar("X", ts("2023-01-01"), 100, 105, 98, 103, 1e6),
        Bar("X", ts("2023-01-02"), 103, 108, 101, 106, 1e6),
        Bar("X", ts("2023-01-03"), 106, 110, 104, 108, 1e6),
    ]
    result = atr(bars, 2)
    assert result > 0


# ─── Kill switch ──────────────────────────────────────────────────────────────

def test_kill_switch_daily_loss():
    from algotrading.risk.kill_switch import KillSwitch
    ks = KillSwitch(max_daily_loss_pct=0.03)
    ks.initialise(100_000)
    triggered, reason = ks.check(96_800, ts("2023-01-01"))   # -3.2% loss
    assert triggered
    assert "Daily loss" in reason


def test_kill_switch_drawdown():
    from algotrading.risk.kill_switch import KillSwitch
    ks = KillSwitch(max_drawdown_pct=0.10)
    ks.initialise(100_000)
    ks.check(100_000, ts("2023-01-01"))   # peak set
    triggered, reason = ks.check(88_000, ts("2023-01-02"))   # -12% from peak
    assert triggered
    assert "Drawdown" in reason


def test_kill_switch_latches():
    from algotrading.risk.kill_switch import KillSwitch
    ks = KillSwitch(max_daily_loss_pct=0.01)
    ks.initialise(100_000)
    ks.check(98_000, ts("2023-01-01"))
    # Even if equity recovers, it stays triggered
    triggered, _ = ks.check(110_000, ts("2023-01-02"))
    assert triggered


# ─── Portfolio ────────────────────────────────────────────────────────────────

def test_portfolio_cash_after_buy():
    from algotrading.backtest.portfolio import Portfolio
    from algotrading.core.types import Direction, Fill
    p = Portfolio(100_000)
    fill = Fill("f1","o1","AAPL",Direction.LONG,100,150.0,1.0,0.0,ts("2023-01-01"))
    p.apply_fill(fill)
    assert p.cash == pytest.approx(100_000 - 150 * 100 - 1.0)


def test_portfolio_equity_with_price_update():
    from algotrading.backtest.portfolio import Portfolio
    from algotrading.core.types import Direction, Fill
    p = Portfolio(100_000)
    fill = Fill("f1","o1","AAPL",Direction.LONG,100,150.0,0.0,0.0,ts("2023-01-01"))
    p.apply_fill(fill)
    p.mark_to_market({"AAPL": 160.0}, ts("2023-01-02"))
    assert p.equity == pytest.approx(100_000 + 100 * 10)  # 1000 profit
