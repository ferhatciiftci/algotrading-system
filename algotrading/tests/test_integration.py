"""
tests/test_integration.py
--------------------------
End-to-end integration test using synthetic OHLCV data.

No external dependencies required (no pyarrow, no network).
Can be run with pytest or directly: python -m algotrading.tests.test_integration
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta, timezone
from typing import List, Optional

try:
    import pytest
    _HAS_PYTEST = True
except ImportError:
    _HAS_PYTEST = False
    class _RaisesCtx:
        def __init__(self, exc, match=None):
            self._exc = exc; self._match = match
        def __enter__(self): return self
        def __exit__(self, exc_type, exc_val, tb):
            if exc_type is None:
                raise AssertionError(f"Expected {self._exc.__name__} to be raised")
            if not issubclass(exc_type, self._exc):
                return False
            if self._match and self._match not in str(exc_val):
                raise AssertionError(
                    f"Exception message '{exc_val}' does not match '{self._match}'"
                )
            return True
    class _PytestStub:
        @staticmethod
        def raises(exc, match=None): return _RaisesCtx(exc, match)
    pytest = _PytestStub()

UTC = timezone.utc


# ---------------------------------------------------------------------------
# Synthetic OHLCV factory
# ---------------------------------------------------------------------------

def make_ohlcv(n_bars=200, start="2022-01-03", symbol="SYNTH",
               trend=0.0002, vol=0.01, seed=42):
    import random
    import pandas as pd
    rng   = random.Random(seed)
    price = 100.0
    dates = [
        datetime.fromisoformat(start).replace(tzinfo=UTC) + timedelta(days=i)
        for i in range(n_bars)
    ]
    rows = []
    for dt in dates:
        ret   = rng.gauss(trend, vol)
        close = price * (1 + ret)
        high  = max(price, close) * (1 + abs(rng.gauss(0, vol / 2)))
        low   = min(price, close) * (1 - abs(rng.gauss(0, vol / 2)))
        rows.append({
            "symbol"   : symbol,
            "timestamp": dt,
            "open"     : round(price, 4),
            "high"     : round(high,  4),
            "low"      : round(low,   4),
            "close"    : round(close, 4),
            "volume"   : float(rng.randint(500_000, 2_000_000)),
            "adj_close": round(close, 4),
            "vwap"     : float("nan"),
        })
        price = close
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# In-memory data handler (no Parquet)
# ---------------------------------------------------------------------------

class InMemoryDataHandler:
    def __init__(self, df, window_size=200):
        self._df     = df.sort_values("timestamp").reset_index(drop=True)
        self._window = window_size
        self._buffers = {}

    def subscribe(self, symbol):
        symbol = symbol.upper()
        self._buffers[symbol] = deque(maxlen=self._window)

    def stream(self, symbol, start, end):
        import pandas as pd
        from algotrading.core.types import Bar
        symbol = symbol.upper()
        if symbol not in self._buffers:
            self.subscribe(symbol)
        # Convert start/end to UTC Timestamps regardless of whether they carry tzinfo
        def _to_utc(dt):
            ts = pd.Timestamp(dt)
            return ts if ts.tzinfo is not None else ts.tz_localize("UTC")
        ts_start = _to_utc(start)
        ts_end   = _to_utc(end)
        mask = (
            (self._df["symbol"] == symbol) &
            (self._df["timestamp"] >= ts_start) &
            (self._df["timestamp"] <= ts_end)
        )
        for _, row in self._df[mask].iterrows():
            ts = row["timestamp"]
            if hasattr(ts, "to_pydatetime"):
                ts = ts.to_pydatetime()
            bar = Bar(
                symbol=symbol, timestamp=ts,
                open=float(row["open"]), high=float(row["high"]),
                low=float(row["low"]),   close=float(row["close"]),
                volume=float(row["volume"]), adjusted=True,
            )
            self._buffers[symbol].append(bar)
            yield bar

    def history(self, symbol, n):
        symbol = symbol.upper()
        buf  = self._buffers.get(symbol, deque())
        bars = list(buf)
        return bars[-n:] if n < len(bars) else bars

    def latest(self, symbol):
        h = self.history(symbol, 1)
        return h[0] if h else None

    def history_df(self, symbol, n):
        import pandas as pd
        bars = self.history(symbol, n)
        if not bars:
            return pd.DataFrame()
        return pd.DataFrame([
            {"timestamp": b.timestamp, "open": b.open, "high": b.high,
             "low": b.low, "close": b.close, "volume": b.volume}
            for b in bars
        ])

    def reset(self):
        for sym in self._buffers:
            self._buffers[sym].clear()


# ---------------------------------------------------------------------------
# Helper: build and run a complete backtest
# ---------------------------------------------------------------------------

def _run_backtest(n_bars=200, initial_capital=100_000.0):
    from algotrading.backtest.commission import FixedPerShare
    from algotrading.backtest.engine     import BacktestConfig, BacktestEngine
    from algotrading.backtest.portfolio  import Portfolio
    from algotrading.backtest.slippage   import VolatilitySlippage
    from algotrading.core.events         import EventBus
    from algotrading.execution.paper_trader import SimulatedExecution
    from algotrading.orchestrator.orchestrator import Orchestrator
    from algotrading.risk.kill_switch    import KillSwitch
    from algotrading.risk.position_sizer import FixedFractional
    from algotrading.risk.risk_manager   import RiskManager
    from algotrading.strategies.trend_volatility import TrendVolatilityStrategy

    SYMBOL = "SYNTH"
    START  = datetime(2022, 1, 3, tzinfo=UTC)
    END    = START + timedelta(days=n_bars)

    df           = make_ohlcv(n_bars=n_bars, symbol=SYMBOL)
    data_handler = InMemoryDataHandler(df)
    bus          = EventBus()
    portfolio    = Portfolio(initial_capital=initial_capital)

    strategy = TrendVolatilityStrategy(
        data_handler=data_handler, bus=bus, symbol=SYMBOL,
        ema_fast=10, ema_slow=30, atr_period=14, vol_threshold=0.04,
    )
    orchestrator = Orchestrator(
        bus=bus, cooldown_bars=3, min_confidence=0.05, allow_short=False,
    )

    kill_switch = KillSwitch(
        max_daily_loss_pct=0.05, max_drawdown_pct=0.20,
        max_consecutive_losses=10,
    )
    kill_switch.initialise(initial_capital, at=START)

    risk_manager = RiskManager(
        portfolio=portfolio, data_handler=data_handler, bus=bus,
        kill_switch=kill_switch,
        position_sizer=FixedFractional(
            max_risk_pct=0.01, atr_stop_mult=2.0, max_position_pct=0.20,
        ),
        max_position_pct=0.20, max_gross_exposure=1.0, max_net_exposure=1.0,
    )

    commission     = FixedPerShare(rate_per_share=0.005, min_per_order=1.0)
    slippage_model = VolatilitySlippage(atr_multiple=0.1, min_bps=5)
    execution      = SimulatedExecution(
        data_handler=data_handler, bus=bus,
        commission=commission, slippage=slippage_model,
    )

    bc = BacktestConfig(
        name="integration_test", symbols=[SYMBOL],
        start=START, end=END,
        initial_capital=initial_capital, random_seed=42,
    )
    engine = BacktestEngine(
        config=bc, data_handler=data_handler,
        strategy=strategy, orchestrator=orchestrator,
        risk_manager=risk_manager, execution=execution,
        commission=commission, slippage=slippage_model,
        bus=bus, portfolio=portfolio,
    )
    result = engine.run()
    return result, portfolio, risk_manager, orchestrator


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestEndToEndBacktest:

    def test_backtest_completes_without_error(self):
        result, *_ = _run_backtest(n_bars=200)
        assert result is not None

    def test_equity_curve_populated(self):
        result, *_ = _run_backtest(n_bars=200)
        ec = result.equity_curve
        assert len(ec) > 0
        assert "equity" in ec.columns

    def test_events_processed(self):
        result, *_ = _run_backtest(n_bars=200)
        assert result.events_processed == 200, (
            f"Expected 200 market events, got {result.events_processed}"
        )

    def test_portfolio_equity_positive(self):
        result, portfolio, *_ = _run_backtest(n_bars=200)
        assert portfolio.equity > 0

    def test_config_fingerprint_deterministic(self):
        from algotrading.backtest.engine import BacktestConfig
        bc = BacktestConfig(
            name="fp_test", symbols=["AAPL", "GOOG"],
            start=datetime(2020, 1, 1, tzinfo=UTC),
            end=datetime(2021, 1, 1, tzinfo=UTC),
            initial_capital=50_000.0, random_seed=7,
        )
        assert bc.fingerprint() == bc.fingerprint()
        assert len(bc.fingerprint()) == 64

    def test_risk_manager_summary(self):
        result, portfolio, risk_manager, *_ = _run_backtest(n_bars=200)
        summary = risk_manager.summary()
        assert "blocked_orders"  in summary
        assert "approved_orders" in summary
        assert summary["blocked_orders"] + summary["approved_orders"] >= 0

    def test_orchestrator_summary(self):
        result, portfolio, _, orchestrator = _run_backtest(n_bars=200)
        summary = orchestrator.summary()
        assert "total_decisions" in summary
        assert summary["total_decisions"] >= 0

    def test_kill_switch_not_triggered_on_normal_run(self):
        from algotrading.risk.kill_switch import KillSwitch
        ks = KillSwitch(
            max_daily_loss_pct=0.05, max_drawdown_pct=0.20,
            max_consecutive_losses=10,
        )
        ks.initialise(100_000.0)
        triggered, reason = ks.check(99_000.0, datetime.now(UTC))
        assert not triggered

    def test_kill_switch_triggers_on_large_loss(self):
        from algotrading.risk.kill_switch import KillSwitch
        START_DT = datetime(2024, 1, 2, 9, 0, tzinfo=UTC)
        ks = KillSwitch(max_daily_loss_pct=0.03)
        ks.initialise(100_000.0, at=START_DT)
        triggered, reason = ks.check(96_000.0, START_DT.replace(hour=15))
        assert triggered
        assert "Daily loss" in reason

    def test_two_runs_produce_same_result(self):
        r1, p1, *_ = _run_backtest(n_bars=150)
        r2, p2, *_ = _run_backtest(n_bars=150)
        assert r1.events_processed == r2.events_processed
        assert abs(p1.equity - p2.equity) < 1e-6, (
            f"Non-deterministic: {p1.equity} vs {p2.equity}"
        )


class TestCSVConnector:

    def test_load_csv_with_standard_columns(self, tmp_path):
        from algotrading.data.connectors.csv_loader import CSVLoader
        csv_file = tmp_path / "test.csv"
        df = make_ohlcv(n_bars=30, symbol="TEST")
        df.to_csv(csv_file, index=False)
        loader = CSVLoader()
        result = loader.load(csv_file, symbol="TEST")
        assert len(result) == 30
        required = {"symbol", "timestamp", "open", "high", "low", "close", "volume"}
        assert required.issubset(set(result.columns))
        assert result["symbol"].iloc[0] == "TEST"

    def test_load_csv_missing_required_column_raises(self, tmp_path):
        import pandas as pd
        from algotrading.data.connectors.csv_loader import CSVLoader
        csv_file = tmp_path / "bad.csv"
        pd.DataFrame({
            "date": ["2022-01-03"],
            "open": [100.0], "high": [101.0],
            "low": [99.0],   "close": [100.5],
        }).to_csv(csv_file, index=False)
        loader = CSVLoader()
        with pytest.raises(ValueError, match="missing required columns"):
            loader.load(csv_file, symbol="BAD")


# ---------------------------------------------------------------------------
# Standalone runner (no pytest needed)
# ---------------------------------------------------------------------------

def _run_all():
    import tempfile, pathlib
    t = TestEndToEndBacktest()
    tests = [
        ("backtest_completes_without_error", t.test_backtest_completes_without_error),
        ("equity_curve_populated",           t.test_equity_curve_populated),
        ("events_processed_200",             t.test_events_processed),
        ("portfolio_equity_positive",        t.test_portfolio_equity_positive),
        ("config_fingerprint_deterministic", t.test_config_fingerprint_deterministic),
        ("risk_manager_summary",             t.test_risk_manager_summary),
        ("orchestrator_summary",             t.test_orchestrator_summary),
        ("kill_switch_not_triggered_normal", t.test_kill_switch_not_triggered_on_normal_run),
        ("kill_switch_triggers_large_loss",  t.test_kill_switch_triggers_on_large_loss),
        ("two_runs_deterministic",           t.test_two_runs_produce_same_result),
    ]
    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception as e:
            import traceback
            print(f"  FAIL  {name}: {e}")
            traceback.print_exc()
            failed += 1

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = pathlib.Path(tmp)
        tc = TestCSVConnector()
        csv_tests = [
            ("csv_standard_columns",      lambda: tc.test_load_csv_with_standard_columns(tmp_path)),
            ("csv_missing_column_raises", lambda: tc.test_load_csv_missing_required_column_raises(tmp_path)),
        ]
        for name, fn in csv_tests:
            try:
                fn()
                print(f"  PASS  {name}")
                passed += 1
            except Exception as e:
                import traceback
                print(f"  FAIL  {name}: {e}")
                traceback.print_exc()
                failed += 1

    print()
    print(f"Results: {passed} passed, {failed} failed")
    return failed


if __name__ == "__main__":
    import sys
    sys.exit(_run_all())
