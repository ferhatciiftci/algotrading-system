"""
tests/test_pipeline.py
-----------------------
Pipeline entegrasyon testleri -- uc bagimsiz katman:

  KATMAN A  In-Memory (HER ZAMAN calisir -- bagimlilik yok)
            Sentetik veri + InMemoryDataHandler + Backtest
            Backtest motorunu ve bilesenlerini kanitlar.
            Parquet veya internet gerektirmez.

  KATMAN B  CSV Storage (pyarrow gerektirir, yoksa ATLANIR)
            Sentetik CSV -> CSVLoader -> RawDataStore(Parquet) -> PITDataHandler -> Backtest
            Tam depolama pipeline'ini kanitlar.
            Internete gerek yok, pyarrow gerekli.

  KATMAN C  Gercek Veri Smoke (yfinance + pyarrow gerektirir, yoksa ATLANIR)
            Yahoo Finance -> Parquet -> PITDataHandler -> Backtest
            Gercek piyasa verisiyle uc-uca pipeline.

Calistirma:
    python -m algotrading.tests.test_pipeline
    pytest algotrading/tests/test_pipeline.py -v
"""

from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path

UTC = timezone.utc


# ---------------------------------------------------------------------------
# Bagimlilık bayraklari
# ---------------------------------------------------------------------------

def _has_pyarrow() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


def _has_yfinance() -> bool:
    try:
        import yfinance  # noqa: F401
        return _has_pyarrow()
    except ImportError:
        return False


_PYARROW_OK  = _has_pyarrow()
_YFINANCE_OK = _has_yfinance()


# ---------------------------------------------------------------------------
# Sentetik OHLCV uretici
# NOT: _make_rows "symbol" parametresi almaz -- CSV'de sutun yoktur,
#      sembol CSVLoader tarafindan yukleme sirasinda atanir.
# ---------------------------------------------------------------------------

def _make_rows(n: int = 252, start: str = "2020-01-02",
               seed: int = 99, trend: float = 0.0003,
               vol: float = 0.012) -> list[dict]:
    """n adet deterministik OHLCV satiri uretir."""
    import random
    rng, price = random.Random(seed), 300.0
    base = datetime.fromisoformat(start).replace(tzinfo=UTC)
    rows = []
    for i in range(n):
        ret   = rng.gauss(trend, vol)
        close = price * (1 + ret)
        high  = max(price, close) * (1 + abs(rng.gauss(0, vol / 2)))
        low   = min(price, close) * (1 - abs(rng.gauss(0, vol / 2)))
        rows.append({
            "date"  : (base + timedelta(days=i)).strftime("%Y-%m-%d"),
            "open"  : round(price, 4),
            "high"  : round(high, 4),
            "low"   : round(low, 4),
            "close" : round(close, 4),
            "volume": str(rng.randint(500_000, 3_000_000)),
        })
        price = close
    return rows


def _write_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["date","open","high","low","close","volume"])
        w.writeheader()
        w.writerows(rows)


def _config(data_dir: Path, symbol: str, start: str, end: str,
            initial_capital: float = 50_000.0) -> dict:
    return {
        "system"      : {"name": "pipeline_test", "random_seed": 42,
                         "env": "backtest", "log_level": "WARNING"},
        "data"        : {"root_dir": str(data_dir), "window_size": 300,
                         "bars_per_year": 252},
        "backtest"    : {"symbols": [symbol], "start": start, "end": end,
                         "initial_capital": initial_capital},
        "strategy"    : {"id": "tv1", "params": {
                             "ema_fast": 10, "ema_slow": 30,
                             "atr_period": 14, "vol_threshold": 0.04}},
        "commission"  : {"model": "fixed_per_share", "rate_per_share": 0.005,
                         "min_per_order": 1.0},
        "slippage"    : {"model": "volatility", "atr_multiple": 0.10, "min_bps": 2.0},
        "risk"        : {"max_risk_pct": 0.01, "atr_stop_mult": 2.0,
                         "max_position_pct": 0.15,
                         "max_gross_exposure": 1.0, "max_net_exposure": 1.0,
                         "atr_period": 14},
        "kill_switch" : {"max_daily_loss_pct": 0.05, "max_drawdown_pct": 0.20,
                         "max_consecutive_losses": 10},
        "orchestrator": {"cooldown_bars": 3, "min_confidence": 0.0, "allow_short": False},
    }


# ---------------------------------------------------------------------------
# In-Memory data handler (Parquet bypass -- Katman A icin)
# ---------------------------------------------------------------------------

class _InMemHandler:
    def __init__(self, df, window: int = 300):
        self._df = df.sort_values("timestamp").reset_index(drop=True)
        self._bufs: dict = {}
        self._win  = window

    def subscribe(self, sym: str) -> None:
        self._bufs[sym.upper()] = deque(maxlen=self._win)

    def stream(self, symbol: str, start, end):
        import pandas as pd
        from algotrading.core.types import Bar
        sym = symbol.upper()
        if sym not in self._bufs:
            self.subscribe(sym)
        def _utc(dt):
            ts = pd.Timestamp(dt)
            return ts if ts.tzinfo is not None else ts.tz_localize("UTC")
        mask = (
            (self._df["symbol"] == sym) &
            (self._df["timestamp"] >= _utc(start)) &
            (self._df["timestamp"] <= _utc(end))
        )
        for _, row in self._df[mask].iterrows():
            ts = row["timestamp"]
            if hasattr(ts, "to_pydatetime"):
                ts = ts.to_pydatetime()
            bar = Bar(symbol=sym, timestamp=ts,
                      open=float(row["open"]),   high=float(row["high"]),
                      low=float(row["low"]),     close=float(row["close"]),
                      volume=float(row["volume"]), adjusted=True)
            self._bufs[sym].append(bar)
            yield bar

    def history(self, symbol: str, n: int) -> list:
        buf = self._bufs.get(symbol.upper(), deque())
        bars = list(buf)
        return bars[-n:] if n < len(bars) else bars

    def latest(self, symbol: str):
        h = self.history(symbol, 1)
        return h[0] if h else None

    def history_df(self, symbol: str, n: int):
        import pandas as pd
        bars = self.history(symbol, n)
        if not bars:
            return pd.DataFrame()
        return pd.DataFrame([
            {"timestamp": b.timestamp, "open": b.open, "high": b.high,
             "low": b.low, "close": b.close, "volume": b.volume}
            for b in bars
        ])

    def reset(self) -> None:
        for s in self._bufs:
            self._bufs[s].clear()


def _run_inmem(n: int = 252, cap: float = 50_000.0):
    """Parquet olmadan tam backtest calistirir."""
    import pandas as pd
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

    SYM   = "INMEM"
    START = datetime(2020, 1, 2, tzinfo=UTC)
    END   = START + timedelta(days=n)
    rows  = _make_rows(n=n)               # BUG DUZELTILDI: symbol parametresi yok
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["date"], utc=True)
    df["symbol"]    = SYM
    for col in ("open","high","low","close","volume"):
        df[col] = df[col].astype(float)

    dh   = _InMemHandler(df)
    bus  = EventBus()
    port = Portfolio(initial_capital=cap)

    strat = TrendVolatilityStrategy(data_handler=dh, bus=bus, symbol=SYM,
                                    ema_fast=10, ema_slow=30,
                                    atr_period=14, vol_threshold=0.04)
    orch  = Orchestrator(bus=bus, cooldown_bars=3, min_confidence=0.0, allow_short=False)
    ks    = KillSwitch(max_daily_loss_pct=0.05, max_drawdown_pct=0.20,
                       max_consecutive_losses=10)
    ks.initialise(cap, at=START)
    rm    = RiskManager(portfolio=port, data_handler=dh, bus=bus, kill_switch=ks,
                        position_sizer=FixedFractional(
                            max_risk_pct=0.01, atr_stop_mult=2.0, max_position_pct=0.15),
                        max_position_pct=0.15, max_gross_exposure=1.0, max_net_exposure=1.0)
    comm  = FixedPerShare(rate_per_share=0.005, min_per_order=1.0)
    slip  = VolatilitySlippage(atr_multiple=0.10, min_bps=2.0)
    exec_ = SimulatedExecution(data_handler=dh, bus=bus, commission=comm, slippage=slip)
    bc    = BacktestConfig(name="inmem", symbols=[SYM], start=START, end=END,
                           initial_capital=cap, random_seed=42)
    engine = BacktestEngine(config=bc, data_handler=dh, strategy=strat,
                            orchestrator=orch, risk_manager=rm, execution=exec_,
                            commission=comm, slippage=slip, bus=bus, portfolio=port)
    result = engine.run()
    return result, port, rm, orch


# ===========================================================================
# Sonuc takipci
# ===========================================================================

class _Results:
    def __init__(self):
        self.passed  = []
        self.failed  = []
        self.skipped = []

    def run(self, name: str, fn):
        """fn() calistir; PASS/FAIL/SKIP kaydet."""
        import io, traceback
        buf = io.StringIO()
        try:
            # stdout'u yakala -- "  SKIP ..." yazan fonksiyonlar icin
            import contextlib
            with contextlib.redirect_stdout(buf):
                fn()
            out = buf.getvalue()
            sys.stdout.write(out)
            if "  SKIP" in out:
                self.skipped.append(name)
            else:
                self.passed.append(name)
        except Exception as e:
            sys.stdout.write(buf.getvalue())
            print(f"  FAIL  {name}: {e}")
            traceback.print_exc()
            self.failed.append(name)

    def summary(self) -> int:
        """0 donerse tum testler gecti/atlandi; >0 hatali var."""
        print()
        print("=" * 62)
        print(f"  GECTI   : {len(self.passed)}")
        print(f"  ATLAMDI : {len(self.skipped)}")
        print(f"  HATALI  : {len(self.failed)}")
        if self.skipped:
            print()
            print("  Atlanan testler (bagimlilik eksik):")
            for s in self.skipped:
                print(f"    - {s}")
        if self.failed:
            print()
            print("  BASARISIZ testler:")
            for f in self.failed:
                print(f"    - {f}")
        if not _PYARROW_OK:
            print()
            print("  UYARI: pyarrow kurulu degil.")
            print("         Gercek pipeline DOGRULANMADI.")
            print("         Kurmak icin: pip install pyarrow")
        elif not _YFINANCE_OK:
            print()
            print("  UYARI: yfinance kurulu degil.")
            print("         Gercek piyasa verisi testi atlamdi.")
            print("         Kurmak icin: pip install yfinance")
        else:
            print()
            print("  Gercek pipeline tam olarak DOGRULANDI.")
        print("=" * 62)
        return len(self.failed)


# ===========================================================================
# KATMAN A -- In-Memory Pipeline (HER ZAMAN calisir)
# ===========================================================================

class TestInMemoryPipeline:
    """Parquet veya internet gerektirmez. Backtest motorunu kanitlar."""

    def test_A1_backtest_tamamlanir(self):
        r, *_ = _run_inmem()
        assert r is not None
        print("  PASS  A1: backtest tamamlandi")

    def test_A2_equity_curve_dolu(self):
        r, *_ = _run_inmem()
        ec = r.equity_curve
        assert len(ec) > 0 and "equity" in ec.columns
        print(f"  PASS  A2: equity_curve | {len(ec)} satir")

    def test_A3_event_sayisi_dogru(self):
        n = 252
        r, *_ = _run_inmem(n=n)
        assert r.events_processed == n, \
            f"Beklenen {n}, alinan {r.events_processed}"
        print(f"  PASS  A3: events_processed = {n}")

    def test_A4_pozitif_equity(self):
        r, port, *_ = _run_inmem()
        assert port.equity > 0
        print(f"  PASS  A4: final equity = {port.equity:.2f}")

    def test_A5_determinizm(self):
        _, p1, *_ = _run_inmem(n=150)
        _, p2, *_ = _run_inmem(n=150)
        assert abs(p1.equity - p2.equity) < 1e-6, \
            f"Determinizm hatasi: {p1.equity} != {p2.equity}"
        print(f"  PASS  A5: determinizm | equity = {p1.equity:.4f}")

    def test_A6_eksik_veri_turkce_hata(self):
        """Parquet yoksa Turkce FileNotFoundError."""
        from algotrading.main import run_backtest
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _config(Path(tmp) / "bos", "YOKTUR", "2022-01-01", "2023-01-01")
            try:
                run_backtest(cfg)
                assert False, "FileNotFoundError bekleniyordu"
            except FileNotFoundError as e:
                msg = str(e)
                assert "Veri bulunamadi" in msg or "Veri bulunamadı" in msg, \
                    f"Turkce hata mesaji yok: {msg}"
                assert "veri indirme komutunu" in msg, \
                    f"Komut ipucu yok: {msg}"
                print("  PASS  A6: Turkce hata mesaji dogru")


# ===========================================================================
# KATMAN B -- CSV Storage Pipeline (pyarrow gerektirir)
# ===========================================================================

class TestCSVStoragePipeline:
    """
    CSV -> Parquet -> PITDataHandler -> Backtest.
    pyarrow ZORUNLU -- eksikse ATLANIR.
    Bu katman gercek depolama pipeline'ini kanitlar.
    """

    def _full_pipeline(self, tmp: Path, n: int = 252):
        from algotrading.data.connectors.csv_loader import CSVLoader
        from algotrading.data.ingestion import RawDataStore
        from algotrading.main import run_backtest

        SYM   = "CSVTEST"
        START = "2020-01-02"
        END   = (datetime.fromisoformat(START).replace(tzinfo=UTC)
                 + timedelta(days=n)).strftime("%Y-%m-%d")

        # 1. Sentetik CSV yaz
        rows     = _make_rows(n=n, start=START)   # symbol parametresi YOK -- dogru
        csv_file = tmp / "CSVTEST.csv"
        _write_csv(rows, csv_file)

        # 2. Parquet deposuna yukle
        data_dir  = tmp / "data"
        raw_store = RawDataStore(data_dir)
        loader    = CSVLoader()
        h = loader.load_and_store(csv_file, SYM, raw_store)
        assert len(h) == 64, f"Hash 64 karakter olmali, alinan: {len(h)}"

        # 3. Parquet olusturuldu mu?
        sym_dir = raw_store.raw_dir / SYM
        pkts = list(sym_dir.rglob("*.parquet"))
        assert len(pkts) > 0, "Parquet dosyasi olusturulmadi"

        # 4. Backtest calistir (PITDataHandler Parquet'ten okur)
        cfg    = _config(data_dir, SYM, START, END)
        result = run_backtest(cfg)
        return result

    def test_B1_csv_parquet_backtest(self):
        if not _PYARROW_OK:
            print(f"  SKIP  B1: pyarrow yok -- pip install pyarrow")
            return
        with tempfile.TemporaryDirectory() as tmp:
            r = self._full_pipeline(Path(tmp))
            assert r is not None
            ec = r.equity_curve
            assert len(ec) > 0, "Equity curve bos"
            assert "equity" in ec.columns
            final = ec["equity"].iloc[-1]
            assert final > 0, f"Negatif equity: {final}"
            print(f"  PASS  B1: CSV->Parquet->backtest | "
                  f"events={r.events_processed} | equity={final:.2f}")

    def test_B2_event_sayisi_eslesir(self):
        if not _PYARROW_OK:
            print(f"  SKIP  B2: pyarrow yok")
            return
        n = 252
        with tempfile.TemporaryDirectory() as tmp:
            r = self._full_pipeline(Path(tmp), n=n)
            assert r.events_processed == n, \
                f"Beklenen {n} event, alinan {r.events_processed}"
            print(f"  PASS  B2: events = {r.events_processed}")

    def test_B3_download_cli_csv(self):
        """download.py CLI ile CSV -> Parquet olusturulur."""
        if not _PYARROW_OK:
            print(f"  SKIP  B3: pyarrow yok")
            return
        with tempfile.TemporaryDirectory() as tmp:
            csv_file = Path(tmp) / "TESTCLI.csv"
            data_dir = Path(tmp) / "data"
            rows = _make_rows(n=60)         # BUG DUZELTILDI: symbol="..." yok
            _write_csv(rows, csv_file)
            proc = subprocess.run(
                [sys.executable, "-m", "algotrading.data.download",
                 "--source",   "csv",
                 "--symbol",   "TESTCLI",
                 "--file",     str(csv_file),
                 "--data-dir", str(data_dir)],
                capture_output=True, text=True,
            )
            assert proc.returncode == 0, \
                f"CLI coktu:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
            pkts = list((data_dir / "raw" / "TESTCLI").rglob("*.parquet"))
            assert len(pkts) > 0, "CLI sonrasi parquet yok"
            print(f"  PASS  B3: download CLI CSV -> {pkts[0].name}")

    def test_B4_determinizm(self):
        if not _PYARROW_OK:
            print(f"  SKIP  B4: pyarrow yok")
            return
        with tempfile.TemporaryDirectory() as t1:
            r1 = self._full_pipeline(Path(t1), n=150)
        with tempfile.TemporaryDirectory() as t2:
            r2 = self._full_pipeline(Path(t2), n=150)
        eq1 = r1.equity_curve["equity"].iloc[-1]
        eq2 = r2.equity_curve["equity"].iloc[-1]
        assert abs(eq1 - eq2) < 1e-6, \
            f"Determinizm hatasi: {eq1} != {eq2}"
        print(f"  PASS  B4: determinizm | equity = {eq1:.4f}")

    def test_B5_turkce_hata_mesaji(self):
        """Parquet yoksa acik Turkce hata."""
        if not _PYARROW_OK:
            print(f"  SKIP  B5: pyarrow yok")
            return
        from algotrading.main import run_backtest
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _config(Path(tmp) / "bos", "YKTST", "2020-01-01", "2021-01-01")
            try:
                run_backtest(cfg)
                assert False, "Hata bekleniyordu"
            except FileNotFoundError as e:
                assert "Veri bulunamad" in str(e)
                print("  PASS  B5: Turkce hata mesaji (storage katmani)")

    def test_B6_veri_yolu_proje_koku(self):
        """data.root_dir 'data' -> PROJECT_ROOT/data olmali."""
        if not _PYARROW_OK:
            print(f"  SKIP  B6: pyarrow yok")
            return
        from algotrading.main import _resolve_data_root, _project_root, load_config
        cfg      = load_config("algotrading/config/default.yaml")
        resolved = _resolve_data_root(cfg)
        expected = _project_root() / "data"
        assert resolved == expected.resolve(), \
            f"Yanlis yol!\n  Elde edilen: {resolved}\n  Beklenen   : {expected.resolve()}"
        print(f"  PASS  B6: veri yolu dogru = {resolved}")


# ===========================================================================
# KATMAN C -- Gercek Veri Smoke Testi (yfinance + pyarrow)
# ===========================================================================

class TestRealDataSmoke:
    """
    Yahoo Finance -> Parquet -> PITDataHandler -> Backtest.
    yfinance VE pyarrow gerektirir -- eksikse ATLANIR.
    """

    def test_C1_spy_backtest_smoke(self):
        if not _YFINANCE_OK:
            print(f"  SKIP  C1: yfinance veya pyarrow yok")
            return
        from algotrading.data.connectors.yfinance_connector import YFinanceConnector
        from algotrading.data.ingestion import RawDataStore
        from algotrading.main import run_backtest

        SYM, START, END = "SPY", "2023-01-01", "2023-06-30"
        CAP = 50_000.0
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data"
            # 1. Indir
            raw_store = RawDataStore(data_dir)
            conn      = YFinanceConnector(interval="1d")
            h = conn.fetch_and_store(SYM, START, END, raw_store)
            assert len(h) == 64

            # 2. Parquet var mi?
            pkts = list((raw_store.raw_dir / SYM).rglob("*.parquet"))
            assert len(pkts) > 0, "Parquet olusturulmadi"

            # 3. Backtest
            cfg    = _config(data_dir, SYM, START, END, initial_capital=CAP)
            result = run_backtest(cfg)
            assert result is not None

            # 4. Sonuclari dogrula
            ec = result.equity_curve
            assert len(ec) > 0, "Equity curve bos"
            final = ec["equity"].iloc[-1]
            assert 0.5 * CAP < final < 2.0 * CAP, \
                f"Beklenmedik equity: {final}"
            assert result.events_processed > 0

            print(f"  PASS  C1: gercek SPY backtest | "
                  f"events={result.events_processed} | equity={final:.2f}")


# ===========================================================================
# Bagımsız calistirici
# ===========================================================================

def run_all() -> int:
    r = _Results()

    print("\n" + "=" * 62)
    print("  KATMAN A -- In-Memory Pipeline (bagimlilik yok)")
    print("=" * 62)
    a = TestInMemoryPipeline()
    r.run("A1_backtest_tamamlanir",    a.test_A1_backtest_tamamlanir)
    r.run("A2_equity_curve_dolu",      a.test_A2_equity_curve_dolu)
    r.run("A3_event_sayisi_dogru",     a.test_A3_event_sayisi_dogru)
    r.run("A4_pozitif_equity",         a.test_A4_pozitif_equity)
    r.run("A5_determinizm",            a.test_A5_determinizm)
    r.run("A6_eksik_veri_turkce_hata", a.test_A6_eksik_veri_turkce_hata)

    status_b = "(pyarrow VAR)" if _PYARROW_OK else "(pyarrow YOK -- atlanacak)"
    print("\n" + "=" * 62)
    print(f"  KATMAN B -- CSV Storage Pipeline {status_b}")
    print("=" * 62)
    b = TestCSVStoragePipeline()
    r.run("B1_csv_parquet_backtest",    b.test_B1_csv_parquet_backtest)
    r.run("B2_event_sayisi",            b.test_B2_event_sayisi_eslesir)
    r.run("B3_download_cli_csv",        b.test_B3_download_cli_csv)
    r.run("B4_determinizm",             b.test_B4_determinizm)
    r.run("B5_turkce_hata",             b.test_B5_turkce_hata_mesaji)
    r.run("B6_veri_yolu_proje_koku",    b.test_B6_veri_yolu_proje_koku)

    status_c = "(yfinance+pyarrow VAR)" if _YFINANCE_OK else "(yfinance/pyarrow YOK -- atlanacak)"
    print("\n" + "=" * 62)
    print(f"  KATMAN C -- Gercek Veri Smoke {status_c}")
    print("=" * 62)
    c = TestRealDataSmoke()
    r.run("C1_spy_backtest_smoke", c.test_C1_spy_backtest_smoke)

    return r.summary()


if __name__ == "__main__":
    sys.exit(run_all())
