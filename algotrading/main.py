"""
main.py
────────
AlgoTrading MVP giriş noktası.

Kullanım:
  python run_backtest.py                                        (proje kökünden)
  python -m algotrading.main --config algotrading/config/default.yaml --mode backtest

Modlar:
  backtest  — tam metriklerle tarihi simülasyon
  paper     — henüz aktif değil (yakında)
  validate  — bağımsız walk-forward doğrulama (yakında)
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

UTC = timezone.utc

logging.basicConfig(
    level    = logging.INFO,
    format   = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers = [logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("algotrading.main")


# ── Yardımcı: config yükleme ──────────────────────────────────────────────────

def load_config(path: str) -> dict:
    """
    YAML config dosyasını yükler.

    Aşağıdaki sırayla denenir:
      1. Tam yol / mutlak yol
      2. Bu dosyaya göre göreceli
      3. Bir üst dizine göre göreceli
    """
    candidates = [
        Path(path),
        Path(__file__).parent / path,
        Path(__file__).parent.parent / path,
    ]
    for p in candidates:
        if p.exists():
            with open(p) as f:
                return yaml.safe_load(f)
    raise FileNotFoundError(
        f"Config dosyası bulunamadı: '{path}'\n"
        f"Denenen yollar: {[str(c) for c in candidates]}"
    )


def _project_root() -> Path:
    """
    Proje kökünü döndürür.
    Bu dosya (main.py) her zaman PROJECT_ROOT/algotrading/main.py konumundadır,
    bu nedenle proje kökü = bu dosyanın iki üst dizinidir.
    """
    return Path(__file__).resolve().parent.parent


def _resolve_data_root(cfg: dict, config_path: str = "") -> Path:
    """
    data.root_dir → PROJECT_ROOT / data.root_dir

    Kural:
      - Mutlak yolsa olduğu gibi kullan.
      - Göreceli yolsa PROJECT_ROOT'a göre çöz.
      - PROJECT_ROOT = algotrading/main.py'nin iki üst dizini.

    Böylece hangi dizinden çalıştırılırsa çalıştırılsın
    `data.root_dir: "data"` her zaman PROJECT_ROOT/data anlamına gelir.
    """
    raw = cfg["data"]["root_dir"]
    p   = Path(raw)
    if p.is_absolute():
        return p
    resolved = (_project_root() / raw).resolve()
    return resolved


# ── Backtest çalıştırıcı ──────────────────────────────────────────────────────

def run_backtest(cfg: dict, config_path: str = "algotrading/config/default.yaml") -> None:
    """
    Tüm bileşenleri config'den bağlayıp tek bir backtest geçişi çalıştırır.
    """
    from algotrading.backtest.commission  import FixedPerShare
    from algotrading.backtest.engine      import BacktestConfig, BacktestEngine
    from algotrading.backtest.portfolio   import Portfolio
    from algotrading.backtest.slippage    import VolatilitySlippage
    from algotrading.core.events          import EventBus
    from algotrading.data.corporate_actions import CorporateActionStore
    from algotrading.data.ingestion       import RawDataStore
    from algotrading.data.pit_handler     import PITDataHandler
    from algotrading.execution.paper_trader import SimulatedExecution
    from algotrading.orchestrator.orchestrator import Orchestrator
    from algotrading.risk.kill_switch     import KillSwitch
    from algotrading.risk.position_sizer  import FixedFractional
    from algotrading.risk.risk_manager    import RiskManager
    from algotrading.strategies.trend_volatility import TrendVolatilityStrategy
    from algotrading.strategies.multi_indicator  import MultiIndicatorStrategy
    from algotrading.validation.metrics   import compute_metrics, compute_trade_metrics

    # ── Veri deposu ───────────────────────────────────────────────────────────
    data_root = _resolve_data_root(cfg, config_path)
    logger.info("Veri deposu : %s", data_root)
    raw_store = RawDataStore(data_root)
    ca_store  = CorporateActionStore(data_root)

    bt_cfg = cfg["backtest"]
    symbols = bt_cfg["symbols"]
    start   = datetime.fromisoformat(bt_cfg["start"]).replace(tzinfo=UTC)
    end     = datetime.fromisoformat(bt_cfg["end"]).replace(tzinfo=UTC)

    # ── Veri kontrolü: eksikse Türkçe hata ver ───────────────────────────────
    for symbol in symbols:
        sym_dir = raw_store.raw_dir / symbol.upper()
        if not sym_dir.exists() or not any(sym_dir.rglob("*.parquet")):
            raise FileNotFoundError(
                f"\n{'='*60}\n"
                f"  Veri bulunamadı: {symbol}\n"
                f"  Beklenen klasör : {sym_dir}\n\n"
                f"  Önce veri indirme komutunu çalıştırın:\n\n"
                f"  python -m algotrading.data.download \\\n"
                f"      --source yfinance \\\n"
                f"      --symbol {symbol} \\\n"
                f"      --start  {bt_cfg['start']} \\\n"
                f"      --end    {bt_cfg['end']}\n"
                f"{'='*60}"
            )

    # ── BacktestConfig ────────────────────────────────────────────────────────
    bc = BacktestConfig(
        name            = cfg["system"]["name"],
        symbols         = symbols,
        start           = start,
        end             = end,
        initial_capital = float(bt_cfg["initial_capital"]),
        random_seed     = int(cfg["system"]["random_seed"]),
    )

    # ── Bileşenler ────────────────────────────────────────────────────────────
    data_handler = PITDataHandler(
        raw_store   = raw_store,
        ca_store    = ca_store,
        window_size = cfg["data"]["window_size"],
    )

    bus       = EventBus()
    portfolio = Portfolio(initial_capital=bc.initial_capital)

    sp           = cfg["strategy"]["params"]
    strategy_id  = cfg["strategy"].get("id", "trend_volatility_v1")

    if strategy_id == "multi_indicator_v1":
        strategy = MultiIndicatorStrategy(
            data_handler  = data_handler,
            bus           = bus,
            symbol        = symbols[0],
            ema_fast      = int(sp.get("ema_fast",      12)),
            ema_slow      = int(sp.get("ema_slow",      26)),
            rsi_period    = int(sp.get("rsi_period",    14)),
            rsi_buy       = float(sp.get("rsi_buy",     55.0)),
            rsi_sell      = float(sp.get("rsi_sell",    45.0)),
            macd_fast     = int(sp.get("macd_fast",     12)),
            macd_slow     = int(sp.get("macd_slow",     26)),
            macd_signal   = int(sp.get("macd_signal",    9)),
            adx_period    = int(sp.get("adx_period",    14)),
            adx_threshold = float(sp.get("adx_threshold", 20.0)),
            atr_period    = int(sp.get("atr_period",    14)),
            vol_threshold = float(sp.get("vol_threshold", 0.035)),
            sl_pct        = float(sp.get("sl_pct",      0.025)),
            tp_pct        = float(sp.get("tp_pct",      0.050)),
            trail_pct     = float(sp.get("trail_pct",   0.015)),
            allow_short   = bool(sp.get("allow_short",  False)),
            cooldown_bars = int(sp.get("cooldown_bars",  3)),
        )
        logger.info("Strateji: MultiIndicatorStrategy (EMA+RSI+MACD+ADX+SL/TP/Trailing)")
    else:
        strategy = TrendVolatilityStrategy(
            data_handler  = data_handler,
            bus           = bus,
            symbol        = symbols[0],
            ema_fast      = int(sp.get("ema_fast",      20)),
            ema_slow      = int(sp.get("ema_slow",      50)),
            atr_period    = int(sp.get("atr_period",    14)),
            vol_threshold = float(sp.get("vol_threshold", 0.025)),
        )
        logger.info("Strateji: TrendVolatilityStrategy (EMA+ATR)")

    oc = cfg["orchestrator"]
    orchestrator = Orchestrator(
        bus            = bus,
        cooldown_bars  = oc["cooldown_bars"],
        min_confidence = oc["min_confidence"],
        allow_short    = oc["allow_short"],
    )

    ks_cfg = cfg["kill_switch"]
    kill_switch = KillSwitch(
        max_daily_loss_pct     = ks_cfg["max_daily_loss_pct"],
        max_drawdown_pct       = ks_cfg["max_drawdown_pct"],
        max_consecutive_losses = ks_cfg["max_consecutive_losses"],
    )
    kill_switch.initialise(bc.initial_capital, at=start)

    rk = cfg["risk"]
    risk_manager = RiskManager(
        portfolio          = portfolio,
        data_handler       = data_handler,
        bus                = bus,
        kill_switch        = kill_switch,
        position_sizer     = FixedFractional(
            max_risk_pct     = rk["max_risk_pct"],
            atr_stop_mult    = rk["atr_stop_mult"],
            max_position_pct = rk["max_position_pct"],
        ),
        max_position_pct   = rk["max_position_pct"],
        max_gross_exposure = rk["max_gross_exposure"],
        max_net_exposure   = rk["max_net_exposure"],
    )

    cm = cfg["commission"]
    sl = cfg["slippage"]
    commission     = FixedPerShare(
        rate_per_share = cm["rate_per_share"],
        min_per_order  = cm["min_per_order"],
    )
    slippage_model = VolatilitySlippage(
        atr_multiple = sl["atr_multiple"],
        min_bps      = sl["min_bps"],
    )

    execution = SimulatedExecution(
        data_handler = data_handler,
        bus          = bus,
        commission   = commission,
        slippage     = slippage_model,
    )

    engine = BacktestEngine(
        config       = bc,
        data_handler = data_handler,
        strategy     = strategy,
        orchestrator = orchestrator,
        risk_manager = risk_manager,
        execution    = execution,
        commission   = commission,
        slippage     = slippage_model,
        bus          = bus,
        portfolio    = portfolio,
    )

    # ── Çalıştır ──────────────────────────────────────────────────────────────
    result = engine.run()

    # ── Veri yeterli mi? ──────────────────────────────────────────────────────
    eq_curve = result.equity_curve
    if len(eq_curve) < 5:
        logger.warning("Yeterli equity verisi yok — metrik hesaplanamadı")
        return result

    # ── Metrikler ─────────────────────────────────────────────────────────────
    metrics = compute_metrics(
        eq_curve["equity"],
        bars_per_year = cfg["data"]["bars_per_year"],
    )
    trade_m = compute_trade_metrics(result.fills)

    orch_sum = orchestrator.summary()
    risk_sum = risk_manager.summary()
    summary  = portfolio.summary()

    # ── Türkçe sonuç çıktısı ──────────────────────────────────────────────────
    sep = "=" * 62
    logger.info(sep)
    logger.info("  BACKTEST SONUÇLARI — %s", bc.name.upper())
    logger.info("  Dönem          : %s → %s", bc.start.date(), bc.end.date())
    logger.info("  Sembol(ler)    : %s", ", ".join(bc.symbols))
    logger.info(sep)
    logger.info("  Başlangıç Serm.: %s USD",  f"{bc.initial_capital:>12,.2f}")
    logger.info("  Bitiş Sermayesi: %s USD",   f"{summary['final_equity']:>12,.2f}")
    logger.info("  Toplam Getiri  : %s%%",      f"{metrics.total_return_pct:>+10.2f}")
    logger.info("  Yıllık Getiri  : %s%%",      f"{metrics.annualised_return:>+10.2f}")
    logger.info("  Yıllık Volatil.: %s%%",      f"{metrics.annualised_vol:>10.2f}")
    logger.info("  Sharpe Oranı   : %s",        f"{metrics.sharpe_ratio:>10.3f}")
    logger.info("  Sortino Oranı  : %s",        f"{metrics.sortino_ratio:>10.3f}")
    logger.info("  Maks. Düşüş    : %s%%",      f"{metrics.max_drawdown_pct:>10.2f}")
    logger.info(sep)
    logger.info("  İşlem Sayısı   : %s",        f"{trade_m.get('num_trades', 0):>10}")
    logger.info("  Kazanan Oran   : %s%%",      f"{trade_m.get('win_rate', 0.0):>10.1f}")
    logger.info("  Risk Vetolu    : %s",        f"{risk_sum['blocked_orders']:>10}")
    logger.info("  Risk Onaylı    : %s",        f"{risk_sum['approved_orders']:>10}")
    logger.info("  Karar Sayısı   : %s",        f"{orch_sum['total_decisions']:>10}")
    if kill_switch.is_triggered:
        logger.warning("  ⚠ KİLİT AKTİF  : %s", kill_switch.trigger_reason)
    logger.info(sep)

    return result


# ── Ana giriş noktası ─────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AlgoTrading MVP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", default="algotrading/config/default.yaml",
        help="YAML konfigürasyon dosyası yolu",
    )
    parser.add_argument(
        "--mode", default="backtest",
        choices=["backtest", "paper", "validate"],
        help="Çalışma modu",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    log_level = getattr(logging, cfg["system"].get("log_level", "INFO").upper(), logging.INFO)
    logging.getLogger().setLevel(log_level)

    logger.info("AlgoTrading MVP başlatılıyor | mod=%s | ortam=%s",
                args.mode, cfg["system"]["env"])

    if args.mode == "backtest":
        run_backtest(cfg, config_path=args.config)
    elif args.mode == "paper":
        logger.warning("Paper modu henüz aktif değil — önce backtest çalıştırın")
        sys.exit(1)
    elif args.mode == "validate":
        l