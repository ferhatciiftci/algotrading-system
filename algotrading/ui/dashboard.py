"""
ui/dashboard.py  —  AlgoTrading MVP  Türkçe Backtest Paneli v4
──────────────────────────────────────────────────────────────
Düzeltmeler (v4):
  - pandas applymap → map (pandas 2.x uyumluluğu)
  - Sıfır / bir işlem ile çökmez
  - Boş aylık getiri ile çökmez
  - Tüm grafik bölümleri try/except ile korunur
  - Kaynak seçici: Yahoo Finance | Binance Kripto | Stooq Günlük
  - Strateji onay modu: Katı | Dengeli | Esnek
  - Güçlü Türkçe nesnel yorumlar

UYARI: Yalnızca geçmiş veri simülasyonu. Canlı işlem yapılmaz.
"""
from __future__ import annotations
import sys, traceback, copy, logging
from datetime import date, datetime, timezone
from pathlib import Path

_UI_DIR       = Path(__file__).resolve().parent
_PKG_DIR      = _UI_DIR.parent
_PROJECT_ROOT = _PKG_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_missing = []
try:    import streamlit as st
except ImportError: _missing.append("streamlit")
try:
    import plotly.graph_objects as go
    import plotly.express as px
except ImportError: _missing.append("plotly")
if _missing:
    print(f"EKSİK: pip install {' '.join(_missing)}")
    sys.exit(1)

import numpy as np
import pandas as pd
import yaml

UTC = timezone.utc
logging.basicConfig(level=logging.WARNING)

# ─── Sayfa yapılandırması ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="AlgoTrading MVP",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("""
<style>
.stAlert p{margin:0}
.block-container{padding-top:1rem}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Yardımcılar
# ─────────────────────────────────────────────────────────────────────────────

def _load_config() -> dict:
    p = _PKG_DIR / "config" / "default.yaml"
    if not p.exists():
        st.error(f"Config bulunamadı: {p}"); st.stop()
    with open(p) as f:
        return yaml.safe_load(f)

def _fmt_pct(v, d=2):
    if v is None or (isinstance(v, float) and v != v): return "—"
    return f"{'+'if v>=0 else ''}{v:.{d}f}%"

def _fmt_usd(v): return f"${v:,.2f}"

def _fmt_f(v, d=3):
    if v is None or (isinstance(v, float) and v != v): return "—"
    return f"{v:.{d}f}"

def _safe_df_map(df: pd.DataFrame, fn):
    """pandas 2.x: DataFrame.map; eski: applymap — her ikisini destekler."""
    try:
        return df.map(fn)
    except AttributeError:
        return df.applymap(fn)  # type: ignore[attr-defined]

def _data_exists(symbol: str, cfg: dict) -> bool:
    try:
        from algotrading.data.ingestion import RawDataStore
        from algotrading.main import _resolve_data_root
        root    = _resolve_data_root(cfg)
        store   = RawDataStore(root)
        sym_dir = store.raw_dir / symbol.upper()
        return sym_dir.exists() and any(sym_dir.rglob("*.parquet"))
    except Exception:
        return False

def _download_data(symbol: str, start: str, end: str, source: str, cfg: dict) -> str:
    from algotrading.data.ingestion import RawDataStore
    from algotrading.main import _resolve_data_root
    root  = _resolve_data_root(cfg)
    store = RawDataStore(root)
    if source == "Yahoo Finance":
        from algotrading.data.connectors.yfinance_connector import YFinanceConnector
        return YFinanceConnector().fetch_and_store(symbol, start, end, store)
    elif source == "Binance Kripto":
        from algotrading.data.connectors.binance_connector import BinanceConnector
        return BinanceConnector(interval="1d").fetch_and_store(symbol, start, end, store)
    elif source == "Stooq Günlük":
        from algotrading.data.connectors.stooq_connector import StooqConnector
        return StooqConnector().fetch_and_store(symbol, start, end, store)
    raise ValueError(f"Bilinmeyen kaynak: {source}")

def _run_backtest(cfg: dict):
    from algotrading.main import run_backtest
    return run_backtest(cfg)


# ─────────────────────────────────────────────────────────────────────────────
# KENAR ÇUBUĞU
# ─────────────────────────────────────────────────────────────────────────────

st.sidebar.title("⚙️ Backtest Ayarları")
st.sidebar.caption("⚠️ Yalnızca geçmiş veri simülasyonu. Canlı işlem yapılmaz.")

cfg_base = _load_config()

# ── Veri & Dönem ──────────────────────────────────────────────────────────────
with st.sidebar.expander("📊 Veri & Dönem", expanded=True):
    source = st.selectbox(
        "Veri Kaynağı",
        ["Yahoo Finance", "Binance Kripto", "Stooq Günlük"],
        help=(
            "Yahoo Finance: Hisse/ETF (SPY, AAPL…)\n"
            "Binance Kripto: BTCUSDT, ETHUSDT…\n"
            "Stooq Günlük: spy.us, aapl.us, ^spx…"
        ),
    )
    _default_sym = {
        "Yahoo Finance" : cfg_base["backtest"]["symbols"][0],
        "Binance Kripto": "BTCUSDT",
        "Stooq Günlük"  : "spy.us",
    }.get(source, "SPY")
    sembol = st.text_input("Sembol", value=_default_sym).strip()
    if source != "Binance Kripto":
        sembol = sembol.upper()

    c1, c2 = st.columns(2)
    with c1:
        baslangic = st.date_input("Başlangıç",
                                  value=date.fromisoformat(cfg_base["backtest"]["start"]))
    with c2:
        bitis = st.date_input("Bitiş",
                              value=date.fromisoformat(cfg_base["backtest"]["end"]))
    sermaye = st.number_input("Başlangıç Sermayesi ($)", 1000, 10_000_000,
                              int(cfg_base["backtest"]["initial_capital"]), 1000)

# ── Strateji ─────────────────────────────────────────────────────────────────
with st.sidebar.expander("📈 Strateji Parametreleri", expanded=True):
    _strat_opts = {
        "multi_indicator_v1" : "📊 Multi-Indicator (EMA+RSI+MACD+ADX)",
        "trend_volatility_v1": "📉 Trend-Volatilite (EMA+ATR)",
    }
    _def_id  = cfg_base["strategy"].get("id", "multi_indicator_v1")
    _def_idx = list(_strat_opts.keys()).index(_def_id) if _def_id in _strat_opts else 0
    strategy_id = st.selectbox("Strateji",
                               options=list(_strat_opts.keys()),
                               format_func=lambda k: _strat_opts[k],
                               index=_def_idx)
    sp = cfg_base["strategy"]["params"]

    # Ortak: EMA
    st.markdown("**EMA**")
    col_a, col_b = st.columns(2)
    with col_a:
        ema_fast = st.number_input("Hızlı", 2, 100, int(sp.get("ema_fast", 12)))
    with col_b:
        ema_slow = st.number_input("Yavaş", 5, 500, int(sp.get("ema_slow", 26)))

    col_c, col_d = st.columns(2)
    with col_c:
        atr_period = st.number_input("ATR Per.", 5, 100, int(sp.get("atr_period", 14)))
    with col_d:
        vol_threshold = st.number_input("Vol. Eşiği", 0.005, 0.20,
                                        float(sp.get("vol_threshold", 0.035)),
                                        0.005, format="%.3f")

    if strategy_id == "multi_indicator_v1":
        # Onay modu
        _mode_opts  = {"balanced":"⚖️ Dengeli (önerilen)",
                       "strict"  :"🔒 Katı (az işlem)",
                       "loose"   :"🔓 Esnek (fazla işlem)"}
        _def_mode   = sp.get("confirmation_mode", "balanced")
        _def_m_idx  = list(_mode_opts.keys()).index(_def_mode) \
                      if _def_mode in _mode_opts else 0
        confirmation_mode = st.selectbox(
            "Onay Modu",
            options=list(_mode_opts.keys()),
            format_func=lambda k: _mode_opts[k],
            index=_def_m_idx,
            help=(
                "Katı: EMA+RSI+MACD+ADX hepsini gerektirir\n"
                "Dengeli: EMA+RSI + MACD veya ADX yeterli\n"
                "Esnek: Yalnızca EMA+RSI yeterli (daha fazla işlem)"
            ),
        )

        st.markdown("**RSI**")
        cr1, cr2, cr3 = st.columns(3)
        with cr1:
            rsi_period = st.number_input("Per.", 2, 50, int(sp.get("rsi_period", 14)))
        with cr2:
            rsi_buy    = st.number_input("Alış <", 30.0, 80.0,
                                         float(sp.get("rsi_buy", 55.0)), 1.0)
        with cr3:
            rsi_sell   = st.number_input("Satış >", 20.0, 70.0,
                                         float(sp.get("rsi_sell", 45.0)), 1.0)

        st.markdown("**MACD**")
        cm1, cm2, cm3 = st.columns(3)
        with cm1:
            macd_fast   = st.number_input("Hızlı", 2, 50, int(sp.get("macd_fast", 12)))
        with cm2:
            macd_slow   = st.number_input("Yavaş", 5, 100, int(sp.get("macd_slow", 26)))
        with cm3:
            macd_signal = st.number_input("Sinyal", 2, 30, int(sp.get("macd_signal", 9)))

        st.markdown("**ADX**")
        ca1, ca2 = st.columns(2)
        with ca1:
            adx_period    = st.number_input("Per.", 5, 50,
                                            int(sp.get("adx_period", 14)))
        with ca2:
            adx_threshold = st.number_input("Eşik", 10.0, 50.0,
                                            float(sp.get("adx_threshold", 20.0)), 1.0)

        st.markdown("**Stop-Loss / TP / Trailing**")
        cs1, cs2, cs3 = st.columns(3)
        with cs1:
            sl_pct    = st.number_input("SL %", 0.5, 20.0,
                                        float(sp.get("sl_pct",    0.025))*100,
                                        0.5, format="%.1f") / 100
        with cs2:
            tp_pct    = st.number_input("TP %", 1.0, 50.0,
                                        float(sp.get("tp_pct",    0.05 ))*100,
                                        0.5, format="%.1f") / 100
        with cs3:
            trail_pct = st.number_input("Trail %", 0.0, 20.0,
                                        float(sp.get("trail_pct", 0.015))*100,
                                        0.5, format="%.1f") / 100

        ce1, ce2 = st.columns(2)
        with ce1:
            cooldown_bars = st.number_input("Bekleme (bar)", 0, 20,
                                            int(sp.get("cooldown_bars", 3)))
        with ce2:
            allow_short = st.checkbox("Açığa Satış",
                                      value=bool(sp.get("allow_short", False)))
    else:
        allow_short = st.checkbox("Açığa Satışa İzin Ver",
                                  value=bool(cfg_base["orchestrator"].get("allow_short", False)))

# ── Risk ──────────────────────────────────────────────────────────────────────
with st.sidebar.expander("🛡️ Risk Parametreleri", expanded=False):
    rk = cfg_base["risk"];  ks = cfg_base["kill_switch"]
    cm = cfg_base["commission"]; sl = cfg_base["slippage"]

    max_risk_pct     = st.slider("İşlem Başına Risk %", 0.1, 5.0,
                                 float(rk["max_risk_pct"])*100, 0.1) / 100
    max_position_pct = st.slider("Maks. Pozisyon %", 1.0, 50.0,
                                 float(rk["max_position_pct"])*100, 1.0) / 100
    atr_stop_mult    = st.number_input("Stop ATR Çarpanı", 0.5, 8.0,
                                       float(rk.get("atr_stop_mult", 2.0)), 0.5)
    max_drawdown_pct = st.slider("Kill Switch DD %", 2.0, 50.0,
                                 float(ks["max_drawdown_pct"])*100, 1.0) / 100
    max_daily_loss   = st.slider("Maks. Günlük Zarar %", 0.5, 10.0,
                                 float(ks["max_daily_loss_pct"])*100, 0.5) / 100
    commission_rate  = st.number_input("Komisyon $/hisse", 0.0, 0.05,
                                       float(cm.get("rate_per_share", 0.005)),
                                       0.001, format="%.4f")
    slippage_mult    = st.number_input("Slippage ATR Çarpanı", 0.0, 1.0,
                                       float(sl.get("atr_multiple", 0.1)),
                                       0.05, format="%.2f")

st.sidebar.divider()

# ── Veri indirme ──────────────────────────────────────────────────────────────
veri_var = _data_exists(sembol, cfg_base)
if not veri_var:
    st.sidebar.warning(f"⚠️ **{sembol}** için yerel veri bulunamadı.")
    _btn_label = f"⬇️ {sembol} İndir ({source})"
    if st.sidebar.button(_btn_label, type="secondary", use_container_width=True):
        with st.spinner(f"{sembol} indiriliyor ({baslangic} → {bitis})…"):
            try:
                _download_data(sembol, str(baslangic), str(bitis), source, cfg_base)
                st.sidebar.success("✅ Veri indirildi!")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"İndirme hatası:\n{e}")
else:
    st.sidebar.success(f"✅ {sembol} verisi mevcut.")
    if st.sidebar.button(f"🔄 {sembol} Veriyi Güncelle", type="secondary",
                         use_container_width=True):
        with st.spinner("Güncelleniyor…"):
            try:
                _download_data(sembol, str(baslangic), str(bitis), source, cfg_base)
                st.sidebar.success("✅ Güncellendi!")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Güncelleme hatası:\n{e}")

calistir = st.sidebar.button("▶  Backtest Çalıştır", type="primary",
                              use_container_width=True, disabled=(not veri_var))
if not veri_var:
    st.sidebar.caption("Önce veriyi indirin, ardından backtest çalıştırın.")


# ─────────────────────────────────────────────────────────────────────────────
# ANA SAYFA
# ─────────────────────────────────────────────────────────────────────────────

st.title("📈 AlgoTrading MVP — Backtest Paneli")
st.caption(
    "⚠️ Bu panel **yalnızca geçmiş veri simülasyonu** sonuçlarını görselleştirir. "
    "Canlı alım-satım yapılmaz ve yapılamaz. Finansal tavsiye değildir."
)

if not calistir:
    if not veri_var:
        st.info(f"👈 Sol panelden **{sembol}** verisini indirin, ardından **Backtest Çalıştır** butonuna tıklayın.")
    else:
        st.info("👈 Parametreleri ayarlayın ve **▶ Backtest Çalıştır** butonuna tıklayın.")
    st.stop()

# Temel doğrulamalar
if baslangic >= bitis:
    st.error("❌ Başlangıç tarihi bitiş tarihinden önce olmalıdır."); st.stop()
if ema_fast >= ema_slow:
    st.error("❌ EMA Hızlı değeri EMA Yavaş değerinden küçük olmalıdır."); st.stop()
if strategy_id == "multi_indicator_v1":
    if macd_fast >= macd_slow:
        st.error("❌ MACD Hızlı değeri MACD Yavaş değerinden küçük olmalıdır."); st.stop()
    if sl_pct > 0 and tp_pct > 0 and sl_pct >= tp_pct:
        st.error("❌ Stop-Loss Take-Profit'ten küçük olmalıdır."); st.stop()

# Config oluştur
cfg = copy.deepcopy(cfg_base)
cfg["backtest"]["symbols"]         = [sembol]
cfg["backtest"]["start"]           = str(baslangic)
cfg["backtest"]["end"]             = str(bitis)
cfg["backtest"]["initial_capital"] = float(sermaye)
cfg["strategy"]["id"]              = strategy_id
spc = cfg["strategy"]["params"]
spc["ema_fast"]      = int(ema_fast)
spc["ema_slow"]      = int(ema_slow)
spc["atr_period"]    = int(atr_period)
spc["vol_threshold"] = float(vol_threshold)
cfg["orchestrator"]["allow_short"]         = allow_short
cfg["risk"]["max_risk_pct"]                = float(max_risk_pct)
cfg["risk"]["max_position_pct"]            = float(max_position_pct)
cfg["risk"]["atr_stop_mult"]               = float(atr_stop_mult)
cfg["kill_switch"]["max_drawdown_pct"]     = float(max_drawdown_pct)
cfg["kill_switch"]["max_daily_loss_pct"]   = float(max_daily_loss)
cfg["commission"]["rate_per_share"]        = float(commission_rate)
cfg["slippage"]["atr_multiple"]            = float(slippage_mult)

if strategy_id == "multi_indicator_v1":
    spc.update({
        "rsi_period":       int(rsi_period),
        "rsi_buy":          float(rsi_buy),
        "rsi_sell":         float(rsi_sell),
        "macd_fast":        int(macd_fast),
        "macd_slow":        int(macd_slow),
        "macd_signal":      int(macd_signal),
        "adx_period":       int(adx_period),
        "adx_threshold":    float(adx_threshold),
        "sl_pct":           float(sl_pct),
        "tp_pct":           float(tp_pct),
        "trail_pct":        float(trail_pct),
        "allow_short":      allow_short,
        "cooldown_bars":    int(cooldown_bars),
        "confirmation_mode":confirmation_mode,
    })

# Backtest çalıştır
with st.spinner(f"⏳ {sembol} backtest çalışıyor ({baslangic} → {bitis})…"):
    try:
        result = _run_backtest(cfg)
    except FileNotFoundError as e:
        st.error(str(e)); st.stop()
    except Exception as e:
        st.error(f"Backtest hatası: {e}")
        with st.expander("Teknik detay"):
            st.code(traceback.format_exc())
        st.stop()

if result is None:
    st.warning("⚠️ Backtest sonuç üretmedi."); st.stop()

# Rapor
try:
    from algotrading.reporting.performance import (
        compute_full_report, generate_commentary,
        compute_bh_return, attach_benchmark,
    )
    _param_count = 4 if strategy_id == "trend_volatility_v1" else 12
    report   = compute_full_report(result, cfg, param_count=_param_count)
    eq_curve = report.equity_curve
    fills    = report.fills
    bh_total, bh_ann, bh_eq = compute_bh_return(eq_curve, float(sermaye))
    attach_benchmark(report, bh_total, bh_ann)
except ValueError as e:
    st.warning(f"⚠️ Rapor oluşturulamadı: {e}"); st.stop()
except Exception as e:
    st.error(f"Rapor hatası: {e}")
    with st.expander("Teknik detay"):
        st.code(traceback.format_exc())
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# BÖLÜM 1 — PERFORMANS ÖZETİ
# ─────────────────────────────────────────────────────────────────────────────

_mode_label = ""
if strategy_id == "multi_indicator_v1":
    _ml = {"strict":"Katı","balanced":"Dengeli","loose":"Esnek"}
    _mode_label = f"  |  Onay: {_ml.get(confirmation_mode, confirmation_mode)}"

st.subheader("📊 Performans Özeti")
st.caption(
    f"Dönem: {baslangic} → {bitis}  |  Sembol: {sembol}  |  "
    f"Sermaye: {_fmt_usd(sermaye)}  |  Strateji: {_strat_opts.get(strategy_id,'?')}"
    f"{_mode_label}"
)

r = report
c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("💰 Başlangıç",     _fmt_usd(sermaye))
c2.metric("🏦 Bitiş",         _fmt_usd(r.final_equity),
          delta=_fmt_pct(r.total_return_pct))
c3.metric("📈 Toplam Getiri", _fmt_pct(r.total_return_pct))
c4.metric("📅 Yıllık Getiri", _fmt_pct(r.ann_return_pct))
c5.metric("📉 Maks. Düşüş",  _fmt_pct(r.max_drawdown_pct))

c6,c7,c8,c9,c10 = st.columns(5)
c6.metric("⚖️ Sharpe",        _fmt_f(r.sharpe_ratio))
c7.metric("📐 Sortino",       _fmt_f(r.sortino_ratio))
c8.metric("🎯 Calmar",        _fmt_f(r.calmar_ratio))
c9.metric("🔄 İşlem Sayısı",  str(r.num_trades))
c10.metric("🏆 Kazanma %",    f"{r.win_rate:.1f}%" if r.num_trades > 0 else "—")

c11,c12,c13,c14,c15 = st.columns(5)
c11.metric("💹 Kâr Faktörü",  _fmt_f(r.profit_factor))
c12.metric("📊 Ort. Kazanç",  _fmt_pct(r.avg_win_pct)  if r.num_trades > 0 else "—")
c13.metric("📊 Ort. Kayıp",   _fmt_pct(r.avg_loss_pct) if r.num_trades > 0 else "—")
c14.metric("🏅 En İyi İşlem", _fmt_pct(r.best_trade_pct)  if r.num_trades > 0 else "—")
c15.metric("⚠️ En Kötü",      _fmt_pct(r.worst_trade_pct) if r.num_trades > 0 else "—")

st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# BÖLÜM 2 — GRAFİKLER
# ─────────────────────────────────────────────────────────────────────────────

st.subheader("📉 Grafik Analizi")

tab1, tab2, tab3, tab4 = st.tabs([
    "Sermaye Eğrisi", "İşlem Noktaları", "Düşüş Grafiği", "Aylık Getiriler"
])

# ── Tab 1 ─────────────────────────────────────────────────────────────────────
with tab1:
    try:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=eq_curve["timestamp"], y=eq_curve["equity"],
            name="Strateji", mode="lines",
            line=dict(color="#2196F3", width=2),
        ))
        if bh_eq is not None and len(bh_eq) > 0:
            fig.add_trace(go.Scatter(
                x=bh_eq.index, y=bh_eq.values,
                name="Buy & Hold", mode="lines",
                line=dict(color="#9E9E9E", width=1.5, dash="dot"),
            ))
        fig.add_hline(y=float(sermaye), line_dash="dash",
                      line_color="#FF9800", line_width=1,
                      annotation_text="Başlangıç")
        fig.update_layout(
            title=f"{sembol} — Sermaye Eğrisi",
            xaxis_title="Tarih", yaxis_title="Sermaye ($)",
            hovermode="x unified", height=430,
            plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(orientation="h", y=1.02),
            yaxis=dict(tickprefix="$", gridcolor="#f5f5f5"),
            xaxis=dict(gridcolor="#f5f5f5"),
        )
        st.plotly_chart(fig, use_container_width=True)

        try:
            bh_ok = bh_total is not None and not (isinstance(bh_total,float) and bh_total!=bh_total)
            if bh_ok:
                diff = r.total_return_pct - bh_total
                ca,cb,cc = st.columns(3)
                ca.metric("Strateji",    _fmt_pct(r.total_return_pct))
                cb.metric("Buy & Hold",  _fmt_pct(bh_total))
                cc.metric("Fark",        _fmt_pct(diff))
        except Exception:
            pass
    except Exception as e:
        st.warning(f"Sermaye grafiği oluşturulamadı: {e}")

# ── Tab 2 ─────────────────────────────────────────────────────────────────────
with tab2:
    try:
        if not fills:
            st.info("Bu backtest döneminde hiç işlem gerçekleşmedi — işaret noktası yok.")
        else:
            eq_ts  = eq_curve["timestamp"].values
            eq_val = eq_curve["equity"].values

            def _snap(ts):
                idx = np.argmin(np.abs(eq_ts - np.datetime64(pd.Timestamp(ts))))
                return float(eq_val[idx])

            long_f  = [f for f in fills if f.direction.value.upper() in ("LONG","BUY")]
            short_f = [f for f in fills if f.direction.value.upper() in ("SHORT","SELL","FLAT")]

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=eq_curve["timestamp"], y=eq_curve["equity"],
                name="Sermaye", mode="lines",
                line=dict(color="#607D8B", width=1.5),
            ))
            if long_f:
                fig2.add_trace(go.Scatter(
                    x=[f.timestamp for f in long_f],
                    y=[_snap(f.timestamp) for f in long_f],
                    mode="markers", name="Alış",
                    marker=dict(symbol="triangle-up", size=12, color="#4CAF50",
                                line=dict(color="white", width=1)),
                ))
            if short_f:
                fig2.add_trace(go.Scatter(
                    x=[f.timestamp for f in short_f],
                    y=[_snap(f.timestamp) for f in short_f],
                    mode="markers", name="Satış/Çıkış",
                    marker=dict(symbol="triangle-down", size=12, color="#F44336",
                                line=dict(color="white", width=1)),
                ))
            fig2.update_layout(
                title=f"{sembol} — İşlem Noktaları",
                xaxis_title="Tarih", yaxis_title="Sermaye ($)",
                hovermode="x unified", height=420,
                plot_bgcolor="white", paper_bgcolor="white",
                legend=dict(orientation="h", y=1.02),
                yaxis=dict(tickprefix="$", gridcolor="#f5f5f5"),
                xaxis=dict(gridcolor="#f5f5f5"),
            )
            st.plotly_chart(fig2, use_container_width=True)
            st.caption(f"🟢 Alış: {len(long_f)}  |  🔴 Satış/Çıkış: {len(short_f)}")
    except Exception as e:
        st.warning(f"İşlem grafiği oluşturulamadı: {e}")

# ── Tab 3 ─────────────────────────────────────────────────────────────────────
with tab3:
    try:
        dd = eq_curve["drawdown"] * 100
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=eq_curve["timestamp"], y=dd,
            name="Drawdown", mode="lines",
            line=dict(color="#F44336", width=1.5),
            fill="tozeroy", fillcolor="rgba(244,67,54,0.12)",
        ))
        fig3.add_hline(y=0, line_color="black", line_width=0.5)
        fig3.add_hline(
            y=-abs(max_drawdown_pct*100),
            line_dash="dash", line_color="#FF9800",
            annotation_text=f"Kill Switch ({-abs(max_drawdown_pct*100):.0f}%)",
            annotation_position="bottom right",
        )
        fig3.update_layout(
            title=f"{sembol} — Drawdown",
            xaxis_title="Tarih", yaxis_title="Düşüş (%)",
            hovermode="x unified", height=390,
            plot_bgcolor="white", paper_bgcolor="white",
            yaxis=dict(ticksuffix="%", gridcolor="#f5f5f5"),
            xaxis=dict(gridcolor="#f5f5f5"),
        )
        st.plotly_chart(fig3, use_container_width=True)
    except Exception as e:
        st.warning(f"Drawdown grafiği oluşturulamadı: {e}")

# ── Tab 4 ─────────────────────────────────────────────────────────────────────
with tab4:
    try:
        eq_m = eq_curve.copy()
        eq_m["timestamp"] = pd.to_datetime(eq_m["timestamp"])
        eq_m = eq_m.set_index("timestamp").sort_index()
        monthly_eq  = eq_m["equity"].resample("ME").last()
        monthly_ret = monthly_eq.pct_change().dropna() * 100

        if len(monthly_ret) < 2:
            st.info("Aylık ısı haritası için en az 2 aylık veri gerekiyor. "
                    "Tarih aralığını genişletin.")
        else:
            mr = monthly_ret.reset_index()
            mr.columns = ["tarih", "getiri"]
            mr["yil"] = mr["tarih"].dt.year
            mr["ay"]  = mr["tarih"].dt.month
            ay_ad = {1:"Oca",2:"Şub",3:"Mar",4:"Nis",5:"May",6:"Haz",
                     7:"Tem",8:"Ağu",9:"Eyl",10:"Eki",11:"Kas",12:"Ara"}
            mr["ay_adi"] = mr["ay"].map(ay_ad)
            pivot  = mr.pivot_table(index="yil", columns="ay_adi",
                                    values="getiri", aggfunc="sum")
            sirali = [ay_ad[i] for i in range(1,13) if ay_ad[i] in pivot.columns]
            pivot  = pivot.reindex(columns=sirali)
            try:
                pivot["Yıllık"] = mr.groupby("yil")["getiri"].sum()
            except Exception:
                pass

            if pivot.empty:
                st.info("Aylık pivot tablosu oluşturulamadı.")
            else:
                fig4 = px.imshow(
                    pivot.values,
                    x=pivot.columns.tolist(),
                    y=pivot.index.astype(str).tolist(),
                    color_continuous_scale="RdYlGn",
                    color_continuous_midpoint=0,
                    text_auto=".1f",
                    labels=dict(color="Getiri (%)"),
                    title="Aylık Getiri Isı Haritası (%)",
                    aspect="auto",
                )
                fig4.update_coloraxes(colorbar_ticksuffix="%")
                fig4.update_layout(height=min(500, 80 + 60 * len(pivot)),
                                   plot_bgcolor="white", paper_bgcolor="white")
                st.plotly_chart(fig4, use_container_width=True)

                with st.expander("Sayısal tablo"):
                    try:
                        # pandas 2.x uyumlu: DataFrame.map
                        fmt = _safe_df_map(
                            pivot,
                            lambda v: f"{v:+.1f}%" if pd.notna(v) else "—"
                        )
                        st.dataframe(fmt, use_container_width=True)
                    except Exception:
                        st.dataframe(pivot.round(2), use_container_width=True)
    except Exception as e:
        st.info(f"Aylık getiri tablosu oluşturulamadı: {e}")

st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# BÖLÜM 3 — İŞLEM GÜNLÜĞÜ
# ─────────────────────────────────────────────────────────────────────────────

st.subheader("📋 İşlem Günlüğü")

if not fills:
    st.info("Bu backtest döneminde hiç işlem gerçekleşmedi.")
    if strategy_id == "multi_indicator_v1":
        st.markdown(
            "💡 **İşlem sayısını artırmak için:** Sol panelde "
            "**Onay Modunu → Esnek** olarak değiştirin veya "
            "RSI Alış eşiğini yükseltin (örn. 60), ADX eşiğini düşürün (örn. 15)."
        )
else:
    try:
        dir_tr = {"long":"Alış","LONG":"Alış","short":"Satış","SHORT":"Satış",
                  "flat":"Çıkış","FLAT":"Çıkış","buy":"Alış","BUY":"Alış",
                  "sell":"Satış","SELL":"Satış"}
        rows = []
        for f in fills:
            rows.append({
                "Tarih"     : f.timestamp.strftime("%Y-%m-%d") if hasattr(f.timestamp,"strftime") else str(f.timestamp),
                "Sembol"    : f.symbol,
                "Yön"       : dir_tr.get(f.direction.value, f.direction.value),
                "Fiyat ($)" : f"{f.fill_price:.4f}",
                "Adet"      : f"{f.quantity:.0f}",
                "Komisyon"  : f"{f.commission:.3f}",
                "Sebep"     : (getattr(f,"reason","") or "Strateji sinyali"),
            })
        df_t   = pd.DataFrame(rows)
        yon_f  = st.selectbox("Filtre",
                              ["Tümü","Alış","Satış","Çıkış"])
        df_vis = df_t if yon_f == "Tümü" else df_t[df_t["Yön"] == yon_f]
        st.dataframe(df_vis, use_container_width=True,
                     height=min(420, 50 + 35*len(df_vis)), hide_index=True)
        st.caption(f"Toplam {len(fills)} işlem  |  Görüntülenen: {len(df_vis)}")
    except Exception as e:
        st.warning(f"İşlem günlüğü gösterilemiyor: {e}")

st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# BÖLÜM 4 — NESNEL TÜRKÇE DEĞERLENDİRME
# ─────────────────────────────────────────────────────────────────────────────

st.subheader("💬 Nesnel Sistem Değerlendirmesi")
st.caption("Bu yorumlar metrik eşiklerine dayanır. Yatırım tavsiyesi değildir.")

# Strateji aşırı filtrelenmiş mi kontrolü
_over_filtered = (r.num_trades < 5)
_enough_trades  = (r.num_trades >= 20)
_bh_ok = (bh_total is not None and not (isinstance(bh_total,float) and bh_total!=bh_total))
_beats_bh = _bh_ok and r.total_return_pct > bh_total
_sharpe_meaningful = (r.num_trades >= 10 and abs(r.sharpe_ratio) > 0.01)

# Dinamik yorumlar (reporting modülünün üstüne ek kontroller)
extra_items = []

if _over_filtered:
    extra_items.append(("🔴", "Strateji Aşırı Filtrelenmiş",
        f"Yalnızca {r.num_trades} işlem gerçekleşti. Bu kadar az işlemle performans "
        "değerlendirmek güvenilmez değildir. Sol panelden **Onay Modunu → Esnek** "
        "yaparak veya RSI/ADX eşiklerini gevşeterek işlem sayısını artırabilirsiniz."))
elif not _enough_trades:
    extra_items.append(("🟠", "İşlem Sayısı Sınırlı",
        f"{r.num_trades} işlem istatistiksel değerlendirme için yeterli olmayabilir. "
        "20+ işlem daha güvenilir sonuç verir."))
else:
    extra_items.append(("✅", "İşlem Sayısı Yeterli",
        f"{r.num_trades} işlemle istatistiksel değerlendirme yapılabilir."))

if r.total_return_pct > 0:
    extra_items.append(("✅", "Sistem Kârlı",
        f"Strateji seçilen dönemde %{r.total_return_pct:.2f} getiri sağladı."))
else:
    extra_items.append(("🔴", "Sistem Kârsız",
        f"Seçilen dönemde %{r.total_return_pct:.2f} kayıp yaşandı. "
        "Parametreleri değiştirin veya farklı dönem deneyin."))

if _bh_ok:
    if _beats_bh:
        extra_items.append(("✅", "Buy & Hold'u Geçiyor",
            f"Strateji ({_fmt_pct(r.total_return_pct)}) > Buy & Hold "
            f"({_fmt_pct(bh_total)}) — {_fmt_pct(r.total_return_pct - bh_total)} fark."))
    else:
        extra_items.append(("🟠", "Buy & Hold Gerisinde",
            f"Buy & Hold ({_fmt_pct(bh_total)}) > Strateji "
            f"({_fmt_pct(r.total_return_pct)}). Pasif yatırım bu dönemde daha iyi."))

if _sharpe_meaningful:
    if r.sharpe_ratio > 1.0:
        extra_items.append(("✅", "Sharpe Oranı Olumlu",
            f"Sharpe {r.sharpe_ratio:.2f} — risk-getiri dengesi iyi."))
    elif r.sharpe_ratio > 0:
        extra_items.append(("🟡", "Sharpe Düşük",
            f"Sharpe {r.sharpe_ratio:.2f} — risk alınan değer üretiliyor "
            "ancak oran düşük."))
    else:
        extra_items.append(("🔴", "Sharpe Negatif",
            f"Sharpe {r.sharpe_ratio:.2f} — risk alınan değer üretilmiyor."))
else:
    extra_items.append(("⚪", "Sharpe Hesaplanamadı",
        "Yeterli işlem olmadan Sharpe oranı anlamlı değil."))

# Overfitting uyarısı
if r.num_trades > 0 and r.num_trades < 15 and r.total_return_pct > 20:
    extra_items.append(("🟠", "Overfitting Riski Yüksek",
        f"Az işlem ({r.num_trades}) ile yüksek getiri (%{r.total_return_pct:.1f}) "
        "tipik bir overfitting işaretidir. Farklı dönem ve sembolde test edin."))

# Yorumları göster
for icon, title, body in extra_items:
    if icon == "✅":
        st.success(f"**{title}** — {body}")
    elif icon in ("🟡","🟠"):
        st.warning(f"**{title}** — {body}")
    elif icon == "🔴":
        st.error(f"**{title}** — {body}")
    else:
        st.info(f"**{title}** — {body}")

# reporting modülünün kommentleri
try:
    commentary = generate_commentary(report)
    if commentary:
        with st.expander("📊 Detaylı Metrik Yorumları"):
            for item in commentary:
                if item.icon == "✅":
                    st.success(f"**{item.title}** — {item.body}")
                elif item.icon in ("🟡","🟠"):
                    st.warning(f"**{item.title}** — {item.body}")
                elif item.icon == "❌":
                    st.error(f"**{item.title}** — {item.body}")
                else:
                    st.info(f"**{item.title}** — {item.body}")
except Exception:
    pass

with st.expander("🔍 Gelişmiş İstatistikler"):
    try:
        gc1, gc2, gc3 = st.columns(3)
        with gc1:
            st.metric("Omega Oranı",           _fmt_f(r.omega_ratio))
            st.metric("Yıllık Volatilite",     _fmt_pct(r.ann_vol_pct))
            st.metric("Maks. Ardışık Kayıp",   str(r.max_consec_losses))
        with gc2:
            st.metric("Maks. DD Süresi (bar)", str(r.max_dd_duration))
            st.metric("Ort. Drawdown",         _fmt_pct(r.avg_drawdown_pct))
        with gc3:
            st.metric("B&H Toplam Getiri",     _fmt_pct(r.bh_return_pct))
            st.metric("B&H Yıllık Getiri",     _fmt_pct(r.bh_ann_return_pct))
    except Exception as e:
        st.info(f"Gelişmiş istatistikler hesaplanamadı: {e}")

st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# BÖLÜM 5 — OPTİMİZASYON
# ─────────────────────────────────────────────────────────────────────────────

st.subheader("🔬 Parametre Optimizasyonu")
st.warning(
    "⚠️ **Overfitting Uyarısı:** Bu optimizasyon geçmiş veriye göre en iyi "
    "parametre setini bulur. Geçmiş başarı gelecek performansı **garanti etmez**."
)

with st.expander("Optimizasyon Ayarları & Çalıştır", expanded=False):
    st.markdown(f"Strateji: `{strategy_id}`")
    oc1, oc2 = st.columns(2)
    with oc1:
        st.markdown("**EMA Hızlı:**")
        ef_min  = st.number_input("EMA H Min",  5,  50,  8, key="ef_min")
        ef_max  = st.number_input("EMA H Maks", 10, 100, 25, key="ef_max")
        ef_step = st.number_input("EMA H Adım", 1,  20,  4,  key="ef_step")
        st.markdown("**EMA Yavaş:**")
        es_min  = st.number_input("EMA Y Min",  10, 100,  20, key="es_min")
        es_max  = st.number_input("EMA Y Maks", 30, 300,  60, key="es_max")
        es_step = st.number_input("EMA Y Adım",  2,  30,  10, key="es_step")
    with oc2:
        if strategy_id == "multi_indicator_v1":
            st.markdown("**RSI Periyodu:**")
            rp_min  = st.number_input("RSI Per Min",  5,  20, 10, key="rp_min")
            rp_max  = st.number_input("RSI Per Maks", 10,  30, 21, key="rp_max")
            rp_step = st.number_input("RSI Per Adım",  1,   5,  7, key="rp_step")
            st.markdown("**ADX Eşiği:**")
            ax_min  = st.number_input("ADX Min",  10.0, 30.0, 15.0, 5.0, key="ax_min")
            ax_max  = st.number_input("ADX Maks", 20.0, 50.0, 30.0, 5.0, key="ax_max")
            ax_step = st.number_input("ADX Adım",  5.0, 20.0,  5.0, 5.0, key="ax_step")
        else:
            st.markdown("**ATR Periyodu:**")
            at_min  = st.number_input("ATR Min",  5,  30, 10, key="at_min")
            at_max  = st.number_input("ATR Maks", 10,  60, 20, key="at_max")
            at_step = st.number_input("ATR Adım",  1,  10,  5, key="at_step")
            st.markdown("**Vol. Eşiği:**")
            vt_min = st.number_input("Vol Min", 0.010, 0.10, 0.015, 0.005, format="%.3f", key="vt_min")
            vt_max = st.number_input("Vol Maks",0.020, 0.20, 0.040, 0.005, format="%.3f", key="vt_max")
            vt_cnt = st.number_input("Vol Adım sayısı", 2, 5, 3, key="vt_cnt")

    max_combos = st.number_input("Maks. kombinasyon", 10, 100, 30)
    opt_btn    = st.button("🔬 Optimizasyonu Başlat", type="secondary")

    if opt_btn:
        try:
            import numpy as _np
            def _arange(mn, mx, step):
                vals, v = [], float(mn)
                while v <= float(mx) + 1e-9:
                    vals.append(round(v, 4)); v += float(step)
                return vals or [mn]

            if strategy_id == "multi_indicator_v1":
                param_grid = {
                    "ema_fast"     : _arange(ef_min, ef_max, ef_step),
                    "ema_slow"     : _arange(es_min, es_max, es_step),
                    "rsi_period"   : _arange(rp_min, rp_max, rp_step),
                    "adx_threshold": _arange(ax_min, ax_max, ax_step),
                }
            else:
                param_grid = {
                    "ema_fast"     : _arange(ef_min, ef_max, ef_step),
                    "ema_slow"     : _arange(es_min, es_max, es_step),
                    "atr_period"   : _arange(at_min, at_max, at_step),
                    "vol_threshold": list(_np.linspace(vt_min, vt_max, int(vt_cnt))),
                }

            from algotrading.optimization.grid_search import run_grid_search
            from algotrading.data.ingestion import RawDataStore
            from algotrading.data.pit_handler import PITDataHandler
            from algotrading.data.corporate_actions import CorporateActionStore
            from algotrading.main import _resolve_data_root

            prog = st.progress(0, text="Optimizasyon başlıyor…")

            def _cb(done, total):
                try:
                    prog.progress(done/total, text=f"Kombinasyon {done}/{total}")
                except Exception:
                    pass

            data_root = _resolve_data_root(cfg)
            raw_store = RawDataStore(data_root)
            ca_store  = CorporateActionStore(data_root)
            dh = PITDataHandler(raw_store=raw_store, ca_store=ca_store,
                                window_size=cfg["data"]["window_size"])

            results = run_grid_search(
                param_grid=param_grid, cfg_base=cfg,
                data_handler=dh, max_combos=int(max_combos),
                top_n=10, progress_cb=_cb,
            )
            prog.empty()

            if not results:
                st.warning("Hiç geçerli sonuç üretilemedi. Tarih aralığını veya parametreleri değiştirin.")
            else:
                st.success(f"✅ {len(results)} en iyi sonuç:")
                rows_opt = []
                for i, res in enumerate(results, 1):
                    row = {"Sıra": i}
                    row.update(res.params)
                    row.update({
                        "Getiri": f"{res.total_return_pct:+.2f}%",
                        "Sharpe": f"{res.sharpe_ratio:.3f}",
                        "DD"    : f"{res.max_drawdown_pct:.1f}%",
                        "İşlem" : res.num_trades,
                        "Kazan.": f"{res.win_rate:.1f}%",
                        "Skor"  : f"{res.score:.3f}",
                    })
                    rows_opt.append(row)
                st.dataframe(pd.DataFrame(rows_opt),
                             use_container_width=True, hide_index=True)
                st.info(
                    "💡 **Skor** = Sharpe (35%) + Getiri (25%) + Düşük DD (25%) "
                    "+ İşlem (10%) + Kâr Faktörü (5%)"
                )
                st.warning(
                    "⚠️ En yüksek skorlu parametreler geçmişe göre optimize edilmiştir. "
                    "Walk-forward analizi ile doğrulayın."
                )
        except Exception as e:
            try: prog.empty()
            except Exception: pass
            st.error(f"Optimizasyon hatası: {e}")
            with st.expander("Detay"):
                st.code(traceback.format_exc())

st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────

st.caption(
    "AlgoTrading MVP v4 • Backtest Paneli • "
    "Veri Kaynakları: Yahoo Finance | Binance | Stooq • "
    "⚠️ Canlı alım-satım yapılmaz. Finansal tavsiye değildir."
)
