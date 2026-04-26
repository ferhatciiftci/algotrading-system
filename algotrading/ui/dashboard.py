"""
ui/dashboard.py  —  AlgoTrading MVP  Türkçe Backtest Paneli v5
──────────────────────────────────────────────────────────────
Yenilikler (v5):
  - Veri kaynağı: Yahoo Finance (varsayılan) | CoinGecko | Stooq
  - Binance yalnızca Gelişmiş/Lokal bölümünde, Cloud'da uyarıyla
  - Sembol ön ayarları: Hisse, CoinGecko kripto, Yahoo kripto
  - Üç ana sekme: Tek Varlık Backtest | Çoklu Varlık Tarama | Araştırma Hafızası
  - Tüm bölümler try/except — ham traceback gösterilmez
  - pandas 2.x applymap → _safe_df_map
  - Araştırma Hafızası: session + JSON kalıcı depolama
"""
from __future__ import annotations
import sys, traceback as _tb, copy, logging, time
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

st.set_page_config(
    page_title="AlgoTrading MVP",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("<style>.stAlert p{margin:0}.block-container{padding-top:.8rem}</style>",
            unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Ön Ayarlar
# ─────────────────────────────────────────────────────────────────────────────
PRESETS = {
    "Hisse / ETF (Yahoo Finance)": [
        "SPY","QQQ","AAPL","MSFT","NVDA","TSLA","AMZN","META","GOOGL","BRK-B"
    ],
    "Kripto — CoinGecko": [
        "bitcoin","ethereum","solana","ripple","cardano",
        "dogecoin","avalanche-2","chainlink","polkadot","litecoin"
    ],
    "Kripto — Yahoo Finance": [
        "BTC-USD","ETH-USD","SOL-USD","XRP-USD","ADA-USD",
        "DOGE-USD","AVAX-USD","LINK-USD","DOT-USD","LTC-USD"
    ],
}

SOURCES = {
    "Yahoo Finance" : "Yahoo Finance (Hisse/ETF/Kripto)",
    "CoinGecko"     : "CoinGecko (Kripto — ücretsiz)",
    "Stooq Günlük"  : "Stooq (Günlük veri — isteğe bağlı)",
    "Binance (Lokal)": "⚠️ Binance — yalnızca lokal/VPS",
}

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
    if v is None or (isinstance(v,float) and v!=v): return "—"
    return f"{'+'if v>=0 else ''}{v:.{d}f}%"
def _fmt_usd(v): return f"${v:,.2f}"
def _fmt_f(v, d=3):
    if v is None or (isinstance(v,float) and v!=v): return "—"
    return f"{v:.{d}f}"

def _safe_df_map(df, fn):
    try: return df.map(fn)
    except AttributeError: return df.applymap(fn)  # type: ignore

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
    src   = source.split()[0]  # "Binance (Lokal)" → "Binance"
    if src == "Yahoo":
        from algotrading.data.connectors.yfinance_connector import YFinanceConnector
        return YFinanceConnector().fetch_and_store(symbol, start, end, store)
    elif src == "CoinGecko":
        from algotrading.data.connectors.coingecko_connector import CoinGeckoConnector
        return CoinGeckoConnector().fetch_and_store(symbol, start, end, store)
    elif src == "Stooq":
        from algotrading.data.connectors.stooq_connector import StooqConnector
        return StooqConnector().fetch_and_store(symbol, start, end, store)
    elif src == "Binance":
        try:
            from algotrading.data.connectors.binance_connector import BinanceConnector
            return BinanceConnector().fetch_and_store(symbol, start, end, store)
        except Exception as e:
            err = str(e)
            if "451" in err or "403" in err or "location" in err.lower():
                raise RuntimeError(
                    "Binance bu cloud ortamında çalışmadı (konum kısıtlaması). "
                    "CoinGecko veya Yahoo Finance kullanın."
                )
            raise
    raise ValueError(f"Bilinmeyen kaynak: {source}")

def _run_backtest(cfg: dict):
    from algotrading.main import run_backtest
    return run_backtest(cfg)

def _get_memory(cfg: dict):
    from algotrading.research.memory import ResearchMemory
    from algotrading.main import _resolve_data_root
    try:
        data_root = _resolve_data_root(cfg)
    except Exception:
        data_root = Path(".")
    if "research_memory" not in st.session_state:
        st.session_state["research_memory"] = ResearchMemory(data_root)
    return st.session_state["research_memory"]

def _composite_score(total_return_pct, max_drawdown_pct, sharpe, num_trades, profit_factor):
    try:
        from algotrading.optimization.grid_search import composite_score
        return composite_score(total_return_pct, max_drawdown_pct, sharpe,
                               num_trades, profit_factor)
    except Exception:
        return 0.0

# ─────────────────────────────────────────────────────────────────────────────
# KENAR ÇUBUĞU
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Ayarlar")
st.sidebar.caption("⚠️ Yalnızca geçmiş simülasyon. Canlı işlem yapılmaz.")
cfg_base = _load_config()

with st.sidebar.expander("📊 Veri & Dönem", expanded=True):
    source = st.selectbox(
        "Veri Kaynağı",
        options=list(SOURCES.keys()),
        format_func=lambda k: SOURCES[k],
        index=0,
        help="Binance Lokal: Streamlit Cloud'da çalışmayabilir.",
    )
    if source == "Binance (Lokal)":
        st.sidebar.warning(
            "⚠️ Binance, Streamlit Cloud'da konum kısıtlaması nedeniyle "
            "çalışmayabilir. Kripto için CoinGecko önerilir."
        )

    # Ön ayar seçici
    preset_key  = st.selectbox("Sembol Ön Ayarı", ["— Manuel giriş —"] + list(PRESETS.keys()))
    if preset_key != "— Manuel giriş —":
        preset_list = PRESETS[preset_key]
        preset_sym  = st.selectbox("Sembol seç", preset_list)
        sembol = preset_sym
    else:
        sembol = st.text_input("Sembol", value=cfg_base["backtest"]["symbols"][0]).strip()
        if source not in ("CoinGecko","Stooq Günlük"):
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

with st.sidebar.expander("📈 Strateji", expanded=True):
    _strat_opts = {
        "multi_indicator_v1" : "📊 Multi-Indicator (EMA+RSI+MACD+ADX)",
        "trend_volatility_v1": "📉 Trend-Volatilite (EMA+ATR)",
    }
    _def_id = cfg_base["strategy"].get("id","multi_indicator_v1")
    strategy_id = st.selectbox("Strateji",
                               options=list(_strat_opts.keys()),
                               format_func=lambda k: _strat_opts[k],
                               index=list(_strat_opts.keys()).index(_def_id)
                               if _def_id in _strat_opts else 0)
    sp = cfg_base["strategy"]["params"]
    col_a, col_b = st.columns(2)
    with col_a: ema_fast = st.number_input("EMA Hızlı", 2, 100, int(sp.get("ema_fast",12)))
    with col_b: ema_slow = st.number_input("EMA Yavaş", 5, 500, int(sp.get("ema_slow",26)))
    col_c, col_d = st.columns(2)
    with col_c: atr_period    = st.number_input("ATR Per.", 5, 100, int(sp.get("atr_period",14)))
    with col_d: vol_threshold = st.number_input("Vol. Eşiği", 0.005, 0.20,
                                                float(sp.get("vol_threshold",0.035)),
                                                0.005, format="%.3f")

    if strategy_id == "multi_indicator_v1":
        _mode_opts = {"balanced":"⚖️ Dengeli (önerilen)","strict":"🔒 Katı","loose":"🔓 Esnek"}
        confirmation_mode = st.selectbox(
            "Onay Modu",
            options=list(_mode_opts.keys()),
            format_func=lambda k: _mode_opts[k],
            index=0,
            help="Dengeli: EMA+RSI+(MACD veya ADX) | Katı: hepsi zorunlu | Esnek: yalnızca EMA+RSI",
        )
        st.markdown("**RSI**")
        r1,r2,r3 = st.columns(3)
        with r1: rsi_period = st.number_input("Per.",2,50,int(sp.get("rsi_period",14)))
        with r2: rsi_buy    = st.number_input("Alış<",30.,80.,float(sp.get("rsi_buy",55.)),1.)
        with r3: rsi_sell   = st.number_input("Satış>",20.,70.,float(sp.get("rsi_sell",45.)),1.)
        st.markdown("**MACD**")
        m1,m2,m3 = st.columns(3)
        with m1: macd_fast   = st.number_input("Hızlı",2,50,int(sp.get("macd_fast",12)))
        with m2: macd_slow   = st.number_input("Yavaş",5,100,int(sp.get("macd_slow",26)))
        with m3: macd_signal = st.number_input("Sinyal",2,30,int(sp.get("macd_signal",9)))
        st.markdown("**ADX**")
        a1,a2 = st.columns(2)
        with a1: adx_period    = st.number_input("Per.",5,50,int(sp.get("adx_period",14)))
        with a2: adx_threshold = st.number_input("Eşik",10.,50.,float(sp.get("adx_threshold",20.)),1.)
        st.markdown("**SL / TP / Trailing**")
        s1,s2,s3 = st.columns(3)
        with s1: sl_pct    = st.number_input("SL%",0.5,20.,float(sp.get("sl_pct",0.025))*100,0.5,format="%.1f")/100
        with s2: tp_pct    = st.number_input("TP%",1.,50.,float(sp.get("tp_pct",0.05))*100,0.5,format="%.1f")/100
        with s3: trail_pct = st.number_input("Trail%",0.,20.,float(sp.get("trail_pct",0.015))*100,0.5,format="%.1f")/100
        e1,e2 = st.columns(2)
        with e1: cooldown_bars = st.number_input("Bekleme(bar)",0,20,int(sp.get("cooldown_bars",3)))
        with e2: allow_short   = st.checkbox("Açığa Satış",value=bool(sp.get("allow_short",False)))
    else:
        allow_short = st.checkbox("Açığa Satışa İzin Ver",
                                  value=bool(cfg_base["orchestrator"].get("allow_short",False)))

with st.sidebar.expander("🛡️ Risk", expanded=False):
    rk=cfg_base["risk"]; ks=cfg_base["kill_switch"]
    cm=cfg_base["commission"]; sl=cfg_base["slippage"]
    max_risk_pct     = st.slider("İşlem Başına Risk %",0.1,5.,float(rk["max_risk_pct"])*100,.1)/100
    max_position_pct = st.slider("Maks. Pozisyon %",1.,50.,float(rk["max_position_pct"])*100,1.)/100
    atr_stop_mult    = st.number_input("Stop ATR Çarpanı",0.5,8.,float(rk.get("atr_stop_mult",2.)),.5)
    max_drawdown_pct = st.slider("Kill Switch DD %",2.,50.,float(ks["max_drawdown_pct"])*100,1.)/100
    max_daily_loss   = st.slider("Maks. Günlük Zarar %",0.5,10.,float(ks["max_daily_loss_pct"])*100,.5)/100
    commission_rate  = st.number_input("Komisyon $/hisse",0.,0.05,float(cm.get("rate_per_share",0.005)),.001,format="%.4f")
    slippage_mult    = st.number_input("Slippage ATR Çarpanı",0.,1.,float(sl.get("atr_multiple",0.1)),.05,format="%.2f")

st.sidebar.divider()
veri_var = _data_exists(sembol, cfg_base)
if not veri_var:
    st.sidebar.warning(f"⚠️ **{sembol}** için yerel veri yok.")
    if st.sidebar.button(f"⬇️ {sembol} İndir ({source.split()[0]})",
                         type="secondary", use_container_width=True):
        with st.spinner(f"{sembol} indiriliyor…"):
            try:
                _download_data(sembol, str(baslangic), str(bitis), source, cfg_base)
                st.sidebar.success("✅ İndirildi!")
                st.rerun()
            except RuntimeError as e:
                st.sidebar.error(str(e))
            except Exception as e:
                st.sidebar.error(f"İndirme hatası: {e}")
else:
    st.sidebar.success(f"✅ {sembol} verisi mevcut.")
    if st.sidebar.button(f"🔄 Güncelle", type="secondary", use_container_width=True):
        with st.spinner("Güncelleniyor…"):
            try:
                _download_data(sembol, str(baslangic), str(bitis), source, cfg_base)
                st.sidebar.success("✅ Güncellendi!")
                st.rerun()
            except RuntimeError as e:
                st.sidebar.error(str(e))
            except Exception as e:
                st.sidebar.error(f"Güncelleme hatası: {e}")

calistir = st.sidebar.button("▶ Backtest Çalıştır", type="primary",
                             use_container_width=True, disabled=(not veri_var))
if not veri_var:
    st.sidebar.caption("Önce veriyi indirin.")

# ─────────────────────────────────────────────────────────────────────────────
# ANA ALAN
# ─────────────────────────────────────────────────────────────────────────────
st.title("📈 AlgoTrading MVP — Backtest Paneli")
st.caption("⚠️ Yalnızca geçmiş simülasyon. Canlı işlem yapılmaz. Finansal tavsiye değildir.")

tab_bt, tab_scan, tab_mem = st.tabs([
    "🎯 Tek Varlık Backtest",
    "📊 Çoklu Varlık Tarama",
    "📚 Araştırma Hafızası",
])

# ══════════════════════════════════════════════════════════════════════════════
# SEKME 1 — TEK VARLIK BACKTEST
# ══════════════════════════════════════════════════════════════════════════════
with tab_bt:
    if not calistir:
        if not veri_var:
            st.info(f"👈 Sol panelden **{sembol}** verisini indirin, ardından **▶ Backtest Çalıştır** tıklayın.")
        else:
            st.info("👈 Sol panelden parametreleri ayarlayın ve **▶ Backtest Çalıştır** tıklayın.")
        st.stop()

    if baslangic >= bitis:
        st.error("❌ Başlangıç tarihi bitiş tarihinden önce olmalıdır."); st.stop()
    if ema_fast >= ema_slow:
        st.error("❌ EMA Hızlı < EMA Yavaş olmalıdır."); st.stop()
    if strategy_id == "multi_indicator_v1":
        if macd_fast >= macd_slow:
            st.error("❌ MACD Hızlı < MACD Yavaş olmalıdır."); st.stop()

    # Config kur
    cfg = copy.deepcopy(cfg_base)
    cfg["backtest"]["symbols"]         = [sembol]
    cfg["backtest"]["start"]           = str(baslangic)
    cfg["backtest"]["end"]             = str(bitis)
    cfg["backtest"]["initial_capital"] = float(sermaye)
    cfg["strategy"]["id"]              = strategy_id
    spc = cfg["strategy"]["params"]
    spc.update({"ema_fast":int(ema_fast),"ema_slow":int(ema_slow),
                "atr_period":int(atr_period),"vol_threshold":float(vol_threshold)})
    cfg["orchestrator"]["allow_short"]       = allow_short
    cfg["risk"]["max_risk_pct"]              = float(max_risk_pct)
    cfg["risk"]["max_position_pct"]          = float(max_position_pct)
    cfg["risk"]["atr_stop_mult"]             = float(atr_stop_mult)
    cfg["kill_switch"]["max_drawdown_pct"]   = float(max_drawdown_pct)
    cfg["kill_switch"]["max_daily_loss_pct"] = float(max_daily_loss)
    cfg["commission"]["rate_per_share"]      = float(commission_rate)
    cfg["slippage"]["atr_multiple"]          = float(slippage_mult)
    if strategy_id == "multi_indicator_v1":
        spc.update({
            "rsi_period":int(rsi_period),"rsi_buy":float(rsi_buy),"rsi_sell":float(rsi_sell),
            "macd_fast":int(macd_fast),"macd_slow":int(macd_slow),"macd_signal":int(macd_signal),
            "adx_period":int(adx_period),"adx_threshold":float(adx_threshold),
            "sl_pct":float(sl_pct),"tp_pct":float(tp_pct),"trail_pct":float(trail_pct),
            "allow_short":allow_short,"cooldown_bars":int(cooldown_bars),
            "confirmation_mode":confirmation_mode,
        })

    with st.spinner(f"⏳ {sembol} backtest çalışıyor…"):
        try:
            result = _run_backtest(cfg)
        except FileNotFoundError as e:
            st.error(str(e)); st.stop()
        except Exception as e:
            st.error(f"Backtest çalışırken hata oluştu: {e}")
            with st.expander("Teknik detay"):
                st.code(_tb.format_exc()[:2000])
            st.stop()

    if result is None:
        st.warning("⚠️ Backtest sonuç üretmedi."); st.stop()

    try:
        from algotrading.reporting.performance import (
            compute_full_report, generate_commentary,
            compute_bh_return, attach_benchmark,
        )
        from algotrading.validation.metrics import compute_metrics, compute_trade_metrics
        _pc = 4 if strategy_id=="trend_volatility_v1" else 12
        report   = compute_full_report(result, cfg, param_count=_pc)
        eq_curve = report.equity_curve
        fills    = report.fills
        bh_total, bh_ann, bh_eq = compute_bh_return(eq_curve, float(sermaye))
        attach_benchmark(report, bh_total, bh_ann)
        _metrics = compute_metrics(eq_curve["equity"], bars_per_year=252)
        _trade_m = compute_trade_metrics(result.fills)
    except Exception as e:
        st.error(f"Rapor oluşturulamadı: {e}"); st.stop()

    # Araştırma hafızasına kaydet
    try:
        mem   = _get_memory(cfg_base)
        score = _composite_score(
            report.total_return_pct, report.max_drawdown_pct,
            report.sharpe_ratio, report.num_trades, report.profit_factor,
        )
        from algotrading.research.memory import build_result
        res_obj = build_result(
            symbol=sembol, source=source.split()[0],
            strategy=strategy_id,
            confirmation_mode=spc.get("confirmation_mode","—"),
            cfg_params=spc, metrics=_metrics,
            trade_metrics=_trade_m, score=score,
            start_date=str(baslangic), end_date=str(bitis),
        )
        mem.add(res_obj)
    except Exception:
        pass

    r = report
    _ml = {"strict":"Katı","balanced":"Dengeli","loose":"Esnek"}
    _mode_lbl = f"  |  Onay: {_ml.get(spc.get('confirmation_mode',''),'')}" if strategy_id=="multi_indicator_v1" else ""
    st.subheader("📊 Performans Özeti")
    st.caption(f"Dönem: {baslangic} → {bitis}  |  {sembol}  |  {_fmt_usd(sermaye)}{_mode_lbl}")

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("💰 Başlangıç",     _fmt_usd(sermaye))
    c2.metric("🏦 Bitiş",         _fmt_usd(r.final_equity), delta=_fmt_pct(r.total_return_pct))
    c3.metric("📈 Toplam Getiri", _fmt_pct(r.total_return_pct))
    c4.metric("📅 Yıllık Getiri", _fmt_pct(r.ann_return_pct))
    c5.metric("📉 Maks. Düşüş",  _fmt_pct(r.max_drawdown_pct))
    c6,c7,c8,c9,c10 = st.columns(5)
    c6.metric("⚖️ Sharpe",  _fmt_f(r.sharpe_ratio))
    c7.metric("📐 Sortino", _fmt_f(r.sortino_ratio))
    c8.metric("🎯 Calmar",  _fmt_f(r.calmar_ratio))
    c9.metric("🔄 İşlem",   str(r.num_trades))
    c10.metric("🏆 Kazan%", f"{r.win_rate:.1f}%" if r.num_trades>0 else "—")
    c11,c12,c13,c14,c15 = st.columns(5)
    c11.metric("💹 Kâr Fakt.",  _fmt_f(r.profit_factor))
    c12.metric("📊 Ort.Kazanç", _fmt_pct(r.avg_win_pct)   if r.num_trades>0 else "—")
    c13.metric("📊 Ort.Kayıp",  _fmt_pct(r.avg_loss_pct)  if r.num_trades>0 else "—")
    c14.metric("🏅 En İyi",     _fmt_pct(r.best_trade_pct)  if r.num_trades>0 else "—")
    c15.metric("⚠️ En Kötü",    _fmt_pct(r.worst_trade_pct) if r.num_trades>0 else "—")

    st.divider()
    gtab1,gtab2,gtab3,gtab4 = st.tabs(["Sermaye Eğrisi","Drawdown + Aylık","İşlem Günlüğü","Yorum + Optimizasyon"])

    with gtab1:
        try:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=eq_curve["timestamp"],y=eq_curve["equity"],name="Strateji",mode="lines",line=dict(color="#2196F3",width=2)))
            if bh_eq is not None and len(bh_eq)>0:
                fig.add_trace(go.Scatter(x=bh_eq.index,y=bh_eq.values,name="Buy&Hold",mode="lines",line=dict(color="#9E9E9E",width=1.5,dash="dot")))
            fig.add_hline(y=float(sermaye),line_dash="dash",line_color="#FF9800",line_width=1,annotation_text="Başlangıç")
            fig.update_layout(title=f"{sembol} — Sermaye Eğrisi",xaxis_title="Tarih",yaxis_title="Sermaye ($)",hovermode="x unified",height=420,plot_bgcolor="white",paper_bgcolor="white",legend=dict(orientation="h",y=1.02),yaxis=dict(tickprefix="$",gridcolor="#f5f5f5"),xaxis=dict(gridcolor="#f5f5f5"))
            st.plotly_chart(fig, use_container_width=True)
            _bh_ok = bh_total is not None and not (isinstance(bh_total,float) and bh_total!=bh_total)
            if _bh_ok:
                diff=r.total_return_pct-bh_total
                ca,cb,cc=st.columns(3)
                ca.metric("Strateji",_fmt_pct(r.total_return_pct))
                cb.metric("Buy&Hold",_fmt_pct(bh_total))
                cc.metric("Fark",_fmt_pct(diff))
        except Exception as e:
            st.info(f"Sermaye grafiği oluşturulamadı: {e}")

        if fills:
            try:
                eq_ts=eq_curve["timestamp"].values; eq_val=eq_curve["equity"].values
                def _snap(ts):
                    idx=np.argmin(np.abs(eq_ts-np.datetime64(pd.Timestamp(ts)))); return float(eq_val[idx])
                lf=[f for f in fills if f.direction.value.upper() in ("LONG","BUY")]
                sf=[f for f in fills if f.direction.value.upper() in ("SHORT","SELL","FLAT")]
                fig2=go.Figure()
                fig2.add_trace(go.Scatter(x=eq_curve["timestamp"],y=eq_curve["equity"],name="Sermaye",mode="lines",line=dict(color="#607D8B",width=1.5)))
                if lf: fig2.add_trace(go.Scatter(x=[f.timestamp for f in lf],y=[_snap(f.timestamp) for f in lf],mode="markers",name="Alış",marker=dict(symbol="triangle-up",size=11,color="#4CAF50",line=dict(color="white",width=1))))
                if sf: fig2.add_trace(go.Scatter(x=[f.timestamp for f in sf],y=[_snap(f.timestamp) for f in sf],mode="markers",name="Satış/Çıkış",marker=dict(symbol="triangle-down",size=11,color="#F44336",line=dict(color="white",width=1))))
                fig2.update_layout(title="İşlem Noktaları",xaxis_title="Tarih",yaxis_title="Sermaye ($)",hovermode="x unified",height=380,plot_bgcolor="white",paper_bgcolor="white",legend=dict(orientation="h",y=1.02),yaxis=dict(tickprefix="$",gridcolor="#f5f5f5"),xaxis=dict(gridcolor="#f5f5f5"))
                st.plotly_chart(fig2, use_container_width=True)
                st.caption(f"🟢 Alış: {len(lf)}  |  🔴 Satış/Çıkış: {len(sf)}")
            except Exception as e:
                st.info(f"İşlem noktaları oluşturulamadı: {e}")
        else:
            st.info("Hiç işlem gerçekleşmedi — işaret noktası yok.")
            if strategy_id=="multi_indicator_v1":
                st.markdown("💡 **Onay Modunu → Esnek** yaparak veya RSI/ADX eşiklerini gevşeterek işlem sayısını artırabilirsiniz.")

    with gtab2:
        try:
            dd=eq_curve["drawdown"]*100
            fig3=go.Figure()
            fig3.add_trace(go.Scatter(x=eq_curve["timestamp"],y=dd,name="Drawdown",mode="lines",line=dict(color="#F44336",width=1.5),fill="tozeroy",fillcolor="rgba(244,67,54,0.12)"))
            fig3.add_hline(y=0,line_color="black",line_width=0.5)
            fig3.add_hline(y=-abs(max_drawdown_pct*100),line_dash="dash",line_color="#FF9800",annotation_text=f"Kill Switch ({-abs(max_drawdown_pct*100):.0f}%)",annotation_position="bottom right")
            fig3.update_layout(title=f"{sembol} — Drawdown",xaxis_title="Tarih",yaxis_title="Düşüş (%)",hovermode="x unified",height=360,plot_bgcolor="white",paper_bgcolor="white",yaxis=dict(ticksuffix="%",gridcolor="#f5f5f5"),xaxis=dict(gridcolor="#f5f5f5"))
            st.plotly_chart(fig3, use_container_width=True)
        except Exception as e:
            st.info(f"Drawdown grafiği oluşturulamadı: {e}")

        try:
            eq_m=eq_curve.copy(); eq_m["timestamp"]=pd.to_datetime(eq_m["timestamp"])
            eq_m=eq_m.set_index("timestamp").sort_index()
            monthly_eq=eq_m["equity"].resample("ME").last()
            monthly_ret=monthly_eq.pct_change().dropna()*100
            if len(monthly_ret)<2:
                st.info("Aylık ısı haritası için en az 2 aylık veri gerekiyor.")
            else:
                mr=monthly_ret.reset_index(); mr.columns=["tarih","getiri"]
                mr["yil"]=mr["tarih"].dt.year; mr["ay"]=mr["tarih"].dt.month
                ay_ad={1:"Oca",2:"Şub",3:"Mar",4:"Nis",5:"May",6:"Haz",7:"Tem",8:"Ağu",9:"Eyl",10:"Eki",11:"Kas",12:"Ara"}
                mr["ay_adi"]=mr["ay"].map(ay_ad)
                pivot=mr.pivot_table(index="yil",columns="ay_adi",values="getiri",aggfunc="sum")
                sirali=[ay_ad[i] for i in range(1,13) if ay_ad[i] in pivot.columns]
                pivot=pivot.reindex(columns=sirali)
                try: pivot["Yıllık"]=mr.groupby("yil")["getiri"].sum()
                except Exception: pass
                if not pivot.empty:
                    fig4=px.imshow(pivot.values,x=pivot.columns.tolist(),y=pivot.index.astype(str).tolist(),color_continuous_scale="RdYlGn",color_continuous_midpoint=0,text_auto=".1f",labels=dict(color="Getiri (%)"),title="Aylık Getiri Isı Haritası (%)",aspect="auto")
                    fig4.update_coloraxes(colorbar_ticksuffix="%")
                    fig4.update_layout(height=min(500,80+60*len(pivot)),plot_bgcolor="white",paper_bgcolor="white")
                    st.plotly_chart(fig4,use_container_width=True)
                    with st.expander("Sayısal tablo"):
                        try:
                            fmt=_safe_df_map(pivot,lambda v:f"{v:+.1f}%" if pd.notna(v) else "—")
                            st.dataframe(fmt,use_container_width=True)
                        except Exception:
                            st.dataframe(pivot.round(2),use_container_width=True)
        except Exception as e:
            st.info(f"Aylık tablo oluşturulamadı: {e}")

    with gtab3:
        if not fills:
            st.info("Hiç işlem gerçekleşmedi.")
        else:
            try:
                dir_tr={"long":"Alış","LONG":"Alış","short":"Satış","SHORT":"Satış","flat":"Çıkış","FLAT":"Çıkış","buy":"Alış","BUY":"Alış","sell":"Satış","SELL":"Satış"}
                rows=[{"Tarih":f.timestamp.strftime("%Y-%m-%d") if hasattr(f.timestamp,"strftime") else str(f.timestamp),"Sembol":f.symbol,"Yön":dir_tr.get(f.direction.value,f.direction.value),"Fiyat":f"{f.fill_price:.4f}","Adet":f"{f.quantity:.0f}","Kom.":f"{f.commission:.3f}","Sebep":getattr(f,"reason","") or "Sinyal"} for f in fills]
                df_t=pd.DataFrame(rows)
                yon_f=st.selectbox("Filtre",["Tümü","Alış","Satış","Çıkış"])
                df_vis=df_t if yon_f=="Tümü" else df_t[df_t["Yön"]==yon_f]
                st.dataframe(df_vis,use_container_width=True,height=min(420,50+35*len(df_vis)),hide_index=True)
                st.caption(f"Toplam {len(fills)} işlem  |  Görüntülenen: {len(df_vis)}")
            except Exception as e:
                st.info(f"İşlem tablosu gösterilemiyor: {e}")

    with gtab4:
        st.subheader("💬 Nesnel Değerlendirme")
        _bh_ok2 = bh_total is not None and not (isinstance(bh_total,float) and bh_total!=bh_total)
        _beats   = _bh_ok2 and r.total_return_pct > bh_total
        items=[]
        if r.num_trades<5: items.append(("🔴","Aşırı Filtrelenmiş",f"{r.num_trades} işlem — analiz için yetersiz. Onay Modunu Esnek yapın."))
        elif r.num_trades<20: items.append(("🟠","Sınırlı İşlem",f"{r.num_trades} işlem — istatistiksel güvenilirlik düşük."))
        else: items.append(("✅","İşlem Yeterli",f"{r.num_trades} işlem — analiz güvenilir."))
        if r.total_return_pct>0: items.append(("✅","Sistem Kârlı",f"%{r.total_return_pct:.2f} getiri."))
        else: items.append(("🔴","Sistem Kârsız",f"%{r.total_return_pct:.2f} kayıp."))
        if _bh_ok2:
            if _beats: items.append(("✅","Buy&Hold'u Geçiyor",f"{_fmt_pct(r.total_return_pct)} > {_fmt_pct(bh_total)}"))
            else: items.append(("🟠","Buy&Hold Gerisinde",f"Buy&Hold {_fmt_pct(bh_total)} > Strateji {_fmt_pct(r.total_return_pct)}"))
        if r.num_trades>=10:
            if r.sharpe_ratio>1: items.append(("✅","Sharpe İyi",f"{r.sharpe_ratio:.2f}"))
            elif r.sharpe_ratio>0: items.append(("🟡","Sharpe Düşük",f"{r.sharpe_ratio:.2f}"))
            else: items.append(("🔴","Sharpe Negatif",f"{r.sharpe_ratio:.2f}"))
        else: items.append(("⚪","Sharpe Hesaplanamadı","Yeterli işlem yok."))
        if r.num_trades<15 and r.total_return_pct>25: items.append(("🟠","Overfitting Riski",f"Az işlem ({r.num_trades}) + yüksek getiri = şüpheli."))

        for icon,title,body in items:
            if icon=="✅": st.success(f"**{title}** — {body}")
            elif icon in ("🟡","🟠"): st.warning(f"**{title}** — {body}")
            elif icon=="🔴": st.error(f"**{title}** — {body}")
            else: st.info(f"**{title}** — {body}")

        try:
            commentary=generate_commentary(report)
            if commentary:
                with st.expander("📊 Detaylı Metrik Yorumları"):
                    for item in commentary:
                        if item.icon=="✅": st.success(f"**{item.title}** — {item.body}")
                        elif item.icon in ("🟡","🟠"): st.warning(f"**{item.title}** — {item.body}")
                        elif item.icon=="❌": st.error(f"**{item.title}** — {item.body}")
                        else: st.info(f"**{item.title}** — {item.body}")
        except Exception: pass

        with st.expander("🔍 Gelişmiş İstatistikler"):
            try:
                gc1,gc2,gc3=st.columns(3)
                with gc1:
                    st.metric("Omega",_fmt_f(r.omega_ratio)); st.metric("Yıllık Vol",_fmt_pct(r.ann_vol_pct)); st.metric("Ardışık Kayıp",str(r.max_consec_losses))
                with gc2:
                    st.metric("DD Süresi(bar)",str(r.max_dd_duration)); st.metric("Ort.DD",_fmt_pct(r.avg_drawdown_pct))
                with gc3:
                    st.metric("B&H Toplam",_fmt_pct(r.bh_return_pct)); st.metric("B&H Yıllık",_fmt_pct(r.bh_ann_return_pct))
            except Exception as e:
                st.info(f"Detaylı istatistikler: {e}")

        st.subheader("🔬 Parametre Optimizasyonu")
        st.warning("⚠️ Overfitting riski — geçmiş veriye göre optimize edilir. Farklı dönemde doğrulayın.")
        with st.expander("Optimizasyon Ayarları & Çalıştır", expanded=False):
            oc1,oc2=st.columns(2)
            with oc1:
                ef_min=st.number_input("EMA H Min",5,50,8,key="ef_min"); ef_max=st.number_input("EMA H Maks",10,100,25,key="ef_max"); ef_step=st.number_input("EMA H Adım",1,20,4,key="ef_step")
                es_min=st.number_input("EMA Y Min",10,100,20,key="es_min"); es_max=st.number_input("EMA Y Maks",30,300,60,key="es_max"); es_step=st.number_input("EMA Y Adım",2,30,10,key="es_step")
            with oc2:
                if strategy_id=="multi_indicator_v1":
                    rp_min=st.number_input("RSI Per Min",5,20,10,key="rp_min"); rp_max=st.number_input("RSI Per Maks",10,30,21,key="rp_max"); rp_step=st.number_input("RSI Per Adım",1,5,7,key="rp_step")
                    ax_min=st.number_input("ADX Min",10.,30.,15.,5.,key="ax_min"); ax_max=st.number_input("ADX Maks",20.,50.,30.,5.,key="ax_max"); ax_step=st.number_input("ADX Adım",5.,20.,5.,5.,key="ax_step")
                else:
                    at_min=st.number_input("ATR Min",5,30,10,key="at_min"); at_max=st.number_input("ATR Maks",10,60,20,key="at_max"); at_step=st.number_input("ATR Adım",1,10,5,key="at_step")
                    vt_min=st.number_input("Vol Min",0.010,0.10,0.015,0.005,format="%.3f",key="vt_min"); vt_max=st.number_input("Vol Maks",0.020,0.20,0.040,0.005,format="%.3f",key="vt_max"); vt_cnt=st.number_input("Vol Adım sayısı",2,5,3,key="vt_cnt")
            max_combos=st.number_input("Maks. kombinasyon",10,100,30)
            opt_btn=st.button("🔬 Optimizasyonu Başlat",type="secondary")
            if opt_btn:
                try:
                    import numpy as _np
                    def _ar(mn,mx,step):
                        v,vals=float(mn),[]
                        while v<=float(mx)+1e-9: vals.append(round(v,4)); v+=float(step)
                        return vals or [mn]
                    if strategy_id=="multi_indicator_v1":
                        pg={"ema_fast":_ar(ef_min,ef_max,ef_step),"ema_slow":_ar(es_min,es_max,es_step),"rsi_period":_ar(rp_min,rp_max,rp_step),"adx_threshold":_ar(ax_min,ax_max,ax_step)}
                    else:
                        pg={"ema_fast":_ar(ef_min,ef_max,ef_step),"ema_slow":_ar(es_min,es_max,es_step),"atr_period":_ar(at_min,at_max,at_step),"vol_threshold":list(_np.linspace(vt_min,vt_max,int(vt_cnt)))}
                    from algotrading.optimization.grid_search import run_grid_search
                    from algotrading.data.ingestion import RawDataStore
                    from algotrading.data.pit_handler import PITDataHandler
                    from algotrading.data.corporate_actions import CorporateActionStore
                    from algotrading.main import _resolve_data_root
                    prog=st.progress(0,text="Başlıyor…")
                    def _cb(d,t):
                        try: prog.progress(d/t,text=f"{d}/{t}")
                        except Exception: pass
                    dr=_resolve_data_root(cfg); rs=RawDataStore(dr); cs=CorporateActionStore(dr)
                    dh=PITDataHandler(raw_store=rs,ca_store=cs,window_size=cfg["data"]["window_size"])
                    results=run_grid_search(pg,cfg,dh,max_combos=int(max_combos),top_n=10,progress_cb=_cb)
                    prog.empty()
                    if not results:
                        st.warning("Geçerli sonuç üretilemedi.")
                    else:
                        st.success(f"✅ {len(results)} sonuç:")
                        rrows=[{"Sıra":i+1,**res.params,"Getiri":f"{res.total_return_pct:+.2f}%","Sharpe":f"{res.sharpe_ratio:.3f}","DD":f"{res.max_drawdown_pct:.1f}%","İşlem":res.num_trades,"Skor":f"{res.score:.3f}"} for i,res in enumerate(results)]
                        st.dataframe(pd.DataFrame(rrows),use_container_width=True,hide_index=True)
                except Exception as e:
                    try: prog.empty()
                    except Exception: pass
                    st.error(f"Optimizasyon hatası: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# SEKME 2 — ÇOKLU VARLIK TARAMA
# ══════════════════════════════════════════════════════════════════════════════
with tab_scan:
    st.subheader("📊 Çoklu Varlık Tarama")
    st.caption("Birden fazla varlık için aynı strateji parametrelerini çalıştırır ve karşılaştırır.")
    st.info("💡 Sol paneldeki **Strateji** ve **Risk** parametreleri burada da kullanılır.")

    scan_src = st.selectbox("Tarama Veri Kaynağı",
                            ["Yahoo Finance","CoinGecko","Stooq Günlük"],
                            key="scan_src")
    _scan_presets = {
        "Hisse / ETF":        ["SPY","QQQ","AAPL","MSFT","NVDA","TSLA"],
        "Kripto (CoinGecko)": ["bitcoin","ethereum","solana","ripple","cardano","dogecoin"],
        "Kripto (Yahoo)":     ["BTC-USD","ETH-USD","SOL-USD","XRP-USD","ADA-USD"],
    }
    scan_cat = st.selectbox("Ön Ayar Grubu",
                             ["— Özel —"] + list(_scan_presets.keys()),
                             key="scan_cat")
    if scan_cat != "— Özel —":
        default_syms = _scan_presets[scan_cat]
        scan_syms = st.multiselect("Semboller", options=default_syms,
                                   default=default_syms[:4], key="scan_syms")
    else:
        raw_input = st.text_input("Semboller (virgülle ayır)",
                                  "SPY,QQQ,AAPL", key="scan_custom")
        scan_syms = [s.strip() for s in raw_input.split(",") if s.strip()]

    sc1, sc2 = st.columns(2)
    with sc1:
        scan_start = st.date_input("Başlangıç",
                                   value=date.fromisoformat(cfg_base["backtest"]["start"]),
                                   key="scan_start")
    with sc2:
        scan_end = st.date_input("Bitiş",
                                 value=date.fromisoformat(cfg_base["backtest"]["end"]),
                                 key="scan_end")
    scan_capital = st.number_input("Sermaye ($)", 1000, 10_000_000,
                                   int(cfg_base["backtest"]["initial_capital"]),
                                   1000, key="scan_cap")

    if st.button("🔍 Tümünü Tara", type="primary", disabled=(not scan_syms)):
        if not scan_syms:
            st.warning("En az bir sembol seçin.")
        else:
            scan_results = []
            prog = st.progress(0, text="Tarama başlıyor…")
            status_box = st.empty()

            for i, sym in enumerate(scan_syms):
                prog.progress((i) / len(scan_syms), text=f"Tarıyor: {sym} ({i+1}/{len(scan_syms)})")
                status_box.info(f"⏳ {sym} işleniyor…")

                scan_cfg = copy.deepcopy(cfg_base)
                scan_cfg["backtest"]["symbols"]         = [sym]
                scan_cfg["backtest"]["start"]           = str(scan_start)
                scan_cfg["backtest"]["end"]             = str(scan_end)
                scan_cfg["backtest"]["initial_capital"] = float(scan_capital)
                scan_cfg["strategy"]["id"]              = strategy_id
                spc2 = scan_cfg["strategy"]["params"]
                spc2.update({"ema_fast":int(ema_fast),"ema_slow":int(ema_slow),
                             "atr_period":int(atr_period),"vol_threshold":float(vol_threshold)})
                scan_cfg["orchestrator"]["allow_short"]       = allow_short
                scan_cfg["risk"]["max_risk_pct"]              = float(max_risk_pct)
                scan_cfg["risk"]["max_position_pct"]          = float(max_position_pct)
                scan_cfg["risk"]["atr_stop_mult"]             = float(atr_stop_mult)
                scan_cfg["kill_switch"]["max_drawdown_pct"]   = float(max_drawdown_pct)
                scan_cfg["kill_switch"]["max_daily_loss_pct"] = float(max_daily_loss)
                scan_cfg["commission"]["rate_per_share"]      = float(commission_rate)
                scan_cfg["slippage"]["atr_multiple"]          = float(slippage_mult)
                if strategy_id == "multi_indicator_v1":
                    spc2.update({
                        "rsi_period":int(rsi_period),"rsi_buy":float(rsi_buy),"rsi_sell":float(rsi_sell),
                        "macd_fast":int(macd_fast),"macd_slow":int(macd_slow),"macd_signal":int(macd_signal),
                        "adx_period":int(adx_period),"adx_threshold":float(adx_threshold),
                        "sl_pct":float(sl_pct),"tp_pct":float(tp_pct),"trail_pct":float(trail_pct),
                        "allow_short":allow_short,"cooldown_bars":int(cooldown_bars),
                        "confirmation_mode":confirmation_mode,
                    })

                row = {"Varlık":sym, "Kaynak":scan_src}
                try:
                    # Veri var mı?
                    if not _data_exists(sym, scan_cfg):
                        _download_data(sym, str(scan_start), str(scan_end),
                                       scan_src, scan_cfg)

                    scan_res = _run_backtest(scan_cfg)
                    if scan_res is None:
                        row.update({"Getiri":"—","Maks.DD":"—","Sharpe":"—",
                                    "İşlem":0,"Kazan%":"—","Skor":"—","Yorum":"Sonuç yok"})
                    else:
                        from algotrading.validation.metrics import compute_metrics, compute_trade_metrics
                        eq = scan_res.equity_curve
                        m  = compute_metrics(eq["equity"], bars_per_year=252)
                        tm = compute_trade_metrics(scan_res.fills)
                        sc = _composite_score(m.total_return_pct, m.max_drawdown_pct,
                                              m.sharpe_ratio, tm.get("num_trades",0),
                                              tm.get("profit_factor",1.0))
                        nt = tm.get("num_trades",0)
                        if   nt<5:               yorum="Yetersiz veri"
                        elif sc>=0.6:            yorum="✅ Güçlü"
                        elif sc>=0.4:            yorum="🟡 Orta"
                        else:                    yorum="🔴 Zayıf"
                        row.update({
                            "Getiri":    f"{m.total_return_pct:+.2f}%",
                            "Maks.DD":   f"{m.max_drawdown_pct:.1f}%",
                            "Sharpe":    f"{m.sharpe_ratio:.3f}",
                            "İşlem":     nt,
                            "Kazan%":    f"{tm.get('win_rate',0.0):.1f}%",
                            "Skor":      f"{sc:.3f}",
                            "Yorum":     yorum,
                        })
                        # Hafızaya kaydet
                        try:
                            mem2 = _get_memory(cfg_base)
                            from algotrading.research.memory import build_result
                            ro = build_result(symbol=sym, source=scan_src.split()[0],
                                              strategy=strategy_id,
                                              confirmation_mode=spc2.get("confirmation_mode","—"),
                                              cfg_params=spc2, metrics=m,
                                              trade_metrics=tm, score=sc,
                                              start_date=str(scan_start), end_date=str(scan_end))
                            mem2.add(ro)
                        except Exception: pass
                except RuntimeError as e:
                    row.update({"Getiri":"—","Maks.DD":"—","Sharpe":"—",
                                "İşlem":0,"Kazan%":"—","Skor":"—","Yorum":str(e)[:60]})
                except Exception as e:
                    row.update({"Getiri":"—","Maks.DD":"—","Sharpe":"—",
                                "İşlem":0,"Kazan%":"—","Skor":"—","Yorum":f"Hata: {str(e)[:50]}"})
                scan_results.append(row)
                time.sleep(0.2)

            prog.progress(1.0, text="Tarama tamamlandı.")
            status_box.empty()

            if scan_results:
                st.success(f"✅ {len(scan_results)} varlık tarandı:")
                df_scan = pd.DataFrame(scan_results)
                st.dataframe(df_scan, use_container_width=True, hide_index=True)
                st.info(
                    "💡 **Skor** = Sharpe(35%) + Getiri(25%) + Düşük DD(25%) + "
                    "İşlem(10%) + Kâr Faktörü(5%). "
                    "Sonuçlar Araştırma Hafızası'na kaydedildi."
                )
            else:
                st.warning("Tarama sonuç üretmedi.")

# ══════════════════════════════════════════════════════════════════════════════
# SEKME 3 — ARAŞTIRMA HAFIZASI
# ══════════════════════════════════════════════════════════════════════════════
with tab_mem:
    st.subheader("📚 Araştırma Hafızası")
    st.caption(
        "Bu oturumda gerçekleştirilen tüm backtest sonuçları burada saklanır. "
        "Streamlit Cloud'da oturum sıfırlandığında kayıtlar temizlenir."
    )

    try:
        mem3  = _get_memory(cfg_base)
        df_mem = mem3.to_dataframe()
    except Exception as e:
        st.error(f"Araştırma hafızası yüklenemedi: {e}")
        df_mem = pd.DataFrame()

    if df_mem.empty:
        st.info("Henüz kayıt yok. Backtest veya Çoklu Tarama çalıştırın.")
    else:
        # Özet istatistikler
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Toplam Kayıt",  len(df_mem))
        try:
            best_score  = df_mem["Skor"].astype(float).max()
            m2.metric("En İyi Skor", f"{best_score:.3f}")
        except Exception: m2.metric("En İyi Skor","—")
        try:
            insuf = (df_mem["İşlem"].astype(int) < 5).sum()
            m3.metric("Yetersiz Veri", str(insuf))
        except Exception: m3.metric("Yetersiz Veri","—")
        try:
            strong = (df_mem["Sonuç"] == "Güçlü").sum()
            m4.metric("Güçlü Sonuç",  str(strong))
        except Exception: m4.metric("Güçlü Sonuç","—")

        st.divider()
        st.markdown("**Tüm Sonuçlar** (en yeni önce)")
        try:
            df_show = df_mem.iloc[::-1].reset_index(drop=True)
            st.dataframe(df_show, use_container_width=True, hide_index=True, height=400)
        except Exception as e:
            st.info(f"Tablo gösterilemedi: {e}")

        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("**🏆 En İyi 5 Sonuç**")
            try:
                best5 = mem3.best(5)
                if best5:
                    rows_b = [{"Sembol":r.symbol,"Kaynak":r.source,
                               "Getiri":f"{r.total_return_pct:+.2f}%",
                               "Sharpe":f"{r.sharpe_ratio:.3f}",
                               "İşlem":r.num_trades,"Skor":f"{r.score:.3f}",
                               "Sonuç":r.verdict} for r in best5]
                    st.dataframe(pd.DataFrame(rows_b), use_container_width=True, hide_index=True)
                else:
                    st.info("Henüz veri yok.")
            except Exception as e:
                st.info(f"En iyi sonuçlar: {e}")

        with col_r:
            st.markdown("**⚠️ En Zayıf 5 Sonuç**")
            try:
                worst5 = mem3.worst(5)
                if worst5:
                    rows_w = [{"Sembol":r.symbol,"Kaynak":r.source,
                               "Getiri":f"{r.total_return_pct:+.2f}%",
                               "Sharpe":f"{r.sharpe_ratio:.3f}",
                               "İşlem":r.num_trades,"Skor":f"{r.score:.3f}",
                               "Sonuç":r.verdict} for r in worst5]
                    st.dataframe(pd.DataFrame(rows_w), use_container_width=True, hide_index=True)
                else:
                    st.info("Henüz yeterli veri yok.")
            except Exception as e:
                st.info(f"En zayıf sonuçlar: {e}")

        insuf_list = mem3.insufficient_data()
        if insuf_list:
            st.markdown(f"**🟠 Yetersiz İşlem ({len(insuf_list)} kayıt)**")
            rows_i = [{"Sembol":r.symbol,"Strateji":r.strategy,"Onay":r.confirmation_mode,
                       "İşlem":r.num_trades,"Öneri":"Onay modunu Esnek yapın veya eşikleri gevşetin"}
                      for r in insuf_list]
            st.dataframe(pd.DataFrame(rows_i), use_container_width=True, hide_index=True)

        if st.button("🗑️ Hafızayı Temizle", type="secondary"):
            try:
                mem3.clear()
                if "research_memory" in st.session_state:
                    del st.session_state["research_memory"]
                st.success("Hafıza temizlendi.")
                st.rerun()
            except Exception as e:
                st.error(f"Temizleme hatası: {e}")

st.divider()
st.caption(
    "AlgoTrading MVP v5 • Veri: Yahoo Finance | CoinGecko | Stooq • "
    "⚠️ Canlı işlem yapılmaz. Finansal tavsiye değildir."
)
