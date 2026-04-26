"""
ui/dashboard.py
───────────────
AlgoTrading MVP — Türkçe Gelişmiş Backtest Paneli v2

Çalıştırma:
    streamlit run algotrading/ui/dashboard.py

ÖNEMLİ: YALNIZCA backtest görselleştirme aracıdır.
         Canlı alım-satım yapılmaz ve yapılamaz.
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
    from plotly.subplots import make_subplots
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

# ─── CSS: minimal ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card{background:#f8f9fa;border-radius:8px;padding:12px 16px;border-left:4px solid #2196F3}
.warn-box{background:#fff3cd;border-radius:6px;padding:10px 14px;border-left:4px solid #ffc107}
.good-box{background:#d4edda;border-radius:6px;padding:10px 14px;border-left:4px solid #28a745}
.bad-box{background:#f8d7da;border-radius:6px;padding:10px 14px;border-left:4px solid #dc3545}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Yardımcı fonksiyonlar
# ─────────────────────────────────────────────────────────────────────────────

def _load_config() -> dict:
    p = _PKG_DIR / "config" / "default.yaml"
    if not p.exists():
        st.error(f"Config bulunamadı: {p}"); st.stop()
    with open(p) as f:
        return yaml.safe_load(f)

def _data_exists(symbol: str, cfg: dict) -> bool:
    from algotrading.data.ingestion import RawDataStore
    from algotrading.main import _resolve_data_root
    try:
        root  = _resolve_data_root(cfg)
        store = RawDataStore(root)
        sym_dir = store.raw_dir / symbol.upper()
        return sym_dir.exists() and any(sym_dir.rglob("*.parquet"))
    except Exception:
        return False

def _download_data(symbol: str, start: str, end: str, cfg: dict) -> str:
    from algotrading.data.ingestion import RawDataStore
    from algotrading.data.connectors.yfinance_connector import YFinanceConnector
    from algotrading.main import _resolve_data_root
    root  = _resolve_data_root(cfg)
    store = RawDataStore(root)
    conn  = YFinanceConnector()
    path  = conn.fetch_and_store(symbol, start, end, store)
    return path

def _run_backtest(cfg: dict):
    from algotrading.main import run_backtest
    return run_backtest(cfg)

def _fmt_pct(v, d=2):
    if v is None or (isinstance(v, float) and (v != v)):
        return "—"
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.{d}f}%"

def _fmt_usd(v):
    return f"${v:,.2f}"

def _fmt_f(v, d=3):
    if v is None or (isinstance(v, float) and (v != v)):
        return "—"
    return f"{v:.{d}f}"


# ─────────────────────────────────────────────────────────────────────────────
# KENAR ÇUBUĞU
# ─────────────────────────────────────────────────────────────────────────────

st.sidebar.title("⚙️ Backtest Ayarları")
st.sidebar.caption("⚠️ Yalnızca geçmiş veri simülasyonu. Canlı işlem yapılmaz.")

cfg_base = _load_config()

with st.sidebar.expander("📊 Veri & Dönem", expanded=True):
    sembol = st.text_input("Sembol", value=cfg_base["backtest"]["symbols"][0],
                           help="Örnek: SPY, AAPL, MSFT").upper().strip()
    c1, c2 = st.columns(2)
    with c1:
        baslangic = st.date_input("Başlangıç", value=date.fromisoformat(cfg_base["backtest"]["start"]))
    with c2:
        bitis     = st.date_input("Bitiş",     value=date.fromisoformat(cfg_base["backtest"]["end"]))
    sermaye = st.number_input("Başlangıç Sermayesi ($)", 1000, 10_000_000,
                              int(cfg_base["backtest"]["initial_capital"]), 1000)

with st.sidebar.expander("📈 Strateji Parametreleri", expanded=True):
    sp = cfg_base["strategy"]["params"]
    ema_fast      = st.number_input("EMA Hızlı (bar)", 2, 100, int(sp["ema_fast"]),
                                    help="Hızlı üstel hareketli ortalama periyodu")
    ema_slow      = st.number_input("EMA Yavaş (bar)", 5, 500, int(sp["ema_slow"]),
                                    help="Yavaş üstel hareketli ortalama periyodu")
    atr_period    = st.number_input("ATR Periyodu", 5, 100, int(sp["atr_period"]),
                                    help="Ortalama Gerçek Aralık hesaplama periyodu")
    vol_threshold = st.number_input("Volatilite Eşiği", 0.005, 0.20,
                                    float(sp["vol_threshold"]), 0.005, format="%.3f",
                                    help="Normalize ATR eşiği — üzerinde yeni pozisyon açılmaz")
    allow_short   = st.checkbox("Açığa Satışa İzin Ver",
                                value=bool(cfg_base["orchestrator"].get("allow_short", False)),
                                help="Kapalıysa sadece long işlem yapılır")

with st.sidebar.expander("🛡️ Risk Parametreleri", expanded=False):
    rk = cfg_base["risk"]
    ks = cfg_base["kill_switch"]
    cm = cfg_base["commission"]
    sl = cfg_base["slippage"]

    max_risk_pct      = st.slider("İşlem Başına Risk (%)", 0.1, 5.0,
                                  float(rk["max_risk_pct"])*100, 0.1,
                                  help="Her işlemde riske atılacak sermaye yüzdesi") / 100
    max_position_pct  = st.slider("Maks. Pozisyon (%)", 1.0, 50.0,
                                  float(rk["max_position_pct"])*100, 1.0,
                                  help="Tek pozisyonun portföy içindeki maksimum payı") / 100
    atr_stop_mult     = st.number_input("Stop-Loss ATR Çarpanı", 0.5, 8.0,
                                        float(rk.get("atr_stop_mult", 2.0)), 0.5,
                                        help="Stop-loss = fiyat ± (ATR × çarpan)")
    max_drawdown_pct  = st.slider("Maks. Drawdown Kill Switch (%)", 2.0, 50.0,
                                  float(ks["max_drawdown_pct"])*100, 1.0,
                                  help="Bu eşik aşılınca sistem kilitlenir") / 100
    max_daily_loss    = st.slider("Maks. Günlük Zarar (%)", 0.5, 10.0,
                                  float(ks["max_daily_loss_pct"])*100, 0.5,
                                  help="Günlük kayıp limiti") / 100
    commission_rate   = st.number_input("Komisyon ($/hisse)", 0.0, 0.05,
                                        float(cm.get("rate_per_share", 0.005)), 0.001,
                                        format="%.4f")
    slippage_mult     = st.number_input("Slippage ATR Çarpanı", 0.0, 1.0,
                                        float(sl.get("atr_multiple", 0.1)), 0.05, format="%.2f",
                                        help="Slippage = ATR × çarpan")

st.sidebar.divider()

# Veri indirme butonu
veri_var = _data_exists(sembol, cfg_base)
if not veri_var:
    st.sidebar.warning(f"⚠️ {sembol} için yerel veri bulunamadı.")
    if st.sidebar.button(f"⬇️  {sembol} Verisini İndir (yfinance)", type="secondary",
                         use_container_width=True):
        with st.spinner(f"{sembol} indiriliyor ({baslangic} → {bitis})…"):
            try:
                _download_data(sembol, str(baslangic), str(bitis), cfg_base)
                st.sidebar.success("✅ Veri indirildi! Backtest çalıştırabilirsiniz.")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"İndirme hatası: {e}")
else:
    st.sidebar.success(f"✅ {sembol} verisi mevcut.")

calistir = st.sidebar.button("▶  Backtest Çalıştır", type="primary", use_container_width=True,
                              disabled=(not veri_var))
if not veri_var and not calistir:
    st.sidebar.caption("Önce veriyi indirin, sonra backtest çalıştırın.")


# ─────────────────────────────────────────────────────────────────────────────
# ANA SAYFA BAŞLIĞI
# ─────────────────────────────────────────────────────────────────────────────

st.title("📈 AlgoTrading MVP — Backtest Paneli")
st.caption("⚠️ Bu panel geçmiş veri simülasyonu (backtest) sonuçlarını görselleştirir. "
           "**Canlı alım-satım yapılmaz ve yapılamaz.** Finansal tavsiye değildir.")

if not calistir:
    if not veri_var:
        st.info(f"👈 Sol panelde **{sembol} Verisini İndir** butonunu kullanın, "
                "ardından **Backtest Çalıştır** butonuna tıklayın.")
    else:
        st.info("👈 Sol panelden parametreleri ayarlayın ve **Backtest Çalıştır** butonuna tıklayın.")
    st.stop()

# Validasyon
if baslangic >= bitis:
    st.error("❌ Başlangıç tarihi bitiş tarihinden önce olmalıdır."); st.stop()
if ema_fast >= ema_slow:
    st.error("❌ EMA Hızlı değeri EMA Yavaş değerinden küçük olmalıdır."); st.stop()

# Config güncelle
cfg = copy.deepcopy(cfg_base)
cfg["backtest"]["symbols"]         = [sembol]
cfg["backtest"]["start"]           = str(baslangic)
cfg["backtest"]["end"]             = str(bitis)
cfg["backtest"]["initial_capital"] = float(sermaye)
cfg["strategy"]["params"]["ema_fast"]      = int(ema_fast)
cfg["strategy"]["params"]["ema_slow"]      = int(ema_slow)
cfg["strategy"]["params"]["atr_period"]    = int(atr_period)
cfg["strategy"]["params"]["vol_threshold"] = float(vol_threshold)
cfg["orchestrator"]["allow_short"]         = allow_short
cfg["risk"]["max_risk_pct"]        = float(max_risk_pct)
cfg["risk"]["max_position_pct"]    = float(max_position_pct)
cfg["risk"]["atr_stop_mult"]       = float(atr_stop_mult)
cfg["kill_switch"]["max_drawdown_pct"]  = float(max_drawdown_pct)
cfg["kill_switch"]["max_daily_loss_pct"]= float(max_daily_loss)
cfg["commission"]["rate_per_share"]     = float(commission_rate)
cfg["slippage"]["atr_multiple"]         = float(slippage_mult)

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
    st.warning("Backtest sonuç üretmedi."); st.stop()

# Rapor oluştur
from algotrading.reporting.performance import (
    compute_full_report, generate_commentary, compute_bh_return, attach_benchmark
)
from algotrading.validation.metrics import compute_metrics, compute_trade_metrics

try:
    report = compute_full_report(result, cfg, param_count=4)
except ValueError as e:
    st.warning(str(e)); st.stop()

eq_curve = report.equity_curve
fills    = report.fills

# Buy & Hold
bh_total, bh_ann, bh_eq = compute_bh_return(eq_curve, float(sermaye))
attach_benchmark(report, bh_total, bh_ann)


# ─────────────────────────────────────────────────────────────────────────────
# BÖLÜM 1 — PERFORMANS ÖZETİ
# ─────────────────────────────────────────────────────────────────────────────

st.subheader("📊 Performans Özeti")
st.caption(f"Dönem: {baslangic} → {bitis}  |  Sembol: {sembol}  |  "
           f"Sermaye: {_fmt_usd(sermaye)}")

# Satır 1
c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("💰 Başlangıç",    _fmt_usd(sermaye))
c2.metric("🏦 Bitiş",        _fmt_usd(report.final_equity),
          delta=_fmt_pct(report.total_return_pct))
c3.metric("📈 Toplam Getiri",_fmt_pct(report.total_return_pct))
c4.metric("📅 Yıllık Getiri",_fmt_pct(report.ann_return_pct))
c5.metric("📉 Maks. Düşüş",  _fmt_pct(report.max_drawdown_pct))

# Satır 2
c6,c7,c8,c9,c10 = st.columns(5)
c6.metric("⚖️ Sharpe",        _fmt_f(report.sharpe_ratio))
c7.metric("📐 Sortino",       _fmt_f(report.sortino_ratio))
c8.metric("🎯 Calmar",        _fmt_f(report.calmar_ratio))
c9.metric("🔄 İşlem Sayısı",  str(report.num_trades))
c10.metric("🏆 Kazanma Oranı",f"{report.win_rate:.1f}%")

# Satır 3
c11,c12,c13,c14,c15 = st.columns(5)
c11.metric("💹 Kâr Faktörü",   _fmt_f(report.profit_factor))
c12.metric("📊 Ort. Kazanç",   _fmt_pct(report.avg_win_pct))
c13.metric("📊 Ort. Kayıp",    _fmt_pct(report.avg_loss_pct))
c14.metric("🏅 En İyi İşlem",  _fmt_pct(report.best_trade_pct))
c15.metric("⚠️ En Kötü İşlem", _fmt_pct(report.worst_trade_pct))

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# BÖLÜM 2 — GRAFİKLER
# ─────────────────────────────────────────────────────────────────────────────

st.subheader("📉 Grafik Analizi")

tab1, tab2, tab3, tab4 = st.tabs([
    "Sermaye Eğrisi", "Fiyat & İşlemler", "Düşüş Grafiği", "Aylık Getiriler"
])

# ── Tab 1: Sermaye Eğrisi + B&H Karşılaştırması ──────────────────────────────
with tab1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=eq_curve["timestamp"], y=eq_curve["equity"],
        name="Strateji", mode="lines",
        line=dict(color="#2196F3", width=2),
        fill="tonexty", fillcolor="rgba(33,150,243,0.06)",
    ))

    # Buy & Hold
    if len(bh_eq) > 0:
        fig.add_trace(go.Scatter(
            x=bh_eq.index, y=bh_eq.values,
            name="Buy & Hold", mode="lines",
            line=dict(color="#9E9E9E", width=1.5, dash="dot"),
        ))

    fig.add_hline(y=float(sermaye), line_dash="dash", line_color="#FF9800",
                  line_width=1, annotation_text="Başlangıç")

    fig.update_layout(
        title=f"{sembol} — Sermaye Eğrisi vs Buy & Hold",
        xaxis_title="Tarih", yaxis_title="Sermaye ($)",
        hovermode="x unified", height=430,
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", y=1.02),
        yaxis=dict(tickprefix="$", gridcolor="#f5f5f5"),
        xaxis=dict(gridcolor="#f5f5f5"),
    )
    st.plotly_chart(fig, use_container_width=True)

    if not (isinstance(bh_total, float) and bh_total != bh_total):
        diff = report.total_return_pct - bh_total
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Strateji Toplam", _fmt_pct(report.total_return_pct))
        col_b.metric("Buy & Hold Toplam", _fmt_pct(bh_total))
        col_c.metric("Fark (Strateji − B&H)", _fmt_pct(diff),
                     delta_color="normal" if diff >= 0 else "inverse")

# ── Tab 2: Fiyat Grafiği + Alış/Satış İşaretleri ─────────────────────────────
with tab2:
    if fills:
        eq_df = eq_curve.copy()
        eq_df["timestamp"] = pd.to_datetime(eq_df["timestamp"])
        eq_ts_arr = eq_df["timestamp"].values
        eq_val_arr = eq_df["equity"].values

        def _snap_eq(ts):
            idx = np.argmin(np.abs(eq_ts_arr - np.datetime64(pd.Timestamp(ts))))
            return float(eq_val_arr[idx])

        long_fills  = [f for f in fills if f.direction.value.upper() in ("LONG","BUY")]
        short_fills = [f for f in fills if f.direction.value.upper() in ("SHORT","SELL","FLAT")]

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=eq_curve["timestamp"], y=eq_curve["equity"],
            name="Sermaye", mode="lines",
            line=dict(color="#607D8B", width=1.5),
        ))
        if long_fills:
            fig2.add_trace(go.Scatter(
                x=[f.timestamp for f in long_fills],
                y=[_snap_eq(f.timestamp) for f in long_fills],
                mode="markers", name="Alış (Long)",
                marker=dict(symbol="triangle-up", size=12, color="#4CAF50",
                            line=dict(color="white", width=1)),
            ))
        if short_fills:
            fig2.add_trace(go.Scatter(
                x=[f.timestamp for f in short_fills],
                y=[_snap_eq(f.timestamp) for f in short_fills],
                mode="markers", name="Satış / Çıkış",
                marker=dict(symbol="triangle-down", size=12, color="#F44336",
                            line=dict(color="white", width=1)),
            ))
        fig2.update_layout(
            title=f"{sembol} — Sermaye Eğrisi Üzerinde İşlem Noktaları",
            xaxis_title="Tarih", yaxis_title="Sermaye ($)",
            hovermode="x unified", height=420,
            plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(orientation="h", y=1.02),
            yaxis=dict(tickprefix="$", gridcolor="#f5f5f5"),
            xaxis=dict(gridcolor="#f5f5f5"),
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.caption(f"🟢 Alış: {len(long_fills)} işaret  |  🔴 Satış/Çıkış: {len(short_fills)} işaret")
    else:
        st.info("Bu backtest döneminde hiç işlem gerçekleşmedi — işaret noktası yok.")

# ── Tab 3: Drawdown Grafiği ───────────────────────────────────────────────────
with tab3:
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
        annotation_font_color="#FF9800",
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

# ── Tab 4: Aylık Getiri Isı Haritası ─────────────────────────────────────────
with tab4:
    eq_m = eq_curve.copy()
    eq_m["timestamp"] = pd.to_datetime(eq_m["timestamp"])
    eq_m = eq_m.set_index("timestamp").sort_index()
    monthly_eq  = eq_m["equity"].resample("ME").last()
    monthly_ret = monthly_eq.pct_change().dropna() * 100

    if len(monthly_ret) >= 2:
        mr = monthly_ret.reset_index()
        mr.columns = ["tarih", "getiri"]
        mr["yil"] = mr["tarih"].dt.year
        mr["ay"]  = mr["tarih"].dt.month
        ay_ad = {1:"Oca",2:"Şub",3:"Mar",4:"Nis",5:"May",6:"Haz",
                 7:"Tem",8:"Ağu",9:"Eyl",10:"Eki",11:"Kas",12:"Ara"}
        mr["ay_adi"] = mr["ay"].map(ay_ad)
        pivot = mr.pivot_table(index="yil", columns="ay_adi", values="getiri", aggfunc="sum")
        sirali = [ay_ad[i] for i in range(1,13) if ay_ad[i] in pivot.columns]
        pivot  = pivot.reindex(columns=sirali)
        pivot["Yıllık"] = mr.groupby("yil")["getiri"].sum()

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
        fig4.update_layout(height=340, plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig4, use_container_width=True)

        with st.expander("Sayısal tablo"):
            fmt = pivot.applymap(lambda v: f"{v:+.1f}%" if pd.notna(v) else "—")
            st.dataframe(fmt, use_container_width=True)
    else:
        st.info("Aylık ısı haritası için en az 2 aylık veri gerekiyor.")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# BÖLÜM 3 — İŞLEM GÜNLÜĞÜ
# ─────────────────────────────────────────────────────────────────────────────

st.subheader("📋 İşlem Günlüğü")

if not fills:
    st.info("Bu backtest döneminde hiç işlem gerçekleşmedi.")
else:
    dir_tr = {
        "long":"Alış (Long)","LONG":"Alış (Long)",
        "short":"Satış (Short)","SHORT":"Satış (Short)",
        "flat":"Çıkış (Flat)","FLAT":"Çıkış (Flat)",
    }
    rows = []
    for f in fills:
        rows.append({
            "Tarih"       : f.timestamp.strftime("%Y-%m-%d") if hasattr(f.timestamp,"strftime") else str(f.timestamp),
            "Sembol"      : f.symbol,
            "Yön"         : dir_tr.get(f.direction.value, f.direction.value),
            "Fiyat ($)"   : f"{f.fill_price:.4f}",
            "Adet"        : f"{f.quantity:.0f}",
            "Komisyon ($)": f"{f.commission:.3f}",
            "Slippage ($)": f"{getattr(f,'slippage',0):.4f}",
            "Sebep"       : (getattr(f,"reason","") or "Strateji sinyali"),
        })
    df_t = pd.DataFrame(rows)
    yon_f = st.selectbox("Filtre", ["Tümü","Alış (Long)","Satış (Short)","Çıkış (Flat)"])
    if yon_f != "Tümü":
        df_t = df_t[df_t["Yön"] == yon_f]
    st.dataframe(df_t, use_container_width=True,
                 height=min(420, 50 + 35*len(df_t)), hide_index=True)
    st.caption(f"Toplam {len(fills)} işlem gösteriliyor: {len(df_t)}")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# BÖLÜM 4 — TÜRKÇE YORUM PANELİ
# ─────────────────────────────────────────────────────────────────────────────

st.subheader("💬 Nesnel Sistem Değerlendirmesi")
st.caption("Aşağıdaki yorumlar metrik eşiklerine dayanır. Yatırım tavsiyesi değildir.")

commentary = generate_commentary(report)
for item in commentary:
    if item.icon == "✅":
        st.success(f"**{item.title}** — {item.body}")
    elif item.icon in ("🟡", "🟠"):
        st.warning(f"**{item.title}** — {item.body}")
    elif item.icon == "❌":
        st.error(f"**{item.title}** — {item.body}")
    else:
        st.info(f"**{item.title}** — {item.body}")

with st.expander("🔍 Gelişmiş İstatistikler"):
    gc1, gc2, gc3 = st.columns(3)
    with gc1:
        st.metric("Omega Oranı",          _fmt_f(report.omega_ratio))
        st.metric("Yıllık Volatilite",    _fmt_pct(report.ann_vol_pct))
        st.metric("Maks. Ardışık Kayıp",  str(report.max_consec_losses))
    with gc2:
        st.metric("Maks. DD Süresi (bar)",str(report.max_dd_duration))
        st.metric("Ort. DD",              _fmt_pct(report.avg_drawdown_pct))
        st.metric("En İyi İşlem",         _fmt_pct(report.best_trade_pct))
    with gc3:
        st.metric("En Kötü İşlem",        _fmt_pct(report.worst_trade_pct))
        st.metric("B&H Yıllık Getiri",    _fmt_pct(report.bh_ann_return_pct))
        st.metric("B&H Toplam Getiri",    _fmt_pct(report.bh_return_pct))

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# BÖLÜM 5 — PARAMETRE OPTİMİZASYONU
# ─────────────────────────────────────────────────────────────────────────────

st.subheader("🔬 Parametre Optimizasyonu")

st.warning(
    "⚠️ **Overfitting Uyarısı:** Aşağıdaki optimizasyon geçmiş veriye göre en iyi "
    "parametre setini arar. **Geçmiş başarı gelecek performansı garanti etmez.** "
    "Sonuçları farklı dönem ve sembollerde doğrulayın."
)

with st.expander("Optimizasyon Ayarları & Çalıştır", expanded=False):
    oc1, oc2 = st.columns(2)
    with oc1:
        st.markdown("**EMA Hızlı aralığı:**")
        ef_min = st.number_input("EMA Hızlı — Min", 5, 50, 8, key="ef_min")
        ef_max = st.number_input("EMA Hızlı — Maks", 10, 100, 25, key="ef_max")
        ef_step= st.number_input("EMA Hızlı — Adım", 1, 20, 4, key="ef_step")

        st.markdown("**EMA Yavaş aralığı:**")
        es_min = st.number_input("EMA Yavaş — Min", 10, 100, 20, key="es_min")
        es_max = st.number_input("EMA Yavaş — Maks", 30, 300, 60, key="es_max")
        es_step= st.number_input("EMA Yavaş — Adım", 2, 30, 10, key="es_step")
    with oc2:
        st.markdown("**ATR Periyodu aralığı:**")
        at_min = st.number_input("ATR — Min", 5, 30, 10, key="at_min")
        at_max = st.number_input("ATR — Maks", 10, 60, 20, key="at_max")
        at_step= st.number_input("ATR — Adım", 1, 10, 5, key="at_step")

        st.markdown("**Volatilite Eşiği aralığı:**")
        vt_min = st.number_input("Vol Eşiği — Min", 0.010, 0.10, 0.015, 0.005, format="%.3f", key="vt_min")
        vt_max = st.number_input("Vol Eşiği — Maks", 0.020, 0.20, 0.040, 0.005, format="%.3f", key="vt_max")
        vt_cnt = st.number_input("Vol Eşiği — Adım sayısı", 2, 5, 3, key="vt_cnt")

    max_combos = st.number_input("Maksimum kombinasyon sayısı", 10, 100, 30,
                                  help="Daha fazla = daha uzun süre ama daha iyi kapsam")
    opt_btn = st.button("🔬 Optimizasyonu Başlat", type="secondary")

    if opt_btn:
        import numpy as _np

        def _arange(mn, mx, step):
            vals = []
            v = mn
            while v <= mx + 1e-9:
                vals.append(round(v, 4))
                v += step
            return vals if vals else [mn]

        param_grid = {
            "ema_fast"     : _arange(ef_min, ef_max, ef_step),
            "ema_slow"     : _arange(es_min, es_max, es_step),
            "atr_period"   : _arange(at_min, at_max, at_step),
            "vol_threshold": list(_np.linspace(vt_min, vt_max, int(vt_cnt))),
        }
        # EMA fast < EMA slow filtresi
        valid_pairs = [(ef, es) for ef in param_grid["ema_fast"]
                                for es in param_grid["ema_slow"] if ef < es]

        total_valid = len(valid_pairs) * len(param_grid["atr_period"]) * len(param_grid["vol_threshold"])
        st.info(f"Geçerli kombinasyon sayısı: ~{total_valid} (maksimum {max_combos} tanesi çalıştırılacak)")

        from algotrading.optimization.grid_search import run_grid_search

        # Veri handler — in-memory olarak yükle (PITDataHandler üzerinden)
        from algotrading.data.ingestion import RawDataStore
        from algotrading.data.pit_handler import PITDataHandler
        from algotrading.data.corporate_actions import CorporateActionStore
        from algotrading.main import _resolve_data_root

        progress_bar = st.progress(0, text="Optimizasyon çalışıyor…")

        def _prog(done, total):
            progress_bar.progress(done/total, text=f"Kombinasyon {done}/{total}")

        try:
            data_root = _resolve_data_root(cfg)
            raw_store = RawDataStore(data_root)
            ca_store  = CorporateActionStore(data_root)
            dh = PITDataHandler(raw_store=raw_store, ca_store=ca_store,
                                window_size=cfg["data"]["window_size"])

            results = run_grid_search(
                param_grid  = param_grid,
                cfg_base    = cfg,
                data_handler= dh,
                max_combos  = int(max_combos),
                top_n       = 10,
                progress_cb = _prog,
            )
            progress_bar.empty()

            if not results:
                st.warning("Hiç geçerli sonuç üretilemedi.")
            else:
                st.success(f"✅ {len(results)} en iyi sonuç (bileşik skora göre sıralanmış):")
                rows_opt = []
                for i, r in enumerate(results, 1):
                    rows_opt.append({
                        "Sıra": i,
                        "EMA Hızlı": r.params.get("ema_fast","—"),
                        "EMA Yavaş": r.params.get("ema_slow","—"),
                        "ATR Per.": r.params.get("atr_period","—"),
                        "Vol Eşiği": r.params.get("vol_threshold","—"),
                        "Toplam Getiri": f"{r.total_return_pct:+.2f}%",
                        "Sharpe": f"{r.sharpe_ratio:.3f}",
                        "Maks. DD": f"{r.max_drawdown_pct:.1f}%",
                        "İşlem": r.num_trades,
                        "Kazan. %": f"{r.win_rate:.1f}%",
                        "Kâr Fakt.": f"{r.profit_factor:.2f}",
                        "Bileşik Skor": f"{r.score:.3f}",
                    })
                st.dataframe(pd.DataFrame(rows_opt), use_container_width=True, hide_index=True)
                st.info(
                    "💡 **Bileşik Skor** = Sharpe (%35) + Getiri (%25) + Düşük DD (%25) + "
                    "İşlem Sayısı (%10) + Kâr Faktörü (%5). "
                    "Sadece en yüksek getiriyi değil, risk-getiri dengesini yansıtır."
                )
                st.warning(
                    "⚠️ Bu sonuçlar **geçmiş veriye** göre optimize edilmiştir. "
                    "En yüksek skorlu parametreler gelecekte en iyi performansı göstermeyebilir. "
                    "Walk-forward analizi ile doğrulamak önerilir."
                )
        except Exception as e:
            progress_bar.empty()
            st.error(f"Optimizasyon hatası: {e}")
            with st.expander("Detay"):
                st.code(traceback.format_exc())

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────

st.caption(
    "AlgoTrading MVP v2 • Backtest Paneli • "
    "⚠️ Bu sistem yalnızca geçmiş veri simülasyonu içindir. "
    "Canlı alım-satım yapılmaz. Finansal tavsiye değildir."
)
