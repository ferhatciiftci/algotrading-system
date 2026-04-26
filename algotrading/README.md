# AlgoTrading MVP

> ⚠️  **ÜRETİM UYARISI**
> Bu sistem **YALNIZCA backtesting** için tasarlanmıştır.
> Canlı (live) alım-satım koda kilitlidir ve aktif edilemez.
> Gerçek parayla işlem yapmadan önce mutlaka bir finansal uzmana danışın.

---

## Hızlı Başlangıç

Tüm adımlar proje kökünden (`run_backtest.py`'nin bulunduğu klasörden) çalıştırılır.

```bash
# 1. Bağımlılıkları yükle
pip install -r algotrading/requirements.txt

# 2. SPY verisi indir (PROJECT_ROOT/data klasörüne kaydedilir)
python -m algotrading.data.download \
    --source yfinance \
    --symbol SPY \
    --start 2020-01-01 \
    --end 2024-01-01

# 3. Backtest çalıştır (terminal çıktısı)
python run_backtest.py

# 4. Görsel dashboard aç (tarayıcıda açılır)
streamlit run algotrading/ui/dashboard.py
```

**Veri yoksa backtest başlamaz** — hangi komutu çalıştırmanız gerektiği Türkçe hata mesajıyla gösterilir.

---

## Dashboard (Görsel Panel)

### Nasıl çalıştırılır?

```bash
# Proje kökünden:
streamlit run algotrading/ui/dashboard.py
```

Tarayıcıda `http://localhost:8501` adresinde otomatik açılır.

### Ne gösterir?

Dashboard sol panelden parametre girişi alır ve mevcut backtest motorunu çağırır:

| Bölüm | İçerik |
|-------|--------|
| **Performans Özeti** | Başlangıç/bitiş sermayesi, toplam getiri, yıllık getiri, maks. düşüş, Sharpe, işlem sayısı, kazanan oranı |
| **Sermaye Eğrisi** | Interaktif equity curve + alış/satış işaret noktaları |
| **Düşüş Grafiği** | Drawdown eğrisi + kill switch eşik çizgisi |
| **Aylık Getiriler** | Yıl × Ay ısı haritası ve sayısal tablo |
| **İşlem Günlüğü** | Tüm fillerin tarih, yön, fiyat, adet, komisyon detayları |
| **Sistem Değerlendirmesi** | Nesnel Türkçe yorumlar (kârlılık, risk, Sharpe, strateji gücü) |

### Önemli notlar

- Dashboard **yalnızca görselleştirme** aracıdır; strateji veya risk kodu içermez.
- Backtest motoru, risk motoru ve strateji **değiştirilmeden** aynı şekilde çalışır.
- **Canlı alım-satım yoktur ve yapılamaz** — bu bir tasarım kararıdır, kod seviyesinde kilitlidir.
- Tüm arayüz metni Türkçedir.

---

## Testler

### Hızlı Test (veri, pyarrow, yfinance gerekmez)

```bash
python -m algotrading.tests.test_integration
```

### Pipeline Testi (3 katman)

```bash
python -m algotrading.tests.test_pipeline
```

| Katman | Gereksinim | Açıklama |
|--------|-----------|----------|
| A (6 test) | Hiçbir şey | In-memory backtest, determinizm, Türkçe hata |
| B (6 test) | `pyarrow` | CSV → Parquet → backtest tam pipeline |
| C (1 test) | `pyarrow` + `yfinance` | Gerçek SPY verisi smoke testi |

> **Not:** SKIP ≠ PASS. Katman B/C atlanıyorsa `pip install pyarrow yfinance` çalıştırın.

---

## Proje Yapısı

```
algotrading_mvp/
├── run_backtest.py               ← Terminal giriş noktası
├── algotrading/
│   ├── main.py                   ← Backtest orchestrasyonu
│   ├── requirements.txt          ← Python bağımlılıkları
│   ├── config/
│   │   └── default.yaml          ← Tüm parametreler
│   ├── core/
│   │   ├── types.py              ← Bar, Signal, Order, Fill, Position (immutable)
│   │   ├── events.py             ← EventBus, pub/sub
│   │   └── clock.py              ← UTC saat
│   ├── data/
│   │   ├── ingestion.py          ← Append-only Parquet deposu
│   │   ├── cleaning.py           ← NaN doldurma, tekrarsızlaştırma
│   │   ├── validator.py          ← 7 doğrulama kontrolü
│   │   ├── pit_handler.py        ← PITDataHandler (lookahead yasak)
│   │   ├── corporate_actions.py  ← Split/temettü düzeltme
│   │   ├── download.py           ← Veri indirme CLI
│   │   └── connectors/
│   │       ├── yfinance_connector.py
│   │       ├── binance_connector.py
│   │       └── csv_loader.py
│   ├── backtest/
│   │   ├── engine.py             ← Event-driven döngü
│   │   ├── portfolio.py          ← Portföy, equity curve
│   │   ├── commission.py         ← Komisyon modelleri
│   │   └── slippage.py           ← Slippage modelleri
│   ├── strategies/
│   │   └── trend_volatility.py   ← EMA crossover + ATR filtresi
│   ├── orchestrator/
│   │   └── orchestrator.py       ← 5 kapılı karar mekanizması
│   ├── risk/
│   │   ├── kill_switch.py        ← Mandal kill switch
│   │   ├── position_sizer.py     ← FixedFractional
│   │   └── risk_manager.py       ← Risk motoru
│   ├── validation/
│   │   ├── metrics.py            ← Sharpe, Sortino, Calmar, PSR
│   │   ├── walk_forward.py       ← Walk-Forward Analysis
│   │   └── overfitting.py        ← PSR, Monte Carlo
│   ├── execution/
│   │   └── paper_trader.py       ← SimulatedExecution
│   ├── ui/
│   │   └── dashboard.py          ← Streamlit Türkçe paneli  ← YENİ
│   └── tests/
│       ├── test_integration.py   ← 12 uçtan uca test
│       └── test_pipeline.py      ← 3 katmanlı pipeline testi
```

---

## Veri Yolu Kuralı

`data.root_dir: "data"` her zaman **proje kökü** baz alınır:

```
run_backtest.py       → PROJECT_ROOT
data/                 → PROJECT_ROOT/data  ← veri buraya indirilir
algotrading/config/default.yaml → data.root_dir: "data"
```

---

## Sistem Akışı

```
Veri İndirme  →  Parquet Deposu  →  PITDataHandler
                                          │
                                    MarketEvent
                                          │
                               TrendVolatilityStrategy
                                          │
                                     SignalEvent
                                          │
                                    Orchestrator
                                          │
                                   DecisionEvent
                                          │
                                    RiskManager
                                          │
                                      OrderEvent
                                          │
                               SimulatedExecution
                                          │
                                      FillEvent
                                          │
                                    Portfolio
                                          │
                              BacktestResult + Türkçe Tablo
                                          │
                              Streamlit Dashboard (görsel)
```

---

## Veri Bağlayıcıları

```bash
# Yahoo Finance
python -m algotrading.data.download \
    --source yfinance --symbol SPY \
    --start 2020-01-01 --end 2024-01-01

# Binance (kripto, API anahtarı gerekmez)
python -m algotrading.data.download \
    --source binance --symbol BTCUSDT \
    --start 2022-01-01 --end 2024-01-01 --interval 1d

# Yerel CSV
python -m algotrading.data.download \
    --source csv --symbol AAPL --file /tam/yol/AAPL.csv
```

---

## Güvenlik Kuralları

| Kural | Durum |
|-------|-------|
| Canlı işlem devre dışı | ✅ Kod seviyesinde kilitli |
| Varsayılan mod = backtest | ✅ |
| Risk motoru bypass edilemez | ✅ |
| API anahtarı gerekmez | ✅ |
| İnsan onayı zorunlu | ✅ `require_human_approval: true` |
| Deterministik | ✅ SHA-256 config parmak izi |

---

## Bulut Deployment Notu

> ⚠️  **Cloudflare Workers** Python desteklemez — uygun değildir.

Önerilen: **Railway** veya **Fly.io** (ayrıntılar: [CLOUD_DEPLOYMENT.md](CLOUD_DEPLOYMENT.md))

---

## Beklenen Terminal Çıktısı

```
==============================================================
  BACKTEST SONUÇLARI — ALGOTRADING_MVP
  Dönem          : 2020-01-02 → 2024-01-01
  Sembol(ler)    : SPY
==============================================================
  Başlangıç Serm.:   100,000.00 USD
  Bitiş Sermayesi:   134,872.31 USD
  Toplam Getiri  :     +34.87%
  Yıllık Getiri  :      +7.82%
  Sharpe Oranı   :       0.549
  Maks. Düşüş    :     -18.44%
  İşlem Sayısı   :         47
  Kazanan Oran   :      57.4%
==============================================================
```

*Gerçek sonuçlar piyasa koşullarına ve config parametrelerine göre değişir.*
