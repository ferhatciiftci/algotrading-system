# AlgoTrading MVP — Backtest Paneli

> ⚠️ **ÖNEMLİ UYARI**
> Bu sistem **YALNIZCA geçmiş veri simülasyonu (backtest)** için tasarlanmıştır.
> Canlı alım-satım koda kilitlidir — etkinleştirilemez.
> **Finansal tavsiye değildir. Gerçek para ile işlem yapmadan önce uzman görüşü alın.**

---

## 🚀 Streamlit Community Cloud'a Ücretsiz Deploy

### Adım 1: GitHub Reposu Oluştur

1. [github.com](https://github.com) adresine gidin ve giriş yapın
2. Sağ üstteki **"+"** → **"New repository"** tıklayın
3. Repository adı: `algotrading-mvp` (veya istediğiniz bir isim)
4. **Public** seçin (Streamlit Cloud ücretsiz planda public repo gerektirir)
5. **"Create repository"** tıklayın

### Adım 2: Dosyaları GitHub'a Yükle

**A — GitHub arayüzü ile (kolay):**
1. ZIP içindeki tüm dosyaları bilgisayarınıza çıkartın
2. GitHub repo sayfasında **"uploading an existing file"** linkine tıklayın
3. Tüm dosya ve klasörleri sürükleyip bırakın
4. **"Commit changes"** tıklayın

**B — Git ile (terminal):**
```bash
git clone https://github.com/KULLANICI_ADINIZ/algotrading-mvp.git
# ZIP içeriğini klasöre kopyalayın
cd algotrading-mvp
git add .
git commit -m "İlk yükleme: AlgoTrading MVP"
git push origin main
```

**Repo yapısı şöyle görünmeli:**
```
algotrading-mvp/
├── streamlit_app.py     ← Cloud giriş noktası
├── requirements.txt
├── README.md
├── .streamlit/
│   └── config.toml
└── algotrading/
    ├── config/
    ├── data/
    ├── ui/
    │   └── dashboard.py
    └── ...
```

### Adım 3: Streamlit Community Cloud'a Bağla

1. [share.streamlit.io](https://share.streamlit.io) adresine gidin
2. **"Sign in with GitHub"** ile giriş yapın
3. **"New app"** butonuna tıklayın
4. Şu ayarları girin:

| Alan | Değer |
|------|-------|
| **Repository** | `KULLANICI_ADINIZ/algotrading-mvp` |
| **Branch** | `main` |
| **Main file path** | `streamlit_app.py` |

5. **"Deploy!"** butonuna tıklayın
6. Birkaç dakika bekleyin — uygulama otomatik kurulup başlar
7. Size özel bir URL alırsınız: `https://KULLANICI.streamlit.app`

---

## 🌐 Cloud'da Kullanım

### Veri yoksa ne yapmalısınız?

Streamlit Cloud ücretsiz planında **dosya sistemi geçicidir** — uygulama yeniden başlatıldığında indirilen veriler silinebilir.

**Endişelenmeyin:** Dashboard içinde butona tıklamak yeterlidir:

1. Sol panel açık değilse **">"** okuna tıklayın
2. **Sembol** ve **tarih aralığını** seçin (varsayılan: SPY, 2020-2024)
3. Sarı uyarı çıkarsa **"⬇️ SPY Verisini İndir"** butonuna tıklayın
4. İndirme tamamlanınca **"▶ Backtest Çalıştır"** aktif olur
5. Tıklayın — sonuçlar saniyeler içinde görünür

> **Not:** Uygulama yeniden başlatıldıktan sonra butona tekrar tıklamak yeterlidir.
> Terminal veya CMD kullanmanıza gerek yoktur.

---

## 💻 Yerel Çalıştırma (Opsiyonel)

```bash
# 1. Bağımlılıkları yükle
pip install -r requirements.txt

# 2. Dashboard başlat
streamlit run streamlit_app.py

# veya doğrudan:
streamlit run algotrading/ui/dashboard.py

# Terminal backtest (dashboard olmadan)
python run_backtest.py
```

---

## 📊 Dashboard Özellikleri

| Bölüm | İçerik |
|-------|--------|
| **Veri İndirme** | Tek tıkla yfinance'den SPY/AAPL/vb. indirme butonu |
| **Performans Özeti** | 15 metrik: getiri, Sharpe, Sortino, Calmar, drawdown, kâr faktörü |
| **Sermaye Eğrisi** | Strateji vs Buy & Hold karşılaştırması |
| **Fiyat & İşlemler** | Alış/satış noktaları işaretlenmiş grafik |
| **Düşüş Grafiği** | Drawdown + kill switch eşik çizgisi |
| **Aylık Getiriler** | Yıl × Ay ısı haritası |
| **İşlem Günlüğü** | Tüm işlemlerin detayı (tarih, yön, fiyat, komisyon) |
| **Nesnel Değerlendirme** | Türkçe yorumlar: kârlılık, risk, B&H karşılaştırması, overfitting |
| **Parametre Optimizasyonu** | Çok amaçlı grid arama, Top 10 sonuç, overfitting uyarısı |

---

## ⚙️ Parametre Açıklamaları

### Strateji

| Parametre | Açıklama |
|-----------|----------|
| **EMA Hızlı** | Kısa vadeli trend (8–25 bar önerilen) |
| **EMA Yavaş** | Uzun vadeli trend (20–100 bar önerilen) |
| **ATR Periyodu** | Volatilite ölçüm penceresi |
| **Volatilite Eşiği** | Yüksek vol'da yeni işlem yapılmaz |
| **Açığa Satış** | Kapalıysa yalnızca long işlem |

### Risk

| Parametre | Açıklama |
|-----------|----------|
| **İşlem Başına Risk %** | Her işlemde riske atılacak sermaye (0.5–2% önerilen) |
| **Maks. Pozisyon %** | Tek pozisyon üst sınırı |
| **Stop-Loss ATR Çarpanı** | Stop = fiyat ± (ATR × çarpan) |
| **Kill Switch Drawdown %** | Bu eşik aşılınca sistem kilitlenir |
| **Günlük Zarar Limiti** | Gün içi maksimum kayıp |
| **Komisyon** | Hisse başına maliyet (gerçekçi tahmin için) |

---

## 📁 Proje Yapısı

```
algotrading-mvp/
├── streamlit_app.py          ← Streamlit Cloud giriş noktası
├── run_backtest.py           ← Yerel terminal giriş noktası
├── requirements.txt          ← Python bağımlılıkları
├── README.md
├── .streamlit/
│   └── config.toml           ← Tema ve sunucu ayarları
└── algotrading/
    ├── main.py               ← Backtest motoru
    ├── config/default.yaml   ← Tüm parametreler
    ├── core/                 ← Temel tipler, event bus
    ├── data/                 ← Veri deposu, yfinance connector
    ├── backtest/             ← Engine, portfolio, komisyon
    ├── strategies/           ← EMA + ATR trend stratejisi
    ├── orchestrator/         ← Karar mekanizması
    ├── risk/                 ← Kill switch, risk motoru
    ├── validation/           ← Performans metrikleri
    ├── reporting/            ← Türkçe analiz ve yorumlar
    ├── optimization/         ← Çok amaçlı parametre optimizasyonu
    ├── ui/
    │   └── dashboard.py      ← Streamlit paneli (694 satır Türkçe)
    └── tests/                ← Entegrasyon testleri
```

---

## ⚠️ Önemli Uyarılar

- **Araştırma sistemi** — üretim alım-satım platformu değildir
- **Cloud depolama geçicidir** — yeniden başlatmada veri silinebilir, indirme butonu yeterli
- **Backtest ≠ Gerçek performans** — slippage, likidite, anlık veri farklılıkları gerçek sonuçları değiştirir
- **Optimizasyon ≠ Garanti** — geçmiş veriye uyarlanmış parametreler gelecekte işe yaramayabilir
- **Cloudflare Workers Python desteklemez** — Streamlit Cloud veya Railway kullanın
