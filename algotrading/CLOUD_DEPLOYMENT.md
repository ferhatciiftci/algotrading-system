# Bulut Deployment Planı

## Neden Cloudflare Workers Çalışmaz?

Cloudflare Workers, JavaScript/WebAssembly çalıştırmak için tasarlanmış
edge bir platformdur. Python desteği **yoktur** (Workers'ın V8 sandbox'ı
yalnızca JS/TS/Wasm kabul eder). Ek olarak:

- Maksimum istek süresi 50ms (CPU) — backtest döngüsü dakikalar alır
- Depolama sistemi (R2/KV) pandas/pyarrow ile uyumlu değil
- `numpy`, `scipy`, `pyarrow` gibi native C extension'lar build edilemez

**Sonuç:** AlgoTrading MVP için Cloudflare Workers uygun değildir.

---

## Önerilen Platformlar

### 1. Railway (En Kolay Başlangıç)

**Uygunluk:** ⭐⭐⭐⭐⭐ Backtest çalıştırma için ideal

```
# railway.toml
[build]
builder = "NIXPACKS"

[deploy]
startCommand = "python run_backtest.py"
healthcheckPath = "/"
restartPolicyType = "ON_FAILURE"
```

- Otomatik Python ortamı tespiti
- `requirements.txt` ile doğrudan kurulum
- Aylık $5 planla yeterli kaynak (2 vCPU, 512 MB RAM)
- GitHub push → otomatik deployment
- Kalıcı depolama için Volume eklenebilir (Parquet dosyaları)

**Kurulum adımları:**
1. `railway.app` → New Project → Deploy from GitHub
2. Repo bağla → root dizini = proje klasörü
3. Environment variable ekle: `PYTHONPATH=/app`
4. Volume mount: `/app/data` (Parquet store için)

---

### 2. Fly.io (Daha Fazla Kontrol)

**Uygunluk:** ⭐⭐⭐⭐ Üretim ortamı için

```dockerfile
# Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "run_backtest.py"]
```

```toml
# fly.toml
app = "algotrading-mvp"
primary_region = "fra"  # Frankfurt

[build]
  dockerfile = "Dockerfile"

[[mounts]]
  source = "data_vol"
  destination = "/app/data"
```

- Gerçek Docker container → tüm Python bağımlılıkları çalışır
- Persistent volume desteği (Parquet store için kritik)
- `fly deploy` ile tek komut deployment
- Ücretsiz tier: 256 MB RAM, 3 shared CPU

---

### 3. Render (Basit Alternatif)

**Uygunluk:** ⭐⭐⭐ Hızlı prototipleme için

- `render.yaml` ile tanımla, GitHub push → deploy
- Background Worker olarak çalıştır (HTTP server gerektirmez)
- Ücretsiz tier mevcut (ama uyuyor — cron için ücretli plan gerekli)

---

## Mimari Önerisi (Üretim)

```
┌─────────────────────────────────────────────────────────────┐
│                      Railway / Fly.io                       │
│                                                             │
│  ┌──────────────┐    ┌──────────────┐   ┌──────────────┐  │
│  │  Downloader  │    │   Backtest   │   │  Streamlit   │  │
│  │  (cron job)  │───▶│   Engine     │──▶│  Dashboard   │  │
│  │  yfinance/   │    │  (main.py)   │   │  (ui/)       │  │
│  │  binance     │    └──────┬───────┘   └──────────────┘  │
│  └──────────────┘           │                              │
│                        ┌────▼────┐                         │
│                        │ Parquet │                         │
│                        │  Store  │                         │
│                        │ (data/) │                         │
│                        └─────────┘                         │
└─────────────────────────────────────────────────────────────┘
```

### Deployment Sırası

1. **Veri indirme** (günlük cron): `python -m algotrading.data.download --source yfinance --symbol SPY --start ... --end ...`
2. **Backtest çalıştırma** (her gün kapanış sonrası): `python run_backtest.py`
3. **Walk-forward validasyon** (haftalık): `python -m algotrading.main --mode validate`
4. **Streamlit dashboard** (sürekli çalışır): `streamlit run algotrading/ui/dashboard.py`

---

## Güvenlik Kuralları (Değiştirilemez)

```yaml
# config/default.yaml
system:
  env: "paper"           # ASLA "live" yapılmamalı
  require_human_approval: true   # Deployment = insan onayı zorunlu
```

- **Canlı alım-satım TAMAMEN DEVRE DIŞI** — kod seviyesinde engel var
- API anahtarı gerekmez (Binance public endpoint, yfinance ücretsiz)
- Risk motoru her zaman aktif, bypass edilemez
- Kill switch tetiklendiğinde sıfırlama için manuel token gerekir

---

## Streamlit UI (İleride)

```bash
# Kurulum
pip install streamlit plotly

# Lokal çalıştırma
streamlit run algotrading/ui/dashboard.py

# Railway'de ek servis olarak
# → Ayrı bir Worker servisi oluştur, port 8501
```

Streamlit, backtest sonuçlarını görselleştirmek için kullanılacak.
Strateji mantığına doğrudan erişmez — sadece kayıtlı sonuçları okur.
