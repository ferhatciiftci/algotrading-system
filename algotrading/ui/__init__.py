"""
ui/
────
Kullanıcı arayüzü katmanı — ileride geliştirilecek.

Planlanan bileşenler:
  dashboard.py   — Streamlit tabanlı backtest sonuç paneli
  charts.py      — Equity curve, drawdown, trade scatter grafikleri
  config_editor.py — YAML konfigürasyon düzenleme formu
  report_viewer.py — Walk-forward ve overfitting raporları

Bu klasör sadece UI bileşenlerini içerir.
İş mantığı (strateji, risk, backtest) buraya TAŞINMAZ.
UI → algotrading.* paketlerini import eder, tersi asla olmaz.

Kurulum (henüz gerekli değil):
    pip install streamlit plotly
"""
