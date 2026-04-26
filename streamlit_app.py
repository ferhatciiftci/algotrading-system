"""
streamlit_app.py
────────────────
Streamlit Community Cloud giriş noktası.

Bu dosya algotrading/ui/dashboard.py içindeki Türkçe backtest
panelini çalıştırır. Ticaret mantığı içermez.

Streamlit Cloud'da ayar:
  Repository  : <github-kullanici>/<repo-adi>
  Branch      : main
  Entry file  : streamlit_app.py

⚠️  Bu sistem YALNIZCA backtest görselleştirme aracıdır.
    Canlı alım-satım yapılmaz.
"""
import sys
from pathlib import Path

# Proje kökünü Python yoluna ekle (Cloud + local uyumlu)
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Dashboard dosyasını çalıştır
# __file__ dashboard.py olarak ayarlanır ki içindeki yol hesapları doğru çalışsın
_dashboard = _ROOT / "algotrading" / "ui" / "dashboard.py"

with open(_dashboard, encoding="utf-8") as _f:
    _src = _f.read()

exec(
    compile(_src, str(_dashboard), "exec"),
    {
        "__file__"     : str(_dashboard),
        "__name__"     : "__main__",
        "__builtins__" : __builtins__,
    },
)
