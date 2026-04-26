"""
ui/charts.py
─────────────
Grafik oluşturma yardımcıları — ileride Streamlit dashboard'unda kullanılacak.
Bağımlılık: plotly
"""

# Şablon — henüz aktif değil.

def equity_curve_fig(equity_series):
    """Equity curve Plotly figürü döndür."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("plotly gerekli: pip install plotly")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=equity_series.index,
        y=equity_series.values,
        mode="lines",
        name="Equity",
        line=dict(color="#00b4d8", width=2),
    ))
    fig.update_layout(
        title="Equity Curve",
        xaxis_title="Tarih",
        yaxis_title="Portföy Değeri (USD)",
        template="plotly_dark",
    )
    return fig


def drawdown_fig(equity_series):
    """Drawdown grafiği Plotly figürü döndür."""
    try:
        import plotly.graph_objects as go
        import pandas as pd
    except ImportError:
        raise ImportError("plotly gerekli: pip install plotly")

    roll_max = equity_series.cummax()
    drawdown = (equity_series - roll_max) / roll_max * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown.values,
        mode="lines",
        name="Drawdown %",
        fill="tozeroy",
        line=dict(color="#ef233c"),
    ))
    fig.update_layout(
        title="Drawdown",
        xaxis_title="Tarih",
        yaxis_title="Drawdown (%)",
        template="plotly_dark",
    )
    return fig
