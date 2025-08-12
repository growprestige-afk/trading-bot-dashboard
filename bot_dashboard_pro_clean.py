import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Bot de Trading - Versão Pro", layout="wide")
st.title("📊 Bot de Trading — Versão Pro (Candlesticks + RSI)")
st.write("Painel com gráfico de velas, volume, RSI e marcações de sinais (RSI < 30 compra, RSI > 70 venda).")

@st.cache_data(ttl=300)
def get_data(ticker: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
    except Exception as e:
        st.error(f"Erro ao obter dados para {ticker}: {e}")
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna()
    return df

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val

def gerar_sinal(rsi_value: float) -> str:
    if pd.isna(rsi_value):
        return "⚪ SEM DADOS"
    if rsi_value < 30:
        return "🟢 COMPRA"
    if rsi_value > 70:
        return "🔴 VENDA"
    return "⚪ MANTER"

def plot_candles_rsi(df: pd.DataFrame):
    d = df.copy()
    d["RSI"] = rsi(d["Close"])
    d.dropna(inplace=True)

    buy_mask = d["RSI"] < 30
    sell_mask = d["RSI"] > 70

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03,
        row_heights=[0.7, 0.3]
    )

    fig.add_trace(go.Candlestick(
        x=d.index, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"],
        name="Preço"
    ), row=1, col=1)

    if "Volume" in d.columns:
        fig.add_trace(go.Bar(x=d.index, y=d["Volume"], name="Volume", opacity=0.3), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=d.index[buy_mask], y=d.loc[buy_mask, "Close"],
        mode="markers", marker=dict(symbol="triangle-up", size=10),
        name="Compra (RSI<30)"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=d.index[sell_mask], y=d.loc[sell_mask, "Close"],
        mode="markers", marker=dict(symbol="triangle-down", size=10),
        name="Venda (RSI>70)"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(x=d.index, y=d["RSI"], name="RSI"), row=2, col=1)

    try:
        fig.add_hline(y=70, line_dash="dash", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", row=2, col=1)
    except Exception:
        fig.update_layout(shapes=[
            dict(type="line", xref="x", yref="y2", x0=d.index.min(), x1=d.index.max(), y0=70, y1=70, line=dict(dash="dash")),
            dict(type="line", xref="x", yref="y2", x0=d.index.min(), x1=d.index.max(), y0=30, y1=30, line=dict(dash="dash")),
        ])

    fig.update_layout(
        xaxis_rangeslider_visible=True,
        legend_orientation="h",
        margin=dict(l=20, r=20, t=20, b=20),
    )
    last_rsi = float(d["RSI"].iloc[-1]) if not d.empty else float("nan")
    return fig, last_rsi

col1, col2, col3 = st.columns([2,2,1])
with col1:
    stock_symbol = st.text_input("Ação (ex.: AAPL)", value="AAPL")
with col2:
    crypto_symbol = st.text_input("Cripto (ex.: BTC-USD)", value="BTC-USD")
with col3:
    period = st.selectbox("Período", ["3mo", "6mo", "1y", "2y"], index=1)

for symbol in [stock_symbol, crypto_symbol]:
    st.subheader(f"📈 {symbol}")
    data = get_data(symbol, period=period, interval="1d")
    if data.empty or not {"Open","High","Low","Close"}.issubset(data.columns):
        st.warning("Sem dados suficientes para este símbolo.")
        continue

    fig, last_rsi = plot_candles_rsi(data)
    sinal = gerar_sinal(last_rsi)
    rsi_txt = f"RSI: {round(last_rsi, 2)}" if pd.notna(last_rsi) else "RSI indisponível"
    st.metric(label=f"Sinal atual ({symbol})", value=sinal, delta=rsi_txt)
    st.plotly_chart(fig, use_container_width=True)
