# forecast_app/plots.py
from typing import Optional
import numpy as np
import pandas as pd
import streamlit as st

from .data_helpers import _parse_dates_any


def _build_pseudo_ohlc_from_close(df: pd.DataFrame, date_col: str, price_col: str, last_n: int = 260):
    tmp = df[[date_col, price_col]].copy()
    tmp[date_col] = _parse_dates_any(tmp[date_col])
    tmp[price_col] = pd.to_numeric(tmp[price_col], errors="coerce")
    tmp = tmp.dropna(subset=[date_col, price_col]).sort_values(date_col).reset_index(drop=True)
    if tmp.empty:
        return tmp

    tmp = tmp.tail(int(last_n)).reset_index(drop=True)
    tmp["close"] = tmp[price_col]
    tmp["open"] = tmp["close"].shift(1)
    tmp.loc[tmp.index[0], "open"] = tmp.loc[tmp.index[0], "close"]

    delta = (tmp["close"] - tmp["open"]).abs()
    wick = delta.rolling(10, min_periods=1).mean() * 0.6
    wick = wick.fillna(0.0)

    tmp["high"] = tmp[["open", "close"]].max(axis=1) + wick
    tmp["low"] = tmp[["open", "close"]].min(axis=1) - wick
    tmp["volume"] = delta
    return tmp


def plot_candlestick_preview(
    df: pd.DataFrame,
    date_col: str,
    price_col: str,
    last_n: int = 260,
    show_volume: bool = True,
    title: Optional[str] = None,
):
    if df is None or df.empty or date_col not in df.columns or price_col not in df.columns:
        st.info("Chưa có dữ liệu để vẽ candlestick.")
        return

    ohlc = _build_pseudo_ohlc_from_close(df, date_col, price_col, last_n=last_n)
    if ohlc is None or ohlc.empty:
        st.info("Không đủ dữ liệu để vẽ.")
        return

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception:
        st.error("Thiếu Plotly. Cài bằng: pip install plotly")
        return

    if title is None:
        title = f"{price_col} - Candlestick (daily)"

    if show_volume:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.78, 0.22])
    else:
        fig = make_subplots(rows=1, cols=1)

    candle = go.Candlestick(
        x=ohlc[date_col],
        open=ohlc["open"],
        high=ohlc["high"],
        low=ohlc["low"],
        close=ohlc["close"],
        name=f"{price_col}",
        increasing_line_color="#14B8A6",
        decreasing_line_color="#F43F5E",
        increasing_fillcolor="#14B8A6",
        decreasing_fillcolor="#F43F5E",
        line=dict(width=1),
        whiskerwidth=0.3,
    )
    fig.add_trace(candle, row=1, col=1)

    if show_volume:
        vol = go.Bar(x=ohlc[date_col], y=ohlc["volume"], name="Volume", marker=dict(color="rgba(2,6,23,0.10)"))
        fig.add_trace(vol, row=2, col=1)

    fig.update_layout(
        template="plotly_white",
        height=620 if show_volume else 520,
        margin=dict(l=10, r=10, t=55, b=10),
        title=dict(text=title, x=0.02, xanchor="left"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(2,6,23,0.05)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(2,6,23,0.05)")
    if show_volume:
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume (proxy)", row=2, col=1, showgrid=False)

    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
