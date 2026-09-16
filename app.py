from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from scipy.stats import norm

st.set_page_config(page_title="Gamma Surface", page_icon="◈", layout="wide")


def _first_number(values: list[Any]) -> float | None:
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number) and number > 0:
            return number
    return None


@st.cache_data(ttl=60, show_spinner=False)
def load_market_data(symbol: str) -> tuple[float | None, list[str], str | None]:
    ticker = yf.Ticker(symbol)
    try:
        fast_info = ticker.fast_info
        spot = _first_number([fast_info.get("lastPrice"), fast_info.get("regularMarketPrice")])
    except Exception:
        spot = None

    if spot is None:
        try:
            history = ticker.history(period="5d", auto_adjust=False)
            spot = _first_number(history["Close"].tolist()[::-1]) if not history.empty else None
        except Exception:
            spot = None

    try:
        expirations = list(ticker.options or [])
    except Exception:
        expirations = []

    if spot is None:
        return None, expirations, f"Could not load a current price for {symbol}."
    if not expirations:
        return spot, [], f"No listed options were returned for {symbol}."
    return spot, expirations, None


@st.cache_data(ttl=60, show_spinner=False)
def load_chain(symbol: str, expiration: str) -> tuple[pd.DataFrame, pd.DataFrame, str | None]:
    try:
        chain = yf.Ticker(symbol).option_chain(expiration)
    except Exception as exc:
        return pd.DataFrame(), pd.DataFrame(), f"Could not load the {expiration} option chain: {exc}"

    calls = chain.calls.copy()
    puts = chain.puts.copy()
    required = {"strike", "openInterest", "impliedVolatility"}
    if not required.issubset(calls.columns) or not required.issubset(puts.columns):
        return pd.DataFrame(), pd.DataFrame(), "The data provider returned an incomplete option chain."

    for frame in (calls, puts):
        frame["openInterest"] = pd.to_numeric(frame["openInterest"], errors="coerce").fillna(0)
        frame["impliedVolatility"] = pd.to_numeric(frame["impliedVolatility"], errors="coerce").fillna(0)
        frame["strike"] = pd.to_numeric(frame["strike"], errors="coerce")
    calls = calls.dropna(subset=["strike"])
    puts = puts.dropna(subset=["strike"])
    return calls, puts, None


def calculate_gamma(spot: float, strike: pd.Series, days_to_expiry: int, volatility: pd.Series) -> pd.Series:
    time = max(days_to_expiry, 1) / 365
    safe_volatility = volatility.clip(lower=0.01)
    d1 = (np.log(spot / strike) + (0.045 + 0.5 * safe_volatility**2) * time) / (safe_volatility * np.sqrt(time))
    return pd.Series(norm.pdf(d1) / (spot * safe_volatility * np.sqrt(time)), index=strike.index)


def build_gex_frame(calls: pd.DataFrame, puts: pd.DataFrame, spot: float, expiration: str) -> pd.DataFrame:
    days_to_expiry = max((date.fromisoformat(expiration) - date.today()).days, 1)
    calls = calls.copy()
    puts = puts.copy()
    calls["Call_GEX"] = calculate_gamma(spot, calls["strike"], days_to_expiry, calls["impliedVolatility"]) * calls["openInterest"] * 100 * spot**2 * 0.01
    puts["Put_GEX"] = -calculate_gamma(spot, puts["strike"], days_to_expiry, puts["impliedVolatility"]) * puts["openInterest"] * 100 * spot**2 * 0.01

    frame = pd.merge(
        calls[["strike", "openInterest", "Call_GEX"]].rename(columns={"openInterest": "Call_OI"}),
        puts[["strike", "openInterest", "Put_GEX"]].rename(columns={"openInterest": "Put_OI"}),
        on="strike",
        how="outer",
    ).fillna(0)
    frame = frame[(frame["strike"] >= spot * 0.90) & (frame["strike"] <= spot * 1.10)].sort_values("strike")
    frame["Net_GEX"] = frame["Call_GEX"] + frame["Put_GEX"]
    return frame.reset_index(drop=True)


def find_levels(frame: pd.DataFrame, spot: float) -> dict[str, float | str]:
    if frame.empty:
        return {"call_wall": spot, "put_wall": spot, "max_pain": spot, "gamma_flip": spot, "regime": "No gamma data"}
    strikes = frame["strike"].to_numpy()
    call_wall = float(frame.loc[frame["Call_GEX"].idxmax(), "strike"])
    put_wall = float(frame.loc[frame["Put_GEX"].idxmin(), "strike"])
    pain = [np.sum(np.maximum(0, strikes - strike) * frame["Call_OI"] + np.maximum(0, strike - strikes) * frame["Put_OI"]) for strike in strikes]
    changes = np.where(np.diff(np.sign(frame["Net_GEX"].to_numpy())) != 0)[0]
    gamma_flip = float(frame.iloc[changes[0]]["strike"]) if len(changes) else spot
    total = float(frame["Net_GEX"].sum())
    return {
        "call_wall": call_wall,
        "put_wall": put_wall,
        "max_pain": float(strikes[np.argmin(pain)]),
        "gamma_flip": gamma_flip,
        "regime": "POSITIVE GAMMA / PINNING" if total >= 0 else "NEGATIVE GAMMA / EXPANSION",
    }


def money(value: float) -> str:
    sign = "-" if value < 0 else ""
    amount = abs(value)
    if amount >= 1_000_000:
        return f"{sign}${amount / 1_000_000:.1f}M"
    if amount >= 1_000:
        return f"{sign}${amount / 1_000:.1f}K"
    return f"{sign}${amount:.0f}"


def style_net_gex(values: pd.Series) -> list[str]:
    max_abs = max(float(values.abs().max()), 1.0)
    styles = []
    for value in values:
        intensity = min(abs(float(value)) / max_abs, 1.0)
        if value >= 0:
            styles.append(f"background-color: rgba(40, 215, 161, {0.12 + intensity * 0.55:.2f}); color: #dce5ef")
        else:
            styles.append(f"background-color: rgba(255, 85, 125, {0.12 + intensity * 0.55:.2f}); color: #dce5ef")
    return styles


def render_chart(frame: pd.DataFrame, spot: float, levels: dict[str, float | str], symbol: str) -> None:
    max_exposure = max(
        float(frame[["Call_GEX", "Put_GEX"]].abs().to_numpy().max()) / 1_000_000,
        1.0,
    )
    chart = go.Figure()
    chart.add_trace(go.Bar(
        y=frame["strike"],
        x=frame["Put_GEX"] / 1_000_000,
        orientation="h",
        name="Put GEX",
        marker_color="#ff557d",
        hovertemplate="Strike %{y:.2f}<br>Put GEX $%{x:.2f}M<extra></extra>",
    ))
    chart.add_trace(go.Bar(
        y=frame["strike"],
        x=frame["Call_GEX"] / 1_000_000,
        orientation="h",
        name="Call GEX",
        marker_color="#28d7a1",
        hovertemplate="Strike %{y:.2f}<br>Call GEX $%{x:.2f}M<extra></extra>",
    ))
    chart.add_vline(x=0, line_color="#5e6b7d", line_width=1)
    chart.add_hline(y=spot, line_color="#f5c84b", line_width=2, annotation_text=f"SPOT ${spot:.2f}", annotation_position="top left")
    for key, color in (("call_wall", "#28d7a1"), ("put_wall", "#ff557d"), ("gamma_flip", "#aa7cff")):
        chart.add_hline(y=float(levels[key]), line_color=color, line_dash="dot", line_width=1, annotation_text=key.replace("_", " ").upper(), annotation_font_color=color)
    chart.update_layout(
        height=570,
        template="plotly_dark",
        paper_bgcolor="#080d14",
        plot_bgcolor="#080d14",
        margin=dict(l=10, r=18, t=42, b=20),
        barmode="relative",
        showlegend=False,
        xaxis=dict(
            title="Dealer GEX ($M)",
            range=[-max_exposure * 1.12, max_exposure * 1.12],
            zeroline=False,
            gridcolor="#1b2735",
        ),
        yaxis=dict(
            title="Strike",
            range=[float(frame["strike"].min()), float(frame["strike"].max())],
            gridcolor="#1b2735",
        ),
        title=f"{symbol} / DEALER GAMMA BY STRIKE",
    )
    st.plotly_chart(chart, use_container_width=True, config={"displayModeBar": False})


def main() -> None:
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root { --ink:#080d14; --panel:#111a24; --line:#243140; --muted:#8796a8; --green:#28d7a1; --red:#ff557d; --gold:#f5c84b; }
    .stApp { background: radial-gradient(circle at 20% -10%, #15243b 0, #080d14 42%); color:#dce5ef; }
    h1,h2,h3,p,div,button { font-family:'Space Grotesk', sans-serif; }
    code, .stMetricValue, [data-testid='stDataFrame'] { font-family:'DM Mono', monospace; }
    [data-testid='stMetric'] { background:#111a24; border:1px solid #243140; padding:14px 16px; border-radius:6px; }
    [data-testid='stMetricLabel'] { color:#8796a8; text-transform:uppercase; letter-spacing:.08em; font-size:.7rem; }
    [data-testid='stMetricValue'] { color:#f5c84b; font-size:1.25rem; }
    section[data-testid='stSidebar'] { background:#0b121b; border-right:1px solid #243140; }
    .terminal-label { color:#8796a8; font:500 .7rem 'DM Mono',monospace; letter-spacing:.14em; text-transform:uppercase; }
    .regime { border-left:3px solid #28d7a1; background:#101b25; padding:12px 16px; margin:4px 0 18px; color:#28d7a1; font:500 .78rem 'DM Mono',monospace; letter-spacing:.08em; }
    </style>
    """, unsafe_allow_html=True)
    st.sidebar.markdown("<div class='terminal-label'>GAMMA SURFACE / PUBLIC DATA</div>", unsafe_allow_html=True)
    symbol = st.sidebar.text_input("Symbol", "SPY", max_chars=8).strip().upper()
    st.sidebar.caption("Yahoo Finance · delayed market data")
    refresh = st.sidebar.button("Refresh data", use_container_width=True, type="primary")
    if refresh:
        load_market_data.clear()
        load_chain.clear()

    st.markdown("<div class='terminal-label'>OPTIONS POSITIONING TERMINAL · LIVE SNAPSHOT</div>", unsafe_allow_html=True)
    st.title(f"{symbol} / Dealer Gamma Surface")
    spot, expirations, market_error = load_market_data(symbol)
    if market_error:
        st.error(market_error)
        return
    expiration = st.sidebar.selectbox("Expiration", expirations, index=0, format_func=lambda value: f"{value}  ·  {(date.fromisoformat(value) - date.today()).days}D")
    calls, puts, chain_error = load_chain(symbol, expiration)
    if chain_error or spot is None:
        st.error(chain_error or "No spot price available.")
        return
    frame = build_gex_frame(calls, puts, spot, expiration)
    levels = find_levels(frame, spot)
    total = float(frame["Net_GEX"].sum())
    st.markdown(f"<div class='regime'>{levels['regime']} <span style='color:#8796a8'>· {expiration} · SOURCE: YAHOO FINANCE</span></div>", unsafe_allow_html=True)
    metrics = st.columns(5)
    metrics[0].metric("Spot", f"${spot:.2f}")
    metrics[1].metric("Net GEX", money(total))
    metrics[2].metric("Gamma flip", f"${float(levels['gamma_flip']):.2f}")
    metrics[3].metric("Call wall", f"${float(levels['call_wall']):.2f}")
    metrics[4].metric("Put wall", f"${float(levels['put_wall']):.2f}")
    render_chart(frame, spot, levels, symbol)
    st.subheader("Strike matrix")
    display = frame[["strike", "Call_GEX", "Net_GEX", "Put_GEX"]].copy()
    display.columns = ["Strike", "Call GEX", "Net GEX", "Put GEX"]
    styled_display = display.style.format(
        {"Strike": "${:.2f}", "Call GEX": money, "Net GEX": money, "Put GEX": money}
    ).set_properties(
        **{"background-color": "#111a24", "color": "#dce5ef", "border-color": "#243140"}
    ).set_table_styles([
        {"selector": "th", "props": [("background-color", "#0b121b"), ("color", "#8796a8"), ("border-color", "#243140")]},
    ]).apply(style_net_gex, subset=["Net GEX"])
    st.dataframe(styled_display, use_container_width=True, hide_index=True, height=390)


if __name__ == "__main__":
    main()
