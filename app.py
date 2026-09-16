from __future__ import annotations

from datetime import date
from html import escape
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
    frame = frame[(frame["strike"] >= spot * 0.96) & (frame["strike"] <= spot * 1.04)]
    if len(frame) > 41:
        frame = frame.loc[(frame["strike"] - spot).abs().nsmallest(41).index]
    frame = frame.sort_values("strike")
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


def style_matrix_rows(row: pd.Series, spot_strike: float, levels: dict[str, float | str]) -> list[str]:
    strike = float(row["Strike"])
    styles = ["" for _ in row]
    if abs(strike - float(levels["max_pain"])) < 0.01:
        styles = ["background-color: rgba(245, 200, 75, 0.22); color: #f5c84b; font-weight: 600" for _ in row]
    if abs(strike - spot_strike) < 0.01:
        styles = ["background-color: rgba(245, 200, 75, 0.10); color: #f5c84b; font-weight: 600" for _ in row]
    return styles


def render_matrix(frame: pd.DataFrame, spot: float, levels: dict[str, float | str]) -> None:
    max_net = max(float(frame["Net_GEX"].abs().max()), 1.0)
    spot_strike = float(frame.loc[(frame["strike"] - spot).abs().idxmin(), "strike"])
    rows = []
    for _, row in frame.iterrows():
        strike = float(row["strike"])
        is_max_pain = abs(strike - float(levels["max_pain"])) < 0.01
        is_spot = abs(strike - spot_strike) < 0.01
        row_class = " gold-row" if is_max_pain else " spot-row" if is_spot else ""
        net = float(row["Net_GEX"])
        net_width = min(abs(net) / max_net * 100, 100)
        net_color = "#28d7a1" if net >= 0 else "#ff557d"
        marker = " ◆ MAX PAIN" if is_max_pain else " ◈ SPOT" if is_spot else ""
        rows.append(
            f"<div class='matrix-row{row_class}'>"
            f"<div class='strike-cell'>{strike:.0f}<span class='row-marker'>{escape(marker)}</span></div>"
            f"<div class='value-cell call-value'>{money(float(row['Call_GEX']))}</div>"
            f"<div class='net-cell'><span class='net-bar' style='width:{net_width:.1f}%;background:{net_color}'></span><span>{money(net)}</span></div>"
            f"<div class='value-cell put-value'>{money(float(row['Put_GEX']))}</div>"
            "</div>"
        )
    st.markdown(
        "<div class='matrix'>"
        "<div class='matrix-head'><div>STRIKE</div><div class='right'>CALL GEX</div><div class='right'>NET GEX</div><div class='right'>PUT GEX</div></div>"
        + "".join(rows)
        + "</div>",
        unsafe_allow_html=True,
    )


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
    chart.add_trace(go.Scatter(
        x=[0],
        y=[float(levels["max_pain"])],
        mode="markers+text",
        marker=dict(symbol="diamond", size=12, color="#f5c84b", line=dict(color="#fff1a6", width=1)),
        text=[f"MAX PAIN ${float(levels['max_pain']):.2f}"],
        textposition="middle right",
        textfont=dict(color="#f5c84b", size=10),
        name="Max pain",
        hovertemplate="Max pain %{y:.2f}<extra></extra>",
    ))
    chart.add_vline(x=0, line_color="#5e6b7d", line_width=1)
    chart.add_hline(y=spot, line_color="#f5c84b", line_width=2, annotation_text=f"SPOT ${spot:.2f}", annotation_position="top left", annotation_font_color="#f5c84b")
    for key, color in (("call_wall", "#28d7a1"), ("put_wall", "#ff557d"), ("gamma_flip", "#aa7cff")):
        chart.add_hline(y=float(levels[key]), line_color=color, line_dash="dot", line_width=1, annotation_text=f"{key.replace('_', ' ').upper()} ${float(levels[key]):.2f}", annotation_font_color=color, annotation_position="top right")
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
            dtick=max(float(frame["strike"].max() - frame["strike"].min()) / 12, 1),
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
    .summary-value { color:#f5c84b; font:600 1.1rem 'DM Mono',monospace; }
    .summary-note { color:#8796a8; font:400 .7rem 'DM Mono',monospace; }
    [data-testid='stMarkdownContainer'] .summary-value { line-height:1.8; }
    .matrix { border:1px solid #243140; border-radius:6px; overflow:hidden; background:#0d151f; font:400 .72rem 'DM Mono',monospace; }
    .matrix-head, .matrix-row { display:grid; grid-template-columns: 1fr 1.2fr 1.6fr 1.2fr; align-items:center; min-height:32px; border-bottom:1px solid #1c2937; }
    .matrix-head { background:#162231; color:#8796a8; font-size:.64rem; letter-spacing:.1em; }
    .matrix-head > div, .matrix-row > div { padding:7px 12px; }
    .matrix-head .right { text-align:right; }
    .matrix-row { color:#dce5ef; }
    .matrix-row:hover { background:#182533; }
    .strike-cell { color:#dce5ef; font-weight:600; }
    .call-value { color:#57e4b4; text-align:right; }
    .put-value { color:#ff7797; text-align:right; }
    .net-cell { position:relative; display:flex; justify-content:flex-end; overflow:hidden; }
    .net-cell > span:last-child { position:relative; z-index:1; }
    .net-bar { position:absolute; right:0; top:3px; bottom:3px; opacity:.32; border-radius:2px; }
    .row-marker { color:#f5c84b; font-size:.58rem; margin-left:8px; }
    .gold-row { background:rgba(245,200,75,.13); box-shadow:inset 3px 0 #f5c84b; }
    .spot-row { background:rgba(245,200,75,.06); box-shadow:inset 3px 0 #8c7430; }
    .summary-strip { display:grid; grid-template-columns:repeat(7, minmax(125px, 1fr)); gap:8px; margin-top:18px; overflow-x:auto; }
    .summary-card { min-height:76px; padding:10px 12px; border:1px solid #243140; border-radius:6px; background:#111a24; }
    .summary-card-label { color:#dce5ef; font:600 .64rem 'Space Grotesk',sans-serif; text-transform:uppercase; white-space:nowrap; }
    .summary-card-value { color:#28d7a1; font:600 1rem 'DM Mono',monospace; margin-top:7px; white-space:nowrap; }
    .summary-card-value.negative { color:#ff557d; }
    .summary-card-value.gold { color:#f5c84b; }
    .summary-card-note { color:#8796a8; font:400 .61rem 'DM Mono',monospace; margin-top:4px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
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
    today_expiration = date.today().isoformat()
    default_expiration = expirations.index(today_expiration) if today_expiration in expirations else 0
    expiration = st.sidebar.selectbox("Expiration", expirations, index=default_expiration, format_func=lambda value: f"{value}  ·  {(date.fromisoformat(value) - date.today()).days}D")
    calls, puts, chain_error = load_chain(symbol, expiration)
    if chain_error or spot is None:
        st.error(chain_error or "No spot price available.")
        return
    frame = build_gex_frame(calls, puts, spot, expiration)
    levels = find_levels(frame, spot)
    total = float(frame["Net_GEX"].sum())
    days_to_expiry = max((date.fromisoformat(expiration) - date.today()).days, 1)
    spot_strike = float(frame.loc[(frame["strike"] - spot).abs().idxmin(), "strike"])
    nearest_call = calls.loc[(calls["strike"] - spot).abs().idxmin()]
    nearest_put = puts.loc[(puts["strike"] - spot).abs().idxmin()]
    atm_iv = float(np.nanmean([nearest_call["impliedVolatility"], nearest_put["impliedVolatility"]]))
    implied_move = spot * atm_iv * np.sqrt(days_to_expiry / 365)
    volt_strike = spot_strike
    try:
        daily_open = float(yf.Ticker(symbol).history(period="5d", auto_adjust=False)["Open"].dropna().iloc[-1])
    except Exception:
        daily_open = spot
    grower = spot - daily_open
    st.markdown(f"<div class='regime'>{levels['regime']} <span style='color:#8796a8'>· {expiration} · SOURCE: YAHOO FINANCE</span></div>", unsafe_allow_html=True)
    metrics = st.columns(5)
    metrics[0].metric("Spot", f"${spot:.2f}")
    metrics[1].metric("Net GEX", money(total))
    metrics[2].metric("Gamma flip", f"${float(levels['gamma_flip']):.2f}")
    metrics[3].metric("Call wall", f"${float(levels['call_wall']):.2f}")
    metrics[4].metric("Put wall", f"${float(levels['put_wall']):.2f}")
    render_chart(frame, spot, levels, symbol)
    st.subheader("Strike matrix")
    render_matrix(frame, spot, levels)
    total_class = "" if total >= 0 else " negative"
    grower_class = "" if grower >= 0 else " negative"
    st.markdown(
        f"<div class='summary-strip'>"
        f"<div class='summary-card'><div class='summary-card-label'>Net GEX · {expiration}</div><div class='summary-card-value{total_class}'>{money(total)}</div><div class='summary-card-note'>{levels['regime']}</div></div>"
        f"<div class='summary-card'><div class='summary-card-label'>Call Wall · {expiration}</div><div class='summary-card-value'>${float(levels['call_wall']):.2f}</div><div class='summary-card-note'>{float(levels['call_wall']) - spot:+.2f} from spot</div></div>"
        f"<div class='summary-card'><div class='summary-card-label'>Volt · {expiration}</div><div class='summary-card-value gold'>${volt_strike:.0f}</div><div class='summary-card-note'>nearest listed strike</div></div>"
        f"<div class='summary-card'><div class='summary-card-label'>Gamma Flip · {expiration}</div><div class='summary-card-value'>${float(levels['gamma_flip']):.2f}</div><div class='summary-card-note'>{float(levels['gamma_flip']) - spot:+.2f} from spot</div></div>"
        f"<div class='summary-card'><div class='summary-card-label'>Δ Grower · all dates</div><div class='summary-card-value{grower_class}'>${spot:.2f} {grower:+.1f}</div><div class='summary-card-note'>derived from daily open</div></div>"
        f"<div class='summary-card'><div class='summary-card-label'>± Move · {expiration}</div><div class='summary-card-value gold'>±${implied_move:.2f}</div><div class='summary-card-note'>ATM IV implied range</div></div>"
        f"<div class='summary-card'><div class='summary-card-label'>ATM IV · {expiration}</div><div class='summary-card-value gold'>{atm_iv * 100:.1f}%</div><div class='summary-card-note'>call/put midpoint</div></div>"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
