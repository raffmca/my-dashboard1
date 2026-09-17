from __future__ import annotations

from datetime import date
from html import escape
from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from scipy.stats import norm

APP_VERSION = "Version 1.2"
S_AND_P_50 = (
    "AAPL MSFT NVDA AMZN META GOOGL AVGO TSLA BRK-B GOOG JPM WMT ORCL V LLY NFLX " \
    "COST JNJ HD PG BAC ABBV CVX KO MRK AMD PEP TMO CRM ACN MCD WFC LIN CSCO IBM " \
    "ABT GE NOW INTU ISRG QCOM TXN AMGN CAT PLTR DIS DHR VZ CMCSA PFE NEE".split()
)

st.set_page_config(page_title="Gamma Surface", page_icon="◈", layout="wide")


def authenticate() -> None:
    configured_password = st.secrets.get("dashboard_password")
    if not configured_password:
        st.error("Dashboard is not configured. Set the dashboard_password secret before running this app.")
        st.stop()
    if st.session_state.get("authenticated", False):
        return
    st.title("Gamma Surface")
    st.caption("Enter the dashboard password to continue.")
    password = st.text_input("Password", type="password")
    if st.button("Unlock dashboard", type="primary"):
        if password == configured_password:
            st.session_state.authenticated = True
            st.rerun()
        st.error("Incorrect password.")
    st.stop()


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


def _next_friday() -> str:
    days_ahead = (4 - date.today().weekday()) % 7
    return (date.today() + pd.Timedelta(days=days_ahead)).isoformat()


def _scan_ticker(symbol: str, expiration: str) -> dict[str, Any] | None:
    try:
        ticker = yf.Ticker(symbol)
        if expiration not in list(ticker.options or []):
            return None
        info = ticker.fast_info
        spot = _first_number([info.get("lastPrice"), info.get("regularMarketPrice")])
        if spot is None:
            return None
        chain = ticker.option_chain(expiration)
        options = pd.concat([chain.calls, chain.puts], ignore_index=True)
        for column in ("volume", "openInterest", "bid", "ask"):
            options[column] = pd.to_numeric(options.get(column, 0), errors="coerce").fillna(0)
        options = options[options["strike"].between(spot * 0.95, spot * 1.05)]
        if options.empty:
            return None
        active = options[options["volume"] > 0]
        dollar_volume = float((active["volume"] * spot * 100).sum())
        open_interest = int(options["openInterest"].sum())
        spread_dollars = float((active["ask"] - active["bid"]).clip(lower=0).mean()) if not active.empty else 0.0
        mid = ((active["ask"] + active["bid"]) / 2).replace(0, np.nan)
        spread_percent = float(((active["ask"] - active["bid"]) / mid).replace([np.inf, -np.inf], np.nan).dropna().mean() * 100) if not active.empty else 0.0
        return {"Ticker": symbol, "Spot": spot, "Option volume": int(active["volume"].sum()), "Dollar volume": dollar_volume, "Open interest": open_interest, "Avg spread": spread_dollars, "Spread %": spread_percent}
    except Exception:
        return None


@st.cache_data(ttl=300, show_spinner=False)
def scan_optionable_universe(expiration: str) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(_scan_ticker, symbol, expiration) for symbol in S_AND_P_50]
        for future in as_completed(futures):
            record = future.result()
            if record:
                records.append(record)
    if not records:
        return pd.DataFrame()
    result = pd.DataFrame(records)
    result["volume_score"] = result["Dollar volume"].rank(pct=True) * 30
    result["spread_score"] = (1 - result["Spread %"].rank(pct=True)) * 20
    result["oi_score"] = result["Open interest"].rank(pct=True) * 15
    result["underlying_score"] = result["Spot"].rank(pct=True) * 0
    result["Tradeability"] = (result["volume_score"] + result["spread_score"] + result["oi_score"]).clip(0, 65)
    return result.sort_values(["Tradeability", "Dollar volume"], ascending=False).head(20).reset_index(drop=True)


def render_universe_scan(expiration: str) -> None:
    st.markdown(f"<div class='layer-title'>TOP 20 0DTE UNIVERSE <span>{expiration} · RANKED BY OPTION LIQUIDITY</span></div>", unsafe_allow_html=True)
    with st.spinner(f"Scanning optionable universe for {expiration}..."):
        ranked = scan_optionable_universe(expiration)
    if ranked.empty:
        st.warning(f"Yahoo returned no option chains for {expiration}.")
        return
    display = ranked[["Ticker", "Spot", "Option volume", "Dollar volume", "Open interest", "Spread %", "Tradeability"]].copy()
    display["Dollar volume"] = display["Dollar volume"] / 1_000_000
    display.columns = ["Ticker", "Spot", "Opt vol", "$ opt vol (M)", "OI", "Spread %", "Score"]
    st.dataframe(display.style.format({"Spot": "${:.2f}", "$ opt vol (M)": "${:.1f}", "Spread %": "{:.2f}%", "Score": "{:.0f}"}), use_container_width=True, hide_index=True, height=520)


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
    frame = frame.sort_values("strike")
    frame["Net_GEX"] = frame["Call_GEX"] + frame["Put_GEX"]
    return frame.reset_index(drop=True)


def focus_strikes(frame: pd.DataFrame, spot: float) -> pd.DataFrame:
    frame = frame[(frame["strike"] >= spot * 0.96) & (frame["strike"] <= spot * 1.04)]
    if len(frame) > 41:
        frame = frame.loc[(frame["strike"] - spot).abs().nsmallest(41).index]
    return frame.reset_index(drop=True)


def find_levels(frame: pd.DataFrame, spot: float) -> dict[str, float | str]:
    if frame.empty:
        return {"call_wall": spot, "put_wall": spot, "max_pain": spot, "gamma_flip": spot, "regime": "No gamma data"}
    strikes = frame["strike"].to_numpy()
    call_wall = float(frame.loc[frame["Call_GEX"].idxmax(), "strike"])
    put_wall = float(frame.loc[frame["Put_GEX"].idxmin(), "strike"])
    pain = [np.sum(np.maximum(0, strikes - strike) * frame["Call_OI"] + np.maximum(0, strike - strikes) * frame["Put_OI"]) for strike in strikes]
    net_gex = frame["Net_GEX"].to_numpy()
    changes = np.where(
        ((net_gex[:-1] < 0) & (net_gex[1:] >= 0))
        | ((net_gex[:-1] > 0) & (net_gex[1:] <= 0))
    )[0]
    if len(changes):
        closest_change = min(
            changes,
            key=lambda index: min(
                abs(float(frame.iloc[index]["strike"]) - spot),
                abs(float(frame.iloc[index + 1]["strike"]) - spot),
            ),
        )
        left = frame.iloc[closest_change]
        right = frame.iloc[closest_change + 1]
        net_delta = float(right["Net_GEX"] - left["Net_GEX"])
        gamma_flip = float(left["strike"] - left["Net_GEX"] * (right["strike"] - left["strike"]) / net_delta) if net_delta else float(left["strike"])
    else:
        gamma_flip = spot
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


@st.cache_data(ttl=300, show_spinner=False)
def load_price_history(symbol: str, period: str = "2y") -> pd.DataFrame:
    history = yf.Ticker(symbol).history(period=period, auto_adjust=False)
    return history[["Open", "High", "Low", "Close", "Volume"]].dropna()


def calculate_institutional_layers(
    calls: pd.DataFrame,
    puts: pd.DataFrame,
    spot: float,
    expiration: str,
    history: pd.DataFrame,
) -> dict[str, Any]:
    days_to_expiry = max((date.fromisoformat(expiration) - date.today()).days, 1)
    time = days_to_expiry / 365
    layers: dict[str, Any] = {}
    for options, sign in ((calls, 1.0), (puts, -1.0)):
        volatility = options["impliedVolatility"].clip(lower=0.01)
        d1 = (np.log(spot / options["strike"]) + (0.045 + 0.5 * volatility**2) * time) / (volatility * np.sqrt(time))
        d2 = d1 - volatility * np.sqrt(time)
        vega = spot * norm.pdf(d1) * np.sqrt(time)
        charm = -norm.pdf(d1) * (2 * 0.045 * time - d2 * volatility * np.sqrt(time)) / (2 * time * volatility * np.sqrt(time))
        vanna = -vega * d2 / (spot * volatility)
        layers.setdefault("vanna", 0.0)
        layers.setdefault("charm", 0.0)
        layers["vanna"] += float((vanna * options["openInterest"] * 100 * sign).sum())
        layers["charm"] += float((charm * options["openInterest"] * 100 * sign).sum())

    if history.empty:
        layers.update({"cta": "Unavailable", "cta_detail": "No price history", "breadth": "Unavailable", "breadth_detail": "No ETF history", "backtest": "Unavailable", "win_rate": 0.0})
    else:
        close = history["Close"]
        sma20 = close.rolling(20).mean().iloc[-1]
        sma50 = close.rolling(50).mean().iloc[-1]
        sma200 = close.rolling(200).mean().iloc[-1]
        cta_up = close.iloc[-1] > sma50 and sma50 > sma200
        layers["cta"] = "TREND UP" if cta_up else "TREND DOWN / MIXED"
        layers["cta_detail"] = f"Price ${close.iloc[-1]:.2f} · SMA20 ${sma20:.2f} · SMA50 ${sma50:.2f}"
        signal = close > close.rolling(50).mean()
        next_returns = close.pct_change().shift(-1)
        samples = next_returns[signal].dropna()
        layers["win_rate"] = float((samples > 0).mean() * 100) if len(samples) else 0.0
        layers["backtest"] = f"{layers['win_rate']:.0f}% hit rate · {len(samples)} signals"
        layers["breadth"] = "SINGLE-ASSET PROXY"
        layers["breadth_detail"] = "Add QQQ / IWM confirmation with Tradier"

    atm_iv = float(np.nanmean([
        calls.loc[(calls["strike"] - spot).abs().idxmin(), "impliedVolatility"],
        puts.loc[(puts["strike"] - spot).abs().idxmin(), "impliedVolatility"],
    ]))
    layers["atm_iv"] = atm_iv
    layers["expected_move"] = spot * atm_iv * np.sqrt(days_to_expiry / 365)
    return layers


def render_institutional_layers(layers: dict[str, Any], spot: float) -> None:
    st.markdown("<div class='layer-title'>INSTITUTIONAL LAYERS <span>YAHOO-DERIVED SAMPLE · NOT INVESTMENT ADVICE</span></div>", unsafe_allow_html=True)
    vanna_class = "layer-negative" if layers["vanna"] < 0 else ""
    charm_class = "layer-negative" if layers["charm"] < 0 else ""
    cta_class = "layer-positive" if layers["cta"] == "TREND UP" else "layer-negative"
    backtest_class = "layer-negative" if layers["win_rate"] < 50 else "layer-positive"
    st.markdown(
        f"<div class='layer-grid'>"
        f"<div class='layer-card'><div class='layer-label'>Vanna Exposure <span>PROXY</span></div><div class='layer-value {vanna_class}'>{money(layers['vanna'])}</div><div class='layer-note'>IV sensitivity × OI</div></div>"
        f"<div class='layer-card'><div class='layer-label'>Charm Exposure <span>PROXY</span></div><div class='layer-value {charm_class}'>{money(layers['charm'])}</div><div class='layer-note'>Time decay × OI</div></div>"
        f"<div class='layer-card'><div class='layer-label'>IV Rank / Expected Move</div><div class='layer-value gold'>{layers['atm_iv'] * 100:.1f}% / ±${layers['expected_move']:.2f}</div><div class='layer-note'>IV rank unavailable from Yahoo</div></div>"
        f"<div class='layer-card'><div class='layer-label'>CTA Trend State</div><div class='layer-value {cta_class}'>{layers['cta']}</div><div class='layer-note'>{layers['cta_detail']}</div></div>"
        f"<div class='layer-card'><div class='layer-label'>Breadth Confirmation</div><div class='layer-value gold'>{layers['breadth']}</div><div class='layer-note'>{layers['breadth_detail']}</div></div>"
        f"<div class='layer-card'><div class='layer-label'>Historical Signal Test</div><div class='layer-value {backtest_class}'>{layers['backtest']}</div><div class='layer-note'>Close above 50-day SMA · next-day return</div></div>"
        "</div>",
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
        marker=dict(symbol="diamond", size=16, color="#f5c84b", line=dict(color="#080d14", width=3)),
        text=[""],
        name="Max pain",
        hovertemplate="Max pain %{y:.2f}<extra></extra>",
    ))
    chart.add_vline(x=0, line_color="#5e6b7d", line_width=1)
    chart.add_hline(y=spot, line_color="#f5c84b", line_width=2, annotation_text=f"SPOT ${spot:.2f}", annotation_position="top left", annotation_font_color="#f5c84b")
    chart.add_hline(y=float(levels["max_pain"]), line_color="#f5c84b", line_dash="dash", line_width=1.5, opacity=0.9)
    chart.add_annotation(
        x=0,
        y=float(levels["max_pain"]),
        text=f"MAX PAIN ${float(levels['max_pain']):.2f}",
        showarrow=True,
        arrowhead=2,
        arrowcolor="#f5c84b",
        ax=0,
        ay=-34,
        bgcolor="#111a24",
        bordercolor="#f5c84b",
        borderwidth=1,
        borderpad=4,
        font=dict(color="#f5c84b", size=10),
    )
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
    authenticate()
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
    .regime.negative { border-left-color:#ff557d; color:#ff557d; }
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
    .summary-strip { display:grid; grid-template-columns:repeat(8, minmax(125px, 1fr)); gap:8px; margin-top:18px; overflow-x:auto; }
    .summary-card { min-height:76px; padding:10px 12px; border:1px solid #243140; border-radius:6px; background:#111a24; }
    .summary-card-label { color:#dce5ef; font:600 .64rem 'Space Grotesk',sans-serif; text-transform:uppercase; white-space:nowrap; }
    .summary-card-value { color:#28d7a1; font:600 1rem 'DM Mono',monospace; margin-top:7px; white-space:nowrap; }
    .summary-card-value.negative { color:#ff557d; }
    .summary-card-value.gold { color:#f5c84b; }
    .summary-card-note { color:#8796a8; font:400 .61rem 'DM Mono',monospace; margin-top:4px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    .layer-title { color:#dce5ef; border-bottom:1px solid #243140; margin:26px 0 10px; padding-bottom:8px; font:600 .8rem 'DM Mono',monospace; letter-spacing:.1em; }
    .layer-title span { color:#8796a8; font-size:.62rem; margin-left:8px; }
    .layer-grid { display:grid; grid-template-columns:repeat(3, minmax(180px, 1fr)); gap:8px; }
    .layer-card { min-height:82px; border:1px solid #243140; border-radius:6px; padding:11px 13px; background:#0f1822; }
    .layer-label { color:#8796a8; font:500 .66rem 'DM Mono',monospace; text-transform:uppercase; }
    .layer-label span { color:#aa7cff; font-size:.56rem; }
    .layer-value { color:#28d7a1; font:600 .95rem 'DM Mono',monospace; margin-top:9px; }
    .layer-value.gold { color:#f5c84b; }
    .layer-value.layer-negative { color:#ff557d; }
    .layer-value.layer-positive { color:#28d7a1; }
    .layer-note { color:#8796a8; font:400 .62rem 'DM Mono',monospace; margin-top:5px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    @media (max-width: 900px) { .layer-grid { grid-template-columns:repeat(2, minmax(160px, 1fr)); } }
    @media (max-width: 560px) { .layer-grid { grid-template-columns:1fr; } }
    </style>
    """, unsafe_allow_html=True)
    st.sidebar.markdown(f"<div class='terminal-label'>GAMMA SURFACE / PUBLIC DATA · {APP_VERSION.upper()}</div>", unsafe_allow_html=True)
    symbol = st.sidebar.text_input("Symbol", "SPY", max_chars=8).strip().upper()
    st.sidebar.caption("Yahoo Finance · delayed market data")
    refresh = st.sidebar.button("Refresh data", use_container_width=True, type="primary")
    if "show_layers" not in st.session_state:
        st.session_state.show_layers = False
    if st.sidebar.button(
        "Hide institutional layers" if st.session_state.show_layers else "Add institutional layers",
        use_container_width=True,
    ):
        st.session_state.show_layers = not st.session_state.show_layers
    if "show_universe" not in st.session_state:
        st.session_state.show_universe = False
    if st.sidebar.button(
        "Hide top-20 scanner" if st.session_state.show_universe else "Scan top 20 optionable",
        use_container_width=True,
    ):
        st.session_state.show_universe = not st.session_state.show_universe
    scanner_expiries = [date.today().isoformat(), _next_friday()]
    scanner_expiry = st.sidebar.selectbox(
        "Universe expiry",
        scanner_expiries,
        format_func=lambda value: f"{value} · {'TODAY' if value == date.today().isoformat() else 'FRIDAY'}",
        disabled=not st.session_state.show_universe,
    )
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
    full_frame = build_gex_frame(calls, puts, spot, expiration)
    frame = focus_strikes(full_frame, spot)
    levels = find_levels(full_frame, spot)
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
    regime_class = " negative" if total < 0 else ""
    st.markdown(f"<div class='regime{regime_class}'>{levels['regime']} <span style='color:#8796a8'>· {expiration} · SOURCE: YAHOO FINANCE</span></div>", unsafe_allow_html=True)
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
        f"<div class='summary-card'><div class='summary-card-label'>Max Pain · {expiration}</div><div class='summary-card-value gold'>${float(levels['max_pain']):.2f}</div><div class='summary-card-note'>{float(levels['max_pain']) - spot:+.2f} from spot</div></div>"
        f"<div class='summary-card'><div class='summary-card-label'>Δ Grower · all dates</div><div class='summary-card-value{grower_class}'>${spot:.2f} {grower:+.1f}</div><div class='summary-card-note'>derived from daily open</div></div>"
        f"<div class='summary-card'><div class='summary-card-label'>± Move · {expiration}</div><div class='summary-card-value gold'>±${implied_move:.2f}</div><div class='summary-card-note'>ATM IV implied range</div></div>"
        f"<div class='summary-card'><div class='summary-card-label'>ATM IV · {expiration}</div><div class='summary-card-value gold'>{atm_iv * 100:.1f}%</div><div class='summary-card-note'>call/put midpoint</div></div>"
        "</div>",
        unsafe_allow_html=True,
    )
    if st.session_state.show_universe:
        render_universe_scan(scanner_expiry)
    if st.session_state.show_layers:
        try:
            history = load_price_history(symbol)
            layers = calculate_institutional_layers(calls, puts, spot, expiration, history)
            render_institutional_layers(layers, spot)
        except Exception as exc:
            st.warning(f"Institutional layers unavailable: {exc}")


if __name__ == "__main__":
    main()
