from __future__ import annotations

import sqlite3
import hmac
from datetime import date, datetime, time, timedelta
from html import escape
from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
import exchange_calendars as xcals
from scipy.stats import norm

APP_VERSION = "Version 3.0"
S_AND_P_50 = (
    "AAPL MSFT NVDA AMZN META GOOGL AVGO TSLA BRK-B GOOG JPM WMT ORCL V LLY NFLX " \
    "COST JNJ HD PG BAC ABBV CVX KO MRK AMD PEP TMO CRM ACN MCD WFC LIN CSCO IBM " \
    "ABT GE NOW INTU ISRG QCOM TXN AMGN CAT PLTR DIS DHR VZ CMCSA PFE NEE".split()
)
TODAY_UNIVERSE = ("QQQ", "SPY", "IWM", "SPXW")
MAG7 = ("AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "TSLA", "AMD", "AVGO", "PLTR", "NFLX")
PREMIUM_SENTIMENT_UNIVERSE = ("SPY", "QQQ", "NVDA", "MU", "AAPL", "TSLA")
YAHOO_SYMBOLS = {"SPXW": "^SPX"}

st.set_page_config(page_title="Gamma Surface", page_icon="◈", layout="wide")


def authenticate() -> None:
    try:
        configured_password = st.secrets.get("dashboard_password")
    except Exception:
        configured_password = None

    if not configured_password:
        return

    if st.session_state.get("authenticated", False):
        return

    st.title("Gamma Surface")
    st.caption("Enter the dashboard password to continue.")
    password = st.text_input("Password", type="password")
    if st.button("Unlock dashboard", type="primary"):
        if hmac.compare_digest(password, str(configured_password)):
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

    expirations: list[str] = []
    options_error: Exception | None = None
    for attempt in range(2):
        try:
            expirations = list(yf.Ticker(symbol).options or [])
            if expirations:
                break
        except Exception as exc:
            options_error = exc

    if spot is None:
        return None, expirations, f"Could not load a current price for {symbol}."
    expirations = _valid_expirations(expirations)
    if not expirations:
        if options_error:
            return spot, [], f"Yahoo Finance temporarily failed to return options for {symbol}. Try Refresh data in a few seconds."
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


def calculate_premium_sentiment(
    calls: pd.DataFrame,
    puts: pd.DataFrame,
    symbol: str,
    expiration: str,
) -> dict[str, Any] | None:
    """Estimate traded option premium from reported volume and bid/ask midpoint."""
    required = {"volume", "bid", "ask", "strike"}
    if not required.issubset(calls.columns) or not required.issubset(puts.columns):
        return None

    def prepare(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        for column in ("volume", "bid", "ask", "strike"):
            result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0)
        result["mid"] = ((result["bid"] + result["ask"]) / 2).clip(lower=0)
        result["premium"] = result["volume"] * result["mid"] * 100
        return result[(result["volume"] > 0) & (result["mid"] > 0)]

    active_calls = prepare(calls)
    active_puts = prepare(puts)
    call_premium = float(active_calls["premium"].sum())
    put_premium = float(active_puts["premium"].sum())
    total_premium = call_premium + put_premium
    if total_premium <= 0:
        return None

    net_premium = call_premium - put_premium
    put_call_ratio = put_premium / call_premium if call_premium else np.inf
    call_share = call_premium / total_premium
    if put_call_ratio >= 1.35:
        sentiment = "BEARISH / HEDGING"
    elif put_call_ratio >= 1.1:
        sentiment = "BEARISH / PROTECTIVE"
    elif put_call_ratio <= 0.75:
        sentiment = "BULLISH / DIRECTIONAL"
    elif put_call_ratio <= 0.9:
        sentiment = "BULLISH / CALL LEAN"
    else:
        sentiment = "NEUTRAL / EQUILIBRIUM"
    return {
        "Ticker": symbol,
        "Expiration": expiration,
        "Call Premium": call_premium,
        "Put Premium": put_premium,
        "Net Premium": net_premium,
        "Call Share": call_share,
        "Put/Call Premium": put_call_ratio,
        "Sentiment": sentiment,
        "Call Contracts": int(active_calls["volume"].sum()),
        "Put Contracts": int(active_puts["volume"].sum()),
        "Quoted Rows": len(active_calls) + len(active_puts),
    }


@st.cache_data(ttl=300, show_spinner=False)
def scan_premium_sentiment(expiration: str, symbols: tuple[str, ...]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {
            executor.submit(load_chain, YAHOO_SYMBOLS.get(symbol, symbol), expiration): symbol
            for symbol in symbols
        }
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                calls, puts, error = future.result()
                if not error:
                    record = calculate_premium_sentiment(calls, puts, symbol, expiration)
                    if record:
                        records.append(record)
            except Exception:
                continue
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).sort_values("Net Premium", ascending=False).reset_index(drop=True)


NYSE_CALENDAR = xcals.get_calendar("XNYS")
NEW_YORK = ZoneInfo("America/New_York")
MARKET_CLOSE = time(16, 0)


def _calendar_holidays(start: date, end: date) -> set[date]:
    holidays = NYSE_CALENDAR.regular_holidays.holidays(start=start.isoformat(), end=end.isoformat())
    return {timestamp.date() for timestamp in holidays}


def _is_trading_day(value: date) -> bool:
    if value.weekday() >= 5:
        return False
    first_session = NYSE_CALENDAR.sessions[0].date()
    last_session = NYSE_CALENDAR.sessions[-1].date()
    if first_session <= value <= last_session:
        return bool(NYSE_CALENDAR.is_session(pd.Timestamp(value)))
    return value not in _calendar_holidays(value - timedelta(days=7), value + timedelta(days=7))


def _previous_trading_day(value: date) -> date:
    if _is_trading_day(value):
        return value
    candidate = value - timedelta(days=1)
    while not _is_trading_day(candidate):
        candidate -= timedelta(days=1)
    return candidate


def _next_trading_day(value: date) -> date:
    candidate = value + timedelta(days=1)
    while not _is_trading_day(candidate):
        candidate += timedelta(days=1)
    return candidate


def _effective_signal_date(now: datetime | None = None) -> date:
    now = now.astimezone(NEW_YORK) if now else datetime.now(NEW_YORK)
    session = now.date()
    if not _is_trading_day(session):
        return _next_trading_day(session - timedelta(days=1))
    if now.time() >= MARKET_CLOSE:
        return _next_trading_day(session)
    return session


def _friday_session(anchor: date | None = None) -> date:
    anchor = anchor or date.today()
    if anchor.weekday() >= 5:
        friday = anchor - timedelta(days=anchor.weekday() - 4)
    else:
        friday = anchor + timedelta(days=4 - anchor.weekday())
    while not _is_trading_day(friday):
        friday -= timedelta(days=1)
    return friday


def _next_friday(anchor: date | None = None) -> date:
    anchor = anchor or date.today()
    current_friday = _friday_session(anchor)
    if current_friday > anchor or (_is_trading_day(anchor) and anchor <= current_friday):
        return current_friday
    return _friday_session(current_friday + timedelta(days=3))


def _valid_expirations(expirations: list[str]) -> list[str]:
    return [expiration for expiration in expirations if _is_trading_day(date.fromisoformat(expiration))]


def get_signal_expiry_options() -> list[str]:
    today = _effective_signal_date()
    options: list[str] = [today.isoformat()]
    friday_date = _friday_session(today)
    if friday_date != today:
        options.append(friday_date.isoformat())
    next_friday = _friday_session(friday_date + timedelta(days=3))
    if next_friday != friday_date and next_friday != today:
        options.append(next_friday.isoformat())
    return options


def get_signal_universe(expiry: str) -> tuple[str, ...]:
    today = _effective_signal_date()
    expiry_date = date.fromisoformat(expiry)
    friday_date = _next_friday(today)
    next_friday = _next_friday(friday_date + timedelta(days=1))
    base = tuple(dict.fromkeys(TODAY_UNIVERSE + MAG7 + tuple(S_AND_P_50)))
    if expiry_date in {today, friday_date, next_friday}:
        return base
    return base


def _scan_ticker(symbol: str, expiration: str) -> dict[str, Any] | None:
    try:
        yahoo_symbol = YAHOO_SYMBOLS.get(symbol, symbol)
        ticker = yf.Ticker(yahoo_symbol)
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
def scan_optionable_universe(expiration: str, symbols: tuple[str, ...]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(_scan_ticker, symbol, expiration) for symbol in symbols]
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


def render_universe_scan(expiration: str, symbols: tuple[str, ...]) -> None:
    st.markdown(f"<div class='layer-title'>TOP 20 0DTE UNIVERSE <span>{expiration} · RANKED BY OPTION LIQUIDITY</span></div>", unsafe_allow_html=True)
    with st.spinner(f"Scanning optionable universe for {expiration}..."):
        ranked = scan_optionable_universe(expiration, symbols)
    if ranked.empty or not symbols:
        st.warning(f"Yahoo returned no option chains for {expiration}.")
        return
    display = ranked[["Ticker", "Spot", "Option volume", "Dollar volume", "Open interest", "Spread %", "Tradeability"]].copy()
    display["Dollar volume"] = display["Dollar volume"] / 1_000_000
    display.columns = ["Ticker", "Spot", "Opt vol", "$ opt vol (M)", "OI", "Spread %", "Score"]

    def format_universe(value: Any) -> str:
        return "background-color: rgba(17, 26, 36, 0.9); color: #dce5ef;"

    styled = display.style.format({"Spot": "${:.2f}", "$ opt vol (M)": "${:.1f}", "Spread %": "{:.2f}%", "Score": "{:.0f}"}).map(format_universe)
    st.dataframe(styled, width="stretch", hide_index=True, height=520)


def evaluate_actionable_signal(symbol: str, expiration: str) -> dict[str, Any] | None:
    yahoo_symbol = YAHOO_SYMBOLS.get(symbol, symbol)
    spot, expirations, error = load_market_data(yahoo_symbol)
    if error or spot is None or expiration not in expirations:
        return None
    calls, puts, error = load_chain(yahoo_symbol, expiration)
    if error:
        return None
    full_frame = build_gex_frame(calls, puts, spot, expiration)
    if full_frame.empty:
        return None
    levels = find_levels(full_frame, spot)
    history = load_price_history(yahoo_symbol, "3mo")
    if history.empty:
        return None
    volume = history["Volume"]
    avg_volume = float(volume.rolling(20).mean().iloc[-1]) if len(volume) >= 20 else 0.0
    relative_volume = float(volume.iloc[-1] / avg_volume) if avg_volume else 0.0
    atr = float((history["High"] - history["Low"]).rolling(14).mean().iloc[-1]) if len(history) >= 14 else 0.0
    total_gex = float(full_frame["Net_GEX"].sum())
    call_wall = float(levels["call_wall"])
    put_wall = float(levels["put_wall"])
    gamma_flip = float(levels["gamma_flip"])
    wall_range = max(call_wall - put_wall, 0.01)
    wall_position = (spot - put_wall) / wall_range
    above_flip = spot > gamma_flip
    breakout_up = spot > call_wall and above_flip and relative_volume >= 1.2
    breakdown_down = spot < put_wall and not above_flip and relative_volume >= 1.2
    if total_gex < 0 and breakout_up:
        setup = "LONG BREAKOUT"
        reasons = ["Negative GEX can amplify movement", "Spot is above Gamma Flip and Call Wall", f"Relative volume is {relative_volume:.1f}x normal"]
        trigger = f"Hold above ${call_wall:.2f}"
        invalidation = f"Close back below ${call_wall:.2f}"
        target = "Upper expected-move band / next resistance"
    elif total_gex < 0 and breakdown_down:
        setup = "SHORT BREAKDOWN"
        reasons = ["Negative GEX can amplify movement", "Spot is below Gamma Flip and Put Wall", f"Relative volume is {relative_volume:.1f}x normal"]
        trigger = f"Hold below ${put_wall:.2f}"
        invalidation = f"Close back above ${put_wall:.2f}"
        target = "Lower expected-move band / next support"
    elif total_gex > 0 and 0.15 <= wall_position <= 0.85:
        setup = "RANGE / FADE"
        reasons = ["Positive GEX favors mean reversion", f"Spot is {wall_position:.0%} through the wall range", "No confirmed wall breakout"]
        trigger = f"Rejection near ${put_wall:.2f} or ${call_wall:.2f}"
        invalidation = "15-minute close beyond the tested wall"
        target = "VWAP / wall-range midpoint"
    else:
        setup = "WAIT"
        reasons = ["Price location and dealer regime are not aligned", f"GEX regime is {'positive' if total_gex >= 0 else 'negative'}", f"Relative volume is {relative_volume:.1f}x normal"]
        trigger = "Wait for wall acceptance or rejection"
        invalidation = "No trade without confirmation"
        target = "Next confirmed level"
    return {"Ticker": symbol, "Setup": setup, "Spot": spot, "GEX": total_gex, "Gamma Flip": gamma_flip, "Put Wall": put_wall, "Call Wall": call_wall, "Rel Vol": relative_volume, "ATR": atr, "Why": reasons, "Trigger": trigger, "Invalidation": invalidation, "Target": target}


def render_actionable_signals(expiration: str, symbols: tuple[str, ...]) -> None:
    st.markdown(f"<div class='layer-title'>ACTIONABLE SIGNAL BOARD <span>{expiration} · EXPLICIT THESIS · NOT INVESTMENT ADVICE</span></div>", unsafe_allow_html=True)
    signals = []
    with st.spinner(f"Building directional signals for {len(symbols)} tickers..."):
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(evaluate_actionable_signal, symbol, expiration) for symbol in symbols]
            for future in as_completed(futures):
                signal = future.result()
                if signal:
                    signals.append(signal)
    if not signals:
        st.warning(f"No complete Yahoo chain was available for {expiration}.")
        return
    for signal in sorted(signals, key=lambda item: (item["Setup"] == "WAIT", -abs(item["GEX"]))):
        setup_class = "signal-red" if "SHORT" in signal["Setup"] or signal["Setup"] == "WAIT" else "signal-green" if "LONG" in signal["Setup"] else "signal-gold"
        reasons = "<br>".join(f"· {escape(reason)}" for reason in signal["Why"])
        st.markdown(f"<div class='signal-card {setup_class}'><div class='signal-top'><b>{signal['Ticker']}</b><strong>{signal['Setup']}</strong><span>${signal['Spot']:.2f} · GEX {money(signal['GEX'])}</span></div><div class='signal-grid'><div><label>WHY</label><p>{reasons}</p></div><div><label>TRIGGER</label><p>{escape(signal['Trigger'])}</p></div><div><label>INVALIDATION</label><p>{escape(signal['Invalidation'])}</p></div><div><label>TARGET</label><p>{escape(signal['Target'])}</p></div></div><div class='signal-meta'>FLIP ${signal['Gamma Flip']:.2f} · PUT WALL ${signal['Put Wall']:.2f} · CALL WALL ${signal['Call Wall']:.2f} · REL VOL {signal['Rel Vol']:.1f}x · ATR ${signal['ATR']:.2f}</div></div>", unsafe_allow_html=True)


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


def focus_strikes(
    frame: pd.DataFrame,
    spot: float,
    required_strikes: list[float] | None = None,
    strike_count: int = 41,
) -> pd.DataFrame:
    frame = frame.copy()
    strike_count = max(1, min(strike_count, len(frame)))
    if len(frame) > strike_count:
        nearest = set((frame["strike"] - spot).abs().nsmallest(strike_count).index)
        required = set()
        for required_strike in required_strikes or []:
            required.add((frame["strike"] - required_strike).abs().idxmin())
        frame = frame.loc[sorted(nearest | required)]
    return frame.sort_values("strike").reset_index(drop=True)


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


def money_or_dash(value: float) -> str:
    return "—" if abs(value) < 0.5 else money(value)


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
            f"<div class='value-cell call-value'>{money_or_dash(float(row['Call_GEX']))}</div>"
            f"<div class='net-cell'><span class='net-bar' style='width:{net_width:.1f}%;background:{net_color}'></span><span>{money_or_dash(net)}</span></div>"
            f"<div class='value-cell put-value'>{money_or_dash(float(row['Put_GEX']))}</div>"
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
    visible_min = float(frame["strike"].min())
    visible_max = float(frame["strike"].max())

    def is_visible(level: str) -> bool:
        value = float(levels[level])
        return visible_min <= value <= visible_max

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
    if is_visible("max_pain"):
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
    if is_visible("max_pain"):
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
        if is_visible(key):
            chart.add_hline(y=float(levels[key]), line_color=color, line_dash="dot", line_width=1, annotation_text=f"{key.replace('_', ' ').upper()} ${float(levels[key]):.2f}", annotation_font_color=color, annotation_position="top right")
    y_tick_step = 2 if spot < 300 else 5 if spot < 1000 else 10
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
            range=[visible_min, visible_max],
            gridcolor="#1b2735",
            dtick=y_tick_step,
        ),
        title=f"{symbol} / DEALER GAMMA BY STRIKE",
    )
    st.plotly_chart(chart, width="stretch", config={"displayModeBar": False})


def render_navigation_styles() -> None:
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root { --ink:#080d14; --panel:#111a24; --line:#243140; --muted:#8796a8; --green:#28d7a1; --red:#ff557d; --gold:#f5c84b; }
    .stApp { background:radial-gradient(circle at 20% -10%, #15243b 0, #080d14 42%); color:#dce5ef; }
    [data-testid='stAppViewContainer'], [data-testid='stHeader'] { background:transparent; }
    h1,h2,h3,p,div,button,label { font-family:'Space Grotesk',sans-serif; }
    code, .stMetricValue, [data-testid='stDataFrame'], .layer-title, .terminal-label { font-family:'DM Mono',monospace; }
    section[data-testid='stSidebar'] { background:#0b121b; border-right:1px solid #243140; }
    section[data-testid='stSidebar'] > div { background:#0b121b; }
    [data-testid='stMetric'] { background:#111a24; border:1px solid #243140; padding:14px 16px; border-radius:6px; }
    [data-testid='stMetricLabel'] { color:#8796a8; text-transform:uppercase; letter-spacing:.08em; font-size:.7rem; }
    [data-testid='stMetricValue'] { color:#f5c84b; font:600 1.25rem 'DM Mono',monospace; }
    .terminal-label { color:#8796a8; font:500 .7rem 'DM Mono',monospace; letter-spacing:.14em; text-transform:uppercase; }
    .layer-title { color:#f5c84b; border-bottom:1px solid #243140; margin:26px 0 10px; padding-bottom:8px; font:600 .8rem 'DM Mono',monospace; letter-spacing:.1em; }
    .layer-title span { color:#8796a8; font-size:.62rem; margin-left:8px; }
    .signal-card { margin:10px 0; border:1px solid #243140; border-left:4px solid #f5c84b; border-radius:6px; background:#111a24; padding:14px 16px; }
    .signal-card.signal-green { border-color:#1d765d; border-left-color:#28d7a1; background:linear-gradient(90deg,rgba(40,215,161,.13),#111a24 42%); }
    .signal-card.signal-red { border-color:#783049; border-left-color:#ff557d; background:linear-gradient(90deg,rgba(255,85,125,.13),#111a24 42%); }
    .signal-card.signal-gold { border-left-color:#f5c84b; background:linear-gradient(90deg,rgba(245,200,75,.10),#111a24 42%); }
    .signal-top { display:grid; grid-template-columns:90px 1fr auto; gap:12px; align-items:center; color:#dce5ef; font:500 .82rem 'DM Mono',monospace; }
    .signal-top b { color:#f5c84b; font-size:1rem; }
    .signal-top strong { color:#28d7a1; letter-spacing:.05em; }
    .signal-red .signal-top strong { color:#ff557d; }
    .signal-gold .signal-top strong { color:#f5c84b; }
    .signal-top span { color:#8796a8; text-align:right; }
    .signal-grid { display:grid; grid-template-columns:2fr 1.2fr 1.2fr 1.2fr; gap:14px; margin-top:13px; }
    .signal-grid label { color:#8796a8; font:500 .61rem 'DM Mono',monospace; letter-spacing:.1em; }
    .signal-grid p { color:#dce5ef; font:400 .7rem 'DM Mono',monospace; line-height:1.65; margin:6px 0 0; }
    .signal-meta { border-top:1px solid #243140; color:#8796a8; font:400 .62rem 'DM Mono',monospace; margin-top:12px; padding-top:9px; }
    .summary-strip { display:grid; grid-template-columns:repeat(8,minmax(125px,1fr)); gap:8px; margin-top:18px; overflow-x:auto; }
    .summary-card { min-height:76px; padding:10px 12px; border:1px solid #243140; border-radius:6px; background:#111a24; }
    .summary-card-label { color:#dce5ef; font:600 .64rem 'Space Grotesk',sans-serif; text-transform:uppercase; white-space:nowrap; }
    .summary-card-value { color:#28d7a1; font:600 1rem 'DM Mono',monospace; margin-top:7px; white-space:nowrap; }
    .summary-card-value.negative { color:#ff557d; }
    .summary-card-value.gold { color:#f5c84b; }
    .summary-card-note { color:#8796a8; font:400 .61rem 'DM Mono',monospace; margin-top:4px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    .layer-grid { display:grid; grid-template-columns:repeat(6,minmax(150px,1fr)); gap:8px; }
    .layer-card { min-height:82px; border:1px solid #243140; border-radius:6px; padding:11px 13px; background:#0f1822; }
    .layer-label { color:#8796a8; font:500 .66rem 'DM Mono',monospace; text-transform:uppercase; }
    .layer-value { color:#28d7a1; font:600 .95rem 'DM Mono',monospace; margin-top:9px; }
    .layer-value.gold { color:#f5c84b; }
    .layer-value.layer-negative { color:#ff557d; }
    .layer-value.layer-positive { color:#28d7a1; }
    .layer-note { color:#8796a8; font:400 .62rem 'DM Mono',monospace; margin-top:5px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    [data-testid='stDataFrame'], [data-testid='stDataFrame'] > div { background:#0d151f !important; }
    [data-testid='stDataFrame'] table, [data-testid='stDataFrame'] th, [data-testid='stDataFrame'] td { background:#0d151f !important; color:#dce5ef !important; border-color:#243140 !important; }
    [data-testid='stTabs'] button { color:#efe6a6 !important; font-family:'DM Mono',monospace !important; }
    @media (max-width:1200px) { .layer-grid { grid-template-columns:repeat(3,minmax(180px,1fr)); } }
    @media (max-width:900px) { .signal-grid { grid-template-columns:1fr 1fr; } .layer-grid { grid-template-columns:repeat(2,minmax(160px,1fr)); } }
    @media (max-width:560px) { .signal-top { grid-template-columns:1fr; } .signal-top span { text-align:left; } .signal-grid,.layer-grid { grid-template-columns:1fr; } }
    .desk-nav { border-right:1px solid #243140; }
    .desk-nav-title { color:#f5c84b; font:600 .76rem 'DM Mono',monospace; letter-spacing:.12em; margin-bottom:12px; }
    .desk-nav-status { color:#8796a8; font:400 .62rem 'DM Mono',monospace; margin:8px 0 14px; }
    [data-testid='stSidebar'] .stButton > button { border:1px solid #243140; background:#111a24; color:#dce5ef; text-align:left; font:600 .74rem 'DM Mono',monospace; }
    [data-testid='stSidebar'] .stButton > button:hover { border-color:#f5c84b; color:#f5c84b; }
    [data-testid='stSidebar'] .nav-active .stButton > button { border:1px solid #f5c84b; color:#f5c84b; background:rgba(245,200,75,.12); box-shadow:inset 3px 0 #f5c84b; }
    .mockup-shell { border:1px solid #243140; background:#0d151f; padding:14px; margin:12px 0; }
    .mockup-rail { border-right:1px solid #243140; padding:10px; color:#8796a8; font:500 .7rem 'DM Mono',monospace; }
    .mockup-rail strong { display:block; color:#f5c84b; border:1px solid #f5c84b; padding:8px; margin-bottom:8px; }
    .mockup-content { min-height:130px; padding:10px; background:#111a24; color:#dce5ef; font:500 .72rem 'DM Mono',monospace; }
    .mockup-card { display:inline-block; min-width:110px; margin:4px; padding:10px; border:1px solid #243140; color:#f5c84b; background:#162231; }
    .premium-scope { margin:10px 0 8px; padding:9px 12px; border:1px solid #243140; background:#111a24; color:#dce5ef; font:500 .68rem 'DM Mono',monospace; letter-spacing:.04em; }
    .premium-scope b { color:#f5c84b; }
    .premium-metric-grid { display:grid; grid-template-columns:repeat(4,minmax(170px,1fr)); gap:8px; }
    .premium-metric-card { min-height:76px; padding:11px 13px; border:1px solid #243140; border-left:3px solid #f5c84b; border-radius:6px; background:#111a24; }
    .premium-metric-card > div { color:#dce5ef; font:600 .63rem 'DM Mono',monospace; }
    .premium-metric-card strong { display:block; color:#f5c84b; font:600 1rem 'DM Mono',monospace; margin-top:7px; }
    .premium-metric-card span { display:block; color:#8796a8; font:400 .6rem 'DM Mono',monospace; margin-top:4px; }
    .premium-metric-grid.bullish .premium-metric-card { border-left-color:#28d7a1; }
    .premium-metric-grid.bullish .premium-metric-card strong { color:#28d7a1; }
    .premium-metric-grid.bearish .premium-metric-card { border-left-color:#ff557d; }
    .premium-metric-grid.bearish .premium-metric-card strong { color:#ff557d; }
    .premium-metric-grid.neutral .premium-metric-card { border-left-color:#f5c84b; }
    @media (max-width:900px) { .premium-metric-grid { grid-template-columns:repeat(2,minmax(170px,1fr)); } }
    @media (max-width:560px) { .premium-metric-grid { grid-template-columns:1fr; } }
    </style>
    """, unsafe_allow_html=True)


def render_navigation() -> tuple[str, dict[str, Any]]:
    if "active_view" not in st.session_state:
        st.session_state.active_view = "gamma"

    def select_view(view: str) -> None:
        st.session_state.active_view = view

    st.sidebar.markdown("<div class='desk-nav-title'>TRADING DESK / RESEARCH CONSOLE</div>", unsafe_allow_html=True)
    st.sidebar.markdown("<div class='desk-nav-status'>ACTIVE VIEW · {} </div>".format(st.session_state.active_view.upper()), unsafe_allow_html=True)
    labels = (("gamma", "GAMMA SURFACE"), ("signals", "SIGNALS"), ("dvzr", "DVZR SCANNER"), ("premium", "PREMIUM SENTIMENT"))
    for view, label in labels:
        active_class = "nav-active" if st.session_state.active_view == view else ""
        st.sidebar.markdown(f"<div class='{active_class}'>", unsafe_allow_html=True)
        st.sidebar.button(label, key=f"nav_{view}", on_click=select_view, args=(view,), use_container_width=True)
        st.sidebar.markdown("</div>", unsafe_allow_html=True)

    controls: dict[str, Any] = {}
    active_view = st.session_state.active_view
    if active_view == "gamma":
        st.sidebar.markdown("---")
        symbol = st.sidebar.text_input("Symbol", "SPY", max_chars=8, key="desk_symbol").strip().upper()
        _, expirations, market_error = load_market_data(symbol)
        if market_error or not expirations:
            st.sidebar.warning(market_error or "No expirations available.")
            controls.update(symbol=symbol, expiration=None, strike_count=41, refresh=False)
        else:
            today_expiration = _effective_signal_date().isoformat()
            default_expiration = expirations.index(today_expiration) if today_expiration in expirations else 0
            expiration = st.sidebar.selectbox("Expiration", expirations, index=default_expiration, key="desk_expiration", format_func=lambda value: f"{value} · {(date.fromisoformat(value) - date.today()).days}D")
            controls["strike_count"] = st.sidebar.slider("Visible strike range", 10, 101, 41, key="desk_strike_count")
            controls.update(symbol=symbol, expiration=expiration, refresh=st.sidebar.button("Refresh data", key="desk_refresh", use_container_width=True))
    elif active_view == "signals":
        st.sidebar.markdown("---")
        expiry_options = get_signal_expiry_options()
        signal_date = _effective_signal_date().isoformat()
        controls["expiration"] = st.sidebar.radio("Signal expiry", expiry_options, format_func=lambda value: f"{value} · {'TODAY' if value == signal_date else 'FRIDAY'}", key="desk_signal_expiry")
    elif active_view == "dvzr":
        st.sidebar.markdown("---")
        controls["filter_symbols"] = st.sidebar.text_input("Symbol filter (optional)", key="desk_dvzr_filter")
        controls["run_scan"] = st.sidebar.button("Run DVZR scanner", type="primary", key="desk_dvzr_run", use_container_width=True)
        controls["show_history"] = st.sidebar.checkbox("Show last-week picks", key="desk_dvzr_history")
        controls["validation_symbol"] = st.sidebar.selectbox("Validation symbol", sorted(set(["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "JPM", "PG", "SPY", "QQQ", "IWM", "XLF", "XLK"])), key="desk_validation_symbol")
        controls["run_validation"] = st.sidebar.button("Run historical validation", key="desk_validation_run", use_container_width=True)
    elif active_view == "premium":
        st.sidebar.markdown("---")
        _, expirations, error = load_market_data("SPY")
        if error or not expirations:
            controls.update(expiration=None, symbols=(), run_scan=False)
            st.sidebar.warning(error or "No benchmark expirations available.")
        else:
            today = _effective_signal_date().isoformat()
            default_expiration = expirations.index(today) if today in expirations else 0
            controls["expiration"] = st.sidebar.selectbox("Analysis expiry", expirations[:20], index=min(default_expiration, 19), key="desk_premium_expiry")
            text = st.sidebar.text_input("Symbols", value=", ".join(PREMIUM_SENTIMENT_UNIVERSE), key="desk_premium_symbols")
            controls["symbols"] = tuple(dict.fromkeys(symbol.strip().upper() for symbol in text.split(",") if symbol.strip()))
            controls["run_scan"] = st.sidebar.button("Run premium sentiment", type="primary", key="desk_premium_run", use_container_width=True)

    controls["show_mockups"] = st.sidebar.checkbox("Show theme mockups", key="desk_show_mockups")
    return active_view, controls


def render_theme_mockups() -> None:
    st.markdown("<div class='layer-title'>THEME MOCKUPS <span>TRADING DESK · RESEARCH WORKBENCH · TERMINAL RAIL</span></div>", unsafe_allow_html=True)
    themes = (
        ("Trading Desk", "GAMMA SURFACE", "Bold metrics, compact rail, immediate action states."),
        ("Research Workbench", "DVZR SCANNER", "Wider controls, evidence-first tables, slower research rhythm."),
        ("Terminal Rail", "PREMIUM SENTIMENT", "Dense monospace navigation with minimal visual noise."),
    )
    for name, active, description in themes:
        st.markdown(f"<div class='mockup-shell'><b>{name.upper()}</b><br><span class='layer-note'>{description}</span><div style='display:grid;grid-template-columns:180px 1fr;margin-top:10px'><div class='mockup-rail'><strong>{active}</strong> SIGNALS<br><br>DVZR<br><br>PREMIUM SENTIMENT</div><div class='mockup-content'><div class='mockup-card'>NET GEX<br><b>+$1.2M</b></div><div class='mockup-card'>SIGNAL<br><b>RANGE / FADE</b></div><div class='mockup-card'>EVIDENCE<br><b>3D HIT 64%</b></div></div></div></div>", unsafe_allow_html=True)


def render_terminal(controls: dict[str, Any] | None = None) -> None:
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
    .layer-title { color:#f5c84b; border-bottom:1px solid #243140; margin:26px 0 10px; padding-bottom:8px; font:600 .8rem 'DM Mono',monospace; letter-spacing:.1em; }
    .layer-title span { color:#8796a8; font-size:.62rem; margin-left:8px; }
    .layer-grid { display:grid; grid-template-columns:repeat(6, minmax(150px, 1fr)); gap:8px; }
    .layer-card { min-height:82px; border:1px solid #243140; border-radius:6px; padding:11px 13px; background:#0f1822; }
    .layer-label { color:#8796a8; font:500 .66rem 'DM Mono',monospace; text-transform:uppercase; }
    .layer-label span { color:#aa7cff; font-size:.56rem; }
    .layer-value { color:#28d7a1; font:600 .95rem 'DM Mono',monospace; margin-top:9px; }
    .layer-value.gold { color:#f5c84b; }
    .layer-value.layer-negative { color:#ff557d; }
    .layer-value.layer-positive { color:#28d7a1; }
    .layer-note { color:#8796a8; font:400 .62rem 'DM Mono',monospace; margin-top:5px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    .signal-card { margin:10px 0; border:1px solid #243140; border-left:4px solid #f5c84b; border-radius:6px; background:#111a24; padding:14px 16px; }
    .signal-card.signal-green { border-color:#1d765d; border-left-color:#28d7a1; background:linear-gradient(90deg, rgba(40,215,161,.13), #111a24 42%); }
    .signal-card.signal-red { border-color:#783049; border-left-color:#ff557d; background:linear-gradient(90deg, rgba(255,85,125,.13), #111a24 42%); }
    .signal-card.signal-gold { border-left-color:#f5c84b; background:linear-gradient(90deg, rgba(245,200,75,.10), #111a24 42%); }
    .signal-top { display:grid; grid-template-columns:90px 1fr auto; gap:12px; align-items:center; color:#dce5ef; font:500 .82rem 'DM Mono',monospace; }
    .signal-top b { color:#f5c84b; font-size:1rem; }
    .signal-top strong { color:#28d7a1; letter-spacing:.05em; }
    .signal-red .signal-top strong { color:#ff557d; }
    .signal-gold .signal-top strong { color:#f5c84b; }
    .signal-top span { color:#8796a8; text-align:right; }
    .signal-grid { display:grid; grid-template-columns:2fr 1.2fr 1.2fr 1.2fr; gap:14px; margin-top:13px; }
    .signal-grid label { color:#8796a8; font:500 .61rem 'DM Mono',monospace; letter-spacing:.1em; }
    .signal-grid p { color:#dce5ef; font:400 .7rem 'DM Mono',monospace; line-height:1.65; margin:6px 0 0; }
    .signal-meta { border-top:1px solid #243140; color:#8796a8; font:400 .62rem 'DM Mono',monospace; margin-top:12px; padding-top:9px; }
    .signal-date-label { color:#f5c84b; font:500 .72rem 'DM Mono',monospace; letter-spacing:.08em; }
    [data-testid='stRadio'] label { color:#f5c84b !important; font:500 .72rem 'DM Mono',monospace !important; }
    [data-testid='stRadio'] label p { color:#f5c84b !important; }
    [data-testid='stDataFrame'] { background:#0d151f !important; }
    [data-testid='stDataFrame'] > div { background:#0d151f !important; }
    [data-testid='stDataFrame'] table, [data-testid='stDataFrame'] th, [data-testid='stDataFrame'] td { background:#0d151f !important; color:#dce5ef !important; border-color:#243140 !important; }
    [data-testid='stTabs'] button { color:#efe6a6 !important; font-family:'DM Mono',monospace !important; }
    [data-testid='stTabs'] button[aria-selected='true'] { color:#f9efb8 !important; background:rgba(249, 239, 184, 0.12) !important; }
    @media (max-width: 1200px) { .layer-grid { grid-template-columns:repeat(3, minmax(180px, 1fr)); } }
    @media (max-width: 900px) { .layer-grid { grid-template-columns:repeat(2, minmax(160px, 1fr)); } }
    @media (max-width: 560px) { .layer-grid { grid-template-columns:1fr; } }
    </style>
    """, unsafe_allow_html=True)
    controls = controls or {}
    symbol = controls.get("symbol", "SPY")
    expiration = controls.get("expiration")
    if controls.get("refresh"):
        load_market_data.clear()
        load_chain.clear()

    st.markdown("<div class='terminal-label'>OPTIONS POSITIONING TERMINAL · LIVE SNAPSHOT</div>", unsafe_allow_html=True)
    st.title(f"{symbol} / Dealer Gamma Surface")
    spot, expirations, market_error = load_market_data(symbol)
    if market_error:
        st.error(market_error)
        return
    if not expiration:
        expiration = expirations[0]
    calls, puts, chain_error = load_chain(symbol, expiration)
    if chain_error or spot is None:
        st.error(chain_error or "No spot price available.")
        return
    full_frame = build_gex_frame(calls, puts, spot, expiration)
    levels = find_levels(full_frame, spot)
    strike_count = min(int(controls.get("strike_count", 41)), len(full_frame))
    frame = focus_strikes(full_frame, spot, [float(levels["max_pain"])], strike_count)
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
    st.caption(f"Visible range: {len(frame)} strikes around spot · Full list: {len(full_frame)} strikes below")
    full_display = full_frame.rename(columns={
        "strike": "Strike",
        "Call_GEX": "Call GEX",
        "Put_GEX": "Put GEX",
        "Net_GEX": "Net GEX",
        "Call_OI": "Call OI",
        "Put_OI": "Put OI",
    })[["Strike", "Call GEX", "Net GEX", "Put GEX", "Call OI", "Put OI"]]
    full_display_style = full_display.style.set_properties(**{
        "background-color": "#0d151f",
        "color": "#dce5ef",
        "border-color": "#243140",
    }).set_table_styles([
        {"selector": "th", "props": [("background-color", "#162231"), ("color", "#f5c84b"), ("border-color", "#243140")]},
        {"selector": "td", "props": [("background-color", "#0d151f"), ("color", "#dce5ef"), ("border-color", "#243140")]},
    ]).format({
        "Strike": "{:.2f}",
        "Call GEX": money,
        "Net GEX": money,
        "Put GEX": money,
        "Call OI": "{:.0f}",
        "Put OI": "{:.0f}",
    })
    st.dataframe(
        full_display_style,
        width="stretch",
        hide_index=True,
        height=420,
    )
    total_class = "" if total >= 0 else " negative"
    grower_class = "" if grower >= 0 else " negative"
    st.markdown(
        f"<div class='summary-strip'>"
        f"<div class='summary-card'><div class='summary-card-label'>Net GEX · {expiration}</div><div class='summary-card-value{total_class}'>{money(total)}</div><div class='summary-card-note'>{levels['regime']}</div></div>"
        f"<div class='summary-card'><div class='summary-card-label'>Call Wall · {expiration}</div><div class='summary-card-value'>${float(levels['call_wall']):.2f}</div><div class='summary-card-note'>{float(levels['call_wall']) - spot:+.2f} from spot</div></div>"
        f"<div class='summary-card'><div class='summary-card-label'>Put Wall · {expiration}</div><div class='summary-card-value'>${float(levels['put_wall']):.2f}</div><div class='summary-card-note'>{float(levels['put_wall']) - spot:+.2f} from spot</div></div>"
        f"<div class='summary-card'><div class='summary-card-label'>Volt · {expiration}</div><div class='summary-card-value gold'>${volt_strike:.0f}</div><div class='summary-card-note'>nearest listed strike</div></div>"
        f"<div class='summary-card'><div class='summary-card-label'>Gamma Flip · {expiration}</div><div class='summary-card-value'>${float(levels['gamma_flip']):.2f}</div><div class='summary-card-note'>{float(levels['gamma_flip']) - spot:+.2f} from spot</div></div>"
        f"<div class='summary-card'><div class='summary-card-label'>Max Pain · {expiration}</div><div class='summary-card-value gold'>${float(levels['max_pain']):.2f}</div><div class='summary-card-note'>{float(levels['max_pain']) - spot:+.2f} from spot</div></div>"
        f"<div class='summary-card'><div class='summary-card-label'>Δ Grower · all dates</div><div class='summary-card-value{grower_class}'>${spot:.2f} {grower:+.1f}</div><div class='summary-card-note'>derived from daily open</div></div>"
        f"<div class='summary-card'><div class='summary-card-label'>± Move · {expiration}</div><div class='summary-card-value gold'>±${implied_move:.2f}</div><div class='summary-card-note'>ATM IV implied range</div></div>"
        f"<div class='summary-card'><div class='summary-card-label'>ATM IV · {expiration}</div><div class='summary-card-value gold'>{atm_iv * 100:.1f}%</div><div class='summary-card-note'>call/put midpoint</div></div>"
        "</div>",
        unsafe_allow_html=True,
    )
    render_institutional_tab(symbol, expiration)
def render_institutional_tab(symbol: str, expiration: str) -> None:
    st.markdown("<div class='layer-title'>INSTITUTIONAL LAYERS <span>YAHOO-DERIVED SAMPLE · NOT INVESTMENT ADVICE</span></div>", unsafe_allow_html=True)
    spot, _, error = load_market_data(symbol)
    if error or spot is None:
        st.warning(error or "No price data available.")
        return
    calls, puts, error = load_chain(symbol, expiration)
    if error:
        st.warning(error)
        return
    try:
        layers = calculate_institutional_layers(calls, puts, spot, expiration, load_price_history(symbol))
        render_institutional_layers(layers, spot)
    except Exception as exc:
        st.warning(f"Institutional layers unavailable: {exc}")


DB_FILE = "dvzr_signals.db"


def init_dvzr_db() -> None:
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS dvzr_signals (
            snapshot_date TEXT,
            symbol TEXT,
            asset_class TEXT,
            close_price REAL,
            z_score REAL,
            signal TEXT,
            days_in_state INTEGER,
            sustainability_score REAL,
            signal_score REAL,
            volatility_ratio REAL,
            event_risk TEXT,
            atr_14 REAL,
            PRIMARY KEY (snapshot_date, symbol)
        )
        """
    )
    existing_columns = {row[1] for row in cursor.execute("PRAGMA table_info(dvzr_signals)")}
    for column_name, column_type in (("signal_score", "REAL"), ("volatility_ratio", "REAL")):
        if column_name not in existing_columns:
            cursor.execute(f"ALTER TABLE dvzr_signals ADD COLUMN {column_name} {column_type}")
    conn.commit()
    conn.close()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_market_data(symbol: str, period: str = "1y") -> pd.DataFrame:
    try:
        data = yf.download(symbol, period=period, progress=False, auto_adjust=False)
        if isinstance(data.columns, pd.MultiIndex):
            data = data.xs(symbol, level=1, axis=1)
        if not data.empty and len(data) >= 200:
            return data
    except Exception:
        return pd.DataFrame()
    return pd.DataFrame()


def _event_risk_flags(df: pd.DataFrame) -> str:
    if df.empty:
        return "none"
    close = pd.to_numeric(df["Close"], errors="coerce")
    volume = pd.to_numeric(df["Volume"], errors="coerce").fillna(0)
    recent_gap = abs(close.pct_change().fillna(0.0)).iloc[-1]
    volume_ratio = float(volume.iloc[-1] / volume.rolling(20).mean().iloc[-1]) if volume.rolling(20).mean().iloc[-1] else 0.0
    flags: list[str] = []
    if recent_gap > 0.06:
        flags.append("gap move")
    if volume_ratio > 1.8:
        flags.append("volume spike")
    if (close.iloc[-1] - close.rolling(20).mean().iloc[-1]) / close.rolling(20).std().iloc[-1] if len(close) >= 20 and pd.notna(close.rolling(20).std().iloc[-1]) and close.rolling(20).std().iloc[-1] != 0 else 0.0 > 2.5:
        flags.append("trend break")
    return "; ".join(flags) if flags else "none"


def score_market_state(df: pd.DataFrame) -> dict[str, Any]:
    close = pd.to_numeric(df["Close"], errors="coerce").dropna()
    if close.empty or len(close) < 200:
        return {
            "signal": "NEUTRAL",
            "days_in_state": 0,
            "z_score": 0.0,
            "sustainability_score": 0.0,
            "signal_score": 0.0,
            "volatility_ratio": 0.0,
            "event_risk": "insufficient data",
            "atr_14": 0.0,
        }

    sma_200 = close.rolling(200).mean()
    sma_20 = close.rolling(20).mean()
    std_20 = close.rolling(20).std().replace(0, np.nan)
    standard_z = ((close - sma_20) / std_20).replace([np.inf, -np.inf], np.nan)
    rolling_median = close.rolling(20).median()
    median_deviation = (close - rolling_median).abs().rolling(20).median().replace(0, np.nan)
    robust_z = (0.6745 * (close - rolling_median) / median_deviation).replace([np.inf, -np.inf], np.nan)
    z_score = robust_z.fillna(standard_z)

    high_low = pd.to_numeric(df["High"], errors="coerce") - pd.to_numeric(df["Low"], errors="coerce")
    prev_close = close.shift(1)
    tr = pd.concat([high_low, (pd.to_numeric(df["High"], errors="coerce") - prev_close).abs(), (pd.to_numeric(df["Low"], errors="coerce") - prev_close).abs()], axis=1).max(axis=1)
    atr_14 = tr.rolling(14).mean()

    current_close = float(close.iloc[-1])
    current_z = float(z_score.iloc[-1])
    current_sma200 = float(sma_200.iloc[-1])
    current_atr = float(atr_14.iloc[-1])
    sma20_slope = float(sma_20.iloc[-1] - sma_20.iloc[-6]) if len(sma_20.dropna()) >= 6 else 0.0
    trend_up = current_close > current_sma200 and sma20_slope > 0
    trend_down = current_close < current_sma200 and sma20_slope < 0

    if trend_up and current_z <= -1.75:
        signal = "BUY (Oversold Dip)"
        state_mask = (z_score <= -1.75) & (close > sma_200)
    elif trend_down and current_z >= 1.75:
        signal = "SELL / SHORT (Overbought)"
        state_mask = (z_score >= 1.75) & (close < sma_200)
    else:
        signal = "NEUTRAL"
        state_mask = pd.Series(False, index=z_score.index)

    days_in_state = 0
    for val in reversed(state_mask.values.tolist()):
        if val:
            days_in_state += 1
        else:
            break

    atr_10 = tr.rolling(10).mean().iloc[-1]
    atr_100 = tr.rolling(100).mean().iloc[-1]
    vol_ratio = atr_10 / atr_100 if atr_100 > 0 else 1.0
    event_risk = _event_risk_flags(df)
    z_strength = min(abs(current_z) / 3.0, 1.0) * 45
    trend_score = 25 if (signal != "NEUTRAL" and ((signal.startswith("BUY") and trend_up) or (signal.startswith("SELL") and trend_down))) else 0
    persistence_score = min(days_in_state, 5) / 5 * 15
    volatility_score = 15 if 0.75 <= vol_ratio <= 1.35 else 5 if vol_ratio <= 1.6 else 0
    event_penalty = 25 if event_risk != "none" else 0
    signal_score = float(np.clip(z_strength + trend_score + persistence_score + volatility_score - event_penalty, 0, 100))
    sustainability_score = float(np.clip(100 - (vol_ratio * 35), 10, 95))

    if event_risk != "none" and signal != "NEUTRAL":
        signal = "WAIT (EVENT RISK)"

    return {
        "signal": signal,
        "days_in_state": days_in_state,
        "z_score": round(current_z, 2),
        "sustainability_score": round(sustainability_score, 1),
        "signal_score": round(signal_score, 1),
        "volatility_ratio": round(float(vol_ratio), 2),
        "event_risk": event_risk,
        "atr_14": round(current_atr, 2),
    }


def calculate_forward_return_validation(
    df: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 3, 5, 10),
) -> pd.DataFrame:
    """Evaluate historical signals using only prices known at each signal date."""
    if df.empty or "Close" not in df or len(df) < 210:
        return pd.DataFrame()

    history = df.reset_index(drop=True).copy()
    close = pd.to_numeric(history["Close"], errors="coerce")
    observations: list[dict[str, Any]] = []
    first_signal = 200
    last_signal = len(history) - max(horizons) - 1
    for signal_index in range(first_signal, last_signal + 1):
        state = score_market_state(history.iloc[: signal_index + 1])
        signal = state["signal"]
        if signal == "NEUTRAL" or signal.startswith("WAIT"):
            continue
        direction = 1 if signal.startswith("BUY") else -1
        score = float(state["signal_score"])
        bucket = "0-39" if score < 40 else "40-59" if score < 60 else "60-79" if score < 80 else "80-100"
        row: dict[str, Any] = {
            "Signal date": history.index[signal_index],
            "Signal": "BUY" if direction == 1 else "SELL",
            "Score bucket": bucket,
            "Signal score": score,
        }
        for horizon in horizons:
            raw_return = float(close.iloc[signal_index + horizon] / close.iloc[signal_index] - 1)
            row[f"{horizon}D return"] = raw_return
            row[f"{horizon}D directional return"] = raw_return * direction
        observations.append(row)

    if not observations:
        return pd.DataFrame()
    return pd.DataFrame(observations)


def summarize_forward_return_validation(observations: pd.DataFrame) -> pd.DataFrame:
    if observations.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    horizons = (1, 3, 5, 10)
    for bucket, group in observations.groupby("Score bucket", sort=False):
        row: dict[str, Any] = {
            "Score bucket": bucket,
            "Signals": len(group),
        }
        for horizon in horizons:
            directional = group[f"{horizon}D directional return"]
            row[f"{horizon}D avg"] = float(directional.mean())
            row[f"{horizon}D median"] = float(directional.median())
            row[f"{horizon}D hit rate"] = float((directional > 0).mean())
        rows.append(row)
    return pd.DataFrame(rows)


@st.cache_data(ttl=900, show_spinner=False)
def get_forward_return_validation(symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    observations = calculate_forward_return_validation(fetch_market_data(symbol, period="2y"))
    return observations, summarize_forward_return_validation(observations)


def analyze_market_state(symbol: str, asset_class: str) -> dict[str, Any] | None:
    symbol = symbol.strip().upper()
    if not symbol:
        return None
    df = fetch_market_data(symbol)
    if df.empty:
        return None
    state = score_market_state(df)
    if not state:
        return None
    return {
        "Symbol": symbol,
        "Asset Class": asset_class,
        "Close Price": round(float(pd.to_numeric(df["Close"], errors="coerce").dropna().iloc[-1]), 2),
        "Z-Score": state["z_score"],
        "Signal": state["signal"],
        "Days in State": state["days_in_state"],
        "Sustainability Score": f"{state['sustainability_score']}%",
        "Signal Score": state["signal_score"],
        "Volatility Ratio": state["volatility_ratio"],
        "Event Risk": state["event_risk"],
        "ATR(14)": state["atr_14"],
    }


def save_signals_to_db(df_results: pd.DataFrame) -> None:
    conn = sqlite3.connect(DB_FILE)
    for _, row in df_results.iterrows():
        sust_val = float(str(row["Sustainability Score"]).replace("%", "")) if isinstance(row["Sustainability Score"], str) else float(row["Sustainability Score"])
        conn.execute(
            """
            INSERT OR REPLACE INTO dvzr_signals (
                snapshot_date, symbol, asset_class, close_price, z_score, signal, days_in_state,
                sustainability_score, signal_score, volatility_ratio, event_risk, atr_14
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                date.today().isoformat(),
                row["Symbol"],
                row["Asset Class"],
                float(row["Close Price"]),
                float(row["Z-Score"]),
                row["Signal"],
                int(row["Days in State"]),
                sust_val,
                float(row.get("Signal Score", 0.0)),
                float(row.get("Volatility Ratio", 0.0)),
                row.get("Event Risk", "none"),
                float(row.get("ATR(14)", 0.0)),
            ),
        )
    conn.commit()
    conn.close()


@st.cache_data(ttl=600, show_spinner=False)
def get_recent_picks() -> pd.DataFrame:
    conn = sqlite3.connect(DB_FILE)
    query = """
        SELECT *
        FROM dvzr_signals
        WHERE snapshot_date >= date('now', '-7 days')
        ORDER BY snapshot_date DESC, symbol ASC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


@st.cache_data(ttl=600, show_spinner=False)
def get_last_20_day_history(symbol: str) -> pd.DataFrame:
    history = fetch_market_data(symbol, period="6mo")
    if history.empty:
        return pd.DataFrame()
    window = history.tail(20).copy()
    if "Close" not in window.columns:
        return pd.DataFrame()
    window["sma_20"] = window["Close"].rolling(20).mean()
    window["z_score"] = ((window["Close"] - window["sma_20"]) / window["Close"].rolling(20).std().replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    return window


def render_dvzr_dashboard(controls: dict[str, Any] | None = None) -> None:
    controls = controls or {}
    init_dvzr_db()
    st.markdown("<div class='terminal-label'>DVZR / DYNAMIC VOLATILITY Z-SCORE REVERSION</div>", unsafe_allow_html=True)
    st.title("Mean Reversion Watchlist")
    st.caption("Yahoo Finance scan · z-score breaches · event-risk filters · SQLite cache")

    default_stocks = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "JPM", "PG"]
    default_etfs = ["SPY", "QQQ", "IWM", "XLF", "XLK"]
    default_futures = ["ES=F", "NQ=F", "CL=F", "GC=F"]
    watchlist_default = ", ".join(default_stocks + default_etfs + default_futures)
    filter_symbols = controls.get("filter_symbols", "")
    run_scan = bool(controls.get("run_scan"))
    show_history = bool(controls.get("show_history"))

    if run_scan:
        selected_symbols = []
        raw_filter = filter_symbols.strip()
        if raw_filter:
            selected_symbols = [x.strip().upper() for x in raw_filter.split(",") if x.strip()]
        else:
            selected_symbols = [s.strip().upper() for s in watchlist_default.split(",") if s.strip()]

        rows: list[dict[str, Any]] = []
        with st.spinner("Scanning for mean-reversion setups and event-risk filters..."):
            for item in selected_symbols:
                if not item:
                    continue
                asset = "Stock" if item not in {"SPY", "QQQ", "IWM", "XLF", "XLK", "ES=F", "NQ=F", "CL=F", "GC=F"} else ("ETF" if "=" not in item else "Futures")
                result = analyze_market_state(item, asset)
                if result:
                    rows.append(result)

        if rows:
            df = pd.DataFrame(rows)
            df = df.sort_values(["Signal"], key=lambda s: s.str.contains("BUY").map({True: 0, False: 1}), ascending=True)
            save_signals_to_db(df)
            st.markdown("<div class='layer-title'>DVZR SCAN RESULTS <span>QUALIFYING SIGNALS & EVENT FILTERS</span></div>", unsafe_allow_html=True)
            buy_count = int((df["Signal"].str.contains("BUY", na=False)).sum())
            wait_count = int((df["Signal"].str.contains("WAIT", na=False)).sum())
            st.metric("Signals scanned", len(df))
            col1, col2, col3 = st.columns(3)
            col1.metric("Qualifying buys", buy_count)
            col2.metric("Event-risk holds", wait_count)
            col3.metric("Neutral", int((df["Signal"] == "NEUTRAL").sum()))
            display = df[["Symbol", "Asset Class", "Close Price", "Z-Score", "Signal", "Signal Score", "Days in State", "Sustainability Score", "Volatility Ratio", "Event Risk", "ATR(14)"]].copy()

            def dvzr_style(value: Any) -> str:
                if "BUY" in str(value):
                    return "background-color: rgba(40, 215, 161, 0.18); color: #dffaf0;"
                if "SELL" in str(value):
                    return "background-color: rgba(255, 85, 125, 0.18); color: #ffd7e2;"
                if "WAIT" in str(value):
                    return "background-color: rgba(245, 200, 75, 0.12); color: #f7e9a7;"
                return "background-color: rgba(17, 26, 36, 0.9); color: #dce5ef;"

            styled_display = display.style.map(dvzr_style, subset=["Signal"]).set_properties(**{"background-color": "rgba(17, 26, 36, 0.9)", "color": "#dce5ef", "border": "1px solid #243140"})
            st.dataframe(styled_display, use_container_width=True, hide_index=True, height=420)
        else:
            st.warning("No valid market data was returned by Yahoo Finance for the selected filter.")

    if show_history:
        st.markdown("<div class='layer-title'>LAST-WEEK DVZR PICKS <span>LAST 7 DAYS OF SIGNAL HISTORY</span></div>", unsafe_allow_html=True)
        history = get_recent_picks()
        if history.empty:
            st.info("No recent DVZR records are saved yet. Run the scanner first to create history.")
        else:
            picks = history[["snapshot_date", "symbol", "asset_class", "close_price", "z_score", "signal", "signal_score", "days_in_state", "volatility_ratio", "event_risk", "sustainability_score"]].copy()
            picks.columns = ["Snapshot", "Symbol", "Asset class", "Close", "Z-Score", "Signal", "Signal score", "Days in state", "Vol ratio", "Event risk", "Sust. score"]
            st.dataframe(picks, use_container_width=True, hide_index=True)

            selected = st.selectbox("Review 20-day history for", sorted(picks["Symbol"].unique().tolist()), index=0)
            hist = get_last_20_day_history(selected)
            if hist.empty:
                st.warning(f"Not enough price history for {selected} to build a 20-day rolling view.")
            else:
                st.line_chart(hist[["Close", "sma_20"]])
                st.dataframe(hist.tail(20)[["Close", "sma_20", "z_score"]].reset_index().rename(columns={"index": "Date"}), use_container_width=True, hide_index=True)

    st.markdown("<div class='layer-title'>FORWARD-RETURN VALIDATION <span>POINT-IN-TIME · DIRECTIONAL HIT RATE</span></div>", unsafe_allow_html=True)
    validation_symbol = controls.get("validation_symbol", sorted(set(default_stocks + default_etfs))[0])
    run_validation = bool(controls.get("run_validation"))
    if run_validation:
        with st.spinner(f"Evaluating historical DVZR signals for {validation_symbol}..."):
            observations, validation = get_forward_return_validation(validation_symbol)
        if validation.empty:
            st.info("No qualifying historical signals were found with enough forward data.")
        else:
            st.caption("Returns are direction-adjusted: positive means the signal direction was correct. Event-risk and neutral states are excluded.")
            percentage_columns = [column for column in validation.columns if column.endswith(("avg", "median", "hit rate"))]
            st.dataframe(
                validation.style.format({column: "{:.1%}" for column in percentage_columns}),
                use_container_width=True,
                hide_index=True,
            )
            detail_columns = ["Signal date", "Signal", "Score bucket", "Signal score", "1D return", "3D return", "5D return", "10D return"]
            st.dataframe(
                observations[detail_columns].tail(100).style.format({column: "{:.2%}" for column in detail_columns[4:]}),
                use_container_width=True,
                hide_index=True,
                height=280,
            )


def render_premium_sentiment(controls: dict[str, Any] | None = None) -> None:
    controls = controls or {}
    st.markdown("<div class='terminal-label'>PREMIUM & DIRECTIONAL SENTIMENT BREAKDOWN</div>", unsafe_allow_html=True)
    st.title("Premium & Directional Sentiment")
    st.caption("Estimated traded option premium · volume × bid/ask midpoint × 100 shares")

    expiration = controls.get("expiration")
    symbols = controls.get("symbols", ())
    run_scan = bool(controls.get("run_scan"))
    if not expiration:
        st.warning("No benchmark option expiration is available from the left pane.")
        return
    if not run_scan:
        st.info("Set the filters and run the scan from the left pane.")
        return
    if not symbols:
        st.warning("Enter at least one symbol.")
        return

    with st.spinner(f"Reading option premium for {len(symbols)} symbols..."):
        sentiment = scan_premium_sentiment(expiration, symbols)
    if sentiment.empty:
        st.warning("No symbols returned usable volume and bid/ask data for this expiry.")
        return

    st.warning(
        "This is a current-chain snapshot, not historical last-week premium. Yahoo Finance does not expose historical option trade premium, so do not compare it directly with the supplied historical example."
    )
    total_call = float(sentiment["Call Premium"].sum())
    total_put = float(sentiment["Put Premium"].sum())
    total = total_call + total_put
    scanned_tickers = ", ".join(sentiment["Ticker"].tolist())
    aggregate_bias = "bullish" if total_call > total_put else "bearish" if total_put > total_call else "neutral"
    st.markdown(
        f"<div class='premium-scope'><b>AGGREGATION SCOPE</b> · {escape(scanned_tickers)} · {expiration} · {len(sentiment)} TICKERS CONSIDERED</div>",
        unsafe_allow_html=True,
    )
    put_call_ratio = total_put / total_call if total_call else 0.0
    st.markdown(
        f"<div class='premium-metric-grid {aggregate_bias}'>"
        f"<div class='premium-metric-card'><div>ALL TICKERS · TOTAL PREMIUM</div><strong>{money(total)}</strong><span>calls + puts</span></div>"
        f"<div class='premium-metric-card'><div>ALL TICKERS · CALL PREMIUM</div><strong>{money(total_call)}</strong><span>call-side proxy</span></div>"
        f"<div class='premium-metric-card'><div>ALL TICKERS · PUT PREMIUM</div><strong>{money(total_put)}</strong><span>put-side proxy</span></div>"
        f"<div class='premium-metric-card'><div>ALL TICKERS · PUT/CALL</div><strong>{put_call_ratio:.2f}</strong><span>{aggregate_bias.upper()} aggregate bias</span></div>"
        "</div>",
        unsafe_allow_html=True,
    )

    display = sentiment[[
        "Ticker", "Call Premium", "Put Premium", "Net Premium", "Call Share",
        "Put/Call Premium", "Sentiment", "Call Contracts", "Put Contracts", "Quoted Rows",
    ]].copy()
    display.columns = [
        "Ticker", "Call Premium", "Put Premium", "Net Premium", "Call Share",
        "Put/Call", "Net Bias & Sentiment", "Call Contracts", "Put Contracts", "Quoted Rows",
    ]
    st.markdown("<div class='layer-title'>NET INSTITUTIONAL PREMIUM SENTIMENT <span>OBSERVED OPTION VOLUME · QUOTED MIDPOINT PROXY</span></div>", unsafe_allow_html=True)

    def sentiment_style(value: Any) -> str:
        text = str(value)
        if text.startswith("BEARISH"):
            return "background-color: rgba(255, 85, 125, 0.18); color: #ffd7e2;"
        if text.startswith("BULLISH"):
            return "background-color: rgba(40, 215, 161, 0.18); color: #dffaf0;"
        return "background-color: rgba(245, 200, 75, 0.14); color: #f7e9a7;"

    def net_premium_style(value: Any) -> str:
        number = float(value)
        if number > 0:
            return "color: #28d7a1; font-weight: 600;"
        if number < 0:
            return "color: #ff557d; font-weight: 600;"
        return "color: #f5c84b; font-weight: 600;"

    styled = display.style.map(sentiment_style, subset=["Net Bias & Sentiment"]).map(net_premium_style, subset=["Net Premium"]).set_properties(**{
        "background-color": "#0d151f",
        "color": "#dce5ef",
        "border-color": "#243140",
    }).format({
        "Call Premium": money,
        "Put Premium": money,
        "Net Premium": money,
        "Call Share": "{:.1%}",
        "Put/Call": "{:.2f}",
        "Call Contracts": "{:,}",
        "Put Contracts": "{:,}",
        "Quoted Rows": "{:,}",
    })
    table_height = min(max(len(display) * 35 + 42, 100), 420)
    st.dataframe(styled, width="stretch", hide_index=True, height=table_height)
    st.caption("Premium is a traded-notional proxy based on reported volume and quoted midpoint; it is not confirmed institutional flow or open/close intent.")


def main() -> None:
    authenticate()
    render_navigation_styles()
    active_view, controls = render_navigation()
    if controls.get("show_mockups"):
        render_theme_mockups()

    if active_view == "gamma":
        render_terminal(controls)
    elif active_view == "signals":
        expiration = controls.get("expiration")
        if expiration:
            st.markdown("<div class='terminal-label'>DIRECTIONAL PLAYBOOKS · EXACT EXPIRY UNIVERSE</div>", unsafe_allow_html=True)
            signal_universe = get_signal_universe(expiration)
            render_actionable_signals(expiration, signal_universe)
            render_universe_scan(expiration, signal_universe)
        else:
            st.warning("No signal expiry is available from the provider.")
    elif active_view == "dvzr":
        render_dvzr_dashboard(controls)
    elif active_view == "premium":
        render_premium_sentiment(controls)


if __name__ == "__main__":
    main()
