import pandas as pd
import numpy as np
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

from app import (
    analyze_market_state,
    calculate_forward_return_validation,
    calculate_premium_sentiment,
    _friday_session,
    _is_trading_day,
    _effective_signal_date,
    _valid_expirations,
    _previous_trading_day,
    score_market_state,
    summarize_forward_return_validation,
)


def test_market_state_returns_expected_fields():
    df = pd.DataFrame(
        {
            "Close": [100, 101, 102, 103, 104, 103, 102, 101, 100, 99, 98, 97, 96, 95, 94],
            "High": [101, 102, 103, 104, 105, 104, 103, 102, 101, 100, 99, 98, 97, 96, 95],
            "Low": [99, 100, 101, 102, 103, 102, 101, 100, 99, 98, 97, 96, 95, 94, 93],
            "Volume": [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000],
        }
    )
    result = score_market_state(df)
    assert "signal" in result
    assert "days_in_state" in result
    assert "z_score" in result
    assert "event_risk" in result


def test_analyze_market_state_handles_missing_data():
    result = analyze_market_state("NOT_A_REAL_SYMBOL_123", "ETF")
    assert result is None


def test_market_state_includes_quality_score_fields():
    close = pd.Series(np.linspace(100, 120, 220))
    df = pd.DataFrame(
        {
            "Close": close,
            "High": close + 1,
            "Low": close - 1,
            "Volume": 1_000,
        }
    )
    result = score_market_state(df)
    assert 0 <= result["signal_score"] <= 100
    assert result["volatility_ratio"] > 0


def test_forward_validation_returns_bucket_summary_without_lookahead():
    close = pd.Series(100 + np.sin(np.arange(260) / 4) * 4 + np.arange(260) * 0.08)
    df = pd.DataFrame({"Close": close, "High": close + 1, "Low": close - 1, "Volume": 1_000})
    observations = calculate_forward_return_validation(df)
    if observations.empty:
        assert summarize_forward_return_validation(observations).empty
    sample = pd.DataFrame(
        {
            "Score bucket": ["60-79", "60-79"],
            "1D directional return": [0.01, -0.005],
            "3D directional return": [0.02, 0.01],
            "5D directional return": [0.03, -0.01],
            "10D directional return": [0.04, 0.02],
        }
    )
    summary = summarize_forward_return_validation(sample)
    assert set(summary.columns) >= {"Score bucket", "Signals", "1D avg", "10D hit rate"}
    if not observations.empty:
        assert observations["Signal date"].max() <= 259


def test_premium_sentiment_uses_volume_midpoint_and_contract_multiplier():
    calls = pd.DataFrame({"strike": [100, 105], "volume": [10, 20], "bid": [2.0, 1.0], "ask": [2.2, 1.2]})
    puts = pd.DataFrame({"strike": [95, 100], "volume": [5, 5], "bid": [3.0, 2.0], "ask": [3.2, 2.2]})
    result = calculate_premium_sentiment(calls, puts, "TEST", "2026-09-25")
    assert result is not None
    assert result["Call Premium"] == 4_300.0
    assert result["Put Premium"] == 2_600.0
    assert result["Net Premium"] == 1_700.0
    assert result["Sentiment"] == "BULLISH / DIRECTIONAL"


def test_non_trading_days_are_excluded_and_adjusted():
    saturday = date(2026, 9, 19)
    independence_day = date(2026, 7, 4)
    assert not _is_trading_day(saturday)
    assert _previous_trading_day(saturday) == date(2026, 9, 18)
    assert _previous_trading_day(independence_day) == date(2026, 7, 2)
    assert _friday_session(independence_day) == date(2026, 7, 2)


def test_future_expirations_do_not_crash_calendar_validation():
    expirations = ["2027-10-15", "2027-12-17", "2027-12-18"]
    assert _valid_expirations(expirations) == ["2027-10-15", "2027-12-17"]


def test_signal_date_advances_after_market_close_and_skips_weekend():
    after_close = datetime(2026, 9, 18, 16, 1, tzinfo=ZoneInfo("America/New_York"))
    assert _effective_signal_date(after_close) == date(2026, 9, 21)
