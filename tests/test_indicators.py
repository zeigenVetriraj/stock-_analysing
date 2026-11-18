"""Unit tests for the Layer 1 Fibonacci and fair value gap calculations.

Synthetic candles only - NO network and NO LLM. Run with:  ``pytest -q``
"""

from __future__ import annotations

import json
import math
import sys
import types

import pandas as pd
import pytest

from src.analysis.analyst_agent import SYSTEM_PROMPT
from src.ingestion.data_fetcher import (
    MarketSnapshot,
    compute_fibonacci_levels,
    detect_fair_value_gaps,
    fetch_market_snapshot,
)

# --- Helpers ---------------------------------------------------------------- #


def _bars(rows, start="2026-01-05"):
    """Daily OHLC frame from (open, high, low, close) tuples."""
    index = pd.date_range(start, periods=len(rows), freq="D", tz="Asia/Kolkata")
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"], index=index)


# --- Fair value gaps -------------------------------------------------------- #

def test_bullish_gap_detected_with_its_zone():
    bars = _bars([
        (100, 102, 99, 101),   # candle 1: high 102
        (101, 110, 101, 109),  # displacement up
        (109, 112, 105, 111),  # candle 3: low 105 > 102 -> gap 102-105
        (111, 113, 106, 112),  # later low 106 stays above the gap
    ])
    gaps = detect_fair_value_gaps(bars, current_price=112.0, timeframe="4h")
    assert len(gaps) == 1
    gap = gaps[0]
    assert gap.timeframe == "4h"
    assert gap.direction == "bullish"
    assert (gap.gap_low, gap.gap_high) == (102.0, 105.0)
    assert gap.filled_pct == 0.0
    assert gap.position == "below"
    assert gap.distance_pct == pytest.approx(6.25)       # (112 - 105) / 112
    assert gap.size_pct == pytest.approx(2.68)           # 3 / 112
    assert gap.formed_at.startswith("2026-01-06")        # the middle candle


def test_bearish_gap_detected_with_its_zone():
    bars = _bars([
        (100, 101, 98, 99),    # candle 1: low 98
        (99, 99, 90, 91),      # displacement down
        (91, 95, 88, 89),      # candle 3: high 95 < 98 -> gap 95-98
        (89, 94, 87, 88),      # later high 94 stays below the gap
    ])
    gaps = detect_fair_value_gaps(bars, current_price=88.0, timeframe="1d")
    assert len(gaps) == 1
    gap = gaps[0]
    assert gap.direction == "bearish"
    assert (gap.gap_low, gap.gap_high) == (95.0, 98.0)
    assert gap.position == "above"
    assert gap.distance_pct == pytest.approx(7.95)       # (95 - 88) / 88


def test_overlapping_candles_have_no_gap():
    bars = _bars([
        (100, 102, 99, 101),
        (101, 103, 100, 102),
        (102, 104, 101, 103),
        (103, 105, 102, 104),
    ])
    assert detect_fair_value_gaps(bars, current_price=104.0, timeframe="1d") == []


def test_partly_filled_gap_reports_fill_percentage():
    bars = _bars([
        (100, 102, 99, 101),
        (101, 110, 101, 109),
        (109, 112, 105, 111),  # gap 102-105
        (111, 111, 103.5, 108),  # dips half way into the gap
    ])
    gaps = detect_fair_value_gaps(bars, current_price=108.0, timeframe="1d")
    assert len(gaps) == 1
    assert gaps[0].filled_pct == pytest.approx(50.0)


def test_fully_filled_gap_is_dropped():
    bars = _bars([
        (100, 102, 99, 101),
        (101, 110, 101, 109),
        (109, 112, 105, 111),  # gap 102-105
        (111, 111, 101.5, 104),  # trades straight through it
    ])
    assert detect_fair_value_gaps(bars, current_price=104.0, timeframe="1d") == []


def test_tiny_gaps_are_ignored_as_noise():
    bars = _bars([
        (100, 100.00, 99, 100),
        (100, 100.50, 100, 100.4),
        (100.4, 100.60, 100.05, 100.5),  # 0.05 gap = ~0.05% of price
    ])
    assert detect_fair_value_gaps(bars, current_price=100.5, timeframe="4h") == []
    assert len(detect_fair_value_gaps(bars, 100.5, timeframe="4h", min_size_pct=0.01)) == 1


def test_nearest_gaps_come_first_and_are_capped():
    bars = _bars([
        (100, 101, 99, 100.5),
        (101, 106, 101, 105),
        (105, 108, 103, 107),  # gap 101-103
        (107, 113, 107, 112),  # gap 106-107
        (112, 116, 110, 115),  # gap 108-110
    ])
    gaps = detect_fair_value_gaps(bars, current_price=115.0, timeframe="1d", max_gaps=2)
    assert [(g.gap_low, g.gap_high) for g in gaps] == [(108.0, 110.0), (106.0, 107.0)]
    assert gaps[0].distance_pct < gaps[1].distance_pct
    assert len(detect_fair_value_gaps(bars, current_price=115.0, timeframe="1d")) == 3


# --- Fibonacci retracements ------------------------------------------------- #

UP_SWING = [
    (100, 101, 95, 96),
    (96, 97, 80, 82),      # swing low 80
    (82, 95, 81, 94),
    (94, 110, 93, 108),
    (108, 120, 107, 118),  # swing high 120
    (118, 119, 109, 110),
]


def test_up_swing_levels_measured_down_from_the_high():
    fib = compute_fibonacci_levels(_bars(UP_SWING), current_price=110.0)
    assert fib is not None
    assert fib.swing_direction == "up"
    assert (fib.swing_low, fib.swing_high) == (80.0, 120.0)
    assert fib.swing_low_date == "2026-01-06"
    assert fib.swing_high_date == "2026-01-09"
    assert fib.levels == {
        "0.236": 110.56, "0.382": 104.72, "0.5": 100.0, "0.618": 95.28, "0.786": 88.56,
    }
    assert fib.retracement_pct == pytest.approx(25.0)
    assert fib.nearest_level == "0.236"
    assert fib.price_vs_nearest_level_pct == pytest.approx(-0.51)


def test_down_swing_levels_measured_up_from_the_low():
    bars = _bars([
        (110, 120, 108, 118),  # swing high 120
        (118, 119, 100, 101),
        (101, 102, 85, 86),
        (86, 88, 80, 82),      # swing low 80
        (82, 91, 81, 90),
    ])
    fib = compute_fibonacci_levels(bars, current_price=90.0)
    assert fib is not None
    assert fib.swing_direction == "down"
    assert fib.levels == {
        "0.236": 89.44, "0.382": 95.28, "0.5": 100.0, "0.618": 104.72, "0.786": 111.44,
    }
    assert fib.retracement_pct == pytest.approx(25.0)
    assert fib.nearest_level == "0.236"


def test_lookback_ignores_older_extremes():
    bars = _bars([(200, 250, 190, 240)] + UP_SWING)
    fib = compute_fibonacci_levels(bars, current_price=110.0, lookback=len(UP_SWING))
    assert fib.lookback_sessions == len(UP_SWING)
    assert fib.swing_high == 120.0
    assert compute_fibonacci_levels(bars, current_price=110.0).swing_high == 250.0


def test_flat_range_has_no_levels():
    bars = _bars([(100, 100, 100, 100)] * 5)
    assert compute_fibonacci_levels(bars, current_price=100.0) is None


# --- Wiring into the snapshot and the analyst prompt ------------------------ #

def test_snapshot_json_and_prompt_carry_the_new_fields():
    snap = MarketSnapshot(ticker="TEST.NS", as_of="2026-01-01T00:00:00+00:00", current_price=100.0)
    payload = json.loads(snap.to_analyst_json())
    assert payload["fibonacci"] is None
    assert payload["fair_value_gaps"] == []
    assert '"fibonacci"' in SYSTEM_PROMPT
    assert '"fair_value_gaps"' in SYSTEM_PROMPT


def _synthetic_daily(sessions=250):
    rows, price = [], 100.0
    for i in range(sessions):
        close = 100 + 20 * math.sin(i / 15) + 0.1 * i + (10 if i >= 245 else 0)
        rows.append((price, max(price, close) + 1, min(price, close) - 1, close, 1_000_000))
        price = close
    index = pd.bdate_range(end="2026-09-28", periods=sessions, tz="Asia/Kolkata")
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close", "Volume"], index=index)


def _synthetic_hourly(days=40):
    stamps, rows, price = [], [], 150.0
    for d, day in enumerate(pd.bdate_range(end="2026-09-28", periods=days)):
        for h, (hh, mm) in enumerate([(9, 15), (10, 15), (11, 15), (12, 15), (13, 15), (14, 15), (15, 15)]):
            close = 150 + 5 * math.sin((d * 7 + h) / 9)
            rows.append((price, max(price, close) + 0.3, min(price, close) - 0.3, close, 50_000))
            stamps.append(pd.Timestamp(day.year, day.month, day.day, hh, mm, tz="Asia/Kolkata"))
            price = close
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close", "Volume"],
                        index=pd.DatetimeIndex(stamps))


def test_fetch_market_snapshot_includes_levels_without_network(monkeypatch):
    daily, hourly = _synthetic_daily(), _synthetic_hourly()

    class FakeTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, period, interval):
            return daily if interval == "1d" else hourly

    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(Ticker=FakeTicker))
    snap = fetch_market_snapshot("TEST.NS")

    assert snap.fibonacci is not None
    assert snap.fibonacci.lookback_sessions == 120
    assert len(snap.fibonacci.levels) == 5
    assert any(g.timeframe == "1d" for g in snap.fair_value_gaps)
    assert all(g.gap_low < g.gap_high and g.filled_pct < 100 for g in snap.fair_value_gaps)
    assert len(snap.recent_4h_candles) == 12
    payload = json.loads(snap.to_analyst_json())
    assert set(payload["fibonacci"]["levels"]) == {"0.236", "0.382", "0.5", "0.618", "0.786"}
