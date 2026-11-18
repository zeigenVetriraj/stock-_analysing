"""Layer 1 - Data ingestion.

Fetches historical price action from Yahoo Finance (``yfinance``) and computes
a small set of swing-trading technical metrics:

* RSI (14)              - momentum oscillator
* SMA 50 / SMA 200      - trend regime (golden / death cross)
* Fibonacci levels      - 23.6-78.6% retracements of the recent daily swing
* Fair value gaps       - unfilled three-candle imbalances on 4h and daily bars
* 4-hour price action   - recent intraday candles (resampled from 1h bars)

The output is a validated :class:`MarketSnapshot` pydantic model, which serialises
cleanly to JSON for the analyst agent (Layer 2).

Note on intervals: Yahoo Finance does not expose a native 4h bar, so the recent
price action is resampled from 1h bars. The 50/200-day SMAs and RSI are computed
on *daily* closes, which is the conventional basis for those swing indicators.

The Fibonacci and fair value gap calculations are pure functions
(:func:`compute_fibonacci_levels`, :func:`detect_fair_value_gaps`) so they can be
unit-tested offline. They are best-effort extras: if either fails, the snapshot
is still produced without it.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional

import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# yfinance is imported lazily inside the fetch so that unit tests and the risk
# layer can import this module without a network-capable environment.

# Yahoo Finance suffixes for Indian exchanges.
_YF_EXCHANGE_SUFFIX = {"NSE": ".NS", "BSE": ".BO"}


def to_yfinance_ticker(symbol: str, exchange: str = "NSE") -> str:
    """Map a plain Indian symbol to a Yahoo Finance ticker.

    ``("RELIANCE", "NSE") -> "RELIANCE.NS"``. If the symbol already carries a
    suffix (contains a dot), it is returned unchanged so US tickers still work.
    """
    sym = symbol.strip().upper()
    if "." in sym:
        return sym
    return f"{sym}{_YF_EXCHANGE_SUFFIX.get(exchange.strip().upper(), '.NS')}"


class Candle(BaseModel):
    """A single OHLCV candle."""

    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float


# Standard Fibonacci retracement ratios reported to the analyst.
FIB_RATIOS = (0.236, 0.382, 0.5, 0.618, 0.786)


class FairValueGap(BaseModel):
    """A three-candle price imbalance that later price has not fully filled."""

    timeframe: str = Field(description="Bar size the gap was found on: '4h' or '1d'")
    direction: Literal["bullish", "bearish"]
    formed_at: str = Field(description="Timestamp of the middle (displacement) candle")
    gap_low: float
    gap_high: float
    size_pct: float = Field(description="Gap height as a % of the current price")
    filled_pct: float = Field(
        description="How much of the gap later candles have traded back into (0-100)"
    )
    position: Literal["above", "below", "inside"] = Field(
        description="Where the gap sits relative to the current price"
    )
    distance_pct: float = Field(
        description="% distance from the current price to the nearest gap edge (0 if inside)"
    )


class FibonacciLevels(BaseModel):
    """Fibonacci retracement levels of the most recent daily swing."""

    lookback_sessions: int
    swing_direction: Literal["up", "down"] = Field(
        description="'up' if the swing low came before the swing high, else 'down'"
    )
    swing_high: float
    swing_high_date: str
    swing_low: float
    swing_low_date: str
    levels: Dict[str, float] = Field(
        description="Retracement ratio -> price, measured back from the end of the swing"
    )
    retracement_pct: float = Field(
        description=(
            "How much of the swing the current price has given back: 0 = at the swing "
            "end, 100 = back at the swing start, negative = beyond the swing end"
        )
    )
    nearest_level: str
    nearest_level_price: float
    price_vs_nearest_level_pct: float = Field(
        description="Signed % distance of the current price from the nearest level"
    )


class MarketSnapshot(BaseModel):
    """Structured market data + technical metrics for one ticker."""

    ticker: str
    as_of: str = Field(description="UTC ISO-8601 timestamp of the snapshot")
    current_price: float
    rsi_14: Optional[float] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    trend_regime: str = "unknown"
    price_vs_sma50_pct: Optional[float] = None
    fibonacci: Optional[FibonacciLevels] = None
    fair_value_gaps: List[FairValueGap] = Field(default_factory=list)
    recent_4h_candles: List[Candle] = Field(default_factory=list)

    def to_analyst_json(self) -> str:
        """Compact JSON payload handed to the LLM analyst."""
        return self.model_dump_json(indent=2)


class DataFetchError(RuntimeError):
    """Raised when market data cannot be retrieved or is unusable."""


def _compute_rsi(close: pd.Series, period: int = 14) -> Optional[float]:
    """Wilder's RSI on a close-price series. Returns the latest value."""
    if len(close) <= period:
        return None
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    latest = rsi.iloc[-1]
    return None if pd.isna(latest) else round(float(latest), 2)


def _latest_sma(close: pd.Series, window: int) -> Optional[float]:
    """Simple moving average over ``window`` closes; None if insufficient data."""
    if len(close) < window:
        return None
    return round(float(close.rolling(window=window).mean().iloc[-1]), 4)


def _resample_4h(intraday: pd.DataFrame) -> pd.DataFrame:
    """Resample 1h OHLCV bars into 4h bars."""
    agg = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }
    return intraday.resample("4h").agg(agg).dropna(how="any")


def detect_fair_value_gaps(
    bars: pd.DataFrame,
    current_price: float,
    *,
    timeframe: str,
    min_size_pct: float = 0.1,
    max_gaps: int = 3,
) -> List[FairValueGap]:
    """Find three-candle fair value gaps (FVGs) that price has not fully filled.

    A *bullish* FVG forms when candle 3's low is above candle 1's high: price
    skipped the zone ``[high1, low3]`` on the way up. A *bearish* FVG forms when
    candle 3's high is below candle 1's low, leaving ``[high3, low1]`` untraded.

    A gap is dropped once later candles trade all the way through it; partly
    filled gaps are kept with their ``filled_pct``. Gaps smaller than
    ``min_size_pct`` of the current price are ignored as noise. Returns at most
    ``max_gaps`` gaps, nearest to the current price first.
    """
    if bars is None or current_price is None or current_price <= 0:
        return []
    bars = bars.dropna(subset=["High", "Low"])
    if len(bars) < 3:
        return []

    highs = bars["High"].to_numpy(dtype=float)
    lows = bars["Low"].to_numpy(dtype=float)
    stamps = list(bars.index)

    gaps: List[FairValueGap] = []
    for i in range(2, len(bars)):
        if lows[i] > highs[i - 2]:
            direction, gap_low, gap_high = "bullish", highs[i - 2], lows[i]
        elif highs[i] < lows[i - 2]:
            direction, gap_low, gap_high = "bearish", highs[i], lows[i - 2]
        else:
            continue

        size = gap_high - gap_low
        if size / current_price * 100 < min_size_pct:
            continue

        # How far later candles have traded back into the gap.
        if direction == "bullish":
            later = lows[i + 1:]
            filled = (gap_high - later.min()) / size if len(later) else 0.0
        else:
            later = highs[i + 1:]
            filled = (later.max() - gap_low) / size if len(later) else 0.0
        filled = min(max(filled, 0.0), 1.0)
        if filled >= 1.0:
            continue

        if current_price > gap_high:
            position, distance = "below", (current_price - gap_high) / current_price * 100
        elif current_price < gap_low:
            position, distance = "above", (gap_low - current_price) / current_price * 100
        else:
            position, distance = "inside", 0.0

        gaps.append(
            FairValueGap(
                timeframe=timeframe,
                direction=direction,
                formed_at=pd.Timestamp(stamps[i - 1]).isoformat(),
                gap_low=round(float(gap_low), 2),
                gap_high=round(float(gap_high), 2),
                size_pct=round(float(size / current_price * 100), 2),
                filled_pct=round(float(filled * 100), 1),
                position=position,
                distance_pct=round(float(distance), 2),
            )
        )

    gaps.sort(key=lambda g: g.distance_pct)
    return gaps[:max_gaps]


def compute_fibonacci_levels(
    daily: pd.DataFrame,
    current_price: float,
    *,
    lookback: int = 120,
) -> Optional[FibonacciLevels]:
    """Fibonacci retracements of the swing inside the last ``lookback`` daily bars.

    The swing runs between the highest high and the lowest low in the window.
    If the low came first the swing is *up* and the levels are measured down
    from the high (``high - ratio * range``); if the high came first the swing is
    *down* and they are measured up from the low (``low + ratio * range``).
    Returns ``None`` when there is no usable range.
    """
    if daily is None or current_price is None or current_price <= 0:
        return None
    window = daily.dropna(subset=["High", "Low"]).tail(lookback)
    if len(window) < 2:
        return None

    high_ts = window["High"].idxmax()
    low_ts = window["Low"].idxmin()
    swing_high = float(window.loc[high_ts, "High"])
    swing_low = float(window.loc[low_ts, "Low"])
    swing_range = swing_high - swing_low
    if not swing_range > 0:
        return None

    if low_ts == high_ts:  # one bar holds both extremes: use its own direction
        up = float(window.loc[high_ts, "Close"]) >= float(window.loc[high_ts, "Open"])
    else:
        up = low_ts < high_ts

    levels: Dict[str, float] = {}
    for ratio in FIB_RATIOS:
        price = swing_high - ratio * swing_range if up else swing_low + ratio * swing_range
        levels[f"{ratio:g}"] = round(price, 2)

    if up:
        retraced = (swing_high - current_price) / swing_range * 100
    else:
        retraced = (current_price - swing_low) / swing_range * 100

    nearest = min(levels, key=lambda k: abs(current_price - levels[k]))
    nearest_price = levels[nearest]

    return FibonacciLevels(
        lookback_sessions=len(window),
        swing_direction="up" if up else "down",
        swing_high=round(swing_high, 2),
        swing_high_date=pd.Timestamp(high_ts).date().isoformat(),
        swing_low=round(swing_low, 2),
        swing_low_date=pd.Timestamp(low_ts).date().isoformat(),
        levels=levels,
        retracement_pct=round(float(retraced), 1),
        nearest_level=nearest,
        nearest_level_price=nearest_price,
        price_vs_nearest_level_pct=round(float((current_price - nearest_price) / nearest_price * 100), 2),
    )


def fetch_market_snapshot(
    ticker: str,
    *,
    daily_period: str = "1y",
    intraday_period: str = "60d",
    recent_candles: int = 12,
    fib_lookback: int = 120,
) -> MarketSnapshot:
    """Fetch price data for ``ticker`` and return a :class:`MarketSnapshot`.

    Parameters
    ----------
    ticker:
        Symbol, e.g. ``"AAPL"``.
    daily_period:
        Look-back for daily bars used by SMA/RSI (needs >=200 sessions for SMA200).
        Daily fair value gaps are searched over this same window.
    intraday_period:
        Look-back for 1h bars that are resampled to 4h price action. 4h fair
        value gaps are searched over this whole window, not just the recent candles.
    recent_candles:
        How many trailing 4h candles to include in the snapshot.
    fib_lookback:
        Number of daily sessions that define the swing for Fibonacci levels.
    """
    symbol = ticker.strip().upper()
    if not symbol:
        raise DataFetchError("Empty ticker symbol.")

    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - environment guard
        raise DataFetchError(
            "yfinance is not installed. Run `pip install -r requirements.txt`."
        ) from exc

    tkr = yf.Ticker(symbol)

    # --- Daily bars: SMA + RSI ---
    daily = tkr.history(period=daily_period, interval="1d")
    if daily is None or daily.empty:
        raise DataFetchError(f"No daily price data returned for '{symbol}'.")
    daily_close = daily["Close"].dropna()

    sma_50 = _latest_sma(daily_close, 50)
    sma_200 = _latest_sma(daily_close, 200)
    rsi_14 = _compute_rsi(daily_close, 14)

    # --- Intraday bars resampled to 4h: recent price action ---
    candles: List[Candle] = []
    current_price: Optional[float] = None
    all_4h: Optional[pd.DataFrame] = None
    try:
        intraday = tkr.history(period=intraday_period, interval="1h")
        if intraday is not None and not intraday.empty:
            all_4h = _resample_4h(intraday)
            bars_4h = all_4h.tail(recent_candles)
            for ts, row in bars_4h.iterrows():
                candles.append(
                    Candle(
                        timestamp=pd.Timestamp(ts).isoformat(),
                        open=round(float(row["Open"]), 4),
                        high=round(float(row["High"]), 4),
                        low=round(float(row["Low"]), 4),
                        close=round(float(row["Close"]), 4),
                        volume=float(row["Volume"]),
                    )
                )
            if not bars_4h.empty:
                current_price = round(float(bars_4h["Close"].iloc[-1]), 4)
    except Exception as exc:  # noqa: BLE001 - intraday is best-effort
        logger.warning("Intraday fetch for %s failed (%s); using daily close.", symbol, exc)

    if current_price is None:
        current_price = round(float(daily_close.iloc[-1]), 4)

    # --- Trend regime ---
    trend_regime = "unknown"
    price_vs_sma50_pct: Optional[float] = None
    if sma_50 is not None and sma_200 is not None:
        trend_regime = "bullish" if sma_50 >= sma_200 else "bearish"
    if sma_50:
        price_vs_sma50_pct = round((current_price - sma_50) / sma_50 * 100, 2)

    # --- Fibonacci retracements + fair value gaps (best-effort extras) ---
    fibonacci: Optional[FibonacciLevels] = None
    try:
        fibonacci = compute_fibonacci_levels(daily, current_price, lookback=fib_lookback)
    except Exception as exc:  # noqa: BLE001 - an extra must never block the pipeline
        logger.warning("Fibonacci levels for %s failed (%s); continuing without them.", symbol, exc)

    fair_value_gaps: List[FairValueGap] = []
    for timeframe, bars in (("4h", all_4h), ("1d", daily)):
        if bars is None or bars.empty:
            continue
        try:
            fair_value_gaps.extend(
                detect_fair_value_gaps(bars, current_price, timeframe=timeframe)
            )
        except Exception as exc:  # noqa: BLE001 - an extra must never block the pipeline
            logger.warning("%s fair value gaps for %s failed (%s); skipping.", timeframe, symbol, exc)

    snapshot = MarketSnapshot(
        ticker=symbol,
        as_of=datetime.now(timezone.utc).isoformat(),
        current_price=current_price,
        rsi_14=rsi_14,
        sma_50=sma_50,
        sma_200=sma_200,
        trend_regime=trend_regime,
        price_vs_sma50_pct=price_vs_sma50_pct,
        fibonacci=fibonacci,
        fair_value_gaps=fair_value_gaps,
        recent_4h_candles=candles,
    )
    logger.info(
        "Snapshot %s: price=%.2f rsi=%s sma50=%s sma200=%s regime=%s",
        symbol, current_price, rsi_14, sma_50, sma_200, trend_regime,
    )
    if fibonacci is not None:
        logger.info(
            "Fibonacci %s: %s-swing %.2f -> %.2f, retraced %.1f%%, nearest %s @ %.2f",
            symbol, fibonacci.swing_direction,
            fibonacci.swing_low if fibonacci.swing_direction == "up" else fibonacci.swing_high,
            fibonacci.swing_high if fibonacci.swing_direction == "up" else fibonacci.swing_low,
            fibonacci.retracement_pct, fibonacci.nearest_level, fibonacci.nearest_level_price,
        )
    for gap in fair_value_gaps:
        logger.info(
            "FVG %s: %s %s %.2f-%.2f (%s price, %.2f%% away, %.0f%% filled)",
            symbol, gap.timeframe, gap.direction, gap.gap_low, gap.gap_high,
            gap.position, gap.distance_pct, gap.filled_pct,
        )
    return snapshot


__all__ = [
    "Candle",
    "FairValueGap",
    "FibonacciLevels",
    "MarketSnapshot",
    "DataFetchError",
    "FIB_RATIOS",
    "compute_fibonacci_levels",
    "detect_fair_value_gaps",
    "fetch_market_snapshot",
    "to_yfinance_ticker",
]
