"""Intraday setups (paper) on 5-minute bars. Deterministic, no AI.

:func:`compute_features` measures one stock's session so far:

* opening range (09:15-09:45), VWAP, EMA 9/20, RSI and the average bar range;
* yesterday's high / low / close, the opening gap and the classic pivot levels
  (P, R1, R2, S1, S2) with the central pivot range (CPR);
* relative volume (today so far against the same bars of the last 5 sessions);
* fair value gaps on 5-minute and 15-minute bars (yesterday + today);
* today's swing (low to high, or high to low) and how much of it price has
  given back, with the Fibonacci retracement levels and extension targets;
* how the stock is doing against the Nifty 50.

:func:`all_setups` then looks for four kinds of trade ("models"):

* ``breakout`` - price is beyond the opening range: trade with the break.
* ``fvg``      - price has come back to an unfilled fair value gap made earlier
                 today: trade in the gap's direction, stop just beyond the gap.
* ``fib``      - price has given back 38.2-61.8% of today's swing: trade in the
                 swing's direction, stop beyond the 78.6% level.
* ``gap_fade`` - the stock with the biggest opening gap up (or down) of the
                 day: trade against the gap.

Every setup carries its own stop (where the idea is proven wrong, never wider
than the rules allow) and a target at a fixed multiple of that risk. The score
is a simple count of things lining up (daily trend, the Nifty, VWAP, volume);
it ranks the shortlist for Claude and is NOT a prediction: see
``intraday_backtest.py`` for what each model actually did on past data.

:func:`simulate_exit` replays the 5-minute bars after the entry to decide how a
paper position ended: at its stop, at its target, or at the square-off time.
:func:`check_exit` does the same replay during the day (for the hourly checks)
and only reports a touch of the stop or the target. If one bar touches both,
the stop is assumed first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config.settings import INTRADAY_MODELS
from src.automation.ledger import Position
from src.ingestion.data_fetcher import FairValueGap, detect_fair_value_gaps

INDEX_TICKER = "^NSEI"          # Nifty 50 on Yahoo
OPENING_RANGE_BARS = 6          # 6 x 5 minutes = 09:15-09:45
BAR_MINUTES = 5
RELVOL_SESSIONS = 5             # compare volume with the same bars of the last 5 sessions
FVG_MIN_SIZE_PCT = {"5m": 0.05, "15m": 0.08}
FIB_RATIOS = (0.236, 0.382, 0.5, 0.618, 0.786)
INDICATOR_BARS = 150            # EMA / RSI look at the last this-many 5-minute closes
REQUIRED_COLUMNS = {"High", "Low", "Close", "Volume"}

MODELS = INTRADAY_MODELS         # ("breakout", "fvg", "fib", "gap_fade")
MODEL_LABELS = {
    "breakout": "opening-range breakout",
    "fvg": "fair value gap retest",
    "fib": "Fibonacci pullback",
    "gap_fade": "gap reversal",
}
MIN_STOP_PCT = 0.3              # a tighter stop is noise on 5-minute bars
STOP_BUFFER_PCT = 0.05          # the stop sits this far beyond the level it leans on
FVG_NEAR_PCT = 0.15             # price within this of a gap counts as "at the gap"
TRADE_GAP_MIN_PCT = 0.15        # a fair value gap smaller than this (% of price) is too small to trade
FIB_ZONE = (0.382, 0.618)       # retracement that counts as a pullback entry
FIB_INVALID = 0.786             # beyond this the swing is considered broken
MIN_SWING_PCT = 1.0             # today's swing must be at least this big
MIN_SWING_AGE_BARS = 2          # the swing's end must be at least this many bars old
MIN_GAP_PCT = 0.5               # smallest opening gap worth fading
VOLUME_FLOOR = 1.2              # relative volume that counts as "busy"


@dataclass
class Level:
    name: str
    price: float
    distance_pct: float            # unsigned % distance from the current price


@dataclass
class IntradayFeatures:
    """Everything measured about one stock's session so far (no opinion yet)."""

    ticker: str
    price: float
    as_of: str                     # HH:MM of the last bar used
    bars_today: int
    today_open: float
    day_high: float
    day_low: float
    or_high: float
    or_low: float
    vwap: float
    last_bar: Optional[pd.Timestamp] = None
    ema9: Optional[float] = None
    ema20: Optional[float] = None
    rsi: Optional[float] = None
    avg_range_pct: Optional[float] = None    # average 5-minute bar range, % of price
    pdh: Optional[float] = None              # yesterday's high / low / close
    pdl: Optional[float] = None
    pdc: Optional[float] = None
    gap_pct: Optional[float] = None
    change_pct: Optional[float] = None
    pivots: Dict[str, float] = field(default_factory=dict)
    cpr_width_pct: Optional[float] = None
    relative_volume: Optional[float] = None
    daily_trend_up: Optional[bool] = None
    index_change_pct: Optional[float] = None
    relative_strength_pct: Optional[float] = None
    fair_value_gaps: List[FairValueGap] = field(default_factory=list)   # nearby, yesterday + today, 5m and 15m
    session_gaps: List[FairValueGap] = field(default_factory=list)      # 5m gaps made by three of today's candles
    fib_levels: Dict[str, float] = field(default_factory=dict)   # retracement ratio -> price, from the swing's end
    swing_up: Optional[bool] = None          # today's swing: low first (up) or high first (down)
    swing_range_pct: Optional[float] = None  # its size, % of price
    swing_retrace: Optional[float] = None    # share of it given back (0 = at the swing's end)
    swing_age_bars: Optional[int] = None     # bars since the swing's end

    @property
    def direction(self) -> Optional[str]:
        """Side of the opening range the price is on ("long" above, "short" below)."""
        if self.price > self.or_high:
            return "long"
        if self.price < self.or_low:
            return "short"
        return None

    @property
    def or_height_pct(self) -> float:
        return (self.or_high - self.or_low) / self.price * 100 if self.price else 0.0

    @property
    def extension_pct(self) -> float:
        """How far price is beyond the opening range, % of price (0 inside it)."""
        if self.price > self.or_high:
            return (self.price - self.or_high) / self.price * 100
        if self.price < self.or_low:
            return (self.or_low - self.price) / self.price * 100
        return 0.0

    def extension_targets(self) -> Dict[str, float]:
        """Fibonacci extensions of the opening range in the breakout direction."""
        height = self.or_high - self.or_low
        if height <= 0 or self.direction is None:
            return {}
        if self.direction == "long":
            return {str(r): round(self.or_low + r * height, 2) for r in (1.272, 1.618, 2.0)}
        return {str(r): round(self.or_high - r * height, 2) for r in (1.272, 1.618, 2.0)}

    def gap_age_bars(self, gap: FairValueGap) -> Optional[int]:
        """Finished bars since the gap's third candle (None if unknown)."""
        if self.last_bar is None:
            return None
        formed = pd.Timestamp(gap.formed_at)
        if formed.tzinfo is None and self.last_bar.tzinfo is not None:
            formed = formed.tz_localize(self.last_bar.tzinfo)
        return int((self.last_bar - formed).total_seconds() // (BAR_MINUTES * 60)) - 1



@dataclass
class IntradaySetup:
    ticker: str
    model: str                        # one of MODELS
    direction: str                    # "long" or "short"
    score: float                      # 50-100: how many things line up (not a prediction)
    price: float
    stop_pct: float                   # % of the price, already inside the allowed range
    target_pct: float
    notes: List[str] = field(default_factory=list)
    level: Optional[Level] = None     # the price level the idea rests on
    features: Optional[IntradayFeatures] = None
    bonus: float = 0.0                # the model's own quality bonus (0 or 10)
    expected_pct: Optional[float] = None   # learned net % per trade (see learning.py); None = not rated
    lessons: List[str] = field(default_factory=list)   # learned lessons that changed expected_pct

    @property
    def label(self) -> str:
        return MODEL_LABELS.get(self.model, self.model)

    @property
    def stop_price(self) -> float:
        sign = 1 if self.direction == "long" else -1
        return round(self.price * (1 - sign * self.stop_pct / 100), 2)

    @property
    def target_price(self) -> float:
        sign = 1 if self.direction == "long" else -1
        return round(self.price * (1 + sign * self.target_pct / 100), 2)

    def as_prompt(self) -> dict:
        """Compact data for Claude."""
        out = {
            "ticker": self.ticker, "model": self.model, "setup": self.direction, "score": self.score,
            "price": self.price, "stop_pct": self.stop_pct, "target_pct": self.target_pct,
            "stop_price": self.stop_price, "target_price": self.target_price,
            "rests_on": _level_dict(self.level), "notes": self.notes,
        }
        f = self.features
        if f is None:
            return out
        out.update({
            "as_of": f.as_of,
            "opening_range": [round(f.or_low, 2), round(f.or_high, 2)],
            "vwap": _r(f.vwap), "ema9": _r(f.ema9), "ema20": _r(f.ema20), "rsi_5m": _r(f.rsi, 1),
            "relative_volume": f.relative_volume,
            "prev_day": {"high": _r(f.pdh), "low": _r(f.pdl), "close": _r(f.pdc)},
            "gap_pct": _r(f.gap_pct), "change_pct": _r(f.change_pct),
            "pivots": {k: round(v, 2) for k, v in f.pivots.items()},
            "market": {"nifty_change_pct": _r(f.index_change_pct), "relative_strength_pct": _r(f.relative_strength_pct)},
            "daily_trend": None if f.daily_trend_up is None else ("up" if f.daily_trend_up else "down"),
            "fair_value_gaps": [
                {"timeframe": g.timeframe, "direction": g.direction, "low": g.gap_low, "high": g.gap_high,
                 "position": g.position, "distance_pct": g.distance_pct, "filled_pct": g.filled_pct}
                for g in f.fair_value_gaps
            ],
            "todays_swing": None if f.swing_up is None else {
                "direction": "up" if f.swing_up else "down", "size_pct": _r(f.swing_range_pct),
                "retraced": _r(f.swing_retrace, 3),
                "fibonacci_levels": f.fib_levels,
            },
        })
        return out


def _r(value: Optional[float], digits: int = 2) -> Optional[float]:
    return round(float(value), digits) if value is not None else None


def _level_dict(level: Optional[Level]) -> Optional[dict]:
    if level is None:
        return None
    return {"name": level.name, "price": round(level.price, 2), "distance_pct": round(level.distance_pct, 2)}


def to_ist(index: pd.Index, tz) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(index)
    return idx.tz_convert(tz) if idx.tz is not None else idx.tz_localize(tz)


def _hhmm(value: str) -> time:
    hours, minutes = (int(x) for x in value.split(":"))
    return time(hours, minutes)


# --------------------------------------------------------------------------- #
# Indicators and levels (pure functions)
# --------------------------------------------------------------------------- #
def ema(close, span: int) -> Optional[float]:
    """Exponential moving average of the latest bar (None with fewer than ``span`` closes)."""
    values = np.asarray(close, dtype=float)[-INDICATOR_BARS:]
    if len(values) < span:
        return None
    alpha, value = 2 / (span + 1), float(values[0])
    for x in values[1:]:
        value = alpha * float(x) + (1 - alpha) * value
    return value


def rsi(close, period: int = 14) -> Optional[float]:
    """Wilder's RSI of the latest bar."""
    values = np.asarray(close, dtype=float)[-INDICATOR_BARS:]
    if len(values) <= period:
        return None
    delta = np.diff(values)
    gains, losses = np.clip(delta, 0.0, None), np.clip(-delta, 0.0, None)
    gain, loss = float(gains[:period].mean()), float(losses[:period].mean())
    for g, x in zip(gains[period:], losses[period:]):
        gain = (gain * (period - 1) + float(g)) / period
        loss = (loss * (period - 1) + float(x)) / period
    if loss == 0:
        return 100.0 if gain > 0 else None
    return round(100 - 100 / (1 + gain / loss), 2)


def pivot_levels(high: float, low: float, close: float) -> Dict[str, float]:
    """Classic floor pivots from the previous session, plus the CPR edges (BC, TC)."""
    pivot = (high + low + close) / 3
    bc = (high + low) / 2
    tc = 2 * pivot - bc
    return {
        "P": pivot,
        "R1": 2 * pivot - low,
        "R2": pivot + (high - low),
        "S1": 2 * pivot - high,
        "S2": pivot - (high - low),
        "CPR_low": min(bc, tc),
        "CPR_high": max(bc, tc),
    }


def daily_trend_up(daily: Optional[pd.DataFrame], day: date, tz) -> Optional[bool]:
    """Was the last daily close *before* ``day`` above its 50-day average?

    Today's unfinished daily bar is left out, so the live job and the backtest
    see the same thing. None without 50 earlier closes.
    """
    if daily is None or daily.empty or "Close" not in daily:
        return None
    close = daily["Close"].dropna()
    if close.empty:
        return None
    close = close[to_ist(close.index, tz).date < day]
    if len(close) < 50:
        return None
    return bool(close.iloc[-1] > close.tail(50).mean())


def index_change_pct(index_bars: Optional[pd.DataFrame], today: date, tz) -> Optional[float]:
    """The index's change today against the previous session's close, in %."""
    if index_bars is None or index_bars.empty or "Close" not in index_bars:
        return None
    frame = index_bars.dropna(subset=["Close"]).copy()
    frame.index = to_ist(frame.index, tz)
    days = frame.index.date
    now, before = frame[days == today], frame[days < today]
    if now.empty or before.empty:
        return None
    return (float(now["Close"].iloc[-1]) / float(before["Close"].iloc[-1]) - 1) * 100


def _gap_frame(stamps: pd.DatetimeIndex, high: np.ndarray, low: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"High": high, "Low": low}, index=stamps)


def _fifteen_minute(stamps: pd.DatetimeIndex, high: np.ndarray, low: np.ndarray) -> pd.DataFrame:
    """15-minute highs and lows from 5-minute bars (labelled by the start of each 15 minutes)."""
    floored = stamps.floor("15min")
    _, first = np.unique(floored.asi8, return_index=True)
    return _gap_frame(floored[first], np.maximum.reduceat(high, first), np.minimum.reduceat(low, first))


def compute_features(
    ticker: str,
    bars: Optional[pd.DataFrame],
    today: date,
    tz,
    *,
    daily_trend_up: Optional[bool] = None,
    index_change: Optional[float] = None,
) -> Optional[IntradayFeatures]:
    """Measure one stock's session so far; None without enough of today's bars.

    ``bars`` must not contain anything after the decision time (the backtest
    truncates them; live data naturally ends at "now").
    """
    if bars is None or bars.empty or not REQUIRED_COLUMNS <= set(bars.columns):
        return None
    stamps = to_ist(bars.index, tz)
    if not stamps.is_monotonic_increasing:
        bars = bars.iloc[np.argsort(stamps.asi8, kind="stable")]
        stamps = to_ist(bars.index, tz)
    high = bars["High"].to_numpy(dtype=float)
    low = bars["Low"].to_numpy(dtype=float)
    close = bars["Close"].to_numpy(dtype=float)
    volume = np.nan_to_num(bars["Volume"].to_numpy(dtype=float))
    opened = bars["Open"].to_numpy(dtype=float) if "Open" in bars.columns else close
    days = stamps.normalize()
    midnight = pd.Timestamp(today).tz_localize(tz)
    keep = np.isfinite(high) & np.isfinite(low) & np.isfinite(close) & np.asarray(days <= midnight)
    if not keep.all():
        stamps, days = stamps[keep], days[keep]
        high, low, close, volume, opened = high[keep], low[keep], close[keep], volume[keep], opened[keep]
    session = days.asi8                                        # one number per trading day
    is_today = np.asarray(days == midnight)
    n = int(is_today.sum())
    if n <= OPENING_RANGE_BARS:
        return None

    t_high, t_low, t_close, t_volume = high[is_today], low[is_today], close[is_today], volume[is_today]
    t_stamps = stamps[is_today]
    or_high, or_low = float(t_high[:OPENING_RANGE_BARS].max()), float(t_low[:OPENING_RANGE_BARS].min())
    price = float(t_close[-1])
    if price <= 0:
        return None
    traded = float(t_volume.sum())
    typical = (t_high + t_low + t_close) / 3
    # An index has no traded volume: its "VWAP" is then the plain average price of the day.
    vwap = float((typical * t_volume).sum() / traded) if traded > 0 else float(typical.mean())
    first_open = float(opened[is_today][0])
    today_open = first_open if np.isfinite(first_open) else float(t_close[0])

    pdh = pdl = pdc = gap = change = None
    rel_volume = None
    earlier_days = np.unique(session[~is_today])
    recent = is_today
    if len(earlier_days):
        previous = session == earlier_days[-1]
        recent = previous | is_today
        pdh, pdl, pdc = float(high[previous].max()), float(low[previous].min()), float(close[previous][-1])
        if pdc > 0:
            gap = (today_open / pdc - 1) * 100
            change = (price / pdc - 1) * 100
        per_day = []
        for key in earlier_days[-RELVOL_SESSIONS:]:
            day_volume = volume[session == key]
            if len(day_volume) >= n:
                per_day.append(float(day_volume[:n].sum()))
        if per_day and sum(per_day) > 0 and traded > 0:
            rel_volume = round(traded / (sum(per_day) / len(per_day)), 2)

    avg_range = float((t_high - t_low)[-14:].mean()) / price * 100

    pivots: Dict[str, float] = {}
    cpr_width = None
    if pdh is not None and pdl is not None and pdc is not None:
        pivots = pivot_levels(pdh, pdl, pdc)
        cpr_width = (pivots["CPR_high"] - pivots["CPR_low"]) / price * 100

    gaps: List[FairValueGap] = []
    r_stamps, r_high, r_low = stamps[recent], high[recent], low[recent]
    for timeframe, source in (("5m", _gap_frame(r_stamps, r_high, r_low)),
                              ("15m", _fifteen_minute(r_stamps, r_high, r_low))):
        try:
            gaps.extend(detect_fair_value_gaps(
                source, price, timeframe=timeframe, min_size_pct=FVG_MIN_SIZE_PCT[timeframe], max_gaps=3,
            ))
        except Exception:  # noqa: BLE001 - an extra must never block the screen
            continue
    gaps.sort(key=lambda g: g.distance_pct)
    # Gaps to trade must be made by three of TODAY's candles: a "gap" that spans
    # the overnight jump is the opening gap, not a move made during the session.
    try:
        session_gaps = detect_fair_value_gaps(
            _gap_frame(t_stamps, t_high, t_low), price, timeframe="5m", min_size_pct=TRADE_GAP_MIN_PCT, max_gaps=5,
        )
    except Exception:  # noqa: BLE001
        session_gaps = []

    # Today's swing: from the day's low to its high (up) or the other way round.
    swing_up = swing_range = swing_retrace = swing_age = None
    fib_levels: Dict[str, float] = {}
    i_high, i_low = int(t_high.argmax()), int(t_low.argmin())
    top, bottom = float(t_high[i_high]), float(t_low[i_low])
    span = top - bottom
    if span > 0 and i_high != i_low:
        swing_up = i_low < i_high
        swing_range = span / price * 100
        swing_retrace = (top - price) / span if swing_up else (price - bottom) / span
        swing_age = n - 1 - (i_high if swing_up else i_low)
        fib_levels = {
            f"{ratio:g}": round(top - ratio * span if swing_up else bottom + ratio * span, 2) for ratio in FIB_RATIOS
        }

    return IntradayFeatures(
        ticker=ticker, price=price, as_of=f"{t_stamps[-1]:%H:%M}", bars_today=n,
        today_open=today_open, day_high=top, day_low=bottom,
        or_high=or_high, or_low=or_low, vwap=vwap, last_bar=t_stamps[-1],
        ema9=ema(close, 9), ema20=ema(close, 20), rsi=rsi(close), avg_range_pct=avg_range,
        pdh=pdh, pdl=pdl, pdc=pdc, gap_pct=gap, change_pct=change,
        pivots=pivots, cpr_width_pct=cpr_width, relative_volume=rel_volume,
        daily_trend_up=daily_trend_up, index_change_pct=index_change,
        relative_strength_pct=(change - index_change) if change is not None and index_change is not None else None,
        fair_value_gaps=gaps, session_gaps=session_gaps, fib_levels=fib_levels,
        swing_up=swing_up, swing_range_pct=swing_range, swing_retrace=swing_retrace, swing_age_bars=swing_age,
    )


# --------------------------------------------------------------------------- #
# Setups
# --------------------------------------------------------------------------- #
def _bracket(distance_pct: float, max_stop_pct: float, reward: float) -> Tuple[float, float]:
    """(stop %, target %): the stop where the idea is wrong, kept inside the allowed range."""
    stop = round(min(max(distance_pct, MIN_STOP_PCT), max_stop_pct), 2)
    return stop, round(stop * reward, 2)


def lines_up(f: IntradayFeatures, long: bool) -> Dict[str, bool]:
    """Which of the four basic conditions agree with a trade in this direction."""
    return {
        "trend": f.daily_trend_up is not None and f.daily_trend_up == long,
        "market": f.index_change_pct is not None and ((f.index_change_pct > 0) if long else (f.index_change_pct < 0)),
        "vwap": (f.price > f.vwap) if long else (f.price < f.vwap),
        "volume": f.relative_volume is not None and f.relative_volume >= VOLUME_FLOOR,
    }


def _confluence(f: IntradayFeatures, long: bool) -> Tuple[float, List[str]]:
    """Up to 40 points for things that line up with the trade, with a note for each."""
    ok = lines_up(f, long)
    notes = []
    if ok["trend"]:
        notes.append("with the daily trend")
    if ok["market"]:
        notes.append(f"with the Nifty ({f.index_change_pct:+.2f}%)")
    if ok["vwap"]:
        notes.append(("above" if long else "below") + " VWAP")
    if ok["volume"]:
        notes.append(f"volume {f.relative_volume:.1f}x normal")
    return 10.0 * sum(ok.values()), notes


def setup_row(s: IntradaySetup, scan: str) -> dict:
    """The facts about one setup that the learner uses, the same for live setups and backtest rows.

    ``scan`` is the time (HH:MM IST) at which the setup was seen.
    """
    f = s.features
    long = s.direction == "long"
    ok = lines_up(f, long) if f is not None else dict.fromkeys(("trend", "market", "vwap", "volume"), False)

    def num(value, digits=3):
        return None if value is None else round(float(value), digits)

    return {
        "ticker": s.ticker, "model": s.model, "direction": s.direction, "scan": scan, "score": s.score,
        "stop_pct": s.stop_pct, "target_pct": s.target_pct,
        "relative_volume": f.relative_volume if f is not None and f.relative_volume is not None else 0.0,
        "gap_pct": num(f.gap_pct) if f else None, "change_pct": num(f.change_pct) if f else None,
        "index_change_pct": num(f.index_change_pct) if f else None, "rsi": num(f.rsi, 1) if f else None,
        "swing_retrace": num(f.swing_retrace) if f else None,
        "f_trend": int(ok["trend"]), "f_market": int(ok["market"]), "f_vwap": int(ok["vwap"]),
        "f_volume": int(ok["volume"]), "f_bonus": int(s.bonus > 0),
    }


def _setup(f: IntradayFeatures, model: str, long: bool, distance_pct: float, level: Optional[Level],
           first_note: str, bonus: float, bonus_note: str, max_stop_pct: float, reward: float) -> IntradaySetup:
    points, notes = _confluence(f, long)
    stop_pct, target_pct = _bracket(distance_pct, max_stop_pct, reward)
    return IntradaySetup(
        ticker=f.ticker, model=model, direction="long" if long else "short",
        score=50 + points + bonus, price=round(f.price, 2), stop_pct=stop_pct, target_pct=target_pct,
        notes=[first_note] + ([bonus_note] if bonus else []) + notes, level=level, features=f, bonus=bonus,
    )


def breakout_setup(f: IntradayFeatures, *, max_stop_pct: float = 1.0, reward: float = 2.0) -> Optional[IntradaySetup]:
    """Price beyond the opening range: trade with the break, stop back at the middle of the range."""
    if f.direction is None:
        return None
    long = f.direction == "long"
    edge = f.or_high if long else f.or_low
    middle = (f.or_high + f.or_low) / 2
    distance = abs(f.price - middle) / f.price * 100
    fresh = f.extension_pct <= 0.5
    return _setup(
        f, "breakout", long, distance,
        Level("opening-range " + ("high" if long else "low"), edge, abs(f.price - edge) / f.price * 100),
        "broke the opening-range " + ("high" if long else "low"),
        10.0 if fresh else 0.0, "fresh break (not stretched)", max_stop_pct, reward,
    )


def fvg_setup(f: IntradayFeatures, *, max_stop_pct: float = 1.0, reward: float = 2.0) -> Optional[IntradaySetup]:
    """Price is back at an unfilled 5-minute fair value gap made earlier today.

    A bullish gap (price jumped up and left a hole) is a long setup once price
    has come back down to it; a bearish gap is the mirror image. The stop sits
    just beyond the far side of the gap: if price trades through the whole gap,
    the idea was wrong.
    """
    best: Optional[Tuple[float, FairValueGap, bool]] = None
    for gap in f.session_gaps:
        age = f.gap_age_bars(gap)
        if age is None or age < 1:              # the move that made the gap is still running
            continue
        long = gap.direction == "bullish"
        at_gap = gap.position == "inside" or (
            gap.position == ("below" if long else "above") and gap.distance_pct <= FVG_NEAR_PCT
        )
        if not at_gap:
            continue
        if best is None or gap.distance_pct < best[0]:
            best = (gap.distance_pct, gap, long)
    if best is None:
        return None
    _, gap, long = best
    far = gap.gap_low if long else gap.gap_high
    distance = abs(f.price - far) / f.price * 100 + STOP_BUFFER_PCT
    in_zone = (
        f.swing_up is not None and f.swing_up == long and f.swing_retrace is not None
        and FIB_ZONE[0] <= f.swing_retrace <= FIB_ZONE[1]
    )
    return _setup(
        f, "fvg", long, distance, Level(f"{gap.direction} 5m fair value gap", far, abs(f.price - far) / f.price * 100),
        f"back at a {gap.direction} fair value gap ({gap.gap_low:g}-{gap.gap_high:g}, {gap.filled_pct:.0f}% filled)",
        10.0 if in_zone else 0.0, "inside the Fibonacci pullback zone too", max_stop_pct, reward,
    )


def fib_setup(f: IntradayFeatures, *, max_stop_pct: float = 1.0, reward: float = 2.0) -> Optional[IntradaySetup]:
    """Price has given back 38.2-61.8% of today's swing: trade in the swing's direction."""
    if f.swing_up is None or f.swing_retrace is None or f.swing_range_pct is None or f.swing_age_bars is None:
        return None
    if f.swing_range_pct < MIN_SWING_PCT or f.swing_age_bars < MIN_SWING_AGE_BARS:
        return None
    if not FIB_ZONE[0] <= f.swing_retrace <= FIB_ZONE[1]:
        return None
    long = f.swing_up
    span = f.day_high - f.day_low
    invalid = f.day_high - FIB_INVALID * span if long else f.day_low + FIB_INVALID * span
    distance = abs(f.price - invalid) / f.price * 100 + STOP_BUFFER_PCT
    support = any(
        g.direction == ("bullish" if long else "bearish")
        and (g.position == "inside" or (g.position == ("below" if long else "above") and g.distance_pct <= 0.3))
        for g in f.session_gaps
    )
    return _setup(
        f, "fib", long, distance, Level("78.6% retracement", invalid, abs(f.price - invalid) / f.price * 100),
        f"gave back {f.swing_retrace * 100:.0f}% of today's {f.swing_range_pct:.1f}% "
        f"{'rise' if long else 'fall'}",
        10.0 if support else 0.0, "a fair value gap backs the pullback", max_stop_pct, reward,
    )


def gap_fade_setups(
    features: Sequence[IntradayFeatures], *, max_stop_pct: float = 1.0, reward: float = 2.0
) -> List[IntradaySetup]:
    """The day's biggest opening gap up (sell it) and biggest gap down (buy it), if big enough."""
    gapped = [f for f in features if f.gap_pct is not None and f.pdc]
    out: List[IntradaySetup] = []
    for long in (False, True):
        pool = [f for f in gapped if (f.gap_pct <= -MIN_GAP_PCT if long else f.gap_pct >= MIN_GAP_PCT)]
        if not pool:
            continue
        f = min(pool, key=lambda x: x.gap_pct) if long else max(pool, key=lambda x: x.gap_pct)
        out.append(_setup(
            f, "gap_fade", long, max_stop_pct, Level("yesterday's close", f.pdc, abs(f.price - f.pdc) / f.price * 100),
            f"biggest gap {'down' if long else 'up'} of the day ({f.gap_pct:+.2f}%)",
            10.0 if abs(f.gap_pct) >= 1.5 else 0.0, "a large gap (1.5% or more)", max_stop_pct, reward,
        ))
    return out


def all_setups(
    features: Sequence[IntradayFeatures], *, models: Sequence[str] = MODELS,
    max_stop_pct: float = 1.0, reward: float = 2.0,
) -> List[IntradaySetup]:
    """Every setup of every enabled model (a stock can appear under more than one model)."""
    unknown = set(models) - set(MODELS)
    if unknown:
        raise ValueError(f"Unknown intraday model(s): {sorted(unknown)}")
    out: List[IntradaySetup] = []
    single = {"breakout": breakout_setup, "fvg": fvg_setup, "fib": fib_setup}
    for name in models:
        if name == "gap_fade":
            out.extend(gap_fade_setups(features, max_stop_pct=max_stop_pct, reward=reward))
            continue
        for f in features:
            setup = single[name](f, max_stop_pct=max_stop_pct, reward=reward)
            if setup is not None:
                out.append(setup)
    return out


def rank_key(s: IntradaySetup) -> Tuple[float, float, float]:
    """Best first: the learned expected result (when rated), then the score, then relative volume."""
    expected = s.expected_pct if s.expected_pct is not None else float("-inf")
    return expected, s.score, (s.features.relative_volume or 0.0) if s.features else 0.0


def shortlist(setups: Sequence[IntradaySetup], *, per_model: int = 3, limit: int = 10) -> List[IntradaySetup]:
    """The best few of each model, one entry per stock, best first (see :func:`rank_key`)."""
    ranked = sorted(setups, key=rank_key, reverse=True)
    taken: Dict[str, int] = {}
    seen, out = set(), []
    for setup in ranked:
        if setup.ticker in seen or taken.get(setup.model, 0) >= per_model:
            continue
        seen.add(setup.ticker)
        taken[setup.model] = taken.get(setup.model, 0) + 1
        out.append(setup)
    return out[:limit]


# --------------------------------------------------------------------------- #
# Exit replay
# --------------------------------------------------------------------------- #
def _bars_after_entry(
    position: Position, bars: Optional[pd.DataFrame], day: date, square_off: str, tz
) -> pd.DataFrame:
    """The 5-minute bars that start after the entry and before the square-off time."""
    if bars is None or bars.empty:
        return pd.DataFrame()
    frame = bars.dropna(subset=["High", "Low", "Close"]).copy()
    frame.index = to_ist(frame.index, tz)
    cutoff = datetime.combine(day, _hhmm(square_off), tzinfo=frame.index.tz)
    frame = frame[(frame.index.date == day) & (frame.index < cutoff)]
    if position.opened_at:
        entered = pd.Timestamp(position.opened_at)
        entered = entered.tz_convert(frame.index.tz) if entered.tzinfo else entered.tz_localize(frame.index.tz)
        frame = frame[frame.index > entered]
    return frame


def _first_hit(position: Position, frame: pd.DataFrame) -> Optional[Tuple[float, str]]:
    """The first bar that touches the stop or the target (the stop wins a tie)."""
    short = position.side == "short"
    for stamp, bar in frame.iterrows():
        high, low = float(bar["High"]), float(bar["Low"])
        stop_hit = high >= position.stop_loss if short else low <= position.stop_loss
        target_hit = low <= position.take_profit if short else high >= position.take_profit
        if stop_hit:
            return position.stop_loss, f"stop-loss hit at {stamp:%H:%M}"
        if target_hit:
            return position.take_profit, f"target hit at {stamp:%H:%M}"
    return None


def check_exit(
    position: Position, bars: Optional[pd.DataFrame], today: date, square_off: str, tz
) -> Optional[Tuple[float, str]]:
    """(price, reason) if the stop or target has been touched so far today, else None."""
    frame = _bars_after_entry(position, bars, today, square_off, tz)
    return _first_hit(position, frame) if not frame.empty else None


def simulate_exit(
    position: Position,
    bars: Optional[pd.DataFrame],
    today: date,
    square_off: str,
    tz,
) -> Tuple[Optional[float], str]:
    """(exit price, reason) for a paper intraday position, replaying 5-minute bars.

    Only bars that start after the entry and before the square-off time count.
    Returns (None, reason) when there are no usable bars at all.
    """
    if bars is None or bars.empty:
        return None, "no price data"
    frame = _bars_after_entry(position, bars, today, square_off, tz)
    if frame.empty:
        return None, "no bars after the entry"
    hit = _first_hit(position, frame)
    if hit is not None:
        return hit
    return round(float(frame["Close"].iloc[-1]), 2), f"squared off at {square_off}"


__all__ = [
    "INDEX_TICKER",
    "MODELS",
    "MODEL_LABELS",
    "IntradayFeatures",
    "IntradaySetup",
    "Level",
    "all_setups",
    "breakout_setup",
    "check_exit",
    "compute_features",
    "daily_trend_up",
    "ema",
    "fib_setup",
    "fvg_setup",
    "gap_fade_setups",
    "index_change_pct",
    "lines_up",
    "pivot_levels",
    "rank_key",
    "rsi",
    "setup_row",
    "shortlist",
    "simulate_exit",
    "to_ist",
]
