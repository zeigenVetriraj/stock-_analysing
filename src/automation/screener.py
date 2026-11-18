"""Deterministic screening (no AI): score many stocks, keep the best few.

Claude is only asked to compare a short list, so the heavy lifting of narrowing
50 (or a few hundred penny) stocks down to ``top_n`` happens here, in plain
Python that is covered by tests.

Technical score (0-100) favours a long swing entry: price above a rising
trend (SMA50 over SMA200), RSI with room to run (not overbought), positive but
not stretched 20-day momentum, price near its 52-week high and rising volume.

Fundamental score (0-100, 50 when unknown) rewards profitability (ROE), low
debt and growing revenue / operating profit, and penalises losses and very
rich valuations.

The combined score is 60% technical, 40% fundamental.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import pandas as pd

from src.ingestion.fundamentals import FundamentalSnapshot

logger = logging.getLogger(__name__)

TECHNICAL_WEIGHT = 0.6
FUNDAMENTAL_WEIGHT = 0.4
MIN_SESSIONS = 60
RS_PER_CRORE = 1e7


@dataclass
class TechnicalScore:
    ticker: str
    price: float
    score: float
    rsi: Optional[float] = None
    sma50: Optional[float] = None
    sma200: Optional[float] = None
    return_20d_pct: Optional[float] = None
    pct_below_high: Optional[float] = None
    avg_turnover_cr: Optional[float] = None
    notes: List[str] = field(default_factory=list)


@dataclass
class RankedStock:
    ticker: str
    price: float
    technical: float
    fundamental: float
    combined: float
    notes: List[str]
    fundamentals: Optional[FundamentalSnapshot] = None


def _rsi(close: pd.Series, period: int = 14) -> Optional[float]:
    if len(close) <= period:
        return None
    delta = close.diff()
    gain = delta.clip(lower=0.0).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    loss = (-delta.clip(upper=0.0)).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    last_gain, last_loss = float(gain.iloc[-1]), float(loss.iloc[-1])
    if last_loss == 0:
        return 100.0 if last_gain > 0 else None
    return round(100 - 100 / (1 + last_gain / last_loss), 2)


def score_technicals(ticker: str, daily: Optional[pd.DataFrame]) -> Optional[TechnicalScore]:
    """Score one stock's daily bars for a long swing entry; None if too little data."""
    if daily is None or daily.empty or "Close" not in daily:
        return None
    bars = daily.dropna(subset=["Close"])
    close = bars["Close"].astype(float)
    if len(close) < MIN_SESSIONS:
        return None

    price = float(close.iloc[-1])
    sma50 = float(close.tail(50).mean())
    sma200 = float(close.tail(200).mean()) if len(close) >= 200 else None
    rsi = _rsi(close)
    ret20 = (price / float(close.iloc[-21]) - 1) * 100 if len(close) > 21 else None
    high = float(bars["High"].tail(250).max()) if "High" in bars else float(close.tail(250).max())
    below_high = (high - price) / high * 100 if high > 0 else None

    turnover = None
    volume_rising = False
    if "Volume" in bars:
        vol = bars["Volume"].astype(float)
        turnover = float((close * vol).tail(20).mean()) / RS_PER_CRORE
        if len(vol) >= 50 and vol.tail(50).mean() > 0:
            volume_rising = vol.tail(10).mean() > vol.tail(50).mean()

    score, notes = 0.0, []
    if price > sma50:
        score += 20
        notes.append("above SMA50")
    if sma200 is not None and sma50 > sma200:
        score += 20
        notes.append("SMA50 over SMA200")
    if rsi is not None:
        if 45 <= rsi <= 65:
            score += 20
            notes.append(f"RSI {rsi:.0f}")
        elif 65 < rsi <= 72:
            score += 10
            notes.append(f"RSI {rsi:.0f} (warm)")
        elif 35 <= rsi < 45:
            score += 8
        elif rsi > 72:
            notes.append(f"RSI {rsi:.0f} overbought")
    if ret20 is not None:
        if 0 < ret20 <= 15:
            score += 15
        elif 15 < ret20 <= 30:
            score += 8
            notes.append(f"+{ret20:.0f}% in 20 days (stretched)")
    if below_high is not None:
        if below_high <= 10:
            score += 15
            notes.append("near 52-week high")
        elif below_high <= 20:
            score += 8
    if volume_rising:
        score += 10
        notes.append("volume rising")

    return TechnicalScore(
        ticker=ticker, price=round(price, 2), score=score, rsi=rsi,
        sma50=round(sma50, 2), sma200=round(sma200, 2) if sma200 else None,
        return_20d_pct=round(ret20, 2) if ret20 is not None else None,
        pct_below_high=round(below_high, 2) if below_high is not None else None,
        avg_turnover_cr=round(turnover, 2) if turnover is not None else None,
        notes=notes,
    )


def score_fundamentals(f: Optional[FundamentalSnapshot]) -> float:
    """0-100; 50 when nothing is known."""
    if f is None:
        return 50.0
    score = 50.0
    if f.return_on_equity_pct is not None:
        if f.return_on_equity_pct >= 15:
            score += 15
        elif f.return_on_equity_pct < 8:
            score -= 10
    if f.debt_to_equity is not None and f.sector != "Financial Services":
        if f.debt_to_equity <= 1:
            score += 10
        elif f.debt_to_equity > 2:
            score -= 15
    if f.revenue_growth_yoy_pct is not None:
        if f.revenue_growth_yoy_pct >= 10:
            score += 10
        elif f.revenue_growth_yoy_pct < 0:
            score -= 10
    if f.operating_profit_growth_yoy_pct is not None:
        if f.operating_profit_growth_yoy_pct >= 10:
            score += 10
        elif f.operating_profit_growth_yoy_pct < -10:
            score -= 10
    if any(flag.startswith("loss-making") for flag in f.flags):
        score -= 30
    if f.trailing_pe is not None and f.trailing_pe > 60:
        score -= 10
    return max(0.0, min(100.0, score))


def rank(
    technicals: Iterable[TechnicalScore],
    fundamentals: Dict[str, Optional[FundamentalSnapshot]],
    top_n: int,
) -> List[RankedStock]:
    ranked = []
    for t in technicals:
        f = fundamentals.get(t.ticker)
        fscore = score_fundamentals(f)
        ranked.append(
            RankedStock(
                ticker=t.ticker, price=t.price, technical=t.score, fundamental=fscore,
                combined=round(TECHNICAL_WEIGHT * t.score + FUNDAMENTAL_WEIGHT * fscore, 1),
                notes=t.notes, fundamentals=f,
            )
        )
    ranked.sort(key=lambda r: r.combined, reverse=True)
    return ranked[:top_n]


def bulk_daily_history(tickers: List[str], period: str = "1y", chunk: int = 40) -> Dict[str, pd.DataFrame]:
    """Daily bars for many Yahoo tickers at once (missing tickers are skipped)."""
    return bulk_history(tickers, period=period, interval="1d", chunk=chunk)


def bulk_intraday_history(tickers: List[str], period: str = "5d", chunk: int = 40) -> Dict[str, pd.DataFrame]:
    """5-minute bars for the last few sessions (for the intraday job)."""
    return bulk_history(tickers, period=period, interval="5m", chunk=chunk)


def bulk_history(tickers: List[str], *, period: str, interval: str, chunk: int = 40) -> Dict[str, pd.DataFrame]:
    """Bars for many Yahoo tickers at once (missing tickers are skipped)."""
    import yfinance as yf

    out: Dict[str, pd.DataFrame] = {}
    for start in range(0, len(tickers), chunk):
        batch = tickers[start:start + chunk]
        try:
            data = yf.download(
                batch, period=period, interval=interval, group_by="ticker",
                auto_adjust=True, threads=True, progress=False,
            )
        except Exception as exc:  # noqa: BLE001 - skip the batch, keep going
            logger.warning("Price download failed for %d tickers (%s).", len(batch), exc)
            continue
        if data is None or data.empty:
            continue
        for ticker in batch:
            try:
                if isinstance(data.columns, pd.MultiIndex):
                    if ticker not in data.columns.get_level_values(0):
                        continue
                    frame = data[ticker]
                else:
                    frame = data
                frame = frame.dropna(how="all")
            except Exception:  # noqa: BLE001
                continue
            if not frame.empty:
                out[ticker] = frame
    missing = sorted(set(tickers) - set(out))
    if missing:
        logger.info("No price data for %d ticker(s): %s", len(missing), ", ".join(missing[:15]))
    return out


__all__ = [
    "RankedStock",
    "TechnicalScore",
    "bulk_daily_history",
    "bulk_history",
    "bulk_intraday_history",
    "rank",
    "score_fundamentals",
    "score_technicals",
]
