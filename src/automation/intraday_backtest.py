"""Backtest of the intraday setup models on saved 5-minute bars (no AI).

It answers one question with numbers: did each kind of setup the bot can trade
(opening-range breakout, fair value gap retest, Fibonacci pullback, gap
reversal) make or lose money on past data, after costs?

For every trading day it stands at each scan time (09:50, 10:20, 10:50, 11:20),
rebuilds exactly what the live job would have seen (only bars that started
before that time), finds the setups with the same code the live job uses,
enters at the next bar's open and replays the rest of the day with the setup's
own stop and target. A bar that touches both the stop and the target counts as
the stop; a bar that opens beyond a level fills at the open. Results are net of
a round-trip cost.

Claude's pick cannot be replayed cheaply, so "one trade a day" uses the
top-ranked setup as a stand-in. Yahoo only serves about 60 days of 5-minute
bars; each run saves what it downloads, so the history grows over time.

    python -m src.automation.intraday_backtest --fetch            # download, save, report
    python -m src.automation.intraday_backtest --data state/backtests   # re-run on saved data
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import zlib
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from src.automation.intraday import (
    INDEX_TICKER,
    MODEL_LABELS,
    MODELS,
    all_setups,
    compute_features,
    daily_trend_up,
    index_change_pct,
    setup_row,
    to_ist,
)

logger = logging.getLogger(__name__)

DEFAULT_COST_PCT = 0.12          # round trip, % of position (brokerage, STT, exchange fees, GST, stamp, slippage)
SCAN_TIMES = ("09:50", "10:20", "10:50", "11:20")
DEFAULT_SQUARE_OFF = "15:15"
MAX_SANE_GAP_PCT = 15.0          # a bigger overnight "gap" is a split or bad data
BARS_REFRESH_DAYS = 7            # a re-download may replace saved 5-minute bars only in its last week
BARS_PREFIX, DAILY_PREFIX = "bars_5m", "daily"   # one gzipped CSV per calendar month
REPORT_FILE, SETUPS_FILE, STATS_FILE = "intraday_backtest.md", "intraday_setups.csv", "intraday_models.json"
PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


# --------------------------------------------------------------------------- #
# Replay
# --------------------------------------------------------------------------- #
def replay_trade(
    long: bool, entry: float, stop_pct: Optional[float], target_pct: Optional[float], bars: pd.DataFrame
) -> Tuple[float, str]:
    """(gross return %, reason) for one trade over ``bars`` (entry bar first)."""
    sign = 1.0 if long else -1.0
    stop = entry * (1 - sign * stop_pct / 100) if stop_pct else None
    target = entry * (1 + sign * target_pct / 100) if target_pct else None
    values = bars[["Open", "High", "Low", "Close"]].to_numpy(dtype=float)
    for opened, high, low, _ in values:
        if stop is not None:
            if (opened <= stop) if long else (opened >= stop):
                return (opened / entry - 1) * 100 * sign, "stop (gap)"
            if (low <= stop) if long else (high >= stop):
                return -stop_pct, "stop"
        if target is not None:
            if (opened >= target) if long else (opened <= target):
                return (opened / entry - 1) * 100 * sign, "target (gap)"
            if (high >= target) if long else (low <= target):
                return target_pct, "target"
    return (float(values[-1][3]) / entry - 1) * 100 * sign, "square-off"


def _clock(day: date, hhmm: str, tz) -> datetime:
    hours, minutes = (int(x) for x in hhmm.split(":"))
    return datetime(day.year, day.month, day.day, hours, minutes, tzinfo=tz)


def _clean(bars: Dict[str, Optional[pd.DataFrame]], tz) -> Dict[str, pd.DataFrame]:
    frames: Dict[str, pd.DataFrame] = {}
    for ticker, frame in bars.items():
        if frame is None or frame.empty:
            continue
        clean = frame.dropna(subset=["Open", "High", "Low", "Close"]).copy()
        clean.index = to_ist(clean.index, tz)
        frames[ticker] = clean.sort_index()
    return frames


def run_backtest(
    bars: Dict[str, pd.DataFrame],
    daily: Dict[str, pd.DataFrame],
    *,
    tz,
    scans: Sequence[str] = SCAN_TIMES,
    square_off: str = DEFAULT_SQUARE_OFF,
    models: Sequence[str] = MODELS,
    max_stop_pct: float = 1.0,
    reward: float = 2.0,
    warmup_days: int = 2,
    days: Optional[Sequence[date]] = None,
) -> List[dict]:
    """One row per setup found: (day, scan time, stock, model) with its outcome.

    ``first`` marks the first scan of the day at which that stock showed that
    model, so a setup that is still there half an hour later is not counted twice.
    ``days``: replay only these days (the earlier bars are still used as history);
    by default every day after the first ``warmup_days``.
    """
    frames = _clean(bars, tz)
    index_frame = frames.pop(INDEX_TICKER, None)
    all_days = sorted({d for frame in frames.values() for d in set(frame.index.date)})
    wanted = all_days[warmup_days:] if days is None else [d for d in all_days if d in set(days)]

    rows: List[dict] = []
    for day in wanted:
        close_at = _clock(day, square_off, tz)
        finished = close_at - timedelta(minutes=30)       # the day's data must reach at least this far
        trends = {ticker: daily_trend_up(daily.get(ticker), day, tz) for ticker in frames}
        seen: set = set()
        for scan in scans:
            decide_at = _clock(day, scan, tz)
            history_from = decide_at - timedelta(days=12)
            market = None
            if index_frame is not None:
                lo, hi = index_frame.index.searchsorted(history_from), index_frame.index.searchsorted(decide_at)
                market = index_change_pct(index_frame.iloc[lo:hi], day, tz)
            features, later = [], {}
            for ticker, frame in frames.items():
                start, now, end = (frame.index.searchsorted(t) for t in (history_from, decide_at, close_at))
                after = frame.iloc[now:end]                   # bars from the decision time to the square-off
                if after.empty or after.index[-1] < finished or float(after["Open"].iloc[0]) <= 0:
                    continue                                  # holiday, half day or today's unfinished session
                known = frame.iloc[start:now]                 # everything the live job would have seen
                f = compute_features(ticker, known, day, tz, daily_trend_up=trends[ticker], index_change=market)
                if f is None or (f.gap_pct is not None and abs(f.gap_pct) > MAX_SANE_GAP_PCT):
                    continue
                features.append(f)
                later[ticker] = after
            for s in all_setups(features, models=models, max_stop_pct=max_stop_pct, reward=reward):
                after = later[s.ticker]
                entry = float(after["Open"].iloc[0])
                long = s.direction == "long"
                gross, why = replay_trade(long, entry, s.stop_pct, s.target_pct, after)
                hold, _ = replay_trade(long, entry, None, None, after)
                stop_hold, _ = replay_trade(long, entry, s.stop_pct, None, after)
                key = (s.ticker, s.model)
                row = {"day": day.isoformat(), "scan": scan, **setup_row(s, scan), "first": int(key not in seen),
                       "entry": round(entry, 2), "ret": round(gross, 4), "why": why,
                       "ret_hold": round(hold, 4), "ret_stop_hold": round(stop_hold, 4),
                       "notes": "; ".join(s.notes)}
                rows.append(row)
                seen.add(key)
    return rows


def coin_flip_rows(
    bars: Dict[str, pd.DataFrame], *, tz, scan: str = SCAN_TIMES[0], square_off: str = DEFAULT_SQUARE_OFF,
    max_stop_pct: float = 1.0, reward: float = 2.0, warmup_days: int = 2,
) -> List[dict]:
    """What no skill at all looks like: every stock, every day, a random side at the first scan.

    The side comes from a fixed hash of the day and the ticker, so the result is
    the same on every run. A model has to beat this to be worth anything.
    """
    frames = _clean(bars, tz)
    frames.pop(INDEX_TICKER, None)
    days = sorted({d for frame in frames.values() for d in set(frame.index.date)})
    rows: List[dict] = []
    for day in days[warmup_days:]:
        decide_at, close_at = _clock(day, scan, tz), _clock(day, square_off, tz)
        for ticker, frame in frames.items():
            now, end = frame.index.searchsorted(decide_at), frame.index.searchsorted(close_at)
            after = frame.iloc[now:end]
            if after.empty or after.index[-1] < close_at - timedelta(minutes=30) or after.index[0] != decide_at:
                continue
            entry = float(after["Open"].iloc[0])
            if entry <= 0:
                continue
            long = zlib.crc32(f"{day.isoformat()}|{ticker}".encode()) % 2 == 0
            gross, why = replay_trade(long, entry, max_stop_pct, max_stop_pct * reward, after)
            rows.append({"day": day.isoformat(), "scan": scan, "ticker": ticker, "model": "coin_flip",
                         "direction": "long" if long else "short", "ret": round(gross, 4), "why": why})
    return rows


def market_backdrop(bars: Dict[str, pd.DataFrame], *, tz, scan: str = SCAN_TIMES[0],
                    square_off: str = DEFAULT_SQUARE_OFF) -> Optional[dict]:
    """What the Nifty itself did over the period (explains why longs or shorts had it easier)."""
    frame = _clean({INDEX_TICKER: bars.get(INDEX_TICKER)}, tz).get(INDEX_TICKER)
    if frame is None or frame.empty:
        return None
    moves = []
    for day, session in frame.groupby(frame.index.date):
        after = session[(session.index >= _clock(day, scan, tz)) & (session.index < _clock(day, square_off, tz))]
        if len(after) < 30:
            continue
        moves.append((float(after["Close"].iloc[-1]) / float(after["Open"].iloc[0]) - 1) * 100)
    if not moves:
        return None
    closes = frame["Close"]
    return {
        "days": len(moves),
        "period_change_pct": round((float(closes.iloc[-1]) / float(closes.iloc[0]) - 1) * 100, 1),
        "avg_intraday_pct": round(sum(moves) / len(moves), 3),
        "up_days_pct": round(sum(1 for m in moves if m > 0) / len(moves) * 100),
    }


# --------------------------------------------------------------------------- #
# Statistics and the report
# --------------------------------------------------------------------------- #
def stats(returns: Iterable[float], cost_pct: float = DEFAULT_COST_PCT) -> dict:
    """Summary of a list of gross returns (% of position), net of ``cost_pct`` each."""
    net = [r - cost_pct for r in returns]
    if not net:
        return {"n": 0, "win_rate": 0.0, "avg": 0.0, "total": 0.0, "profit_factor": None, "max_drawdown": 0.0}
    wins = [r for r in net if r > 0]
    losses = [-r for r in net if r < 0]
    equity = peak = drawdown = 0.0
    for r in net:
        equity += r
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)
    return {
        "n": len(net),
        "win_rate": round(len(wins) / len(net) * 100, 1),
        "avg": round(sum(net) / len(net), 3),
        "total": round(sum(net), 2),
        "profit_factor": round(sum(wins) / sum(losses), 2) if losses else None,
        "max_drawdown": round(drawdown, 2),
    }


def day_stats(rows: Sequence[dict], cost_pct: float = DEFAULT_COST_PCT, column: str = "ret") -> dict:
    """Like :func:`stats`, plus how sure we can be.

    Trades on the same day move together (the whole market rises or falls), so
    the uncertainty is measured on daily averages: ``se`` is the standard error
    of the average day and ``t`` = average / se. Roughly, |t| below 2 means the
    result cannot be told apart from luck.
    """
    out = stats([r[column] for r in rows], cost_pct)
    by_day: Dict[str, List[float]] = {}
    for r in rows:
        by_day.setdefault(r["day"], []).append(r[column] - cost_pct)
    daily = [sum(v) / len(v) for v in by_day.values()]
    out["days"] = len(daily)
    out["se"] = out["t"] = None
    if len(daily) >= 3:
        mean = sum(daily) / len(daily)
        variance = sum((x - mean) ** 2 for x in daily) / (len(daily) - 1)
        if variance > 0:
            out["se"] = round(math.sqrt(variance / len(daily)), 3)
            out["t"] = round(mean / math.sqrt(variance / len(daily)), 2)
    return out


def verdict(s: dict, first_half: dict, second_half: dict, cost_pct: float = DEFAULT_COST_PCT) -> str:
    """Plain-words reading of a model's numbers (``s`` is net of costs)."""
    if s["n"] < 30 or s.get("t") is None:
        return "too few trades to say"
    if s["t"] <= -2:
        # An average loss no bigger than the costs means the trades themselves were a coin flip.
        return "no edge: lost about the trading costs" if s["avg"] >= -1.5 * cost_pct else "lost money"
    if s["t"] >= 2 and first_half["avg"] > 0 and second_half["avg"] > 0:
        return "made money in both halves"
    if s["avg"] > 0:
        return "small gain, could be luck"
    return "no clear edge"


def split_halves(rows: Sequence[dict]) -> Tuple[List[dict], List[dict]]:
    days = sorted({r["day"] for r in rows})
    if len(days) < 2:
        return list(rows), []
    middle = days[len(days) // 2]
    return [r for r in rows if r["day"] < middle], [r for r in rows if r["day"] >= middle]


def daily_pick(rows: Sequence[dict], model: Optional[str] = None) -> List[dict]:
    """One trade a day: the top-ranked setup at the first scan of the day that has one."""
    by_day: Dict[str, List[dict]] = {}
    for r in rows:
        if model is None or r["model"] == model:
            by_day.setdefault(r["day"], []).append(r)
    picked = []
    for day in sorted(by_day):
        first_scan = min(r["scan"] for r in by_day[day])
        candidates = [r for r in by_day[day] if r["scan"] == first_scan]
        picked.append(max(candidates, key=lambda r: (r["score"], r["relative_volume"])))
    return picked


def _record(label: str, mine: Sequence[dict], cost_pct: float) -> dict:
    a, b = split_halves(mine)
    s = day_stats(mine, cost_pct)
    ha, hb = stats([r["ret"] for r in a], cost_pct), stats([r["ret"] for r in b], cost_pct)
    longs = stats([r["ret"] for r in mine if r["direction"] == "long"], cost_pct)
    shorts = stats([r["ret"] for r in mine if r["direction"] == "short"], cost_pct)
    return {
        "label": label, "trades": s["n"], "days": s["days"], "win_rate": s["win_rate"],
        "avg_net_pct": s["avg"], "se": s["se"], "t": s["t"],
        "first_half_avg": ha["avg"], "second_half_avg": hb["avg"],
        "long_trades": longs["n"], "long_avg": longs["avg"], "short_trades": shorts["n"], "short_avg": shorts["avg"],
        "verdict": verdict(s, ha, hb, cost_pct),
    }


def model_table(rows: Sequence[dict], cost_pct: float = DEFAULT_COST_PCT) -> Dict[str, dict]:
    """Per-model statistics on each stock's first appearance of the day."""
    out: Dict[str, dict] = {}
    for model in MODELS:
        mine = [r for r in rows if r["model"] == model and r["first"]]
        if mine:
            out[model] = _record(MODEL_LABELS[model], mine, cost_pct)
    return out


def _table_line(m: dict, reading: Optional[str] = None) -> str:
    return (
        f"| {m['label']} | {m['trades']} | {m['win_rate']:.0f}% | {m['avg_net_pct']:+.3f}% | {_pm(m['se'])} | "
        f"{m['first_half_avg']:+.3f}% | {m['second_half_avg']:+.3f}% | "
        f"{m['long_avg']:+.3f}% ({m['long_trades']}) | {m['short_avg']:+.3f}% ({m['short_trades']}) | "
        f"{reading or m['verdict']} |"
    )


def _pm(value: Optional[float]) -> str:
    return "-" if value is None else f"±{value:.2f}"


def summarise(
    rows: Sequence[dict], *, market: Optional[dict] = None, coin_flip: Optional[Sequence[dict]] = None,
    cost_pct: float = DEFAULT_COST_PCT, scans: Sequence[str] = SCAN_TIMES, title: str = "Intraday backtest",
) -> str:
    """The markdown report."""
    if not rows:
        return f"# {title}\n\nNo setups found - not enough data.\n"
    days = sorted({r["day"] for r in rows})
    tickers = {r["ticker"] for r in rows}
    out = [
        f"# {title}",
        "",
        f"- Period: {days[0]} to {days[-1]} ({len(days)} trading days), {len(tickers)} stocks, "
        f"scans at {', '.join(scans)} IST.",
        f"- Entry at the next 5-minute bar's open, the setup's own stop and target, everything closed at "
        f"{DEFAULT_SQUARE_OFF}.",
        f"- Returns are % of the position, after {cost_pct:.2f}% round-trip costs. "
        "A 10% position turns -0.10% into -0.01% on the account.",
    ]
    if market:
        out.append(
            f"- The market in this period: Nifty {market['period_change_pct']:+.1f}%; from 09:50 to 15:15 it moved "
            f"{market['avg_intraday_pct']:+.3f}% on an average day and rose on {market['up_days_pct']}% of days. "
            "A falling market flatters short trades and hurts long ones (and the other way round)."
        )
    out += [
        "- How to read \"±\": it is the uncertainty of the average. If the average is smaller than about twice "
        "the ±, it cannot be told apart from luck.",
        "",
        "## 1. Each kind of setup (every stock, first time it appears in the day)",
        "",
        "| setup | trades | winners | avg per trade | ± | first half | second half | longs | shorts | reading |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    table = model_table(rows, cost_pct)
    for m in table.values():
        out.append(_table_line(m))
    if coin_flip:
        out.append(_table_line(_record(f"coin flip at {scans[0]} (for comparison)", coin_flip, cost_pct),
                               "what no skill looks like"))
        out += ["", "The coin flip buys or sells every stock at random with the widest stop and target. "
                    "A setup is only worth trading if it clearly beats that line."]

    out += ["", "## 2. One trade a day (the top-ranked setup at the first scan that has one)", "",
            "| | trades | winners | avg per trade | ± | total | worst run |", "|---|---|---|---|---|---|---|"]
    for label, picked in [("All setups together (what the bot shortlists)", daily_pick(rows))] + [
        (f"Only {MODEL_LABELS[m]}", daily_pick(rows, m)) for m in table
    ]:
        s = day_stats(picked, cost_pct)
        out.append(f"| {label} | {s['n']} | {s['win_rate']:.0f}% | {s['avg']:+.3f}% | {_pm(s['se'])} | "
                   f"{s['total']:+.1f}% | -{s['max_drawdown']:.1f}% |")

    out += ["", "## 3. By time of day (all setups, first appearance)", "",
            "| scan | trades | winners | avg per trade | ± |", "|---|---|---|---|---|"]
    for scan in scans:
        s = day_stats([r for r in rows if r["scan"] == scan and r["first"]], cost_pct)
        if s["n"]:
            out.append(f"| {scan} | {s['n']} | {s['win_rate']:.0f}% | {s['avg']:+.3f}% | {_pm(s['se'])} |")

    out += ["", "## 4. Does a higher score mean a better trade? (all setups, first appearance)", "",
            "| score | trades | winners | avg per trade | ± |", "|---|---|---|---|---|"]
    for low, high in ((50, 70), (70, 85), (85, 101)):
        s = day_stats([r for r in rows if r["first"] and low <= r["score"] < high], cost_pct)
        if s["n"]:
            out.append(f"| {low}-{min(high, 100)} | {s['n']} | {s['win_rate']:.0f}% | {s['avg']:+.3f}% | {_pm(s['se'])} |")

    out += ["", "## 5. Stop and target, or just hold to 15:15? (all setups, first appearance)", "",
            "| exit | trades | winners | avg per trade | ± |", "|---|---|---|---|---|"]
    firsts = [r for r in rows if r["first"]]
    for label, column in (("The setup's stop and target", "ret"), ("No stop, hold to 15:15", "ret_hold")):
        s = day_stats(firsts, cost_pct, column)
        out.append(f"| {label} | {s['n']} | {s['win_rate']:.0f}% | {s['avg']:+.3f}% | {_pm(s['se'])} |")
    out.append("")
    return "\n".join(out)


def headline(rows: Sequence[dict], cost_pct: float = DEFAULT_COST_PCT,
             coin_flip: Optional[Sequence[dict]] = None) -> str:
    """A few lines for the Telegram message."""
    if not rows:
        return "*Intraday backtest*: no setups found (no data?)."
    days = sorted({r["day"] for r in rows})
    out = [f"*Intraday backtest* - {days[0]} to {days[-1]} ({len(days)} trading days)",
           f"Average result per trade after {cost_pct:.2f}% costs:"]
    for m in model_table(rows, cost_pct).values():
        out.append(f"- {m['label']}: {m['trades']} trades, {m['win_rate']:.0f}% winners, "
                   f"{m['avg_net_pct']:+.2f}% a trade - {m['verdict']}")
    if coin_flip:
        flip = stats([r["ret"] for r in coin_flip], cost_pct)
        out.append(f"- for comparison, a coin flip: {flip['n']} trades, {flip['win_rate']:.0f}% winners, "
                   f"{flip['avg']:+.2f}% a trade")
    pick = day_stats(daily_pick(rows), cost_pct)
    out.append(f"One trade a day (top-ranked setup): {pick['n']} trades, {pick['win_rate']:.0f}% winners, "
               f"{pick['avg']:+.2f}% a trade.")
    out.append("(% of the position; a 10% position turns -0.10% into -0.01% on the account.)")
    out.append(f"Full report: state/backtests/{REPORT_FILE}")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Data in and out
# --------------------------------------------------------------------------- #
def frames_to_table(frames: Dict[str, pd.DataFrame], tz) -> pd.DataFrame:
    """Long table (ticker, ts in UTC, OHLCV) from a {ticker: bars} dict."""
    parts = []
    for ticker in sorted(frames):
        frame = frames[ticker]
        if frame is None or frame.empty or not {"Open", "High", "Low", "Close"} <= set(frame.columns):
            continue
        part = frame[[c for c in PRICE_COLUMNS if c in frame.columns]].dropna(
            subset=["Open", "High", "Low", "Close"]).copy()
        if part.empty:
            continue
        if "Volume" not in part:
            part["Volume"] = 0
        part["Volume"] = part["Volume"].fillna(0).astype("int64")
        stamps = to_ist(part.index, tz).tz_convert("UTC")
        part = part.reset_index(drop=True)
        part.insert(0, "ts", stamps.strftime("%Y-%m-%dT%H:%M:%SZ"))
        part.insert(0, "ticker", ticker)
        parts.append(part)
    if not parts:
        return pd.DataFrame(columns=["ticker", "ts", *PRICE_COLUMNS])
    return pd.concat(parts, ignore_index=True)


def save_frames(frames: Dict[str, pd.DataFrame], folder: Path, prefix: str, tz, *,
                refresh_days: Optional[int] = None) -> List[Path]:
    """Save bars as ``<prefix>_<YYYY-MM>.csv.gz``, merging with what is already there.

    Yahoo only keeps about 60 days of 5-minute bars, so each run adds the new
    days to the months already saved and the history grows over time.

    ``refresh_days``: a re-download replaces saved bars only in its last that-many
    days (to complete a session that was still running, or a bar that was not
    finished); older saved bars are kept as they are, so finished history stops
    changing and the files are only rewritten when something is new. None = the
    newest download always wins (used for the small daily file).
    """
    table = frames_to_table(frames, tz)
    if table.empty:
        return []
    folder.mkdir(parents=True, exist_ok=True)
    fresh_from = ""
    if refresh_days is not None:
        newest = pd.to_datetime(table["ts"], utc=True).max()
        fresh_from = (newest - pd.Timedelta(days=refresh_days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    months = pd.to_datetime(table["ts"], utc=True).dt.tz_convert(tz).dt.strftime("%Y-%m")
    written = []
    for month, part in table.groupby(months.to_numpy()):
        path = folder / f"{prefix}_{month}.csv.gz"
        before = path.read_bytes() if path.exists() else None
        if before is not None:
            saved = pd.read_csv(path)
            settled = saved[saved["ts"] < fresh_from]                     # saved bars that must not change
            part = pd.concat([saved[saved["ts"] >= fresh_from], part, settled], ignore_index=True)
        part = part.drop_duplicates(["ticker", "ts"], keep="last").sort_values(["ticker", "ts"])
        part.to_csv(path, index=False, float_format="%.4f", compression={"method": "gzip", "mtime": 0})
        if path.read_bytes() != before:
            written.append(path)
    return written


def load_frames(folder: Path, prefix: str) -> Dict[str, pd.DataFrame]:
    """Every saved month of ``prefix`` as {ticker: bars indexed by UTC time}."""
    files = sorted(folder.glob(f"{prefix}_*.csv.gz"))
    if not files:
        return {}
    table = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    table = table.drop_duplicates(["ticker", "ts"], keep="last")
    table["ts"] = pd.to_datetime(table["ts"], utc=True)
    return {
        str(ticker): group.drop(columns=["ticker"]).set_index("ts").sort_index()
        for ticker, group in table.groupby("ticker")
    }


def write_setups(rows: Sequence[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: List[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def fetch_data(tickers: List[str]) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """5-minute bars (about 60 days, Yahoo's limit) and a year of daily bars."""
    from src.automation.screener import bulk_daily_history, bulk_history

    wanted = list(dict.fromkeys(tickers + [INDEX_TICKER]))
    bars: Dict[str, pd.DataFrame] = {}
    for period in ("60d", "59d", "1mo"):
        bars = bulk_history(wanted, period=period, interval="5m")
        if len(bars) >= max(1, len(wanted) // 2):
            break
        logger.warning("Only %d of %d tickers for period %s; trying a shorter one.", len(bars), len(wanted), period)
    return bars, bulk_daily_history(tickers)


def run(
    out_dir: Path, *, tz, tickers: Optional[List[str]] = None, fetch: bool = False,
    cost_pct: float = DEFAULT_COST_PCT, models: Sequence[str] = MODELS,
    max_stop_pct: float = 1.0, reward: float = 2.0, scans: Sequence[str] = SCAN_TIMES,
) -> Tuple[str, List[dict], List[dict]]:
    """Fetch (and save) fresh data if asked, run the backtest on everything saved, write the report.

    Returns (report, setup rows, coin-flip rows).
    """
    if fetch:
        bars, daily = fetch_data(tickers or [])
        save_frames(bars, out_dir, BARS_PREFIX, tz, refresh_days=BARS_REFRESH_DAYS)
        save_frames(daily, out_dir, DAILY_PREFIX, tz)
    bars, daily = load_frames(out_dir, BARS_PREFIX), load_frames(out_dir, DAILY_PREFIX)
    rows = run_backtest(bars, daily, tz=tz, scans=scans, models=models, max_stop_pct=max_stop_pct, reward=reward)
    flips = coin_flip_rows(bars, tz=tz, scan=scans[0], max_stop_pct=max_stop_pct, reward=reward)
    report = summarise(rows, market=market_backdrop(bars, tz=tz), coin_flip=flips, cost_pct=cost_pct, scans=scans)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / REPORT_FILE).write_text(report, encoding="utf-8")
    (out_dir / STATS_FILE).write_text(json.dumps(model_table(rows, cost_pct), indent=2) + "\n", encoding="utf-8")
    write_setups(rows, out_dir / SETUPS_FILE)
    return report, rows, flips


def main(argv=None) -> int:
    from src.automation.jobs import IST
    from src.automation.ledger import STATE_DIR
    from src.automation.universe import load_universe
    from src.ingestion.data_fetcher import to_yfinance_ticker

    parser = argparse.ArgumentParser(description="Backtest the intraday setup models on 5-minute bars.")
    parser.add_argument("--fetch", action="store_true", help="Download fresh data from Yahoo and save it.")
    parser.add_argument("--data", default=str(STATE_DIR / "backtests"), help="Folder for the data and the report.")
    parser.add_argument("--cost", type=float, default=DEFAULT_COST_PCT, help="Round-trip cost, %% of position.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
    for noisy in ("yfinance", "urllib3", "peewee"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    tickers = [to_yfinance_ticker(s) for s in load_universe().large_caps]
    report, rows, _ = run(Path(args.data), tz=IST, tickers=tickers, fetch=args.fetch, cost_pct=args.cost)
    print(report)
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
