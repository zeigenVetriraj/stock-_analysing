"""Tests for the intraday backtest: the trade replay, the statistics, the saved
data files and the no-look-ahead rule.

Synthetic bars only - NO network. Run with:  ``pytest -q``
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from src.automation.intraday import INDEX_TICKER
from src.automation.intraday_backtest import (
    BARS_PREFIX,
    DAILY_PREFIX,
    REPORT_FILE,
    SETUPS_FILE,
    STATS_FILE,
    coin_flip_rows,
    daily_pick,
    day_stats,
    headline,
    load_frames,
    market_backdrop,
    model_table,
    replay_trade,
    run,
    run_backtest,
    save_frames,
    split_halves,
    stats,
    summarise,
    verdict,
)
from src.automation.jobs import IST
from tests.test_automation import bars as daily_bars
from tests.test_intraday import OPENING, PULLBACK, candles, session

DAYS = [date(2026, 9, 28), date(2026, 9, 29), date(2026, 9, 30), date(2026, 10, 1)]   # Mon-Thu, two months
REST = [103.0 + i * 0.01 for i in range(60)]                                           # quiet until 15:20
UP = OPENING + [101.2, 101.5] + [101.8, 102.2, 102.6, 103.0, 103.9, 104.5] + [104.5] * 60     # breaks out, then runs
DOWN = OPENING + [99.2, 98.9] + [98.6, 98.2, 97.8, 97.4, 96.6, 96.0] + [96.0] * 60
FLAT = OPENING + [100.3] * 68
SCANS = ("09:55", "10:10")


def frame(per_day) -> pd.DataFrame:
    return pd.concat([session(day, closes, volume=1000.0 + 500 * i) for i, (day, closes) in enumerate(per_day)])


def pullback(day: date) -> pd.DataFrame:
    """The fair-value-gap pullback from the intraday tests, a rise to 102.6, then back to 100 by the close."""
    rise = [101.1 + 0.06 * i for i in range(1, 26)]
    fall = [102.6 - (2.6 / 37) * i for i in range(1, 38)]
    return pd.concat([session(day, OPENING), candles(day, PULLBACK, start="09:45"),
                      session(day, rise + fall, start="10:10", spread=0.05)])


def market():
    stocks = {
        "UP.NS": frame([(d, UP) for d in DAYS]),
        "DOWN.NS": frame([(d, DOWN) for d in DAYS]),
        "FLAT.NS": frame([(d, FLAT) for d in DAYS]),
        "PULL.NS": pd.concat([pullback(d) for d in DAYS]),
        INDEX_TICKER: frame([(d, [c * 250 for c in UP]) for d in DAYS]),
    }
    daily = {"UP.NS": daily_bars(end=DAYS[-1], step=0.5), "DOWN.NS": daily_bars(end=DAYS[-1], start=300, step=-0.5),
             "FLAT.NS": daily_bars(end=DAYS[-1]), "PULL.NS": daily_bars(end=DAYS[-1], step=0.5)}
    return stocks, daily


# --- the trade replay ------------------------------------------------------------- #

def path(*ohlc) -> pd.DataFrame:
    return pd.DataFrame(list(ohlc), columns=["Open", "High", "Low", "Close"])


@pytest.mark.parametrize(
    "long, bars, expected",
    [
        (True, path((100, 100.5, 99.8, 100.4), (100.4, 102.3, 100.2, 102.1)), (2.0, "target")),
        (True, path((100, 100.5, 98.9, 99.2)), (-1.0, "stop")),
        (True, path((100, 102.5, 98.5, 101.0)), (-1.0, "stop")),                 # both touched: the stop counts
        (True, path((100, 100.2, 99.9, 100.1), (98.0, 98.4, 97.5, 98.2)), (-2.0, "stop (gap)")),   # opens through the stop
        (True, path((100, 100.6, 99.7, 100.5), (100.5, 100.9, 100.3, 100.8)), (0.8, "square-off")),
        (False, path((100, 100.3, 97.9, 98.1)), (2.0, "target")),                # shorts mirror
        (False, path((100, 101.2, 99.8, 101.0)), (-1.0, "stop")),
    ],
)
def test_replay_trade(long, bars, expected):
    gross, why = replay_trade(long, 100.0, 1.0, 2.0, bars)
    assert (round(gross, 4), why) == expected


def test_replay_without_a_stop_or_target_holds_to_the_end():
    gross, why = replay_trade(True, 100.0, None, None, path((100, 105, 90, 101.5)))
    assert (round(gross, 4), why) == (1.5, "square-off")


# --- statistics ------------------------------------------------------------------- #

def test_stats_are_net_of_costs():
    s = stats([2.0, -1.0, 2.0, -1.0], cost_pct=0.1)
    assert (s["n"], s["win_rate"], s["avg"], s["total"]) == (4, 50.0, 0.4, 1.6)
    assert s["profit_factor"] == round(3.8 / 2.2, 2) and s["max_drawdown"] == 1.1
    assert stats([])["n"] == 0


def trades(*per_day):
    """Rows for day_stats: each argument is one day's list of gross returns."""
    return [{"day": f"2026-01-{d + 1:02d}", "ret": r} for d, returns in enumerate(per_day) for r in returns]


def test_uncertainty_is_measured_on_daily_averages():
    """Ten trades on one day are one observation, not ten."""
    one_day = day_stats(trades([1.0] * 10), cost_pct=0.0)
    assert (one_day["n"], one_day["days"], one_day["se"], one_day["t"]) == (10, 1, None, None)
    s = day_stats(trades([1.0, 3.0], [0.0], [2.0], [-2.0]), cost_pct=0.0)   # daily averages 2, 0, 2, -2
    assert (s["days"], s["avg"]) == (4, 0.8)
    assert s["se"] == pytest.approx(0.957, abs=0.001) and s["t"] == pytest.approx(0.52, abs=0.01)


def test_verdict_needs_enough_trades_and_both_halves():
    good, bad = {"avg": 0.3}, {"avg": -0.1}
    assert verdict({"n": 10, "t": 5.0, "avg": 0.5}, good, good) == "too few trades to say"
    assert verdict({"n": 100, "t": 3.0, "avg": 0.3}, good, good) == "made money in both halves"
    assert verdict({"n": 100, "t": 3.0, "avg": 0.3}, good, bad) == "small gain, could be luck"
    assert verdict({"n": 100, "t": 1.0, "avg": 0.1}, good, good) == "small gain, could be luck"
    assert verdict({"n": 100, "t": -1.0, "avg": -0.1}, bad, bad) == "no clear edge"
    assert verdict({"n": 100, "t": -3.0, "avg": -0.13}, bad, bad, 0.12) == "no edge: lost about the trading costs"
    assert verdict({"n": 100, "t": -3.0, "avg": -0.4}, bad, bad, 0.12) == "lost money"


def test_daily_pick_takes_the_top_setup_of_the_first_scan():
    rows = [
        {"day": "d1", "scan": "10:20", "ticker": "LATE", "model": "fib", "score": 100, "relative_volume": 1.0},
        {"day": "d1", "scan": "09:50", "ticker": "A", "model": "breakout", "score": 70, "relative_volume": 1.0},
        {"day": "d1", "scan": "09:50", "ticker": "B", "model": "fvg", "score": 90, "relative_volume": 1.0},
        {"day": "d2", "scan": "09:50", "ticker": "C", "model": "breakout", "score": 60, "relative_volume": 3.0},
        {"day": "d2", "scan": "09:50", "ticker": "D", "model": "breakout", "score": 60, "relative_volume": 1.0},
    ]
    assert [r["ticker"] for r in daily_pick(rows)] == ["B", "C"]                 # tie on d2: higher volume
    assert [r["ticker"] for r in daily_pick(rows, "breakout")] == ["A", "C"]
    assert [r["ticker"] for r in daily_pick(rows, "fib")] == ["LATE"]
    first, second = split_halves(rows)
    assert {r["day"] for r in first} == {"d1"} and {r["day"] for r in second} == {"d2"}


# --- saved data ------------------------------------------------------------------- #

def test_saved_bars_come_back_the_same_and_merge_without_duplicates(tmp_path):
    stocks, _ = market()
    first = {t: f[f.index.date <= DAYS[1]] for t, f in stocks.items()}
    files = save_frames(first, tmp_path, BARS_PREFIX, IST)
    assert [f.name for f in files] == ["bars_5m_2026-09.csv.gz"]
    before = files[0].read_bytes()
    save_frames(first, tmp_path, BARS_PREFIX, IST)
    assert files[0].read_bytes() == before                                       # reproducible: no git churn

    assert save_frames(first, tmp_path, BARS_PREFIX, IST) == []                  # nothing new: nothing rewritten
    save_frames(stocks, tmp_path, BARS_PREFIX, IST)                              # a later, overlapping download
    assert sorted(f.name for f in tmp_path.iterdir()) == ["bars_5m_2026-09.csv.gz", "bars_5m_2026-10.csv.gz"]
    loaded = load_frames(tmp_path, BARS_PREFIX)
    assert set(loaded) == set(stocks)
    back = loaded["UP.NS"]
    assert len(back) == len(stocks["UP.NS"]) and not back.index.has_duplicates
    assert list(back.index.tz_convert(IST)) == list(stocks["UP.NS"].index)
    assert back["Close"].tolist() == pytest.approx(stocks["UP.NS"]["Close"].tolist())
    assert load_frames(tmp_path, "nothing") == {}


def test_a_redownload_only_changes_the_last_few_days(tmp_path):
    """Yahoo revises bars slightly; finished history must stay as it was saved."""
    stocks, _ = market()
    up = {"UP.NS": stocks["UP.NS"]}
    save_frames(up, tmp_path, BARS_PREFIX, IST, refresh_days=2)
    revised = {"UP.NS": stocks["UP.NS"] * 1.01}                                  # every bar "revised" by 1%
    changed = save_frames(revised, tmp_path, BARS_PREFIX, IST, refresh_days=2)
    back = load_frames(tmp_path, BARS_PREFIX)["UP.NS"]
    close, days = back["Close"], back.index.tz_convert(IST).date
    original = stocks["UP.NS"]["Close"]
    assert close[days == DAYS[0]].tolist() == pytest.approx(original[original.index.date == DAYS[0]].tolist())
    assert close[days == DAYS[-1]].tolist() == pytest.approx((original[original.index.date == DAYS[-1]] * 1.01).tolist())
    assert [f.name for f in changed] == ["bars_5m_2026-09.csv.gz", "bars_5m_2026-10.csv.gz"] and len(back) == len(original)


def test_daily_bars_keep_their_dates_through_a_save(tmp_path):
    _, daily = market()
    save_frames(daily, tmp_path, DAILY_PREFIX, IST)
    back = load_frames(tmp_path, DAILY_PREFIX)["UP.NS"]
    assert [ts.date() for ts in back.index.tz_convert(IST)] == [ts.date() for ts in daily["UP.NS"].index]
    assert back.index[-1].tz_convert(IST).date() == DAYS[-1] and DAYS[-1] - timedelta(days=1) in {
        ts.date() for ts in back.index.tz_convert(IST)}


# --- the backtest ----------------------------------------------------------------- #

def test_backtest_finds_each_model_and_enters_at_the_next_bar():
    stocks, daily = market()
    rows = run_backtest(stocks, daily, tz=IST, scans=SCANS, warmup_days=1)
    assert {r["day"] for r in rows} == {d.isoformat() for d in DAYS[1:]} and "FLAT.NS" not in {r["ticker"] for r in rows}

    up = next(r for r in rows if r["ticker"] == "UP.NS" and r["scan"] == "09:55" and r["model"] == "breakout")
    assert up["direction"] == "long" and up["entry"] == 101.5 and up["first"] == 1          # open of the 09:55 bar
    assert (up["stop_pct"], up["target_pct"], up["ret"], up["why"]) == (1.0, 2.0, 2.0, "target")
    assert up["ret_hold"] == pytest.approx((104.5 / 101.5 - 1) * 100, abs=0.001) and up["index_change_pct"] is not None
    down = next(r for r in rows if r["ticker"] == "DOWN.NS" and r["model"] == "breakout")
    assert down["direction"] == "short" and down["ret"] == 2.0

    again = next(r for r in rows if r["ticker"] == "UP.NS" and r["scan"] == "10:10" and r["model"] == "breakout")
    assert again["first"] == 0                                    # still a breakout at the second scan: not counted twice

    gap = next(r for r in rows if r["ticker"] == "PULL.NS" and r["model"] == "fvg")
    assert (gap["scan"], gap["direction"], gap["first"], gap["entry"]) == ("10:10", "long", 1, 101.16)
    assert gap["stop_pct"] == pytest.approx(0.54, abs=0.01) and gap["why"] == "target"
    assert gap["ret"] == pytest.approx(2 * gap["stop_pct"])
    assert {r["model"] for r in rows if r["ticker"] == "PULL.NS" and r["scan"] == "10:10"} == {"breakout", "fvg", "fib"}

    assert {r["model"] for r in run_backtest(stocks, daily, tz=IST, scans=SCANS, warmup_days=1, models=["fib"])} == {"fib"}


def test_backtest_never_looks_ahead():
    """Changing the bars after the decision time must not change what was found or how it was scored."""
    stocks, daily = market()
    crashed = {t: f.copy() for t, f in stocks.items()}
    for f in crashed.values():                                                   # last day, from 10:00 on
        late = (f.index.date == DAYS[-1]) & (f.index.time >= pd.Timestamp("10:00").time())
        f.loc[late, ["Open", "High", "Low", "Close"]] = f.loc[late, ["Open", "High", "Low", "Close"]] * 0.9
    last = str(DAYS[-1])
    a = [r for r in run_backtest(stocks, daily, tz=IST, scans=("09:55",), warmup_days=1) if r["day"] == last]
    b = [r for r in run_backtest(crashed, daily, tz=IST, scans=("09:55",), warmup_days=1) if r["day"] == last]
    outcome = ("ret", "why", "ret_hold", "ret_stop_hold")
    assert a and [{k: v for k, v in r.items() if k not in outcome} for r in a] == [
        {k: v for k, v in r.items() if k not in outcome} for r in b]
    assert [r["ret"] for r in a] != [r["ret"] for r in b]


def test_the_coin_flip_is_the_same_every_time_and_takes_both_sides():
    stocks, _ = market()
    a = coin_flip_rows(stocks, tz=IST, scan="09:55", warmup_days=1)
    assert a == coin_flip_rows(stocks, tz=IST, scan="09:55", warmup_days=1)
    assert len(a) == 4 * 3 and {r["direction"] for r in a} == {"long", "short"}          # 4 stocks, 3 days
    up = [r for r in a if r["ticker"] == "UP.NS"]                                         # UP always rises 2%+
    assert all(r["ret"] == (2.0 if r["direction"] == "long" else -1.0) for r in up)


def test_an_unfinished_day_is_left_out():
    stocks, daily = market()
    cut = {t: f[~((f.index.date == DAYS[-1]) & (f.index.time >= pd.Timestamp("11:30").time()))] for t, f in stocks.items()}
    rows = run_backtest(cut, daily, tz=IST, scans=SCANS, warmup_days=1)
    assert rows and str(DAYS[-1]) not in {r["day"] for r in rows}


def test_report_and_files(tmp_path):
    stocks, daily = market()
    save_frames(stocks, tmp_path, BARS_PREFIX, IST)
    save_frames(daily, tmp_path, DAILY_PREFIX, IST)
    report, rows, flips = run(tmp_path, tz=IST, scans=SCANS)
    assert rows and (tmp_path / REPORT_FILE).read_text(encoding="utf-8") == report
    assert len(flips) == 4 * 2 and "coin flip at 09:55 (for comparison)" in report      # 4 stocks, 2 days after warm-up
    assert (tmp_path / SETUPS_FILE).read_text(encoding="utf-8").startswith("day,scan,ticker,model,direction")
    for text in ("Each kind of setup", "One trade a day", "By time of day", "higher score", "hold to 15:15",
                 "opening-range breakout", "fair value gap retest", "Fibonacci pullback", "gap reversal",
                 "The market in this period"):
        assert text in report
    table = model_table(rows)
    assert set(table) == {"breakout", "fvg", "fib", "gap_fade"}
    assert table["breakout"]["verdict"] == "too few trades to say"
    assert set(__import__("json").loads((tmp_path / STATS_FILE).read_text(encoding="utf-8"))) == set(table)
    assert market_backdrop(stocks, tz=IST)["up_days_pct"] == 100 and market_backdrop({}, tz=IST) is None
    text = headline(rows, coin_flip=flips)
    assert "opening-range breakout" in text and "One trade a day" in text and "a coin flip: 8 trades" in text
    assert "no setups" in headline([]) and "coin flip" not in headline(rows)
    assert "No setups found" in summarise([])
