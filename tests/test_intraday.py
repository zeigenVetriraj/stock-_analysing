"""Tests for the paper intraday job: the measurements, the four setup models
(breakout, fair value gap, Fibonacci, gap reversal), the bar-by-bar exit replay,
short positions and charges in the ledger, and the entry / square-off runs.

Fakes only - NO network, NO LLM, NO orders. Run with:  ``pytest -q``
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd
import pytest

from config.settings import INTRADAY_MODELS, IntradayRules
from src.analysis.analyst_agent import AnalystError
from src.automation.intraday import (
    MODELS,
    all_setups,
    breakout_setup,
    check_exit,
    compute_features,
    daily_trend_up,
    fib_setup,
    fvg_setup,
    gap_fade_setups,
    index_change_pct,
    pivot_levels,
    shortlist,
    simulate_exit,
)
from src.automation.jobs import IST, intraday_trading_rules, record_line, run_job
from src.automation.ledger import Ledger, Position, exit_reason, ledger_path
from src.automation import picker as picker_module
from src.automation.picker import PickDecision, parse_pick
from tests.test_automation import RULES, SHIPPED, TODAY, Recorder, bars, deps, settings

PREV = TODAY - timedelta(days=1)


def session(day: date, closes, *, start="09:15", volume=1000.0, spread=0.2) -> pd.DataFrame:
    """5-minute bars from ``start`` with the given closes (open = previous close)."""
    t0 = datetime.combine(day, datetime.strptime(start, "%H:%M").time(), tzinfo=IST)
    rows, index, prev = [], [], closes[0]
    for i, close in enumerate(closes):
        rows.append({"Open": prev, "High": max(prev, close) + spread, "Low": min(prev, close) - spread,
                     "Close": close, "Volume": volume})
        index.append(t0 + timedelta(minutes=5 * i))
        prev = close
    return pd.DataFrame(rows, index=pd.DatetimeIndex(index))


def two_days(today_closes, *, today_volume=2000.0, prev_level=100.0):
    earlier = session(PREV, [prev_level] * 75)
    today = session(TODAY, today_closes, volume=today_volume)
    return pd.concat([earlier, today])


OPENING = [100.0, 100.4, 100.1, 100.5, 100.2, 100.3]         # 09:15-09:40, range ~99.8-100.7


def candles(day: date, rows, *, start="09:15", volume=1000.0) -> pd.DataFrame:
    """5-minute bars from explicit (open, high, low, close) rows."""
    t0 = datetime.combine(day, datetime.strptime(start, "%H:%M").time(), tzinfo=IST)
    index = pd.DatetimeIndex([t0 + timedelta(minutes=5 * i) for i in range(len(rows))])
    frame = pd.DataFrame(list(rows), columns=["Open", "High", "Low", "Close"], index=index)
    frame["Volume"] = volume
    return frame


def features(closes, **kwargs):
    return compute_features("X.NS", two_days(closes), TODAY, IST, **kwargs)


# A jump that leaves a bullish fair value gap (100.6-101.2), then a pullback into it.
PULLBACK = [
    (100.3, 100.6, 100.2, 100.5),     # 09:45
    (100.5, 101.8, 100.5, 101.7),     # 09:50  the jump
    (101.7, 102.2, 101.2, 102.0),     # 09:55  its low (101.2) is above the 09:45 high (100.6): the gap
    (102.0, 102.1, 101.4, 101.5),     # 10:00
    (101.5, 101.6, 101.0, 101.1),     # 10:05  back inside the gap
]


def pullback_day(extra=PULLBACK, flip=False) -> pd.DataFrame:
    """Yesterday flat at 100, today's opening range, then ``extra`` (mirrored around 100 if ``flip``)."""
    today = pd.concat([session(TODAY, OPENING, volume=2000.0), candles(TODAY, extra, start="09:45", volume=2000.0)])
    if flip:
        mirrored = today.copy()
        mirrored["Open"], mirrored["Close"] = 200 - today["Open"], 200 - today["Close"]
        mirrored["High"], mirrored["Low"] = 200 - today["Low"], 200 - today["High"]
        today = mirrored
    return pd.concat([session(PREV, [100.0] * 75), today])


# --- measurements ------------------------------------------------------------------ #

def test_features_measure_the_session():
    f = features(OPENING + [101.2, 101.5], daily_trend_up=True, index_change=0.4)
    assert (f.price, f.as_of, f.bars_today) == (101.5, "09:50", 8)
    assert (round(f.or_high, 2), round(f.or_low, 2)) == (100.7, 99.8)
    assert (f.pdh, f.pdl, f.pdc) == (100.2, 99.8, 100.0) and f.gap_pct == 0.0
    assert f.change_pct == pytest.approx(1.5) and f.relative_strength_pct == pytest.approx(1.1)
    assert f.relative_volume == 2.0 and f.direction == "long"
    assert f.pivots["P"] == pytest.approx(100.0) and f.pivots["R1"] == pytest.approx(100.2)
    assert f.swing_up is True and f.swing_retrace == pytest.approx((101.7 - 101.5) / (101.7 - 99.8))
    assert f.swing_age_bars == 0
    assert features(OPENING[:4]) is None                                  # too early: no opening range yet


def test_pivot_levels_are_the_classic_floor_pivots():
    p = pivot_levels(110.0, 100.0, 105.0)
    assert (p["P"], p["R1"], p["S1"], p["R2"], p["S2"]) == (105.0, 110.0, 100.0, 115.0, 95.0)
    assert (p["CPR_low"], p["CPR_high"]) == (105.0, 105.0)
    p = pivot_levels(110.0, 100.0, 108.0)
    assert p["CPR_low"] == 105.0 and p["CPR_high"] == pytest.approx(107.0)


def test_daily_trend_ignores_todays_unfinished_bar():
    rising = bars(step=0.5)                                               # ends today
    assert daily_trend_up(rising, TODAY, IST) is True
    crash = rising.copy()
    crash.loc[crash.index[-1], "Close"] = 1.0                             # today's bar collapses: must not matter
    assert daily_trend_up(crash, TODAY, IST) is True
    assert daily_trend_up(bars(start=300, step=-0.5), TODAY, IST) is False
    assert daily_trend_up(rising.tail(30), TODAY, IST) is None            # under 50 earlier closes
    assert daily_trend_up(None, TODAY, IST) is None


def test_index_change_compares_with_yesterdays_close():
    nifty = two_days([100.0, 100.5, 101.0])
    assert index_change_pct(nifty, TODAY, IST) == pytest.approx(1.0)
    assert index_change_pct(session(TODAY, [100.0, 101.0]), TODAY, IST) is None     # no yesterday
    assert index_change_pct(None, TODAY, IST) is None


# --- the setup models -------------------------------------------------------------- #

def test_breakout_above_the_range_is_a_long_setup():
    setup = breakout_setup(features(OPENING + [101.2, 101.5], daily_trend_up=True, index_change=0.4))
    assert (setup.model, setup.direction, setup.price) == ("breakout", "long", 101.5)
    assert setup.notes[0] == "broke the opening-range high" and "with the daily trend" in setup.notes
    assert setup.score == 90                           # 50 + trend + Nifty + VWAP + volume (stretched: no bonus)
    assert (setup.stop_pct, setup.target_pct) == (1.0, 2.0)      # the middle of the range is further: capped
    assert setup.stop_price == 100.48 and setup.target_price == 103.53


def test_breakdown_below_the_range_is_a_short_setup():
    setup = breakout_setup(features(OPENING + [99.2, 98.9], daily_trend_up=False))
    assert setup.direction == "short" and "with the daily trend" in setup.notes
    assert setup.stop_price > setup.price > setup.target_price


def test_inside_the_range_is_no_breakout():
    assert breakout_setup(features(OPENING + [100.3])) is None


def test_a_tight_range_gives_a_tight_stop_but_never_below_the_floor():
    f = features([100.0, 100.05, 100.0, 100.05, 100.0, 100.05, 100.4])   # range 99.8-100.25, price just above
    setup = breakout_setup(f)
    assert setup.stop_pct == pytest.approx(0.37, abs=0.01) and setup.target_pct == pytest.approx(2 * setup.stop_pct)
    assert breakout_setup(f, max_stop_pct=0.2).stop_pct == 0.2            # the rules' cap always wins
    assert "fresh break (not stretched)" in setup.notes


def test_fair_value_gap_retest_is_a_long_setup_with_the_stop_beyond_the_gap():
    f = compute_features("X.NS", pullback_day(), TODAY, IST)
    gap = f.session_gaps[0]
    assert (gap.direction, gap.gap_low, gap.gap_high, gap.position) == ("bullish", 100.6, 101.2, "inside")
    assert f.gap_age_bars(gap) == 2
    setup = fvg_setup(f)
    assert (setup.model, setup.direction, setup.price) == ("fvg", "long", 101.1)
    assert setup.level.price == 100.6 and setup.stop_pct == pytest.approx(0.54, abs=0.01)
    assert setup.stop_price < 100.6 < setup.price and setup.target_pct == pytest.approx(2 * setup.stop_pct)
    assert "inside the Fibonacci pullback zone too" in setup.notes


def test_a_gap_that_is_still_being_made_is_not_a_retest():
    f = compute_features("X.NS", pullback_day(PULLBACK[:3]), TODAY, IST)   # the jump has only just happened
    assert f.session_gaps and fvg_setup(f) is None


def test_the_overnight_jump_is_not_a_fair_value_gap_to_trade():
    """Yesterday closed at 104, today opens at 100: three candles across the night leave a 'gap', but it is
    the opening gap, not a move made today."""
    frame = pd.concat([session(PREV, [104.0] * 75), session(TODAY, OPENING + [100.3, 100.4], volume=2000.0)])
    f = compute_features("X.NS", frame, TODAY, IST)
    assert any(g.direction == "bearish" for g in f.fair_value_gaps)        # still listed as context
    assert f.session_gaps == [] and fvg_setup(f) is None


def test_a_filled_gap_is_no_setup():
    through = PULLBACK[:4] + [(101.5, 101.6, 100.4, 100.5)]               # trades right through the gap
    f = compute_features("X.NS", pullback_day(through), TODAY, IST)
    assert fvg_setup(f) is None


def test_fibonacci_pullback_trades_with_the_swing():
    f = compute_features("X.NS", pullback_day(), TODAY, IST)
    assert f.swing_up and f.swing_retrace == pytest.approx((102.2 - 101.1) / (102.2 - 99.8))
    setup = fib_setup(f)
    assert (setup.model, setup.direction) == ("fib", "long")
    assert setup.level.price == pytest.approx(102.2 - 0.786 * 2.4)        # the 78.6% level
    assert setup.stop_pct == pytest.approx(0.83, abs=0.01) and setup.stop_price < setup.level.price
    assert setup.notes[0].startswith("gave back 46% of today's 2.4% rise")
    assert "a fair value gap backs the pullback" in setup.notes

    shallow = compute_features("X.NS", pullback_day(PULLBACK[:4]), TODAY, IST)      # only 29% given back
    assert fib_setup(shallow) is None
    small = features(OPENING + [100.3, 100.2])                                      # swing under 1%
    assert fib_setup(small) is None


def test_short_setups_mirror_the_long_ones():
    f = compute_features("X.NS", pullback_day(flip=True), TODAY, IST)
    gap, fib = fvg_setup(f), fib_setup(f)
    assert gap.direction == fib.direction == "short"
    assert gap.level.price == pytest.approx(99.4) and gap.stop_price > 99.4 > gap.price
    assert fib.stop_price > fib.level.price > fib.price > fib.target_price


def gapped(ticker, open_level):
    closes = [open_level + d for d in (0.0, 0.1, -0.1, 0.2, 0.0, 0.1, 0.05)]
    return compute_features(ticker, two_days(closes), TODAY, IST)


def test_gap_reversal_takes_only_the_biggest_gap_each_way():
    stocks = [gapped("UP2.NS", 102.0), gapped("UP1.NS", 101.0), gapped("FLAT.NS", 100.2),
              gapped("DN1.NS", 99.2), gapped("DN3.NS", 97.0)]
    setups = gap_fade_setups(stocks)
    assert [(s.ticker, s.direction) for s in setups] == [("UP2.NS", "short"), ("DN3.NS", "long")]
    short = setups[0]
    assert short.notes[0] == "biggest gap up of the day (+2.00%)" and "a large gap (1.5% or more)" in short.notes
    assert (short.stop_pct, short.target_pct) == (1.0, 2.0) and short.level.name == "yesterday's close"
    assert gap_fade_setups([gapped("FLAT.NS", 100.2)]) == []              # under 0.5%: nothing to fade


def test_models_can_be_switched_off_and_unknown_ones_are_refused():
    stocks = [compute_features("PULL.NS", pullback_day(), TODAY, IST), gapped("UP2.NS", 102.0)]
    found = {(s.ticker, s.model) for s in all_setups(stocks)}
    assert found == {("PULL.NS", "breakout"), ("PULL.NS", "fvg"), ("PULL.NS", "fib"), ("UP2.NS", "gap_fade")}
    assert {s.model for s in all_setups(stocks, models=["fib"])} == {"fib"}
    with pytest.raises(ValueError):
        all_setups(stocks, models=["astrology"])
    assert MODELS == INTRADAY_MODELS == SHIPPED.intraday.models and SHIPPED.intraday.cost_pct == 0.12
    with pytest.raises(ValueError):
        IntradayRules(max_position_pct=0.1, stop_loss_pct=0.01, take_profit_pct=0.02, models=["astrology"])


def test_shortlist_keeps_one_entry_per_stock_and_a_few_per_model():
    many = [compute_features(f"B{i}.NS", two_days(OPENING + [101.2, 101.5 + i / 10]), TODAY, IST) for i in range(5)]
    stocks = many + [compute_features("PULL.NS", pullback_day(), TODAY, IST)]
    picked = shortlist(all_setups(stocks), per_model=3, limit=10)
    assert len({s.ticker for s in picked}) == len(picked)                 # PULL.NS appears once, not three times
    assert sum(1 for s in picked if s.model == "breakout") == 3
    assert [s.score for s in picked] == sorted((s.score for s in picked), reverse=True)
    assert len(shortlist(all_setups(stocks), limit=2)) == 2
    prompt = picked[0].as_prompt()
    assert {"model", "setup", "stop_price", "target_price", "rests_on", "fair_value_gaps", "todays_swing",
            "pivots", "market"} <= set(prompt)


# --- exit replay ------------------------------------------------------------------ #

def trade(side="long", entry=100.0, stop=99.0, target=102.0) -> Position:
    return Position(symbol="X.NS", strategy="intraday", side=side, quantity=10, entry_price=entry,
                    stop_loss=stop, take_profit=target, opened_on=TODAY.isoformat(), max_hold_days=1,
                    opened_at=datetime.combine(TODAY, datetime.min.time().replace(hour=9, minute=50),
                                               tzinfo=IST).isoformat())


def afternoon(closes, start="09:55", spread=0.1):
    return session(TODAY, closes, start=start, spread=spread)


@pytest.mark.parametrize(
    "side, closes, expected",
    [
        ("long", [100.2, 101.0, 102.5], (102.0, "target hit at 10:05")),
        ("long", [99.5, 98.5], (99.0, "stop-loss hit at 10:00")),
        ("long", [100.4, 101.8, 98.8], (99.0, "stop-loss hit at 10:05")),   # stop after a near miss
        ("short", [99.5, 97.5], (98.0, "target hit at 10:00")),             # short target below
        ("short", [100.5, 101.5], (101.0, "stop-loss hit at 10:00")),       # short stop above
    ],
)
def test_exit_replay(side, closes, expected):
    stop, target = (99.0, 102.0) if side == "long" else (101.0, 98.0)
    assert simulate_exit(trade(side, stop=stop, target=target), afternoon(closes), TODAY, "15:15", IST) == expected


def test_a_bar_touching_both_levels_counts_as_the_stop():
    frame = afternoon([100.0])
    frame.loc[:, ["High", "Low"]] = [[102.5, 98.5]]
    assert simulate_exit(trade(), frame, TODAY, "15:15", IST) == (99.0, "stop-loss hit at 09:55")


def test_square_off_uses_the_last_bar_before_the_cutoff():
    frame = pd.concat([afternoon([100.5] * 3, start="15:00"), afternoon([150.0], start="15:20")])
    price, reason = simulate_exit(trade(), frame, TODAY, "15:15", IST)
    assert (price, reason) == (100.5, "squared off at 15:15")


# --- ledger and picker ------------------------------------------------------------ #

def test_short_positions_make_money_when_price_falls():
    ledger = Ledger(mode="paper", starting_cash=10_000, cash=10_000)
    short = trade("short", stop=101.0, target=98.0)
    ledger.open(short)
    assert ledger.cash == 9_000 and ledger.equity({"X.NS": 99.0}) == 10_010
    assert exit_reason(short, 101.2, TODAY) == "stop-loss hit" and exit_reason(short, 97.9, TODAY) == "target hit"
    closed = ledger.close(short, 98.0, TODAY, "target hit")
    assert (closed.pnl, closed.pnl_pct, ledger.cash) == (20.0, 2.0, 10_020)


def test_charges_come_off_a_paper_intraday_trade():
    ledger = Ledger(mode="paper", starting_cash=10_000, cash=10_000)
    position = trade("long").model_copy(update={"setup": "fib"})
    ledger.open(position)
    closed = ledger.close(position, 102.0, TODAY, "target hit", cost_pct=0.12)     # +20 gross on a 1,000 position
    assert (closed.charges, closed.pnl, closed.pnl_pct, closed.setup) == (1.2, 18.8, 1.88, "fib")
    assert ledger.cash == 10_018.8 and ledger.realised_pnl() == 18.8
    old = Ledger.model_validate({"mode": "paper", "closed": [{                    # ledgers saved before charges existed
        "symbol": "A.NS", "strategy": "intraday", "quantity": 1, "entry_price": 1, "exit_price": 2,
        "opened_on": "2026-01-01", "closed_on": "2026-01-01", "reason": "r", "pnl": 1, "pnl_pct": 100}]})
    assert old.closed[0].charges == 0 and old.closed[0].setup == ""


def test_only_the_intraday_picker_may_sell():
    text = '{"action": "SELL", "ticker": "TCS.NS", "target_price": 3000, "reasoning": "r"}'
    with pytest.raises(AnalystError):
        parse_pick(text, ["TCS.NS"])
    assert parse_pick(text, ["TCS.NS"], allow_sell=True).action == "SELL"
    rules = intraday_trading_rules(RULES)
    assert rules.allow_short_selling and rules.max_position_pct == 0.10 and rules.stop_loss_pct == 0.01


def test_the_practice_prompt_changes_only_the_rules(monkeypatch):
    normal, practice = picker_module.INTRADAY_SYSTEM_PROMPT, picker_module.INTRADAY_PRACTICE_PROMPT
    assert "HOLD is a good answer" in normal and "Be sceptical" in normal
    assert "PAPER PRACTICE" in practice and "HOLD is a good answer" not in practice and "Be sceptical" not in practice
    assert practice.startswith(normal.split("Rules you MUST follow:")[0])
    assert practice.endswith(normal.split("Output format (CRITICAL):")[1])

    seen = {}

    def fake_request(client, *, model, system, user_content, parse, label):
        seen.update(system=system, user=user_content)
        return parse('{"action": "SELL", "ticker": "BBB.NS", "target_price": 97, "reasoning": "r"}')

    monkeypatch.setattr(picker_module, "request_json_decision", fake_request)
    agent = picker_module.PickerAgent(settings(), client=object())
    setups = [{"ticker": "BBB.NS", "setup": "short"}]
    assert agent.pick_intraday(setups, practice=True).action == "SELL"
    assert seen["system"] == practice and "paper practice account" in seen["user"]
    agent.pick_intraday(setups)
    assert seen["system"] == normal and "or HOLD" in seen["user"]


# --- the runs ----------------------------------------------------------------------- #

class IntradayPicker:
    def __init__(self, decision):
        self.decision, self.seen, self.practice = decision, None, None

    def __call__(self, _settings):
        return self

    def pick_intraday(self, setups, practice=False):
        self.seen, self.practice = setups, practice
        return self.decision


INDEX = "^NSEI"
NINE_FIFTY = datetime.combine(TODAY, datetime.min.time().replace(hour=9, minute=50), tzinfo=IST)
FIVE_MIN = {"AAA.NS": two_days(OPENING + [101.2, 101.5]), "BBB.NS": two_days(OPENING + [99.2, 98.9]),
            "CCC.NS": two_days(OPENING + [100.3])}
DAILY = {"AAA.NS": bars(step=0.5), "BBB.NS": bars(start=300, step=-0.5), "CCC.NS": bars()}


def intraday_deps(tmp_path, picker, *, now=NINE_FIFTY, five_min=FIVE_MIN, s=None, rules=RULES):
    return deps(tmp_path, history=DAILY, picker=picker, s=s, now=now, rules=rules,
                intraday_history=lambda tickers: {t: f for t, f in five_min.items() if t in tickers})


def test_entry_run_opens_one_paper_short(tmp_path):
    picker = IntradayPicker(PickDecision(action="SELL", ticker="BBB.NS", target_price=97.0, reasoning="clean breakdown"))
    d = intraday_deps(tmp_path, picker)
    outcome = run_job("intraday-open", d)

    assert sorted(s["ticker"] for s in picker.seen) == ["AAA.NS", "BBB.NS"]    # CCC is inside its range
    assert outcome.traded and outcome.opened.side == "short" and outcome.opened.setup == "breakout"
    assert outcome.opened.quantity == int(1_000_000 * 0.10 // 98.9)
    assert outcome.opened.stop_loss > 98.9 > outcome.opened.take_profit
    text = d.notifier.messages[0]
    assert "Paper-shorted" in text and "opening-range breakout" in text and "Why it qualified" in text

    again = run_job("intraday-open", intraday_deps(tmp_path, picker))     # one trade a day
    assert not again.traded and "limit is 1" in again.report.render()


def test_entry_uses_the_setups_own_stop_and_target(tmp_path):
    """A fair value gap retest: the stop sits just beyond the gap, far tighter than the 1% cap."""
    five_min = {"AAA.NS": pullback_day(), INDEX: two_days([100.0, 100.2, 100.4])}
    picker = IntradayPicker(PickDecision(action="BUY", ticker="AAA.NS", target_price=102.0, reasoning="gap holds"))
    now = datetime.combine(TODAY, datetime.min.time().replace(hour=10, minute=8), tzinfo=IST)
    rules = RULES.model_copy(update={"intraday": RULES.intraday.model_copy(update={"models": ("fvg",)})})
    d = intraday_deps(tmp_path, picker, now=now, five_min=five_min, rules=rules)
    outcome = run_job("intraday-open", d)

    seen = picker.seen[0]
    assert (seen["model"], seen["setup"], seen["market"]["nifty_change_pct"]) == ("fvg", "long", 0.4)
    opened = outcome.opened
    assert opened.setup == "fvg" and opened.entry_price == 101.1
    assert opened.stop_loss == pytest.approx(101.1 * (1 - seen["stop_pct"] / 100), abs=0.01) and opened.stop_loss < 100.6
    assert opened.take_profit == pytest.approx(101.1 * (1 + 2 * seen["stop_pct"] / 100), abs=0.01)
    assert "fair value gap retest" in d.notifier.messages[0]


def test_the_backtest_record_reaches_claude_and_the_message(tmp_path):
    record = {"breakout": {"label": "opening-range breakout", "trades": 1611, "win_rate": 39.0,
                           "avg_net_pct": -0.106, "verdict": "no edge: lost about the trading costs"}}
    picker = IntradayPicker(PickDecision(action="SELL", ticker="BBB.NS", target_price=97.0, reasoning="clean"))
    d = intraday_deps(tmp_path, picker)
    d.track_record = lambda: record
    run_job("intraday-open", d)
    assert picker.seen[0]["backtest"] == {"trades": 1611, "win_rate": 39.0, "avg_net_pct": -0.106,
                                          "verdict": "no edge: lost about the trading costs"}
    assert ("Backtest record of this kind of setup: 1611 trades, 39% winners, -0.11% a trade after costs "
            "(no edge: lost about the trading costs).") in d.notifier.messages[0]
    assert record_line(None) == "" and record_line({}) == ""


def test_practice_mode_takes_the_best_setup_every_day(tmp_path):
    """The shipped paper account practises: Claude is asked to trade the best setup and may hold only on a
    clear problem. Switched off, the job asks for clean setups only."""
    assert SHIPPED.intraday.practice is True and RULES.intraday.practice is True
    picker = IntradayPicker(PickDecision(action="SELL", ticker="BBB.NS", target_price=97.0, reasoning="best one"))
    d = intraday_deps(tmp_path, picker)
    assert run_job("intraday-open", d).traded and picker.practice is True
    assert "Practice trade: the paper account takes the best setup every day" in d.notifier.messages[0]

    (tmp_path / "hold").mkdir()
    held = IntradayPicker(PickDecision(action="HOLD", reasoning="every price is stale"))
    d = intraday_deps(tmp_path / "hold", held)
    run_job("intraday-open", d)
    assert "practice mode trades every day, but Claude saw a clear problem" in d.notifier.messages[0]

    (tmp_path / "off").mkdir()
    off = RULES.model_copy(update={"intraday": RULES.intraday.model_copy(update={"practice": False})})
    choosy = IntradayPicker(PickDecision(action="HOLD", reasoning="nothing clean"))
    d = intraday_deps(tmp_path / "off", choosy, rules=off)
    run_job("intraday-open", d)
    assert choosy.practice is False and "HOLD - no clean setup right now." in d.notifier.messages[0]


def test_stale_prices_block_the_entry(tmp_path):
    picker = IntradayPicker(PickDecision(action="BUY", ticker="AAA.NS", target_price=104, reasoning="r"))
    late = NINE_FIFTY.replace(hour=10, minute=45)                         # the last bar is 55 minutes old
    outcome = run_job("intraday-open", intraday_deps(tmp_path, picker, now=late))
    assert not outcome.traded and picker.seen is None and "Prices are stale" in outcome.report.render()


def test_no_entry_outside_the_window(tmp_path):
    picker = IntradayPicker(PickDecision(action="BUY", ticker="AAA.NS", target_price=104, reasoning="r"))
    outcome = run_job("intraday-open", intraday_deps(tmp_path, picker, now=NINE_FIFTY.replace(hour=12)))
    assert not outcome.traded and picker.seen is None
    assert "Outside the entry window" in outcome.report.render()


def test_square_off_run_closes_at_the_target(tmp_path):
    ledger = Ledger(mode="paper", starting_cash=1_000_000, cash=1_000_000)
    ledger.open(trade("long", entry=100.0, stop=99.0, target=102.0).model_copy(update={"symbol": "AAA.NS"}))
    ledger.save(ledger_path("paper", tmp_path))
    later = {"AAA.NS": afternoon([100.5, 101.4, 102.3, 101.0])}
    outcome = run_job("intraday-close", intraday_deps(tmp_path, IntradayPicker(None), five_min=later))
    assert [(t.exit_price, t.reason) for t in outcome.closed] == [(102.0, "target hit at 10:05")]
    assert (outcome.closed[0].charges, outcome.closed[0].pnl) == (1.2, 18.8)        # 0.12% of the 1,000 position
    text = outcome.report.render()
    assert "Intraday so far: 1 trade(s), 1 winner(s)" in text and "est. charges" in text


def test_live_mode_never_trades_intraday(tmp_path):
    picker = IntradayPicker(PickDecision(action="BUY", ticker="AAA.NS", target_price=104, reasoning="r"))
    s = settings(AUTOTRADE_MODE="live", DHAN_ENV="live", DHAN_ENABLE_LIVE_TRADING="true",
                 DHAN_CLIENT_ID="1", DHAN_ACCESS_TOKEN="t")
    outcome = run_job("intraday-open", intraday_deps(tmp_path, picker, s=s))
    assert not outcome.traded and picker.seen is None and "paper mode only" in outcome.report.render()


def test_daily_job_leaves_todays_intraday_trade_alone(tmp_path):
    ledger = Ledger(mode="paper", starting_cash=1_000_000, cash=1_000_000)
    ledger.open(trade("long", entry=500.0, stop=499.0, target=502.0).model_copy(update={"symbol": "AAA.NS"}))
    ledger.save(ledger_path("paper", tmp_path))
    outcome = run_job("daily", deps(tmp_path, history={"AAA.NS": bars(step=0.5)}))
    assert outcome.closed == []
    assert Ledger.load(ledger_path("paper", tmp_path), mode="paper", starting_cash=0).count("intraday") == 1


def test_notifier_gets_a_message_even_when_nothing_happens(tmp_path):
    d = intraday_deps(tmp_path, IntradayPicker(None))
    run_job("intraday-close", d)
    assert isinstance(d.notifier, Recorder) and "No open intraday position" in d.notifier.messages[0]


# --- hourly checks -------------------------------------------------------------- #

def _seed_open_trade(tmp_path) -> None:
    ledger = Ledger(mode="paper", starting_cash=1_000_000, cash=1_000_000)
    ledger.open(trade("long", entry=100.0, stop=99.0, target=102.0).model_copy(update={"symbol": "AAA.NS"}))
    ledger.save(ledger_path("paper", tmp_path))


def test_check_exit_only_reports_a_touch():
    assert check_exit(trade(), afternoon([100.5, 101.0]), TODAY, "15:15", IST) is None
    assert check_exit(trade(), afternoon([100.5, 102.4]), TODAY, "15:15", IST) == (102.0, "target hit at 10:00")


def test_hourly_check_closes_and_alerts_when_the_target_is_hit(tmp_path):
    _seed_open_trade(tmp_path)
    d = intraday_deps(tmp_path, IntradayPicker(None), five_min={"AAA.NS": afternoon([100.5, 101.4, 102.3])})
    outcome = run_job("intraday-check", d)
    assert [t.reason for t in outcome.closed] == ["target hit at 10:05"]
    assert len(d.notifier.messages) == 1 and "Intraday alert" in d.notifier.messages[0]

    # The 15:10 square-off then reports it as closed earlier, with the running total.
    d2 = intraday_deps(tmp_path, IntradayPicker(None), five_min={})
    text = run_job("intraday-close", d2).report.render()
    assert "(closed earlier today)" in text and "No open intraday position" not in text
    assert "Intraday so far: 1 trade(s), 1 winner(s)" in text


def test_hourly_check_is_quiet_when_nothing_happened(tmp_path):
    _seed_open_trade(tmp_path)
    d = intraday_deps(tmp_path, IntradayPicker(None), five_min={"AAA.NS": afternoon([100.5, 101.0])})
    outcome = run_job("intraday-check", d)
    assert outcome.closed == [] and d.notifier.messages == []
    assert Ledger.load(ledger_path("paper", tmp_path), mode="paper", starting_cash=0).count("intraday") == 1

    d_empty = intraday_deps(tmp_path / "none", IntradayPicker(None))
    run_job("intraday-check", d_empty)
    assert d_empty.notifier.messages == []
