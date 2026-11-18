"""Tests for the futures intraday book: lots, margin accounting, NSE lot sizes, its own folder.

Fakes only - NO network, NO LLM, NO orders. Run with:  ``pytest -q``
"""

from __future__ import annotations

import json

import pytest

from src.automation.futures_book import (
    current_lot_sizes,
    parse_lot_sizes,
    run_futures_backtest,
    run_futures_intraday,
    size_futures,
)
from src.automation.intraday import IntradaySetup, compute_features
from src.automation.jobs import IST
from src.automation.ledger import Ledger, Position
from src.automation.picker import PickDecision
from src.risk.guardrails import RiskViolation
from tests.test_automation import RULES, SHIPPED, TODAY, deps
from tests.test_intraday import DAILY, IntradayPicker
from tests.test_journal import AT, FULL, MORNING

FR = SHIPPED.futures_intraday
# Nifty is priced from ^NSEI (also the market index), Bank Nifty from ^NSEBANK.
MORNING_IDX = {"^NSEI": MORNING["AAA.NS"], "^NSEBANK": MORNING["BBB.NS"]}
FULL_IDX = {"^NSEI": FULL["AAA.NS"], "^NSEBANK": FULL["BBB.NS"]}
DAILY_IDX = {"^NSEI": DAILY["AAA.NS"], "^NSEBANK": DAILY["BBB.NS"]}


def futures_run(tmp_path, kind, picker, five_min, now, lots=None, **extra):
    d = deps(tmp_path, history=DAILY_IDX, picker=picker, now=now, rules=extra.pop("rules", RULES),
             intraday_history=lambda tickers: {t: f for t, f in five_min.items() if t in tickers}, **extra)
    return run_futures_intraday(kind, d, lot_sizes_source=lambda: lots if lots is not None else {}), d


def setup(price=25_000.0, stop_pct=0.4, ticker="NIFTY-FUT.NS"):
    return IntradaySetup(ticker=ticker, model="breakout", direction="long", score=80, price=price,
                         stop_pct=stop_pct, target_pct=stop_pct * 2)


def test_shipped_futures_rules():
    assert FR.enabled and FR.practice
    assert FR.lot_sizes == {"NIFTY": 65, "BANKNIFTY": 30, "FINNIFTY": 60, "MIDCPNIFTY": 120}
    assert (FR.stop_loss_pct, FR.cost_pct, FR.margin_pct, FR.max_lots) == (0.01, 0.04, 12.0, 3)


def test_only_the_margin_is_blocked_and_profit_is_the_whole_contract():
    ledger = Ledger(mode="paper", starting_cash=1_000_000, cash=1_000_000)
    pos = Position(symbol="NIFTY-FUT.NS", strategy="intraday", quantity=65, entry_price=25_000, stop_loss=24_900,
                   take_profit=25_200, opened_on=TODAY.isoformat(), max_hold_days=1, margin=195_000)
    ledger.open(pos)
    assert ledger.cash == 805_000 and pos.cost == 1_625_000                    # contract worth 16.25 lakh
    assert ledger.equity({"NIFTY-FUT.NS": 25_100}) == 1_000_000 + 6_500       # 100 points x 65
    trade = ledger.close(pos, 25_200, TODAY, "target hit", cost_pct=0.04)
    assert trade.charges == 650.0 and trade.pnl == 13_000 - 650
    assert ledger.cash == 1_000_000 + 13_000 - 650


def test_lots_follow_the_risk_budget_and_the_caps():
    ledger = Ledger(mode="paper", starting_cash=1_000_000, cash=1_000_000)
    sizes = {"NIFTY-FUT": 65}
    one = size_futures(setup(stop_pct=0.4), ledger, FR, sizes)             # 100 points x 65 = 6,500 a lot
    assert (one.lots, one.quantity, one.margin) == (1, 65, 195_000.0) and "1 lot(s) of 65" in one.note
    tight = size_futures(setup(stop_pct=0.1), ledger, FR, sizes)           # 1,625 a lot: 6 lots, capped at 3
    assert tight.lots == 3 and tight.quantity == 195
    wide = size_futures(setup(stop_pct=1.0), ledger, FR, sizes)            # 16,250: over budget, under twice
    assert wide.lots == 1
    with pytest.raises(RiskViolation, match="more than twice"):
        size_futures(setup(price=60_000, stop_pct=1.0), ledger, FR, sizes)  # 39,000 a lot
    poor = Ledger(mode="paper", starting_cash=100_000, cash=100_000)
    with pytest.raises(RiskViolation, match="margin"):
        size_futures(setup(stop_pct=0.1), poor, FR, sizes)                   # margin 195,000 > cash
    with pytest.raises(RiskViolation, match="No lot size"):
        size_futures(setup(ticker="FINNIFTY-FUT.NS"), ledger, FR, sizes)


def test_lot_sizes_come_from_nse_when_available():
    text = ("UNDERLYING,SYMBOL,NOV-26,DEC-26\nDerivatives on Index,,,\n"
            "NIFTY 50                    ,NIFTY     ,75   ,75\nNIFTY BANK,BANKNIFTY ,  ,35\n")
    assert parse_lot_sizes(text) == {"NIFTY": 75, "BANKNIFTY": 35}
    assert current_lot_sizes(FR, {"NIFTY": 75}) == {"NIFTY-FUT": 75, "BANKNIFTY-FUT": 30, "FINNIFTY-FUT": 60,
                                                    "MIDCPNIFTY-FUT": 120}
    assert current_lot_sizes(FR, {})["MIDCPNIFTY-FUT"] == 120


def test_vwap_of_an_index_without_volume_is_its_average_price():
    frame = FULL["AAA.NS"].copy()
    frame["Volume"] = 0.0
    f = compute_features("^NSEI", frame, TODAY, IST)
    today = frame[frame.index.date == TODAY]
    assert f.vwap == pytest.approx(((today["High"] + today["Low"] + today["Close"]) / 3).mean())
    assert f.relative_volume is None


def test_futures_entry_and_square_off_use_their_own_account(tmp_path):
    picker = IntradayPicker(PickDecision(action="SELL", ticker="BANKNIFTY-FUT.NS", target_price=97, reasoning="r"))
    outcome, d = futures_run(tmp_path, "intraday-open", picker, MORNING_IDX, AT(9, 52), lots={"BANKNIFTY": 35})

    assert sorted(s["ticker"] for s in picker.seen) == ["BANKNIFTY-FUT.NS", "NIFTY-FUT.NS"] and picker.practice
    opened = outcome.opened
    assert outcome.traded and opened.side == "short" and opened.quantity == 3 * 35 and opened.margin
    assert opened.margin == pytest.approx(opened.cost * 0.12, abs=0.05)           # NSE's lot size was used
    text = d.notifier.messages[0]
    assert text.startswith("*Futures intraday entry*") and "3 lot(s) of 35 = 105 units" in text
    assert "No prices for FINNIFTY-FUT, MIDCPNIFTY-FUT this run." in text      # the fake has only two indices
    folder = tmp_path / "futures"
    ledger = json.loads((folder / "paper_ledger.json").read_text())
    assert ledger["cash"] == pytest.approx(1_000_000 - opened.margin) and not (tmp_path / "paper_ledger.json").exists()

    outcome, d = futures_run(tmp_path, "intraday-close", IntradayPicker(None), FULL_IDX, AT(15, 20))
    (trade,) = outcome.closed
    expected = round((opened.entry_price - trade.exit_price) * 105 - trade.charges, 2)
    assert trade.pnl == pytest.approx(expected, abs=0.02) and trade.charges == round(opened.cost * 0.04 / 100, 2)
    assert d.notifier.messages[0].startswith("*Futures intraday square-off*") and "*Today's review*" in d.notifier.messages[0]
    review = json.loads((folder / "journal" / f"{TODAY.isoformat()}.json").read_text())["review"]
    assert review["shortlists"][0]["pick_rank"] is not None and review["rows"]
    assert json.loads((folder / "paper_ledger.json").read_text())["cash"] == pytest.approx(1_000_000 + trade.pnl)


def test_the_futures_book_can_be_switched_off(tmp_path):
    off = RULES.model_copy(update={"futures_intraday": FR.model_copy(update={"enabled": False})})
    assert futures_run(tmp_path, "intraday-open", IntradayPicker(None), MORNING_IDX, AT(9, 52), rules=off)[0] is None
    assert run_futures_backtest(tmp_path, off, None) is None

