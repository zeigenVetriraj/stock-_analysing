"""Tests for the penny intraday book: its own stock list, paper account, journal and learning folder,
and the circuit-limit check. The large-stock book must never be touched.

Fakes only - NO network, NO LLM, NO orders. Run with:  ``pytest -q``
"""

from __future__ import annotations

import json

import pytest

from src.automation.intraday import IntradayFeatures, IntradaySetup
from src.automation.penny_book import (
    LIST_FILE,
    circuit_problem,
    load_or_build_list,
    run_penny_backtest,
    run_penny_intraday,
)
from src.automation.picker import PickDecision
from src.automation.universe import (
    PennyCandidate,
    average_turnover_cr,
    parse_price_bands,
    penny_intraday_list,
)
from tests.test_automation import RULES, SHIPPED, TODAY, bars, deps
from tests.test_intraday import DAILY, IntradayPicker
from tests.test_journal import AT, FULL, MORNING

BHAVCOPY = """SYMBOL, SERIES, DATE1, PREV_CLOSE, OPEN_PRICE, HIGH_PRICE, LOW_PRICE, LAST_PRICE, CLOSE_PRICE, AVG_PRICE, TTL_TRD_QNTY, TURNOVER_LACS, NO_OF_TRADES, DELIV_QTY, DELIV_PER
AAA, EQ, x, 100, 100, 101, 99, 100, 100, 100, 1, 2000, 1, 1, 40
BBB, EQ, x, 100, 100, 101, 99, 100, 100, 100, 1, 2000, 1, 1, 40
CCC, EQ, x, 100, 100, 101, 99, 100, 100, 100, 1, 2000, 1, 1, 40
DDD, BE, x, 100, 100, 101, 99, 100, 100, 100, 1, 2000, 1, 1, 40
"""
# The test stocks trade near 100, so the price range is widened for them.
PENNY = SHIPPED.penny_intraday.model_copy(update={"max_price": 500.0})
TEST_RULES = RULES.model_copy(update={"penny_intraday": PENNY})


def penny_run(tmp_path, kind, picker, five_min, now, bands=None, **extra):
    d = deps(tmp_path, history=DAILY, picker=picker, now=now, rules=extra.pop("rules", TEST_RULES),
             bhavcopy=extra.pop("bhavcopy", lambda day: (day, BHAVCOPY)),
             price_bands=lambda: bands if bands is not None else {"AAA": 1.0, "BBB": 20.0, "CCC": None},
             intraday_history=lambda tickers: {t: f for t, f in five_min.items() if t in tickers}, **extra)
    return run_penny_intraday(kind, d), d


def test_shipped_penny_rules():
    pr = SHIPPED.penny_intraday
    assert pr.enabled and pr.practice and (pr.min_price, pr.max_price, pr.max_stocks) == (2.0, 50.0, 50)
    assert (pr.stop_loss_pct, pr.take_profit_pct, pr.cost_pct) == (0.02, 0.04, 0.3)
    assert SHIPPED.intraday.cost_pct == 0.12                      # the large-stock book is unchanged


def test_price_bands_and_the_most_traded_penny_stocks():
    text = "Symbol,Series,Security Name,Band,Remarks\nABC,EQ,Abc,5,\nXYZ,EQ,Xyz,No Band,\nQQ,BE,Q,5,\n"
    assert parse_price_bands(text) == {"ABC": 5.0, "XYZ": None}
    daily = {"A.NS": bars(volume=2e6), "B.NS": bars(volume=4e6), "C.NS": bars(volume=1e4)}
    assert average_turnover_cr(daily["A.NS"]) == pytest.approx(daily["A.NS"]["Close"].tail(20).mean() * 2e6 / 1e7)
    candidates = [PennyCandidate(s, 10.0, 5.0, None) for s in ("A", "B", "C", "D")]
    picked = penny_intraday_list(candidates, daily, PENNY.model_copy(update={"max_stocks": 5}), lambda s: s + ".NS")
    assert [s for s, _ in picked] == ["B", "A"]                   # C trades too little, D has no data


def test_circuit_limits():
    features = IntradayFeatures(ticker="X.NS", price=104.0, as_of="10:00", bars_today=9, today_open=101.0,
                                day_high=104.5, day_low=100.5, or_high=103.0, or_low=100.5, vwap=102.0, pdc=100.0)
    setup = IntradaySetup(ticker="X.NS", model="breakout", direction="long", score=80, price=104.0,
                          stop_pct=1.0, target_pct=2.0, features=features)
    assert circuit_problem(setup, 5.0, 0.5) == "too close to its daily circuit limit"   # target 106.08 > ~104.5
    assert circuit_problem(setup, 10.0, 0.5) is None
    assert circuit_problem(setup, None, 0.5) is None                                     # no band
    no_close = IntradaySetup(ticker="X.NS", model="breakout", direction="long", score=80, price=104.0,
                             stop_pct=1.0, target_pct=2.0, features=None)
    assert circuit_problem(no_close, 5.0, 0.5) == "no previous close to place its circuit limits"


def test_penny_entry_uses_its_own_account_list_and_journal(tmp_path):
    picker = IntradayPicker(PickDecision(action="SELL", ticker="BBB.NS", target_price=97.0, reasoning="r"))
    outcome, d = penny_run(tmp_path, "intraday-open", picker, MORNING, AT(9, 52))

    assert outcome.traded and outcome.opened.symbol == "BBB.NS" and picker.practice is True
    assert [s["ticker"] for s in picker.seen] == ["BBB.NS"]          # AAA sits outside its 1% circuit band
    text = d.notifier.messages[0]
    assert text.startswith("*Penny intraday entry*") and "Skipped (too close to its daily circuit limit): AAA." in text
    penny = tmp_path / "penny"
    assert (penny / "paper_ledger.json").exists() and not (tmp_path / "paper_ledger.json").exists()
    assert (penny / "journal" / f"{TODAY.isoformat()}.json").exists() and not (tmp_path / "journal").exists()
    listing = json.loads((penny / LIST_FILE).read_text())
    assert [s["symbol"] for s in listing["stocks"]] == ["AAA", "BBB", "CCC"]  # most traded first, no BE stock
    assert {s["symbol"]: s["band_pct"] for s in listing["stocks"]} == {"AAA": 1.0, "BBB": 20.0, "CCC": None}


def test_the_penny_square_off_reviews_the_penny_day(tmp_path):
    picker = IntradayPicker(PickDecision(action="SELL", ticker="BBB.NS", target_price=97.0, reasoning="r"))
    penny_run(tmp_path, "intraday-open", picker, MORNING, AT(9, 52))
    gone = lambda day: (_ for _ in ()).throw(RuntimeError("NSE down"))   # noqa: E731 - the saved list is reused
    outcome, d = penny_run(tmp_path, "intraday-close", IntradayPicker(None), FULL, AT(15, 20), bhavcopy=gone)

    assert [t.symbol for t in outcome.closed] == ["BBB.NS"] and outcome.closed[0].charges > 0
    text = d.notifier.messages[0]
    assert text.startswith("*Penny intraday square-off*") and "*Today's review*" in text
    review = json.loads((tmp_path / "penny" / "journal" / f"{TODAY.isoformat()}.json").read_text())["review"]
    assert review["shortlists"][0]["pick_rank"] == 1 and review["rows"]


def test_hourly_check_needs_no_list_and_the_book_can_be_switched_off(tmp_path):
    def gone(day):
        raise RuntimeError("NSE down")
    outcome, d = penny_run(tmp_path, "intraday-check", IntradayPicker(None), MORNING, AT(11, 3), bhavcopy=gone)
    assert outcome is not None and not outcome.closed and d.notifier.messages == []
    off = TEST_RULES.model_copy(update={"penny_intraday": PENNY.model_copy(update={"enabled": False})})
    assert penny_run(tmp_path, "intraday-open", IntradayPicker(None), MORNING, AT(9, 52), rules=off)[0] is None


def test_the_list_is_built_once_a_day(tmp_path):
    calls = []

    def bhavcopy(day):
        calls.append(day)
        return day, BHAVCOPY

    d = deps(tmp_path, history=DAILY, bhavcopy=bhavcopy, price_bands=dict)
    first = load_or_build_list(d, PENNY)
    again = load_or_build_list(d, PENNY)
    assert first == again and len(calls) == 1 and first["bands"].startswith("not available")
    assert {s["band_pct"] for s in first["stocks"]} == {5.0}                 # the default band when NSE fails


def test_the_weekly_penny_review_waits_for_a_list(tmp_path):
    sent = []

    class Notifier:
        def send(self, text):
            sent.append(text)

    assert "No penny stock list yet" in run_penny_backtest(tmp_path, TEST_RULES, Notifier()) and len(sent) == 1
