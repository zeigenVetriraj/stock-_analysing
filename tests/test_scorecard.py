"""Tests for the weekly scorecard: four checks, each met only with real evidence.

Synthetic ledgers and journals only - NO network. Run with:  ``pytest -q``
"""

from __future__ import annotations

import json
import random
from datetime import date, timedelta

from src.automation import scorecard
from src.automation.journal import save_day
from src.automation.ledger import ClosedTrade, Ledger, ledger_path

START = date(2026, 8, 3)


def book(tmp_path, n, *, mean, spread=0.5, pick_edge=0.0, seed=3):
    """``n`` practice days: one trade each (net % ~ mean +- spread) and a reviewed shortlist."""
    rng = random.Random(seed)
    ledger = Ledger(mode="paper", starting_cash=1_000_000, cash=1_000_000)
    for i in range(n):
        day = (START + timedelta(days=i)).isoformat()
        pct = round(rng.gauss(mean, spread), 3)
        ledger.closed.append(ClosedTrade(
            symbol="X.NS", strategy="intraday", quantity=100, entry_price=1000, exit_price=1000, opened_on=day,
            closed_on=day, reason="r", pnl=pct * 1000, pnl_pct=pct))
        others = [rng.gauss(0, 0.5) for _ in range(4)]
        picked = sum(others) / 4 + pick_edge + rng.gauss(0, 0.2)
        outcomes = [{"ret": picked, "picked": True}] + [{"ret": x, "picked": False} for x in others]
        outcomes.sort(key=lambda o: o["ret"], reverse=True)
        rank = next(k for k, o in enumerate(outcomes, start=1) if o["picked"])
        save_day(tmp_path, START + timedelta(days=i), {"date": day, "entries": [], "review": {
            "shortlists": [{"outcomes": outcomes, "pick_rank": rank}]}})
    ledger.closed.insert(0, ClosedTrade(                    # before the journal began: not counted
        symbol="OLD.NS", strategy="intraday", quantity=1, entry_price=1, exit_price=1, opened_on="2026-01-02",
        closed_on="2026-01-02", reason="r", pnl=-99_999, pnl_pct=-50))
    ledger.save(ledger_path("paper", tmp_path))
    return ledger


def test_a_young_book_is_not_proven(tmp_path):
    """Nine trades: too early, whatever the average."""
    card = scorecard.build(book(tmp_path, 9, mean=0.2), tmp_path)
    assert card["trades"] == 9 and card["since"] == "2026-08-03" and not card["proven"]
    assert card["checks"][0]["met"] is False and card["checks"][3]["met"] is False
    assert card["more_trades_needed"] is None or card["more_trades_needed"] > 0


def test_a_strong_long_record_meets_all_four(tmp_path):
    card = scorecard.build(book(tmp_path, 80, mean=0.4, spread=0.4, pick_edge=0.3), tmp_path)
    assert card["proven"] and all(c["met"] for c in card["checks"]) and card["picks"] == 80
    assert card["max_drop_pct"] < 1 and "all four checks are met" in scorecard.text(card)


def test_a_losing_record_says_more_trades_will_not_help(tmp_path):
    card = scorecard.build(book(tmp_path, 60, mean=-0.2), tmp_path)
    assert not card["checks"][1]["met"] and card["more_trades_needed"] is None
    assert "results have to improve before it can be proven" in scorecard.text(card)


def test_trades_needed_follows_the_luck_margin(tmp_path):
    card = scorecard.build(book(tmp_path, 20, mean=0.1, spread=1.0, seed=8), tmp_path)
    if card["avg_pct"] > 0:
        sd = card["se"] * 20 ** 0.5
        assert card["more_trades_needed"] >= max(int((2 * sd / card["avg_pct"]) ** 2) - 20, scorecard.MIN_TRADES - 20)


def test_weekly_saves_the_card_and_handles_an_empty_book(tmp_path):
    book(tmp_path, 5, mean=0.1)
    text = scorecard.weekly(tmp_path)
    assert text.startswith("*Scorecard* (practice trades since 2026-08-03)") and text.count(" - not yet") >= 3
    assert json.loads((tmp_path / "learning" / "scorecard.json").read_text())["trades"] == 5
    empty = tmp_path / "empty"
    assert "No practice trades yet" in scorecard.weekly(empty)


def test_a_young_losing_book_is_too_early_to_judge(tmp_path):
    card = scorecard.build(book(tmp_path, 9, mean=-0.1), tmp_path)
    assert "Too early to judge" in card["outlook"] and "(judged after 50 trades)" in card["checks"][3]["value"]
