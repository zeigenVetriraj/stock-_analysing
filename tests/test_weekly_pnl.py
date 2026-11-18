"""Tests for the weekend profit and loss table. Synthetic ledgers only - NO network."""

from __future__ import annotations

from datetime import date

from src.automation.journal import save_day
from src.automation.ledger import ClosedTrade, Ledger, ledger_path
from src.automation.weekly_pnl import book_result, weekly_pnl
from tests.test_automation import RULES

SATURDAY = date(2026, 11, 7)


def trade(day, pnl):
    return ClosedTrade(symbol="X.NS", strategy="intraday", quantity=1, entry_price=1, exit_price=1,
                       opened_on=day, closed_on=day, reason="r", pnl=pnl, pnl_pct=0.1)


def book(folder, trades, first="2026-10-21"):
    ledger = Ledger(mode="paper", starting_cash=1_000_000, cash=1_000_000 + sum(t.pnl for t in trades),
                    closed=[trade("2026-10-01", -999.0)] + trades)          # before the journal: not counted
    ledger.save(ledger_path("paper", folder))
    save_day(folder, date.fromisoformat(first), {"date": first, "entries": []})


def test_each_book_and_the_total(tmp_path):
    book(tmp_path, [trade("2026-10-28", 500.0), trade("2026-11-03", 1_224.0), trade("2026-11-05", -800.0)])
    book(tmp_path / "penny", [trade("2026-11-02", 3_108.0)], first="2026-10-30")
    book(tmp_path / "futures", [trade("2026-11-04", -10_000.0), trade("2026-11-06", 21_192.0)], first="2026-10-30")

    large = book_result(tmp_path, date(2026, 11, 2), SATURDAY)
    assert (large["week_trades"], large["week_won"], large["week_pnl"], large["all_pnl"]) == (2, 1, 424.0, 924.0)

    lines = weekly_pnl(tmp_path, RULES, SATURDAY).split("```")[1].strip().splitlines()
    assert lines[0].split()[:2] == ["Book", "Trades"]
    assert lines[2].split() == ["Large", "2", "(1", "won)", "+Rs", "424", "+Rs", "924"]
    assert lines[3].split() == ["Penny", "1", "(1", "won)", "+Rs", "3,108", "+Rs", "3,108"]
    assert lines[4].split() == ["Futures", "2", "(1", "won)", "+Rs", "11,192", "+Rs", "11,192"]
    assert lines[6].split() == ["Total", "5", "(3", "won)", "+Rs", "14,724", "+Rs", "15,224"]


def test_a_book_switched_off_is_left_out(tmp_path):
    off = RULES.model_copy(update={"futures_intraday": RULES.futures_intraday.model_copy(update={"enabled": False})})
    text = weekly_pnl(tmp_path, off, SATURDAY)
    assert "Futures" not in text and "02 Nov - 07 Nov 2026" in text and "Total" in text
