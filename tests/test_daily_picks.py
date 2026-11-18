"""Tests for the daily picks table (one row per book, four columns).

Synthetic journals only - NO network. Run with:  ``pytest -q``
"""

from __future__ import annotations

from src.automation.daily_picks import picks_table, profit_at_target
from src.automation.journal import save_day
from tests.test_automation import RULES, TODAY


def opened(symbol, side, qty, entry, stop, target):
    return {"symbol": symbol, "side": side, "quantity": qty, "entry_price": entry, "stop_loss": stop,
            "take_profit": target}


def test_profit_at_target_is_after_charges():
    assert profit_at_target(opened("A.NS", "long", 100, 1000.0, 990.0, 1020.0), 0.12) == 2000 - 120
    assert profit_at_target(opened("B.NS", "short", 10, 100.0, 101.0, 98.0), 0.3) == 20 - 3


def test_one_row_per_book(tmp_path):
    save_day(tmp_path, TODAY, {"date": "", "entries": [
        {"note": "traded", "opened": opened("JSWSTEEL.NS", "long", 79, 1250.2, 1237.7, 1275.21)}]})
    save_day(tmp_path / "penny", TODAY, {"date": "", "entries": [
        {"note": "traded", "opened": opened("SUZLON.NS", "short", 2894, 34.55, 35.24, 33.17)}]})
    save_day(tmp_path / "futures", TODAY, {"date": "", "entries": [
        {"note": "Claude could not make a pick: Intraday picker reply was still unusable", "opened": None}]})
    text = picks_table(tmp_path, RULES, TODAY)
    lines = text.split("```")[1].strip().splitlines()
    assert lines[0].split() == ["Pick", "Stop-loss", "Target", "Profit", "at", "target"]
    assert lines[2].startswith("Large BUY JSWSTEEL 79 @ 1,250.20") and "1,237.70" in lines[2] and "+Rs 1,857" in lines[2]
    assert lines[3].startswith("Penny SELL SUZLON 2894 @ 34.55") and lines[3].endswith("+Rs 3,694")
    assert lines[4].startswith("Futures: no trade (Claude's reply unreadable)")
    assert text.startswith("*Today's picks*") and "after estimated charges" in text


def test_books_without_a_run_or_switched_off(tmp_path):
    off = RULES.model_copy(update={"penny_intraday": RULES.penny_intraday.model_copy(update={"enabled": False})})
    lines = picks_table(tmp_path, off, TODAY).split("```")[1].strip().splitlines()
    assert [line.split(":")[0] for line in lines[2:]] == ["Large", "Futures"]
    assert "no trade (no entry run)" in lines[2]
