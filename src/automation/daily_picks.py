"""One daily table of the three books' picks, sent after the 9:50 entry run.

Four columns: the pick (book, buy or short, name), its stop-loss, its target, and the profit if the
target is hit (after the book's estimated charges). Built from each book's journal for the day, so
it is right even when a run is repeated; a book that did not trade says why.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import List, Tuple

from config.settings import TradingRules
from src.automation.jobs import inr, plain
from src.automation.journal import load_day

# (label, sub-folder of state/, the rules block with the book's charges)
BOOKS = (("Large", "", "intraday"), ("Penny", "penny", "penny_intraday"), ("Futures", "futures", "futures_intraday"))


def _price(value: float) -> str:
    return f"{value:,.2f}"


def profit_at_target(opened: dict, cost_pct: float) -> float:
    """Rupee profit if the target is hit, after ``cost_pct`` charges on the position's value."""
    qty, entry, target = opened["quantity"], opened["entry_price"], opened["take_profit"]
    move = (target - entry) if opened.get("side", "long") == "long" else (entry - target)
    return move * qty - entry * qty * cost_pct / 100


def book_row(label: str, folder: Path, cost_pct: float, today: date) -> Tuple[str, str, str, str]:
    """(pick, stop, target, profit at target) for one book today."""
    entries = load_day(folder, today).get("entries", [])
    traded = [e for e in entries if e.get("opened")]
    if not traded:
        note = (entries[-1].get("note") or "no pick") if entries else "no entry run"
        if note.startswith("Claude could not make a pick"):
            note = "Claude's reply unreadable"
        elif note.startswith("Risk layer refused it"):
            note = "refused by the risk check"
        elif note.startswith("Judge skipped"):
            note = "the judge skipped every setup"
        return f"{label}: no trade ({note[:40]})", "-", "-", "-"
    o = traded[-1]["opened"]
    side = "BUY" if o.get("side", "long") == "long" else "SELL"
    pick = f"{label} {side} {plain(o['symbol'])} {o['quantity']} @ {_price(o['entry_price'])}"
    profit = profit_at_target(o, cost_pct)
    return pick, _price(o["stop_loss"]), _price(o["take_profit"]), ("+" if profit >= 0 else "") + inr(profit, 0)


def picks_table(state_dir: Path, rules: TradingRules, today: date) -> str:
    """The Telegram message: a monospace table with one row per book that is switched on."""
    rows: List[Tuple[str, str, str, str]] = []
    for label, sub, block in BOOKS:
        book_rules = getattr(rules, block, None)
        if book_rules is None or not book_rules.enabled:
            continue
        rows.append(book_row(label, state_dir / sub if sub else state_dir, book_rules.cost_pct, today))
    header = ("Pick", "Stop-loss", "Target", "Profit at target")
    widths = [max(len(r[i]) for r in rows + [header]) for i in range(4)]

    def line(r) -> str:
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(r)).rstrip()

    table = "\n".join([line(header), line(tuple("-" * w for w in widths))] + [line(r) for r in rows])
    return (f"*Today's picks* - {today.strftime('%a %d %b %Y')} (paper)\n```\n{table}\n```\n"
            "Profit at target is after estimated charges. Each trade closes at 15:15 if neither level is hit.")


__all__ = ["BOOKS", "book_row", "picks_table", "profit_at_target"]
