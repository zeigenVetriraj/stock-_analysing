"""The weekend profit and loss message: each book's result this week and since it began, and the total.

Sent by the Saturday job (``automate.py intraday-backtest``) before the weekly reviews. The week is
Monday to the day of the run; "since start" counts a book's practice trades from the first day of
its journal. Profits and losses are after the estimated charges, as in the paper ledgers.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

from config.settings import TradingRules
from src.automation.daily_picks import BOOKS
from src.automation.jobs import inr
from src.automation.journal import journal_dir
from src.automation.ledger import Ledger, ledger_path


def _signed(amount: float) -> str:
    return ("+" if amount >= 0 else "") + inr(amount, 0)


def _first_day(folder: Path) -> Optional[str]:
    days = sorted(p.stem for p in journal_dir(folder).glob("*.json")) if journal_dir(folder).exists() else []
    return days[0] if days else None


def book_result(folder: Path, monday: date, today: date, mode: str = "paper") -> dict:
    """One book's numbers: this week's and all practice trades, and its account value."""
    ledger = Ledger.load(ledger_path(mode, folder), mode=mode, starting_cash=0.0)
    first = _first_day(folder)
    practice = [t for t in ledger.closed if t.strategy == "intraday" and (first is None or t.opened_on >= first)]
    week = [t for t in practice if monday.isoformat() <= t.closed_on <= today.isoformat()]
    return {
        "week_trades": len(week), "week_won": sum(1 for t in week if t.pnl > 0),
        "week_pnl": round(sum(t.pnl for t in week), 2),
        "all_trades": len(practice), "all_pnl": round(sum(t.pnl for t in practice), 2),
        "account": round(ledger.cash + sum(p.blocked for p in ledger.positions), 2), "since": first,
    }


def weekly_pnl(state_dir: Path, rules: TradingRules, today: date) -> str:
    """The Telegram message: a table with one row per book that is switched on, and the total."""
    monday = today - timedelta(days=today.weekday())
    rows: List[Tuple[str, str, str, str]] = []
    week_total = all_total = 0.0
    trades = won = 0
    for label, sub, block in BOOKS:
        book_rules = getattr(rules, block, None)
        if book_rules is None or not book_rules.enabled:
            continue
        r = book_result(state_dir / sub if sub else state_dir, monday, today)
        rows.append((label, f"{r['week_trades']} ({r['week_won']} won)", _signed(r["week_pnl"]), _signed(r["all_pnl"])))
        week_total += r["week_pnl"]
        all_total += r["all_pnl"]
        trades += r["week_trades"]
        won += r["week_won"]
    header = ("Book", "Trades", "This week", "Since start")
    total = ("Total", f"{trades} ({won} won)", _signed(week_total), _signed(all_total))
    widths = [max(len(r[i]) for r in rows + [header, total]) for i in range(4)]

    def line(r) -> str:
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(r)).rstrip()

    rule = line(tuple("-" * w for w in widths))
    table = "\n".join([line(header), rule] + [line(r) for r in rows] + [rule, line(total)])
    span = f"{monday.strftime('%d %b')} - {today.strftime('%d %b %Y')}"
    return (f"*Weekly profit and loss* - {span} (paper)\n```\n{table}\n```\n"
            "After estimated charges. 'Since start' counts each book's practice trades from its first day.")


__all__ = ["book_result", "weekly_pnl"]
