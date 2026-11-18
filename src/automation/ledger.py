"""Positions opened by the automated jobs, kept in a small JSON file.

* Paper mode (default): ``state/paper_ledger.json`` holds a practice account -
  cash, open positions and closed trades - starting from SIMULATED_TOTAL_EQUITY.
  The GitHub workflow commits it back after each run, so results build up.
* Live mode: ``state/live_ledger.json`` records only the positions the bot
  itself bought, so it never touches shares you bought yourself.

Exit rules are deterministic: a position closes when the price reaches its
stop-loss or target, or after ``max_hold_days`` calendar days. Prices are
checked once per run. Intraday positions can be short (``side="short"``): their
stop sits above the entry and their target below, and they are squared off the
same day by the intraday job.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from config.settings import PROJECT_ROOT

logger = logging.getLogger(__name__)

STATE_DIR = PROJECT_ROOT / "state"

Mode = Literal["paper", "live"]


def ledger_path(mode: str, state_dir: Path = STATE_DIR) -> Path:
    return state_dir / f"{mode}_ledger.json"


class Position(BaseModel):
    symbol: str                      # Yahoo ticker, e.g. RELIANCE.NS
    strategy: str                    # "swing", "penny" or "intraday"
    side: str = "long"               # "long" or "short" (intraday only)
    quantity: int = Field(gt=0)
    entry_price: float = Field(gt=0)
    stop_loss: float = Field(gt=0)
    take_profit: float = Field(gt=0)
    opened_on: str                   # YYYY-MM-DD
    max_hold_days: int = Field(gt=0)
    opened_at: Optional[str] = None  # ISO timestamp (intraday)
    setup: str = ""                  # intraday: the kind of setup traded (breakout, fvg, fib, gap_fade)
    reasoning: str = ""
    # Futures: only this margin is blocked from cash while the position is open (None: the full cost).
    margin: Optional[float] = Field(default=None, gt=0)

    @property
    def cost(self) -> float:
        """The position's full value (for futures, the contract value)."""
        return round(self.quantity * self.entry_price, 2)

    @property
    def blocked(self) -> float:
        """Cash set aside while the position is open: its margin for futures, else its full cost."""
        return round(self.margin, 2) if self.margin is not None else self.cost

    def days_held(self, today: date) -> int:
        return (today - date.fromisoformat(self.opened_on)).days

    def pnl_at(self, price: float) -> float:
        move = price - self.entry_price if self.side == "long" else self.entry_price - price
        return round(move * self.quantity, 2)


class ClosedTrade(BaseModel):
    symbol: str
    strategy: str
    side: str = "long"
    quantity: int
    entry_price: float
    exit_price: float
    opened_on: str
    closed_on: str
    reason: str
    pnl: float                       # after charges
    pnl_pct: float
    charges: float = 0.0             # estimated trading costs taken off (paper intraday)
    setup: str = ""


class Ledger(BaseModel):
    mode: str = "paper"
    starting_cash: float = 0.0
    cash: float = 0.0
    positions: List[Position] = Field(default_factory=list)
    closed: List[ClosedTrade] = Field(default_factory=list)

    # ---- persistence ---- #
    @classmethod
    def load(cls, path: Path, *, mode: str, starting_cash: float) -> "Ledger":
        if path.exists():
            return cls.model_validate(json.loads(path.read_text(encoding="utf-8")))
        return cls(mode=mode, starting_cash=starting_cash, cash=starting_cash)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")

    # ---- queries ---- #
    def holds(self, symbol: str) -> bool:
        return any(p.symbol == symbol for p in self.positions)

    def count(self, strategy: str) -> int:
        return sum(1 for p in self.positions if p.strategy == strategy)

    def equity(self, prices: Dict[str, float]) -> float:
        # Cash already excludes what each position blocks (its cost, or its margin for futures).
        value = sum(p.blocked + p.pnl_at(prices.get(p.symbol, p.entry_price)) for p in self.positions)
        return round(self.cash + value, 2)

    def realised_pnl(self) -> float:
        return round(sum(t.pnl for t in self.closed), 2)

    # ---- changes ---- #
    def open(self, position: Position) -> None:
        if self.mode == "paper":
            if position.blocked > self.cash + 1e-6:
                raise ValueError(
                    f"Not enough paper cash for {position.symbol}: need {position.blocked}, have {self.cash}."
                )
            self.cash = round(self.cash - position.blocked, 2)
        self.positions.append(position)

    def close(self, position: Position, price: float, today: date, reason: str, *,
              cost_pct: float = 0.0) -> ClosedTrade:
        """Close a position. ``cost_pct`` (% of the position) is taken off as estimated charges."""
        self.positions.remove(position)
        charges = round(position.cost * cost_pct / 100, 2)
        pnl = round(position.pnl_at(price) - charges, 2)
        if self.mode == "paper":
            self.cash = round(self.cash + position.blocked + pnl, 2)
        trade = ClosedTrade(
            symbol=position.symbol, strategy=position.strategy, side=position.side, quantity=position.quantity,
            entry_price=position.entry_price, exit_price=round(price, 2),
            opened_on=position.opened_on, closed_on=today.isoformat(), reason=reason,
            pnl=pnl, pnl_pct=round(pnl / position.cost * 100, 2), charges=charges, setup=position.setup,
        )
        self.closed.append(trade)
        return trade


def exit_reason(position: Position, price: Optional[float], today: date) -> Optional[str]:
    """Why a position should close now, or None to keep holding."""
    if price is not None:
        short = position.side == "short"
        if (price >= position.stop_loss) if short else (price <= position.stop_loss):
            return "stop-loss hit"
        if (price <= position.take_profit) if short else (price >= position.take_profit):
            return "target hit"
    if position.days_held(today) >= position.max_hold_days:
        return f"held {position.days_held(today)} days (time limit)"
    return None


__all__ = ["ClosedTrade", "Ledger", "Position", "STATE_DIR", "exit_reason", "ledger_path"]
