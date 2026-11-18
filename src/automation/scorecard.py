"""The weekly scorecard: is a book's practice record good enough to trust yet?

Four checks on the book's own practice trades (since its journal began), each "met" or "not yet":

1. **Enough trades:** at least :data:`MIN_TRADES`.
2. **Profit beyond luck:** the average result per trade after charges is above zero by at least
   twice its uncertainty (t >= 2).
3. **Claude's picks beat the shortlist:** on the days Claude traded, its pick did better than the
   average shortlisted setup, by at least twice the uncertainty, over :data:`MIN_PICKS` picks.
4. **Losses stay small:** the worst drop from a high point is at most :data:`MAX_DRAWDOWN_PCT` of the
   account (judged only once there are enough trades).

It also estimates how many more trades the current average would need to pass check 2. The card
is sent with each book's Saturday review and saved as ``learning/scorecard.json``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from src.automation.debate import judge_summary
from src.automation.journal import journal_dir
from src.automation.ledger import Ledger, ledger_path

MIN_TRADES = 50
MIN_PICKS = 30
MAX_DRAWDOWN_PCT = 10.0
T_PROOF = 2.0


def mean_se(values: Sequence[float]) -> Tuple[Optional[float], Optional[float]]:
    """Average and its standard error (None where there are too few values)."""
    n = len(values)
    if n == 0:
        return None, None
    mean = sum(values) / n
    if n < 2:
        return mean, None
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    return mean, math.sqrt(variance / n)


def _pick_edges(state_dir: Path, judge_rows: Optional[List[dict]] = None) -> Tuple[Optional[str], List[float]]:
    """(first journal day, pick minus shortlist average for each shortlist Claude traded from).

    ``judge_rows`` (if given) is extended with the reviews' records of the judge's calls.
    """
    folder = journal_dir(state_dir)
    paths = sorted(folder.glob("*.json")) if folder.exists() else []
    edges: List[float] = []
    for path in paths:
        try:
            review = json.loads(path.read_text(encoding="utf-8")).get("review") or {}
        except (OSError, ValueError):
            continue
        if judge_rows is not None:
            judge_rows.extend(review.get("judge") or [])
        for run in review.get("shortlists", []):
            outcomes, rank = run.get("outcomes", []), run.get("pick_rank")
            if outcomes and rank:
                average = sum(o["ret"] for o in outcomes) / len(outcomes)
                edges.append(outcomes[rank - 1]["ret"] - average)
    return (paths[0].stem if paths else None), edges


def _passes(mean: Optional[float], se: Optional[float]) -> bool:
    return mean is not None and se is not None and mean > 0 and mean >= T_PROOF * se


def build(ledger: Ledger, state_dir: Path) -> dict:
    """The scorecard of one book (its ledger and its journal folder)."""
    judge_rows: List[dict] = []
    first, edges = _pick_edges(state_dir, judge_rows)
    trades = [t for t in ledger.closed if t.strategy == "intraday" and (first is None or t.opened_on >= first)]
    returns = [t.pnl_pct for t in trades]
    n = len(trades)
    avg, se = mean_se(returns)
    equity = peak = drop = 0.0
    run = longest = 0
    for t in trades:
        equity += t.pnl
        peak = max(peak, equity)
        drop = max(drop, peak - equity)
        run = run + 1 if t.pnl < 0 else 0
        longest = max(longest, run)
    drop_pct = drop / ledger.starting_cash * 100 if ledger.starting_cash else 0.0
    pick_avg, pick_se = mean_se(edges)

    def pm(value: Optional[float]) -> str:
        return f" (±{value:.2f})" if value is not None else ""

    checks = [
        {"name": "Enough practice trades", "value": f"{n} of {MIN_TRADES}", "met": n >= MIN_TRADES},
        {"name": "Average after charges clearly above zero",
         "value": "no trades yet" if avg is None else f"{avg:+.2f}% a trade{pm(se)}", "met": _passes(avg, se)},
        {"name": "Claude's picks beat the average shortlisted setup",
         "value": "no picks reviewed yet" if pick_avg is None
         else f"{pick_avg:+.2f}% a trade{pm(pick_se)} over {len(edges)} pick(s)",
         "met": len(edges) >= MIN_PICKS and _passes(pick_avg, pick_se)},
        {"name": "Losses stay small",
         "value": f"worst drop Rs {drop:,.0f} ({drop_pct:.1f}% of the account), longest losing run {longest}"
         + ("" if n >= MIN_TRADES else f" (judged after {MIN_TRADES} trades)"),
         "met": n >= MIN_TRADES and drop_pct <= MAX_DRAWDOWN_PCT},
    ]
    more = None
    if avg is None:
        outlook = "No practice trades yet."
    elif avg <= 0 and n < MIN_TRADES:
        outlook = ("Too early to judge: the average so far is at or below zero but within the luck margin. "
                   "Keep collecting trades.")
    elif avg <= 0:
        outlook = (f"After {n} trades the average is not above zero, so this book has not shown a profit; "
                   "results have to improve before it can be proven.")
    else:
        sd = se * math.sqrt(n) if se is not None else None
        needed = math.ceil((T_PROOF * sd / avg) ** 2) if sd else MIN_TRADES
        more = max(needed, MIN_TRADES) - n
        outlook = (f"At the current average, about {more} more trade(s) would be needed to tell it apart from luck."
                   if more > 0 else "The numbers are past the luck margin.")
    proven = all(c["met"] for c in checks)
    return {
        "since": first, "trades": n, "avg_pct": None if avg is None else round(avg, 3),
        "se": None if se is None else round(se, 3), "max_drop": round(drop, 2), "max_drop_pct": round(drop_pct, 2),
        "longest_losing_run": longest, "pick_edge_pct": None if pick_avg is None else round(pick_avg, 3),
        "pick_edge_se": None if pick_se is None else round(pick_se, 3), "picks": len(edges),
        "checks": checks, "more_trades_needed": more, "outlook": outlook, "proven": proven,
        "judge": judge_summary(judge_rows),
    }


def text(card: dict) -> str:
    """The scorecard as a few Telegram lines."""
    since = f" since {card['since']}" if card.get("since") else ""
    lines = [f"*Scorecard* (practice trades{since})"]
    lines += [f"- {c['name']}: {c['value']} - {'met' if c['met'] else 'not yet'}" for c in card["checks"]]
    verdict = "all four checks are met on paper." if card["proven"] else "not proven yet. " + card["outlook"]
    lines.append("Verdict: " + verdict)
    judge = card.get("judge")
    if judge:
        t, s = judge["TRADE"], judge["SKIP"]

        def part(x, ended):
            return f"{x['n']} ({x['won']} won, {ended} {x['avg_pct']:+.2f}% on average)" if x["n"] else "0"

        lines.append(f"Judge (bull vs bear): approved {part(t, 'ended')}; skipped {part(s, 'would have ended')}. "
                     "The debate helps if the skipped ones would have done worse.")
    return "\n".join(lines)


def weekly(state_dir: Path, mode: str = "paper") -> str:
    """Build, save (learning/scorecard.json) and return the scorecard text of the book in ``state_dir``."""
    ledger = Ledger.load(ledger_path(mode, state_dir), mode=mode, starting_cash=0.0)
    card = build(ledger, state_dir)
    folder = state_dir / "learning"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "scorecard.json").write_text(json.dumps(card, indent=1) + "\n", encoding="utf-8")
    return text(card)


__all__ = ["MAX_DRAWDOWN_PCT", "MIN_PICKS", "MIN_TRADES", "build", "mean_se", "text", "weekly"]
