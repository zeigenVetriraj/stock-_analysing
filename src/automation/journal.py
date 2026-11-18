"""The intraday journal: what the bot saw, chose and got, one small file per trading day.

``state/journal/YYYY-MM-DD.json`` holds

* ``entries``: every entry run - its time, the shortlist shown to Claude (each
  setup's facts and learned expected result), Claude's decision and the paper
  position opened, if any;
* ``review``: written at the square-off - how every shortlisted setup would have
  ended, where Claude's pick ranked among them, how all of the day's setups did
  for each kind of setup (the same replay as the backtest), and the day's trades.

:func:`running_totals` reads all the days for the end-of-day message: Claude's
picks against the rest of the shortlist, and each kind of setup's record since
the journal began - the bot's own forward test, kept apart from the backtest.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from src.automation.intraday import MODEL_LABELS, MODELS, IntradaySetup, setup_row, simulate_exit
from src.automation.ledger import Position

COMPACT_KEYS = ("scan", "ticker", "model", "direction", "score", "stop_pct", "gap_pct", "rsi",
                "f_trend", "f_market", "f_vwap", "f_volume", "f_bonus", "ret", "ret_stop_hold")


def journal_dir(state_dir: Path) -> Path:
    return state_dir / "journal"


def day_path(state_dir: Path, day: date) -> Path:
    return journal_dir(state_dir) / f"{day.isoformat()}.json"


def load_day(state_dir: Path, day: date) -> dict:
    try:
        return json.loads(day_path(state_dir, day).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"date": day.isoformat(), "entries": []}


def save_day(state_dir: Path, day: date, data: dict) -> None:
    path = day_path(state_dir, day)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")


def entry_record(run_at, scan: str, setups: Sequence[IntradaySetup], pick=None,
                 position: Optional[Position] = None, note: str = "", *, practice: bool = False,
                 debates: Optional[List[dict]] = None) -> dict:
    """One entry run, as stored in the journal (``practice``: Claude was asked to trade the best setup)."""
    return {
        "run_at": run_at.isoformat(), "scan": scan,
        "shortlist": [
            {**setup_row(s, scan), "label": s.label, "price": s.price, "stop_price": s.stop_price,
             "target_price": s.target_price, "expected_pct": s.expected_pct, "notes": s.notes}
            for s in setups
        ],
        "pick": None if pick is None else {"action": pick.action, "ticker": pick.ticker, "reasoning": pick.reasoning},
        "opened": None if position is None else json.loads(position.model_dump_json()),
        "note": note,
        "practice": practice,
        "debates": list(debates or []),
    }


def evaluate_shortlist(entry: dict, bars: Dict, day: date, square_off: str, tz, cost_pct: float) -> List[dict]:
    """How each shortlisted setup of one entry run would have ended (net % after costs), best first.

    Each setup is replayed from the moment of the entry run, at the price it had then, with its own stop
    and target - exactly as a paper position would have been.
    """
    picked = None
    pick = entry.get("pick") or {}
    if pick.get("action") in ("BUY", "SELL"):
        picked = pick.get("ticker")
    out = []
    for c in entry.get("shortlist", []):
        long = c["direction"] == "long"
        position = Position(
            symbol=c["ticker"], strategy="intraday", side=c["direction"], quantity=1, entry_price=c["price"],
            stop_loss=c["stop_price"], take_profit=c["target_price"], opened_on=day.isoformat(),
            opened_at=entry["run_at"], max_hold_days=1,
        )
        price, reason = simulate_exit(position, bars.get(c["ticker"]), day, square_off, tz)
        if price is None:
            continue
        gross = (price / c["price"] - 1) * 100 * (1 if long else -1)
        out.append({"ticker": c["ticker"], "model": c["model"], "label": c.get("label", c["model"]),
                    "direction": c["direction"], "ret": round(gross - cost_pct, 3), "reason": reason,
                    "picked": c["ticker"] == picked})
    out.sort(key=lambda o: (o["ret"], o["picked"]), reverse=True)   # a tie goes to Claude's pick
    return out


def model_summary(rows: Sequence[dict], cost_pct: float) -> Dict[str, dict]:
    """Each kind of setup's count, winners and average net % among ``rows`` (first appearances)."""
    out: Dict[str, dict] = {}
    for model in MODELS:
        net = [float(r["ret"]) - cost_pct for r in rows if r["model"] == model and int(r.get("first", 1))]
        if net:
            out[model] = {"n": len(net), "won": sum(1 for x in net if x > 0), "avg_pct": round(sum(net) / len(net), 3)}
    return out


def compact(rows: Sequence[dict]) -> List[dict]:
    """The fields of the day's setup rows worth keeping in the journal (first appearances only)."""
    return [{k: r.get(k) for k in COMPACT_KEYS} for r in rows if int(r.get("first", 1))]


def running_totals(state_dir: Path, cost_pct: float) -> dict:
    """Totals over every reviewed day in the journal."""
    folder = journal_dir(state_dir)
    days, picks, shortlists, rows = [], [], [], []
    for path in sorted(folder.glob("*.json")) if folder.exists() else []:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        review = data.get("review")
        if not review:
            continue
        days.append(data.get("date", path.stem))
        for run in review.get("shortlists", []):
            outcomes = run.get("outcomes", [])
            if not outcomes:
                continue
            shortlists.append(sum(o["ret"] for o in outcomes) / len(outcomes))
            picks.extend(o["ret"] for o in outcomes if o.get("picked"))
        rows.extend(review.get("rows", []))
    by_model = {}
    for model in MODELS:
        net = [float(r["ret"]) - cost_pct for r in rows if r.get("model") == model]
        if net:
            by_model[model] = {"label": MODEL_LABELS[model], "n": len(net), "won": sum(1 for x in net if x > 0),
                               "avg_pct": round(sum(net) / len(net), 3)}
    return {
        "first_day": days[0] if days else None, "days": len(days),
        "picks": len(picks), "pick_avg_pct": round(sum(picks) / len(picks), 3) if picks else None,
        "shortlist_avg_pct": round(sum(shortlists) / len(shortlists), 3) if shortlists else None,
        "models": by_model,
    }


__all__ = [
    "compact",
    "day_path",
    "entry_record",
    "evaluate_shortlist",
    "journal_dir",
    "load_day",
    "model_summary",
    "running_totals",
    "save_day",
]
