"""The penny intraday book: the intraday job run a second time, on penny stocks.

It uses exactly the same code as the large-stock book (the four setups, Claude's pick, practice
mode, the risk layer, the journal, the 3:20 review and the weekly lessons) but everything it keeps
lives under ``state/penny/``, so it never mixes with the large-stock account:

* ``state/penny/paper_ledger.json``: its own paper account (the same starting cash);
* ``state/penny/stocks.json``: today's list of penny stocks, with each one's circuit limit;
* ``state/penny/journal/``, ``state/penny/backtests/``, ``state/penny/learning/``.

The list is rebuilt once a trading day (at the entry run, or at the square-off if the entry did not
run): EQ-series stocks from NSE's bhavcopy inside the price range, the most traded by their 20-day
average traded value. A setup is skipped unless its entry, stop and target all sit inside the
stock's daily circuit limits with a margin, because a stock frozen at its limit cannot be sold.
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Optional

from config.settings import PennyIntradayRules, TradingRules
from src.automation.intraday import IntradaySetup
from src.automation.jobs import (
    IST,
    JobDeps,
    JobOutcome,
    plain,
    read_track_record,
    run_job,
)
from src.automation.learning import load as load_playbook
from src.automation.universe import (
    parse_bhavcopy,
    penny_candidates,
    penny_intraday_list,
)
from src.ingestion.data_fetcher import to_yfinance_ticker

logger = logging.getLogger(__name__)

PENNY_DIR = "penny"
LIST_FILE = "stocks.json"
INTRADAY_KINDS = ("intraday-open", "intraday-check", "intraday-close")


def penny_folder(state_dir: Path) -> Path:
    return state_dir / PENNY_DIR


def circuit_problem(setup: IntradaySetup, band_pct: Optional[float], margin_pct: float) -> Optional[str]:
    """A reason to skip ``setup`` if its entry, stop or target is too close to (or beyond) a circuit limit.

    ``band_pct`` is the stock's daily price band (None: no band, as for stocks with derivatives).
    The limits are measured from yesterday's close.
    """
    if band_pct is None:
        return None
    previous = setup.features.pdc if setup.features is not None else None
    if not previous:
        return "no previous close to place its circuit limits"
    upper = previous * (1 + band_pct / 100) * (1 - margin_pct / 100)
    lower = previous * (1 - band_pct / 100) * (1 + margin_pct / 100)
    prices = (setup.price, setup.stop_price, setup.target_price)
    if max(prices) >= upper or min(prices) <= lower:
        return "too close to its daily circuit limit"
    return None


def build_list(deps: JobDeps, pr: PennyIntradayRules) -> dict:
    """Today's penny stocks: bhavcopy filter, 20-day average traded value, circuit limits."""
    day, text = deps.bhavcopy(deps.today)
    candidates = penny_candidates(parse_bhavcopy(text), pr, limit=max(150, pr.max_stocks * 3))
    daily = deps.bulk_history([to_yfinance_ticker(c.symbol) for c in candidates]) if candidates else {}
    picked = penny_intraday_list(candidates, daily, pr, to_yfinance_ticker)
    bands = deps.price_bands()
    stocks = [
        {"symbol": symbol, "avg_turnover_cr": value,
         "band_pct": bands.get(symbol, pr.default_band_pct)}
        for symbol, value in picked
    ]
    logger.info("Penny list: %d stocks from the %s bhavcopy (%d price bands read).", len(stocks), day, len(bands))
    return {
        "date": deps.today.isoformat(), "bhavcopy_date": day.isoformat(),
        "bands": "NSE" if bands else f"not available; {pr.default_band_pct:g}% assumed",
        "stocks": stocks,
    }


def load_or_build_list(deps: JobDeps, pr: PennyIntradayRules) -> dict:
    """Today's list, built once and kept in state/penny/stocks.json for the later runs of the day."""
    path = penny_folder(deps.state_dir) / LIST_FILE
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved.get("date") == deps.today.isoformat() and saved.get("stocks"):
            return saved
    except (OSError, ValueError):
        pass
    listing = build_list(deps, pr)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(listing, indent=1) + "\n", encoding="utf-8")
    return listing


def penny_deps(deps: JobDeps, pr: PennyIntradayRules, listing: Optional[dict] = None) -> JobDeps:
    """The large-stock job's dependencies, pointed at the penny book's folder, stocks and rules."""
    folder = penny_folder(deps.state_dir)
    stocks = (listing or {}).get("stocks", [])
    bands = {s["symbol"]: s.get("band_pct") for s in stocks}

    def entry_filter(setup: IntradaySetup) -> Optional[str]:
        return circuit_problem(setup, bands.get(plain(setup.ticker), pr.default_band_pct), pr.circuit_margin_pct)

    return replace(
        deps,
        state_dir=folder,
        universe=deps.universe.model_copy(update={"large_caps": [s["symbol"] for s in stocks]}),
        rules=deps.rules.model_copy(update={"intraday": pr}),
        playbook=lambda: load_playbook(folder / "learning"),
        track_record=lambda: read_track_record(folder / "backtests"),
        entry_filter=entry_filter,
        book_label="Penny",
    )


def run_penny_intraday(kind: str, deps: JobDeps) -> Optional[JobOutcome]:
    """Run one intraday job on the penny book (None when the book is switched off)."""
    pr = deps.rules.penny_intraday
    if pr is None or not pr.enabled or kind not in INTRADAY_KINDS:
        return None
    listing = None
    if kind == "intraday-open":
        listing = load_or_build_list(deps, pr)
    elif kind == "intraday-close":
        try:
            listing = load_or_build_list(deps, pr)
        except Exception as exc:  # noqa: BLE001 - the square-off must happen even without a list
            logger.warning("No penny list for the review (%s); squaring off anyway.", exc)
    return run_job(kind, penny_deps(deps, pr, listing))


def run_penny_backtest(state_dir: Path, rules: TradingRules, notifier) -> Optional[str]:
    """The weekly backtest and lessons for the penny book (on the latest list); the message sent."""
    from src.automation import intraday_backtest, learning, scorecard

    pr = rules.penny_intraday
    if pr is None or not pr.enabled:
        return None
    folder = penny_folder(state_dir)
    try:
        listing = json.loads((folder / LIST_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        listing = {}
    tickers = [to_yfinance_ticker(s["symbol"]) for s in listing.get("stocks", [])]
    if not tickers:
        text = "*Penny weekly review*\nNo penny stock list yet (it is built at the first penny entry run)."
        notifier.send(text)
        return text
    _, rows, flips = intraday_backtest.run(
        folder / "backtests", tz=IST, tickers=tickers, fetch=True, cost_pct=pr.cost_pct,
        max_stop_pct=pr.stop_loss_pct * 100, reward=pr.take_profit_pct / pr.stop_loss_pct,
    )
    playbook, _ = learning.build(rows, flips, folder / "learning", cost_pct=pr.cost_pct)
    text = ("*Penny stocks* - " + intraday_backtest.headline(rows, pr.cost_pct, coin_flip=flips)
            + "\n\n*Penny weekly review*\n" + learning.lessons_summary(playbook)
            + "\nFull review: state/penny/learning/learning_report.md\n\n" + scorecard.weekly(folder))
    notifier.send(text)
    return text


__all__ = [
    "LIST_FILE",
    "PENNY_DIR",
    "build_list",
    "circuit_problem",
    "load_or_build_list",
    "penny_deps",
    "penny_folder",
    "run_penny_backtest",
    "run_penny_intraday",
]
