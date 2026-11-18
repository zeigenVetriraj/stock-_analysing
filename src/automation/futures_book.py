"""The futures intraday book: the intraday job run on Nifty, Bank Nifty, Fin Nifty and Midcap Nifty futures.

It uses the same code as the large-stock book (the four setups, Claude's pick, practice mode, the
journal, the 3:20 review and the weekly lessons). Only three things differ:

* **Instruments:** ``NIFTY-FUT``, ``BANKNIFTY-FUT``, ``FINNIFTY-FUT`` and ``MIDCPNIFTY-FUT``, priced at the
  Nifty 50, Nifty Bank, Nifty Financial Services and Nifty Midcap Select index levels from Yahoo (a future trades very close to its index during the day; Yahoo has no Indian
  futures prices).
* **Sizing, in lots** (:func:`size_futures`): enough lots that the loss at the stop is about
  ``risk_per_trade_pct`` of the account, at most ``max_lots``, with the margin capped. Lot sizes come
  from NSE's list every entry run, or from the config if it cannot be read.
* **Margin accounting:** only the margin is blocked from the paper cash, while profit or loss is the
  whole contract's move - that is the leverage.

Everything it keeps lives under ``state/futures/`` (paper ledger, journal, backtests, learning), so it
never mixes with the share books.
"""

from __future__ import annotations

import csv
import io
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Dict, Optional

from config.settings import FuturesIntradayRules, TradingRules
from src.automation.intraday import INDEX_TICKER, IntradaySetup
from src.automation.jobs import (
    IST,
    JobDeps,
    JobOutcome,
    inr,
    plain,
    read_track_record,
    run_job,
)
from src.automation.learning import load as load_playbook
from src.automation.ledger import Ledger
from src.automation.universe import BROWSER_HEADERS
from src.ingestion.data_fetcher import to_yfinance_ticker
from src.risk.guardrails import RiskViolation

logger = logging.getLogger(__name__)

FUTURES_DIR = "futures"
# The book's name for each future -> (its NSE symbol, the Yahoo index it is priced from).
INSTRUMENTS: Dict[str, tuple] = {
    "NIFTY-FUT": ("NIFTY", "^NSEI"),
    "BANKNIFTY-FUT": ("BANKNIFTY", "^NSEBANK"),
    "FINNIFTY-FUT": ("FINNIFTY", "NIFTY_FIN_SERVICE.NS"),
    "MIDCPNIFTY-FUT": ("MIDCPNIFTY", "NIFTY_MID_SELECT.NS"),
}
YAHOO_OF = {to_yfinance_ticker(name): yahoo for name, (_, yahoo) in INSTRUMENTS.items()}
LOT_SIZES_URL = "https://nsearchives.nseindia.com/content/fo/fo_mktlots.csv"
INTRADAY_KINDS = ("intraday-open", "intraday-check", "intraday-close")


def futures_folder(state_dir: Path) -> Path:
    return state_dir / FUTURES_DIR


def aliased(fetch: Callable[[list], Dict]) -> Callable[[list], Dict]:
    """Wrap a price fetcher so the futures names are fetched as their Yahoo indices."""
    def wrapped(tickers):
        wanted = list(dict.fromkeys(YAHOO_OF.get(t, t) for t in tickers))
        frames = fetch(wanted)
        return {t: frames[YAHOO_OF.get(t, t)] for t in tickers if YAHOO_OF.get(t, t) in frames}
    return wrapped


def parse_lot_sizes(text: str) -> Dict[str, int]:
    """NSE's ``fo_mktlots.csv``: symbol -> lot size of the nearest expiry (the first number on its row)."""
    sizes: Dict[str, int] = {}
    for row in csv.reader(io.StringIO(text)):
        cells = [c.strip() for c in row]
        if len(cells) < 3 or not cells[1]:
            continue
        for value in cells[2:]:
            if value.isdigit() and int(value) > 0:
                sizes[cells[1].upper()] = int(value)
                break
    return sizes


def fetch_lot_sizes(*, get: Optional[Callable] = None) -> Dict[str, int]:
    """Current lot sizes from NSE ({} if they cannot be downloaded: the config's are used then)."""
    if get is None:
        import requests

        get = requests.get
    try:
        resp = get(LOT_SIZES_URL, headers=BROWSER_HEADERS, timeout=30)
    except Exception as exc:  # noqa: BLE001 - best effort
        logger.warning("Could not download NSE's lot sizes (%s).", exc)
        return {}
    if getattr(resp, "status_code", 0) != 200:
        logger.warning("NSE's lot sizes: HTTP %s.", getattr(resp, "status_code", "?"))
        return {}
    return parse_lot_sizes(resp.text)


@dataclass
class FuturesOrder:
    """A sized futures trade, in the shape the intraday entry job expects."""

    quantity: int                 # units (lots x lot size)
    reference_price: float
    stop_loss_price: float
    take_profit_price: float
    notional: float               # contract value
    margin: float                 # blocked from the paper cash
    lots: int
    lot_size: int
    note: str


def size_futures(setup: IntradaySetup, ledger: Ledger, fr: FuturesIntradayRules,
                 lot_sizes: Dict[str, int]) -> FuturesOrder:
    """Lots for ``setup``: risk at the stop near ``risk_per_trade_pct`` of the account, margin capped.

    Raises :class:`RiskViolation` if even one lot would risk more than twice the budget, or the
    margin for one lot does not fit.
    """
    name = plain(setup.ticker)
    lot_size = lot_sizes.get(name)
    if not lot_size:
        raise RiskViolation(f"No lot size known for {name}.")
    price = setup.price
    account = ledger.cash + sum(p.blocked for p in ledger.positions)
    budget = account * fr.risk_per_trade_pct / 100
    risk_per_lot = price * setup.stop_pct / 100 * lot_size
    lots = int(budget // risk_per_lot) if risk_per_lot > 0 else 0
    if lots < 1:
        if risk_per_lot > 2 * budget:
            raise RiskViolation(
                f"one lot of {name} would risk {inr(risk_per_lot, 0)} at the stop, more than twice the "
                f"{inr(budget, 0)} limit ({fr.risk_per_trade_pct:g}% of the account)"
            )
        lots = 1
    margin_per_lot = price * lot_size * fr.margin_pct / 100
    affordable = int(min(account * fr.max_margin_pct / 100, ledger.cash) // margin_per_lot)
    lots = min(lots, fr.max_lots, affordable)
    if lots < 1:
        raise RiskViolation(f"not enough paper cash for the margin of one lot of {name} ({inr(margin_per_lot, 0)})")
    quantity = lots * lot_size
    notional, margin = round(price * quantity, 2), round(margin_per_lot * lots, 2)
    return FuturesOrder(
        quantity=quantity, reference_price=price, stop_loss_price=setup.stop_price,
        take_profit_price=setup.target_price, notional=notional, margin=margin, lots=lots, lot_size=lot_size,
        note=(f"{lots} lot(s) of {lot_size} = {quantity} units; contract value {inr(notional, 0)}, margin blocked "
              f"{inr(margin, 0)} ({fr.margin_pct:g}%); risk at the stop about {inr(risk_per_lot * lots, 0)}. "
              "Priced at the index level."),
    )


def futures_deps(deps: JobDeps, fr: FuturesIntradayRules, lot_sizes: Dict[str, int]) -> JobDeps:
    """The share job's dependencies, pointed at the futures book's folder, instruments and sizing."""
    folder = futures_folder(deps.state_dir)
    return replace(
        deps,
        state_dir=folder,
        universe=deps.universe.model_copy(update={"large_caps": list(INSTRUMENTS)}),
        rules=deps.rules.model_copy(update={"intraday": fr}),
        intraday_history=aliased(deps.intraday_history),
        bulk_history=aliased(deps.bulk_history),
        fundamentals=lambda ticker, price: None,            # an index has no results dates
        playbook=lambda: load_playbook(folder / "learning"),
        track_record=lambda: read_track_record(folder / "backtests"),
        entry_filter=None,
        sizer=lambda setup, ledger: size_futures(setup, ledger, fr, lot_sizes),
        book_label="Futures",
    )


def current_lot_sizes(fr: FuturesIntradayRules, fetched: Dict[str, int]) -> Dict[str, int]:
    """Lot size per instrument: NSE's when known, else the config's."""
    sizes = {}
    for name, (symbol, _) in INSTRUMENTS.items():
        size = fetched.get(symbol) or fr.lot_sizes.get(symbol)
        if size:
            sizes[name] = int(size)
    return sizes


def run_futures_intraday(kind: str, deps: JobDeps, *,
                         lot_sizes_source: Callable[[], Dict[str, int]] = fetch_lot_sizes) -> Optional[JobOutcome]:
    """Run one intraday job on the futures book (None when the book is switched off)."""
    fr = deps.rules.futures_intraday
    if fr is None or not fr.enabled or kind not in INTRADAY_KINDS:
        return None
    fetched = lot_sizes_source() if kind == "intraday-open" else {}
    return run_job(kind, futures_deps(deps, fr, current_lot_sizes(fr, fetched)))


def run_futures_backtest(state_dir: Path, rules: TradingRules, notifier) -> Optional[str]:
    """The weekly backtest and lessons for the futures book; returns the message sent."""
    from src.automation import intraday_backtest as bt
    from src.automation import learning, scorecard
    from src.automation.screener import bulk_daily_history, bulk_history

    fr = rules.futures_intraday
    if fr is None or not fr.enabled:
        return None
    folder = futures_folder(state_dir)
    out = folder / "backtests"
    yahoo = list(dict.fromkeys([y for _, y in INSTRUMENTS.values()] + [INDEX_TICKER]))
    five_min: Dict = {}
    for period in ("60d", "59d", "1mo"):
        five_min = bulk_history(yahoo, period=period, interval="5m")
        if len(five_min) >= len(yahoo):
            break

    def rename(frames: Dict) -> Dict:
        return {to_yfinance_ticker(name): frames[y] for name, (_, y) in INSTRUMENTS.items() if y in frames}

    bars = rename(five_min)
    if INDEX_TICKER in five_min:
        bars[INDEX_TICKER] = five_min[INDEX_TICKER]
    bt.save_frames(bars, out, bt.BARS_PREFIX, IST, refresh_days=bt.BARS_REFRESH_DAYS)
    bt.save_frames(rename(bulk_daily_history([y for _, y in INSTRUMENTS.values()])), out, bt.DAILY_PREFIX, IST)
    _, rows, flips = bt.run(out, tz=IST, fetch=False, cost_pct=fr.cost_pct,
                            max_stop_pct=fr.stop_loss_pct * 100, reward=fr.take_profit_pct / fr.stop_loss_pct)
    playbook, _ = learning.build(rows, flips, folder / "learning", cost_pct=fr.cost_pct)
    text = ("*Index futures* - " + bt.headline(rows, fr.cost_pct, coin_flip=flips)
            + "\n\n*Futures weekly review*\n" + learning.lessons_summary(playbook)
            + "\nFull review: state/futures/learning/learning_report.md\n\n" + scorecard.weekly(folder))
    notifier.send(text)
    return text


__all__ = [
    "FUTURES_DIR",
    "INSTRUMENTS",
    "FuturesOrder",
    "aliased",
    "current_lot_sizes",
    "fetch_lot_sizes",
    "futures_deps",
    "futures_folder",
    "parse_lot_sizes",
    "run_futures_backtest",
    "run_futures_intraday",
    "size_futures",
]
