"""The automated jobs: daily swing, weekly penny, paper intraday and a report-only run.

Each run:

1. Loads the ledger for the trading mode (paper by default).
2. Reads your Dhan holdings (read-only) when a valid token is configured.
3. Downloads daily prices for the universe, held stocks and the watchlist.
   If NSE did not trade today (weekend / holiday) nothing is traded.
4. Closes bot positions that hit their stop-loss, target or time limit.
5. Screens the universe (technicals, then fundamentals for the best), keeps
   ``top_n``, and lets Claude pick at most one stock.
6. Sends the pick through the hardcoded risk layer, then opens it (paper) or
   places the order (live, only when every live lock is satisfied).
7. Sends a WhatsApp report and saves the ledger.

Live trading needs ALL of: AUTOTRADE_MODE=live, DHAN_ENV=live,
DHAN_ENABLE_LIVE_TRADING=true, Dhan credentials, and a runner that is not a
GitHub-hosted machine (Dhan only accepts orders from a whitelisted static IP).
If any is missing, a live run reports but does not trade.

Intraday (``intraday-open`` after the opening range, hourly ``intraday-check``
alerts, ``intraday-close`` near 15:15) is paper-only for now: live intraday
needs broker-side stop orders.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

from config.settings import Settings, TradingRules
from src.automation.debate import DebateAgent, data_packet
from src.automation.intraday import (
    INDEX_TICKER,
    MODEL_LABELS,
    MODELS,
    IntradaySetup,
    all_setups,
    check_exit,
    compute_features,
    daily_trend_up,
    index_change_pct,
    setup_row,
    shortlist,
    simulate_exit,
    to_ist,
)
from src.automation.intraday_backtest import STATS_FILE, run_backtest
from src.automation.journal import (
    compact,
    entry_record,
    evaluate_shortlist,
    load_day,
    model_summary,
    running_totals,
    save_day,
)
from src.automation.learning import (
    HOLD_TARGET_PCT,
    exit_for,
    expected_pct,
    lessons_summary,
)
from src.automation.learning import load as load_playbook
from src.automation.ledger import (
    STATE_DIR,
    ClosedTrade,
    Ledger,
    Position,
    exit_reason,
    ledger_path,
)
from src.automation.picker import PickDecision, PickerAgent
from src.automation.screener import (
    RankedStock,
    bulk_daily_history,
    bulk_intraday_history,
    rank,
    score_technicals,
)
from src.automation.universe import (
    Universe,
    fetch_latest_bhavcopy,
    fetch_price_bands,
    parse_bhavcopy,
    penny_candidates,
)
from src.ingestion.data_fetcher import (
    MarketSnapshot,
    fetch_market_snapshot,
    to_yfinance_ticker,
)
from src.ingestion.fundamentals import FundamentalSnapshot, fetch_fundamentals
from src.risk.guardrails import (
    AccountState,
    RiskInterceptor,
    RiskViolation,
    ValidatedOrder,
    check_earnings_blackout,
    get_account_state,
)

logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))
RESULTS_HORIZON_DAYS = 30
NSE_OPEN, NSE_CLOSE = time(9, 15), time(15, 30)   # regular session, IST
MAX_PRICE_AGE = timedelta(minutes=20)             # older 5-minute bars than this = stale prices, no entry


def today_ist() -> date:
    return datetime.now(IST).date()


# --------------------------------------------------------------------------- #
# Small pure helpers
# --------------------------------------------------------------------------- #
def inr(amount: float, decimals: int = 2) -> str:
    """Indian digit grouping: 1234567.8 -> 'Rs 12,34,567.80'."""
    sign = "-" if amount < 0 else ""
    whole, _, frac = f"{abs(amount):.{decimals}f}".partition(".")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        whole = ",".join(groups + [tail])
    return f"{sign}Rs {whole}" + (f".{frac}" if decimals else "")


def plain(ticker: str) -> str:
    return ticker.split(".")[0]


def latest_prices(history: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    prices = {}
    for ticker, frame in history.items():
        close = frame["Close"].dropna() if "Close" in frame else pd.Series(dtype=float)
        if not close.empty:
            prices[ticker] = round(float(close.iloc[-1]), 2)
    return prices


def is_market_open(history: Dict[str, pd.DataFrame], today: date) -> bool:
    """True when at least one stock has a daily bar dated today (IST)."""
    for frame in history.values():
        if frame.empty:
            continue
        last = pd.Timestamp(frame.index[-1])
        if last.tzinfo is not None:
            last = last.tz_convert(IST)
        if last.date() == today:
            return True
    return False


def in_market_hours(now: datetime) -> bool:
    """True during NSE's regular session (09:15-15:30 IST). Holidays are checked separately."""
    local = now.astimezone(IST) if now.tzinfo else now
    return NSE_OPEN <= local.time() <= NSE_CLOSE


def live_blockers(settings: Settings, runner_environment: str) -> List[str]:
    """Reasons a live run may not trade (empty list = all locks satisfied)."""
    problems = []
    if settings.dhan_env != "live":
        problems.append("DHAN_ENV is not 'live'")
    if not settings.dhan_enable_live_trading:
        problems.append("DHAN_ENABLE_LIVE_TRADING is not true")
    if not settings.has_dhan:
        problems.append("Dhan credentials are missing")
    if runner_environment.strip().lower() == "github-hosted":
        problems.append(
            "this is a GitHub-hosted runner with no static IP; Dhan only accepts orders "
            "from a whitelisted static IP, so use a self-hosted runner"
        )
    return problems


def intraday_trading_rules(rules: TradingRules) -> TradingRules:
    """The shared rules with the intraday limits swapped in (short selling allowed)."""
    i = rules.intraday
    if i is None:
        raise ValueError("No intraday rules configured.")
    return rules.model_copy(update={
        "max_position_pct": i.max_position_pct,
        "stop_loss_pct": i.stop_loss_pct,
        "take_profit_pct": i.take_profit_pct,
        "allow_short_selling": True,
        "max_open_positions": i.max_trades_per_day,
        "max_hold_days": 1,
    })


def _clock(value: str) -> time:
    hours, minutes = (int(x) for x in value.split(":"))
    return time(hours, minutes)


def penny_trading_rules(rules: TradingRules) -> TradingRules:
    """The shared rules with the stricter penny limits swapped in."""
    p = rules.penny
    if p is None:
        raise ValueError("No penny rules configured.")
    return rules.model_copy(update={
        "max_position_pct": p.max_position_pct,
        "stop_loss_pct": p.stop_loss_pct,
        "take_profit_pct": p.take_profit_pct,
        "max_open_positions": p.max_open_positions,
        "max_hold_days": p.max_hold_days,
    })


# --------------------------------------------------------------------------- #
# Dependencies (real by default, replaceable in tests)
# --------------------------------------------------------------------------- #
def _safe_fundamentals(ticker: str, price: Optional[float]) -> Optional[FundamentalSnapshot]:
    try:
        return fetch_fundamentals(ticker, current_price=price)
    except Exception as exc:  # noqa: BLE001 - best-effort
        logger.info("Fundamentals unavailable for %s (%s).", ticker, exc)
        return None


def _safe_snapshot(ticker: str) -> Optional[MarketSnapshot]:
    try:
        return fetch_market_snapshot(ticker)
    except Exception as exc:  # noqa: BLE001 - best-effort
        logger.info("Snapshot unavailable for %s (%s).", ticker, exc)
        return None


def _dhan_holdings(settings: Settings) -> List[dict]:
    from src.execution.dhan_client import DhanClient

    return DhanClient(settings).get_holdings()


def _resolve_security(symbol: str) -> str:
    from src.execution.instruments import DhanInstruments

    return DhanInstruments().resolve(symbol, "NSE")


def _executor(settings: Settings):
    from src.execution.order_executor import OrderExecutor

    return OrderExecutor(settings, dry_run=False)


def _playbook() -> Optional[dict]:
    """What the weekly review has learned (state/learning/playbook.json), or None before it first runs."""
    return load_playbook(STATE_DIR / "learning")


def read_track_record(folder: Path) -> Dict[str, dict]:
    """How each intraday model did in the backtest saved in ``folder`` ({} if it has not been run)."""
    try:
        return json.loads((folder / STATS_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _track_record() -> Dict[str, dict]:
    """How each intraday model did in the latest backtest ({} if it has not been run)."""
    return read_track_record(STATE_DIR / "backtests")


def record_line(record: Optional[dict]) -> str:
    """One sentence on a model's backtest record ("" if there is none)."""
    if not record or not record.get("trades"):
        return ""
    return (
        f"Backtest record of this kind of setup: {record['trades']} trades, {record['win_rate']:.0f}% winners, "
        f"{record['avg_net_pct']:+.2f}% a trade after costs ({record['verdict']})."
    )


@dataclass
class JobDeps:
    settings: Settings
    rules: TradingRules
    universe: Universe
    today: date
    notifier: object
    state_dir: Path = STATE_DIR
    runner_environment: str = ""
    bulk_history: Callable[[List[str]], Dict[str, pd.DataFrame]] = bulk_daily_history
    intraday_history: Callable[[List[str]], Dict[str, pd.DataFrame]] = bulk_intraday_history
    now: Optional[datetime] = None
    fundamentals: Callable[[str, Optional[float]], Optional[FundamentalSnapshot]] = _safe_fundamentals
    snapshot: Callable[[str], Optional[MarketSnapshot]] = _safe_snapshot
    bhavcopy: Callable[[date], Tuple[date, str]] = fetch_latest_bhavcopy
    holdings: Callable[[Settings], List[dict]] = _dhan_holdings
    picker_factory: Callable[[Settings], object] = PickerAgent
    # Bull vs bear, then a judge, before each intraday trade (when the rules' debate block is on).
    debater_factory: Callable[[Settings], object] = DebateAgent
    resolve_security: Callable[[str], str] = _resolve_security
    executor_factory: Callable[[Settings], object] = _executor
    track_record: Callable[[], Dict[str, dict]] = _track_record
    playbook: Callable[[], Optional[dict]] = _playbook
    price_bands: Callable[[], Dict[str, Optional[float]]] = fetch_price_bands
    # Intraday entry: a setup this returns a reason for is skipped (the penny book's circuit check).
    entry_filter: Optional[Callable[[IntradaySetup], Optional[str]]] = None
    # "Penny" for the penny intraday book: prefixes the message titles.
    book_label: str = ""
    # Intraday entry: sizes the trade instead of the shares risk layer (the futures book, in lots).
    # Called with (setup, ledger); returns an object like ValidatedOrder plus ``margin`` and ``note``,
    # or raises RiskViolation.
    sizer: Optional[Callable[[IntradaySetup, Ledger], object]] = None


# --------------------------------------------------------------------------- #
# The report
# --------------------------------------------------------------------------- #
@dataclass
class Report:
    title: str
    today: date
    mode: str
    warnings: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    holdings: List[str] = field(default_factory=list)
    positions: List[str] = field(default_factory=list)
    account: str = ""
    exits: List[str] = field(default_factory=list)
    review: List[str] = field(default_factory=list)
    experience: List[str] = field(default_factory=list)
    shortlist: List[str] = field(default_factory=list)
    debate: List[str] = field(default_factory=list)
    pick: List[str] = field(default_factory=list)
    results: List[str] = field(default_factory=list)
    watchlist: List[str] = field(default_factory=list)

    def render(self) -> str:
        out = [f"*{self.title}* - {self.today.strftime('%a %d %b %Y')} ({self.mode})"]
        sections = [
            ("Heads up", self.warnings),
            ("", self.notes),
            ("Your Dhan holdings", self.holdings),
            (f"Bot positions ({self.mode})", self.positions + ([self.account] if self.account else [])),
            ("Closed today", self.exits),
            ("Today's review", self.review),
            ("Experience so far", self.experience),
            ("Today's shortlist", self.shortlist),
            ("Debate (bull vs bear, then the judge)", self.debate),
            ("Pick", self.pick),
            ("Watchlist", self.watchlist),
            ("Results coming up", self.results),
        ]
        for heading, lines in sections:
            if not lines:
                continue
            out.append("")
            if heading:
                out.append(f"*{heading}*")
            out.extend(lines)
        out.append("")
        out.append("_Automated analysis, not financial advice._")
        return "\n".join(out)


# --------------------------------------------------------------------------- #
# The job
# --------------------------------------------------------------------------- #
@dataclass
class JobOutcome:
    report: Report
    pick: Optional[PickDecision] = None
    order: Optional[ValidatedOrder] = None
    opened: Optional[Position] = None
    closed: List[ClosedTrade] = field(default_factory=list)
    traded: bool = False


def run_job(kind: str, deps: JobDeps) -> JobOutcome:
    """Run ``daily`` (swing), ``penny``, ``intraday-open``, ``intraday-close`` or ``report``."""
    if kind in ("intraday-open", "intraday-check", "intraday-close"):
        return _run_intraday(kind, deps)
    if kind == "close-out":
        return _run_close_out(deps)
    if kind not in ("daily", "penny", "report"):
        raise ValueError(f"Unknown job {kind!r}.")
    s, rules, today = deps.settings, deps.rules, deps.today
    mode = s.autotrade_mode
    titles = {"daily": "Daily swing report", "penny": "Weekly penny report", "report": "Portfolio report"}
    report = Report(title=titles[kind], today=today, mode=mode)
    outcome = JobOutcome(report=report)

    trading_allowed = kind != "report"
    if mode == "live" and trading_allowed:
        blockers = live_blockers(s, deps.runner_environment)
        if blockers:
            trading_allowed = False
            report.warnings.append("Live trading is OFF for this run: " + "; ".join(blockers) + ".")

    path = ledger_path(mode, deps.state_dir)
    ledger = Ledger.load(path, mode=mode, starting_cash=s.simulated_total_equity)

    # --- candidates -------------------------------------------------------- #
    strategy = "penny" if kind == "penny" else "swing"
    strategy_rules = rules
    candidates: List[str] = []
    if kind == "daily":
        if rules.swing_enabled:
            candidates = [to_yfinance_ticker(sym) for sym in deps.universe.large_caps]
        else:
            report.notes.append("The swing job is switched off in config/trading_rules.json; no new swing trades.")
    elif kind == "penny":
        if rules.penny is None or not rules.penny.enabled:
            report.notes.append("The penny job is switched off in config/trading_rules.json.")
        else:
            strategy_rules = penny_trading_rules(rules)
            try:
                day, text = deps.bhavcopy(today)
                found = penny_candidates(parse_bhavcopy(text), rules.penny, deps.universe.penny_max_candidates)
                candidates = [to_yfinance_ticker(c.symbol) for c in found]
                report.notes.append(
                    f"{len(candidates)} liquid NSE stocks between Rs {rules.penny.min_price:g} and "
                    f"Rs {rules.penny.max_price:g} (NSE list of {day.strftime('%d %b')})."
                )
            except Exception as exc:  # noqa: BLE001 - UniverseError or a parsing problem
                report.warnings.append(f"Could not get NSE's stock list ({exc}); no penny screen this week.")

    # --- Dhan holdings (read-only) ------------------------------------------ #
    dhan_rows: List[dict] = []
    if s.has_dhan:
        try:
            dhan_rows = deps.holdings(s)
        except Exception as exc:  # noqa: BLE001
            report.holdings.append(
                f"Unavailable ({type(exc).__name__}). The Dhan token lasts 24 hours - update DHAN_ACCESS_TOKEN."
            )
    else:
        report.holdings.append("Not connected (no DHAN_ACCESS_TOKEN).")
    holding_tickers = [
        to_yfinance_ticker(str(r.get("tradingSymbol", "")))
        for r in dhan_rows if r.get("tradingSymbol")
    ]

    watch = [to_yfinance_ticker(sym) for sym in deps.universe.watchlist]
    wanted = list(dict.fromkeys(candidates + [p.symbol for p in ledger.positions] + holding_tickers + watch))
    history = deps.bulk_history(wanted) if wanted else {}
    prices = latest_prices(history)
    market_open = is_market_open(history, today)
    if trading_allowed and not market_open:
        report.notes.append("NSE did not trade today (weekend, holiday or no price data), so nothing was traded.")
        trading_allowed = False

    fundamentals: Dict[str, Optional[FundamentalSnapshot]] = {}

    # --- exits --------------------------------------------------------------- #
    if trading_allowed:
        for position in list(ledger.positions):
            if position.strategy == "intraday" and position.opened_on == today.isoformat():
                continue  # the intraday square-off run handles today's intraday trade
            price = prices.get(position.symbol)
            reason = exit_reason(position, price, today)
            if reason is None or price is None:
                continue
            if mode == "live":
                try:
                    sell = ValidatedOrder(
                        ticker=position.symbol, action="EXIT", side="sell", is_tradeable=True,
                        quantity=position.quantity, reference_price=price,
                        notional=round(price * position.quantity, 2), reasoning=reason,
                    )
                    deps.executor_factory(s).execute(
                        sell, security_id=deps.resolve_security(plain(position.symbol)), exchange_segment="NSE_EQ"
                    )
                except Exception as exc:  # noqa: BLE001
                    report.warnings.append(f"Could not sell {plain(position.symbol)} ({reason}): {exc}")
                    continue
            trade = ledger.close(position, price, today, reason)
            outcome.closed.append(trade)
            report.exits.append(
                f"- {plain(trade.symbol)} {reason}: {trade.pnl_pct:+.1f}% ({inr(trade.pnl)})"
            )

    # --- screen, pick, risk, trade ------------------------------------------ #
    if trading_allowed and candidates:
        now = deps.now or datetime.now(IST)
        opened_today = [
            x.symbol for x in list(ledger.positions) + list(ledger.closed)
            if x.strategy == strategy and x.opened_on == today.isoformat()
        ]
        if opened_today:
            # One new position per strategy per day, however many times the job runs.
            report.notes.append(
                f"Already opened a {strategy} position today ({plain(opened_today[0])}); not screening again."
            )
        elif mode == "live" and not in_market_hours(now):
            report.notes.append(
                f"This run started at {now.astimezone(IST):%H:%M} IST, outside market hours "
                f"({NSE_OPEN:%H:%M}-{NSE_CLOSE:%H:%M}); no new position."
            )
        else:
            if not in_market_hours(now):
                report.warnings.append(
                    f"This run started at {now.astimezone(IST):%H:%M} IST, outside market hours: any paper "
                    "buy below uses the last traded price (the day's close), which you could not get live."
                )
            _screen_and_trade(kind, strategy, strategy_rules, candidates, history, prices, fundamentals,
                              ledger, deps, outcome)

    # --- report sections ----------------------------------------------------- #
    _describe_holdings(dhan_rows, prices, history, strategy_rules, report)
    _describe_positions(ledger, prices, today, report)
    for ticker in watch:
        if ticker in prices:
            report.watchlist.append(f"- {plain(ticker)} {inr(prices[ticker])}{_trend_note(history.get(ticker))}")
    _describe_results(
        list(dict.fromkeys(holding_tickers + [p.symbol for p in ledger.positions] + watch
                           + ([outcome.pick.ticker] if outcome.pick and outcome.pick.ticker else []))),
        fundamentals, prices, deps, today, report,
    )

    ledger.save(path)
    deps.notifier.send(report.render())
    return outcome


def _screen_and_trade(kind, strategy, strategy_rules, candidates, history, prices, fundamentals,
                      ledger, deps, outcome) -> None:
    report, s, u = outcome.report, deps.settings, deps.universe
    technicals = [t for t in (score_technicals(c, history.get(c)) for c in candidates) if t is not None]
    if kind == "penny" and deps.rules.penny is not None:
        p = deps.rules.penny
        technicals = [
            t for t in technicals
            if p.min_price <= t.price <= p.max_price and (t.avg_turnover_cr or 0) >= p.min_avg_turnover_cr
        ]
    technicals.sort(key=lambda t: t.score, reverse=True)
    shortlist = technicals[: u.fundamentals_shortlist]
    for t in shortlist:
        fundamentals[t.ticker] = deps.fundamentals(t.ticker, t.price)
    ranked: List[RankedStock] = rank(shortlist, fundamentals, u.top_n)
    if not ranked:
        report.notes.append("No stock passed the screen today.")
        return
    for i, r in enumerate(ranked, start=1):
        report.shortlist.append(
            f"{i}. {plain(r.ticker)} {r.combined:.0f} (tech {r.technical:.0f} / fund {r.fundamental:.0f}) {inr(r.price)}"
        )

    snapshots = {r.ticker: snap for r in ranked if (snap := deps.snapshot(r.ticker)) is not None}
    try:
        pick = deps.picker_factory(s).pick(strategy, ranked, snapshots)
    except Exception as exc:  # noqa: BLE001
        report.warnings.append(f"Claude could not make a pick today ({exc}).")
        return
    outcome.pick = pick
    if pick.action == "HOLD" or not pick.ticker:
        report.pick.append("HOLD - no clean setup today.")
        report.pick.append(pick.reasoning)
        return

    ticker = pick.ticker
    price = prices.get(ticker) or (snapshots[ticker].current_price if ticker in snapshots else None)
    report.pick.append(f"BUY {plain(ticker)} at about {inr(price or 0)}, target {inr(pick.target_price or 0)}")
    report.pick.append(pick.reasoning)
    if price is None:
        report.pick.append("No live price for the pick; nothing traded.")
        return
    if ledger.holds(ticker):
        report.pick.append("Already held by the bot; not buying more.")
        return

    mode = s.autotrade_mode
    if mode == "live":
        base = get_account_state(s)
        account = base.model_copy(update={"open_positions": ledger.count(strategy)})
    else:
        account = AccountState(
            total_equity=max(ledger.equity(prices), 1.0), cash_balance=max(ledger.cash, 0.0),
            open_positions=ledger.count(strategy), source="paper",
        )
    funda = fundamentals.get(ticker)
    try:
        order = RiskInterceptor(strategy_rules).validate(
            pick.to_analyst_decision(price), current_price=price, account_state=account,
            next_earnings_date=funda.next_earnings_date if funda else None,
        )
    except RiskViolation as exc:
        report.pick.append(f"Risk layer refused it: {exc}")
        return
    outcome.order = order
    if not order.is_tradeable:
        return

    position = Position(
        symbol=ticker, strategy=strategy, quantity=order.quantity, entry_price=order.reference_price,
        stop_loss=order.stop_loss_price, take_profit=order.take_profit_price,
        opened_on=deps.today.isoformat(), max_hold_days=strategy_rules.max_hold_days,
        reasoning=pick.reasoning,
    )
    if mode == "live":
        try:
            result = deps.executor_factory(s).execute(
                order, security_id=deps.resolve_security(plain(ticker)), exchange_segment="NSE_EQ"
            )
        except Exception as exc:  # noqa: BLE001
            report.pick.append(f"Order failed: {exc}")
            return
        if not getattr(result, "submitted", False):
            report.pick.append(f"Order not sent ({getattr(result, 'status', 'unknown')}).")
            return
    ledger.open(position)
    outcome.opened, outcome.traded = position, True
    report.pick.append(
        f"{'Bought' if mode == 'live' else 'Paper-bought'} {order.quantity} x {inr(order.reference_price)} "
        f"= {inr(order.notional)}; stop {inr(order.stop_loss_price)}, target {inr(order.take_profit_price)}"
    )


def _trend_note(frame: Optional[pd.DataFrame]) -> str:
    if frame is None or "Close" not in frame:
        return ""
    close = frame["Close"].dropna()
    if len(close) < 50:
        return ""
    return " (above SMA50)" if close.iloc[-1] > close.tail(50).mean() else " (below SMA50)"


def _describe_holdings(rows, prices, history, rules, report: Report) -> None:
    if not rows:
        if not report.holdings:
            report.holdings.append("None.")
        return
    for row in rows:
        sym = str(row.get("tradingSymbol", "")).upper()
        qty = int(float(row.get("totalQty") or 0))
        avg = float(row.get("avgCostPrice") or 0)
        ticker = to_yfinance_ticker(sym)
        price = prices.get(ticker)
        if price is None or avg <= 0:
            report.holdings.append(f"- {sym} {qty} @ {inr(avg)} (no price)")
            continue
        change = (price / avg - 1) * 100
        status = ""
        if change <= -rules.stop_loss_pct * 100:
            status = " - below the stop-loss level"
        elif change >= rules.take_profit_pct * 100:
            status = " - past the target level"
        report.holdings.append(
            f"- {sym} {qty} @ {inr(avg)} -> {inr(price)} ({change:+.1f}%, {inr((price - avg) * qty)})"
            f"{_trend_note(history.get(ticker))}{status}"
        )


def _describe_positions(ledger: Ledger, prices, today: date, report: Report) -> None:
    for p in ledger.positions:
        price = prices.get(p.symbol, p.entry_price)
        report.positions.append(
            f"- {plain(p.symbol)} ({p.strategy}{' short' if p.side == 'short' else ''}) {p.quantity} @ "
            f"{inr(p.entry_price)} -> {inr(price)} ({p.pnl_at(price) / p.cost * 100:+.1f}%), stop {inr(p.stop_loss)} / target "
            f"{inr(p.take_profit)}, day {p.days_held(today)}"
        )
    if ledger.mode == "paper":
        report.account = (
            f"Paper account: equity {inr(ledger.equity(prices), 0)}, cash {inr(ledger.cash, 0)}, "
            f"realised P&L {inr(ledger.realised_pnl(), 0)} over {len(ledger.closed)} closed trade(s)"
        )
    elif not ledger.positions:
        report.positions.append("None.")


def disabled_strategies(rules: TradingRules) -> List[str]:
    """Automated strategies that are switched off in the rules file."""
    off = []
    if not rules.swing_enabled:
        off.append("swing")
    if rules.penny is None or not rules.penny.enabled:
        off.append("penny")
    if rules.intraday is None or not rules.intraday.enabled:
        off.append("intraday")
    return off


def _run_close_out(deps: JobDeps) -> JobOutcome:
    """Close every open paper position that belongs to a switched-off strategy.

    Used once when a strategy is stopped, so its positions do not sit in the
    ledger unmanaged. Each closes at the latest price. Paper mode only.
    """
    s, today = deps.settings, deps.today
    mode = s.autotrade_mode
    report = Report(title="Close-out", today=today, mode=mode)
    outcome = JobOutcome(report=report)
    path = ledger_path(mode, deps.state_dir)
    ledger = Ledger.load(path, mode=mode, starting_cash=s.simulated_total_equity)
    off = disabled_strategies(deps.rules)
    targets = [p for p in ledger.positions if p.strategy in off]
    try:
        if mode != "paper":
            report.warnings.append("Close-out only works on the paper account; sell live positions yourself.")
        elif not targets:
            report.notes.append("No open positions belong to a switched-off strategy.")
        else:
            prices = latest_prices(deps.bulk_history([p.symbol for p in targets]))
            for position in targets:
                price = prices.get(position.symbol)
                if price is None:
                    report.warnings.append(f"No price for {plain(position.symbol)}; it stays open.")
                    continue
                trade = ledger.close(position, price, today, f"{position.strategy} strategy switched off")
                outcome.closed.append(trade)
                report.exits.append(
                    f"- {plain(trade.symbol)} ({trade.strategy}) {trade.quantity} @ {inr(trade.entry_price)} -> "
                    f"{inr(trade.exit_price)}: {trade.pnl_pct:+.2f}% ({inr(trade.pnl)})"
                )
            report.account = (
                f"Paper account: cash {inr(ledger.cash, 0)}, realised P&L {inr(ledger.realised_pnl(), 0)} "
                f"over {len(ledger.closed)} closed trade(s)"
            )
    finally:
        ledger.save(path)
        deps.notifier.send(report.render())
    return outcome


def _run_intraday(kind: str, deps: JobDeps) -> JobOutcome:
    s, rules, today = deps.settings, deps.rules, deps.today
    now = deps.now or datetime.now(IST)
    mode = s.autotrade_mode
    title = {"intraday-open": "Intraday entry", "intraday-check": "Intraday alert"}.get(kind, "Intraday square-off")
    if deps.book_label:
        title = f"{deps.book_label} {title[0].lower()}{title[1:]}"
    report = Report(title=title, today=today, mode=mode)
    outcome = JobOutcome(report=report)
    path = ledger_path(mode, deps.state_dir)
    ledger = Ledger.load(path, mode=mode, starting_cash=s.simulated_total_equity)
    try:
        ir = rules.intraday
        if ir is None or not ir.enabled:
            report.notes.append("The intraday job is switched off in config/trading_rules.json.")
        elif mode == "live":
            report.warnings.append(
                "Intraday runs in paper mode only for now (live intraday needs broker-side stop orders "
                "and a static IP). Nothing was traded."
            )
        elif kind == "intraday-close":
            _intraday_close(ledger, ir, deps, outcome)
        elif kind == "intraday-check":
            _intraday_check(ledger, ir, deps, outcome)
        else:
            _intraday_open(ledger, rules, ir, deps, now, outcome)
    finally:
        ledger.save(path)
        # Hourly checks stay quiet unless a stop or target was hit.
        if kind != "intraday-check" or outcome.closed:
            deps.notifier.send(report.render())
    return outcome


def _intraday_open(ledger: Ledger, rules: TradingRules, ir, deps: JobDeps, now: datetime, outcome: JobOutcome) -> None:
    report, today, u = outcome.report, deps.today, deps.universe
    now_ist = now.astimezone(IST) if now.tzinfo else now
    if not _clock(ir.entry_after) <= now_ist.time() <= _clock(ir.entry_cutoff):
        report.notes.append(
            f"Outside the entry window ({ir.entry_after}-{ir.entry_cutoff} IST, now {now_ist:%H:%M}); no new intraday trade."
        )
        return
    done_today = sum(1 for p in ledger.positions if p.strategy == "intraday" and p.opened_on == today.isoformat())
    done_today += sum(1 for t in ledger.closed if t.strategy == "intraday" and t.opened_on == today.isoformat())
    if done_today >= ir.max_trades_per_day:
        report.notes.append(f"Already took {done_today} intraday trade(s) today; the limit is {ir.max_trades_per_day}.")
        return

    tickers = [to_yfinance_ticker(sym) for sym in u.large_caps]
    bars = deps.intraday_history(tickers + [INDEX_TICKER])
    market = index_change_pct(bars.get(INDEX_TICKER), today, IST)   # Nifty 50 today, % (None if unavailable)
    daily = deps.bulk_history(tickers)
    features = []
    for ticker in tickers:
        f = compute_features(
            ticker, bars.get(ticker), today, IST,
            daily_trend_up=daily_trend_up(daily.get(ticker), today, IST), index_change=market,
        )
        if f is not None:
            features.append(f)
    if len(tickers) <= 10:                 # a small book (futures): name any instrument without prices
        missing = [plain(t) for t in tickers if t not in bars]
        if missing:
            report.warnings.append("No prices for " + ", ".join(missing) + " this run.")
    if not features:
        report.notes.append("No 5-minute bars for today yet (holiday, or NSE opened less than 30 minutes ago).")
        return
    stale = [f for f in features if f.last_bar is not None and now_ist - f.last_bar > MAX_PRICE_AGE]
    if len(stale) > len(features) / 2:
        report.warnings.append(
            f"Prices are stale (last 5-minute bar {features[0].as_of} IST, now {now_ist:%H:%M}); no intraday trade."
        )
        return
    # Each setup sets its own stop, no wider than the rules allow; the target is a fixed multiple of it.
    found = all_setups(features, models=ir.models, max_stop_pct=ir.stop_loss_pct * 100,
                       reward=ir.take_profit_pct / ir.stop_loss_pct)
    if deps.entry_filter is not None:
        skipped: Dict[str, List[str]] = {}
        kept = []
        for x in found:
            why = deps.entry_filter(x)
            if why:
                skipped.setdefault(why, [])
                if plain(x.ticker) not in skipped[why]:
                    skipped[why].append(plain(x.ticker))
            else:
                kept.append(x)
        found = kept
        for why, names in skipped.items():
            report.notes.append(f"Skipped ({why}): " + ", ".join(names) + ".")
    # What the weekly review has learned ranks the setups (an empty playbook leaves the plain ranking).
    playbook, scan = deps.playbook(), f"{now_ist:%H:%M}"
    for x in found:
        x.expected_pct, x.lessons = expected_pct(setup_row(x, scan), playbook)
        if exit_for(x.model, playbook) == "hold":
            x.target_pct = HOLD_TARGET_PCT
            x.notes.append("holding to 15:15 (learned: it beat the target for this kind of setup)")
    # A stock whose results are due inside the earnings blackout would be refused by the risk layer
    # after Claude picked it, wasting the day, so it never reaches the shortlist.
    funda_cache: Dict[str, Optional[FundamentalSnapshot]] = {}
    blackout_rules = intraday_trading_rules(rules)

    def results_soon(x) -> bool:
        if x.ticker not in funda_cache:
            funda_cache[x.ticker] = deps.fundamentals(x.ticker, x.price)
        f = funda_cache[x.ticker]
        try:
            check_earnings_blackout(blackout_rules, f.next_earnings_date if f else None, today)
        except RiskViolation:
            return True
        return False

    left_out: List[str] = []
    while True:
        setups = shortlist([x for x in found if x.ticker not in left_out], limit=u.top_n)
        newly = [x.ticker for x in setups if results_soon(x)]
        if not newly:
            break
        left_out += newly
    if left_out:
        report.notes.append(
            f"Left out (results due within {rules.earnings_blackout_days} days): "
            + ", ".join(plain(t) for t in left_out) + "."
        )
    if not setups:
        report.notes.append(
            "No setups right now (no breakout, fair value gap retest, Fibonacci pullback or big gap); "
            "no intraday trade."
        )
        return
    for i, x in enumerate(setups, start=1):
        volume = f", volume {x.features.relative_volume:.1f}x" if x.features.relative_volume else ""
        expected = f", expected {x.expected_pct:+.2f}%" if x.expected_pct is not None else ""
        report.shortlist.append(
            f"{i}. {plain(x.ticker)} {x.direction} {x.score:.0f} - {x.label} ({inr(x.price)}{volume}{expected})"
        )

    debates: List[dict] = []

    def journal(note: str, pick=None, position: Optional[Position] = None) -> None:
        try:
            data = load_day(deps.state_dir, today)
            data["entries"].append(entry_record(now_ist, scan, setups, pick, position, note,
                                                practice=ir.practice, debates=debates))
            save_day(deps.state_dir, today, data)
        except Exception as exc:  # noqa: BLE001 - the journal must never stop a trade
            logger.warning("Could not write the journal (%s).", exc)

    records = deps.track_record()
    candidates = []
    for x in setups:
        item = x.as_prompt()
        record = records.get(x.model)
        if record:
            item["backtest"] = {k: record.get(k) for k in ("trades", "win_rate", "avg_net_pct", "verdict")}
        if x.expected_pct is not None:
            item["experience"] = {"expected_net_pct": x.expected_pct,
                                  "coin_flip_pct": (playbook or {}).get("baseline_pct"),
                                  "lessons_applied": x.lessons}
        candidates.append(item)
    try:
        pick = deps.picker_factory(deps.settings).pick_intraday(candidates, practice=ir.practice)
    except Exception as exc:  # noqa: BLE001
        report.warnings.append(f"Claude could not make an intraday pick ({exc}).")
        journal(f"Claude could not make a pick: {exc}")
        return
    outcome.pick = pick
    if pick.action == "HOLD" or not pick.ticker:
        if ir.practice:
            report.pick.extend(["HOLD - practice mode trades every day, but Claude saw a clear problem:",
                                pick.reasoning])
        else:
            report.pick.extend(["HOLD - no clean setup right now.", pick.reasoning])
        journal("HOLD", pick)
        return

    setup = next(x for x in setups if x.ticker == pick.ticker)
    debate_rules = getattr(rules, "debate", None)
    if debate_rules is not None and debate_rules.enabled:
        # Claude's pick is debated first, then the next setups on the shortlist; the first one the judge
        # approves is traded. If the debate itself cannot run, Claude's pick is traded without it.
        by_ticker = {c["ticker"]: c for c in candidates}
        queue = [setup] + [x for x in setups if x.ticker != setup.ticker]
        approved = None
        try:
            debater = deps.debater_factory(deps.settings)
            for x in queue[: debate_rules.max_candidates]:
                packet = data_packet(by_ticker[x.ticker], x, bars.get(x.ticker), bars.get(INDEX_TICKER), today, IST)
                result = debater.debate(packet, rounds=debate_rules.rounds)
                debates.append(result.record())
                report.debate.extend(result.lines(plain(x.ticker)))
                if result.decision == "TRADE":
                    approved = (x, result)
                    break
        except Exception as exc:  # noqa: BLE001 - a failed debate must not cost the day's trade
            logger.warning("The debate failed: %s", exc)
            if not debates:
                report.warnings.append(f"The bull vs bear debate could not run ({exc}); Claude's pick is traded "
                                       "without it.")
                approved = (setup, None)
            else:
                report.warnings.append(f"The debate stopped early ({exc}).")
        if approved is None:
            report.pick.append(f"The judge said SKIP to all {len(debates)} setup(s) debated; no trade today.")
            journal("Judge skipped every setup debated", pick)
            return
        chosen, verdict = approved
        if chosen.ticker != setup.ticker:
            pick = PickDecision(
                action="BUY" if chosen.direction == "long" else "SELL", ticker=chosen.ticker,
                target_price=chosen.target_price,
                reasoning=f"Claude's first pick was skipped by the judge; this one was approved "
                          f"({verdict.confidence}%): {verdict.reasoning}",
            )
            outcome.pick, setup = pick, chosen
        elif verdict is not None:
            pick = pick.model_copy(update={
                "reasoning": f"{pick.reasoning} Judge approved ({verdict.confidence}%): {verdict.reasoning}"})
            outcome.pick = pick
    price, short = setup.price, pick.action == "SELL"
    report.pick.append(f"{'SHORT' if short else 'BUY'} {plain(pick.ticker)} at about {inr(price)} - {setup.label}")
    if ir.practice:
        report.pick.append("Practice trade: the paper account takes the best setup every day to build experience.")
    report.pick.append("Why it qualified: " + "; ".join(setup.notes) + ".")
    report.pick.append(pick.reasoning)
    if record_line(records.get(setup.model)):
        report.pick.append(record_line(records.get(setup.model)))
    if setup.expected_pct is not None and playbook:
        report.pick.append(
            f"Experience: setups like this are expected to make {setup.expected_pct:+.2f}% a trade after costs "
            f"(a coin flip: {playbook.get('baseline_pct', 0):+.2f}%)."
            + (f" Lessons used: {'; '.join(setup.lessons)}." if setup.lessons else "")
        )
    try:
        if deps.sizer is not None:
            order = deps.sizer(setup, ledger)
        else:
            funda = funda_cache[pick.ticker] if pick.ticker in funda_cache else deps.fundamentals(pick.ticker, price)
            account = AccountState(
                total_equity=max(ledger.equity({f.ticker: f.price for f in features}), 1.0),
                cash_balance=max(ledger.cash, 0.0), open_positions=ledger.count("intraday"), source="paper",
            )
            trade_rules = intraday_trading_rules(rules).model_copy(update={
                "stop_loss_pct": setup.stop_pct / 100, "take_profit_pct": setup.target_pct / 100,
            })
            order = RiskInterceptor(trade_rules).validate(
                pick.to_analyst_decision(price), current_price=price, account_state=account,
                next_earnings_date=funda.next_earnings_date if funda else None,
            )
    except RiskViolation as exc:
        report.pick.append(f"Risk layer refused it: {exc}")
        journal(f"Risk layer refused it: {exc}", pick)
        return
    outcome.order = order
    position = Position(
        symbol=pick.ticker, strategy="intraday", side="short" if short else "long",
        quantity=order.quantity, entry_price=order.reference_price,
        stop_loss=order.stop_loss_price, take_profit=order.take_profit_price,
        opened_on=today.isoformat(), opened_at=now.isoformat(), max_hold_days=1,
        setup=setup.model, reasoning=pick.reasoning, margin=getattr(order, "margin", None),
    )
    ledger.open(position)
    outcome.opened, outcome.traded = position, True
    journal("traded", pick, position)
    report.pick.append(
        f"{'Paper-shorted' if short else 'Paper-bought'} {order.quantity} x {inr(order.reference_price)} "
        f"= {inr(order.notional)}; stop {inr(order.stop_loss_price)} ({setup.stop_pct:.2f}%), "
        f"target {inr(order.take_profit_price)} ({setup.target_pct:.2f}%). "
        f"Squared off at {ir.square_off} unless the stop or target hits first."
    )
    if getattr(order, "note", ""):
        report.pick.append(order.note)


def _intraday_line(trade: ClosedTrade) -> str:
    kind = f" ({MODEL_LABELS.get(trade.setup, trade.setup)})" if trade.setup else ""
    charges = f", after {inr(trade.charges, 0)} est. charges" if trade.charges else ""
    return (
        f"- {'SHORT' if trade.side == 'short' else 'BUY'} {plain(trade.symbol)}{kind} {trade.quantity} @ "
        f"{inr(trade.entry_price)} -> {inr(trade.exit_price)}, {trade.reason}: "
        f"{trade.pnl_pct:+.2f}% ({inr(trade.pnl)}{charges})"
    )


def _intraday_summary(ledger: Ledger) -> str:
    history = [t for t in ledger.closed if t.strategy == "intraday"]
    if not history:
        return ""
    wins = sum(1 for t in history if t.pnl > 0)
    text = (
        f"Intraday so far: {len(history)} trade(s), {wins} winner(s), P&L {inr(sum(t.pnl for t in history), 0)} "
        f"after charges. Paper account cash {inr(ledger.cash, 0)}"
    )
    by_setup: Dict[str, List[ClosedTrade]] = {}
    for t in history:
        if t.setup:
            by_setup.setdefault(t.setup, []).append(t)
    if by_setup:
        text += "\nBy setup: " + "; ".join(
            f"{MODEL_LABELS.get(name, name)} {len(ts)} trade(s), {inr(sum(t.pnl for t in ts), 0)}"
            for name, ts in sorted(by_setup.items())
        )
    return text


def _intraday_check(ledger: Ledger, ir, deps: JobDeps, outcome: JobOutcome) -> None:
    """Close today's intraday trade early if its stop or target has been touched."""
    report, today = outcome.report, deps.today
    open_trades = [p for p in ledger.positions if p.strategy == "intraday" and p.opened_on == today.isoformat()]
    if not open_trades:
        return
    bars = deps.intraday_history([p.symbol for p in open_trades])
    for position in open_trades:
        hit = check_exit(position, bars.get(position.symbol), today, ir.square_off, IST)
        if hit is None:
            continue
        price, reason = hit
        trade = ledger.close(position, price, today, reason, cost_pct=ir.cost_pct)
        outcome.closed.append(trade)
        report.exits.append(_intraday_line(trade))
    if outcome.closed:
        report.account = _intraday_summary(ledger)


def _intraday_close(ledger: Ledger, ir, deps: JobDeps, outcome: JobOutcome) -> None:
    report = outcome.report
    open_trades = [p for p in ledger.positions if p.strategy == "intraday"]
    if not open_trades:
        report.notes.append("No open intraday position.")
    else:
        bars = deps.intraday_history([p.symbol for p in open_trades])
        for position in open_trades:
            frame = bars.get(position.symbol)
            price, reason = simulate_exit(position, frame, date.fromisoformat(position.opened_on), ir.square_off, IST)
            if price is None and frame is not None and "Close" in frame and not frame["Close"].dropna().empty:
                price, reason = round(float(frame["Close"].dropna().iloc[-1]), 2), "squared off at the last price"
            if price is None:
                report.warnings.append(f"No prices for {plain(position.symbol)}; it stays open until the next run.")
                continue
            trade = ledger.close(position, price, deps.today, reason, cost_pct=ir.cost_pct)
            outcome.closed.append(trade)
            report.exits.append(_intraday_line(trade))
    earlier = [
        t for t in ledger.closed
        if t.strategy == "intraday" and t.closed_on == deps.today.isoformat() and t not in outcome.closed
    ]
    if earlier:
        report.exits.extend(_intraday_line(t) + " (closed earlier today)" for t in earlier)
        report.notes = [n for n in report.notes if n != "No open intraday position."]
    try:
        _end_of_day_review(ledger, ir, deps, outcome)
    except Exception as exc:  # noqa: BLE001 - the review must never block the square-off
        logger.exception("End-of-day review failed.")
        report.warnings.append(f"The end-of-day review failed ({type(exc).__name__}: {exc}).")
    report.account = _intraday_summary(ledger)


def _index_move(frame: Optional[pd.DataFrame], day: date, start: str, end: str) -> Optional[float]:
    """The Nifty's % move between two times of ``day`` (None without bars)."""
    if frame is None or frame.empty:
        return None
    bars = frame.dropna(subset=["Open", "Close"]).copy()
    bars.index = to_ist(bars.index, IST)
    bars = bars[(bars.index.date == day) & (bars.index.time >= _clock(start)) & (bars.index.time < _clock(end))]
    if bars.empty:
        return None
    return round((float(bars["Close"].iloc[-1]) / float(bars["Open"].iloc[0]) - 1) * 100, 2)


def _end_of_day_review(ledger: Ledger, ir, deps: JobDeps, outcome: JobOutcome) -> None:
    """Replay the day: every shortlisted setup, all of today's setups, and the running totals."""
    report, today, u = outcome.report, deps.today, deps.universe
    tickers = [to_yfinance_ticker(sym) for sym in u.large_caps]
    bars = deps.intraday_history(tickers + [INDEX_TICKER])
    rows = run_backtest(
        bars, deps.bulk_history(tickers), tz=IST, days=[today], square_off=ir.square_off, models=MODELS,
        max_stop_pct=ir.stop_loss_pct * 100, reward=ir.take_profit_pct / ir.stop_loss_pct,
    )
    data = load_day(deps.state_dir, today)
    if not rows and not data["entries"]:
        return                                            # no session today (holiday) or no data yet

    shortlists, judge_rows = [], []
    for entry in data["entries"]:
        outcomes = evaluate_shortlist(entry, bars, today, ir.square_off, IST, ir.cost_pct)
        rank = next((i for i, o in enumerate(outcomes, start=1) if o["picked"]), None)
        ended = {o["ticker"]: o["ret"] for o in outcomes}
        for d in entry.get("debates") or []:
            judge_rows.append({"ticker": d["ticker"], "decision": d["decision"], "ret": ended.get(d["ticker"])})
        shortlists.append({"run_at": entry["run_at"], "scan": entry.get("scan"), "outcomes": outcomes,
                           "pick_rank": rank})
        if not outcomes:
            continue
        average = sum(o["ret"] for o in outcomes) / len(outcomes)
        best = outcomes[0]
        best_text = f"best: {plain(best['ticker'])} {best['direction']} ({best['label']}) {best['ret']:+.2f}%"
        if rank is not None:
            mine = outcomes[rank - 1]
            report.review.append(
                f"- Claude's pick {plain(mine['ticker'])} ({mine['label']}) ranked {rank} of {len(outcomes)} on the "
                f"{entry.get('scan', '')} shortlist ({mine['ret']:+.2f}% after costs); the shortlist averaged "
                f"{average:+.2f}%; {best_text}."
            )
        else:
            report.review.append(
                f"- At {entry.get('scan', '')} Claude did not trade ({entry.get('note') or 'no pick'}); the shortlist "
                f"would have averaged {average:+.2f}% after costs; {best_text}."
            )

    if judge_rows:
        report.review.append("- Judge: " + "; ".join(
            f"{r['decision']} {plain(r['ticker'])}"
            + (f" ({'ended' if r['decision'] == 'TRADE' else 'would have ended'} {r['ret']:+.2f}% after costs)"
               if r["ret"] is not None else "")
            for r in judge_rows) + ".")
    summary = model_summary(rows, ir.cost_pct)
    if summary:
        report.review.append("- All of today's setups (every stock, after costs): " + "; ".join(
            f"{MODEL_LABELS[m]} {s['n']} ({s['won']} won, {s['avg_pct']:+.2f}%)" for m, s in summary.items()
        ) + ".")
    nifty = _index_move(bars.get(INDEX_TICKER), today, "09:50", ir.square_off)
    if nifty is not None:
        report.review.append(f"- Nifty 50 from 09:50 to {ir.square_off}: {nifty:+.2f}%.")

    trades = [t for t in ledger.closed if t.strategy == "intraday" and t.closed_on == today.isoformat()]
    data["review"] = {
        "built_at": datetime.now(IST).isoformat(timespec="seconds"), "shortlists": shortlists,
        "setups": summary, "rows": compact(rows), "nifty_pct": nifty, "judge": judge_rows,
        "trades": [json.loads(t.model_dump_json()) for t in trades],
    }
    save_day(deps.state_dir, today, data)

    totals = running_totals(deps.state_dir, ir.cost_pct)
    if totals["days"]:
        since = [t for t in ledger.closed if t.strategy == "intraday" and t.closed_on >= totals["first_day"]]
        wins = sum(1 for t in since if t.pnl > 0)
        report.experience.append(
            f"- Since {totals['first_day']}: {totals['days']} trading day(s) reviewed, {len(since)} paper trade(s), "
            f"{wins} winner(s), {inr(sum(t.pnl for t in since), 0)} after charges."
        )
        if totals["picks"]:
            report.experience.append(
                f"- Claude's picks: {totals['pick_avg_pct']:+.2f}% a trade on average, against "
                f"{totals['shortlist_avg_pct']:+.2f}% for the whole shortlist ({totals['picks']} pick(s))."
            )
        if totals["models"]:
            report.experience.append("- Every setup seen, by kind: " + "; ".join(
                f"{m['label']} {m['n']} ({m['won'] / m['n']:.0%} won, {m['avg_pct']:+.2f}%)"
                for m in totals["models"].values()
            ) + ".")
    report.experience.append("- " + lessons_summary(deps.playbook()))


def _describe_results(tickers, fundamentals, prices, deps: JobDeps, today: date, report: Report) -> None:
    upcoming = []
    for ticker in tickers:
        if ticker not in fundamentals:
            fundamentals[ticker] = deps.fundamentals(ticker, prices.get(ticker))
        f = fundamentals.get(ticker)
        day = f.next_earnings() if f else None
        if day is not None and 0 <= (day - today).days <= RESULTS_HORIZON_DAYS:
            upcoming.append((day, ticker))
    for day, ticker in sorted(upcoming):
        days = (day - today).days
        when = "today" if days == 0 else f"in {days} day{'s' if days != 1 else ''}"
        report.results.append(f"- {plain(ticker)}: {day.strftime('%d %b')} ({when}, Yahoo estimate)")


__all__ = [
    "IST",
    "JobDeps",
    "JobOutcome",
    "Report",
    "in_market_hours",
    "disabled_strategies",
    "inr",
    "is_market_open",
    "latest_prices",
    "live_blockers",
    "intraday_trading_rules",
    "penny_trading_rules",
    "record_line",
    "run_job",
    "today_ist",
]
