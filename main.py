"""Orchestrator - stitches the four layers together sequentially.

Pipeline:  Ingest (L1) -> Analyze (L2) -> Risk Intercept/Validate (L3) -> Execute (L4)

Market: Indian equities via Dhan. Prices/analysis come from Yahoo Finance
(real prices), plus company fundamentals from Yahoo (best-effort: a run carries
on without them); orders go to the Dhan **sandbox** by default (no real money).

Usage
-----
    python main.py --symbol RELIANCE                 # safe dry-run (default)
    python main.py --symbol TCS --exchange NSE       # choose exchange
    python main.py --symbol INFY --submit            # send to Dhan sandbox
    python main.py --symbol SBIN --qty 1000          # propose a size; L3 downsizes

    python main.py --symbol RELIANCE --options       # buy-only options, dry-run
    python main.py --symbol RELIANCE --options --lots 2 --sim-account

Options mode (``--options``) buys a call for a bullish view or a put for a
bearish one, never sells/writes options. It needs a live Dhan access token with
an active Data API plan for the option chain.

Safety
------
* Default is dry-run: prints the exact order + risk plan, sends nothing.
* ``--submit`` sends to whatever ``DHAN_ENV`` selects. Sandbox = mock money.
* Live (real money) also requires DHAN_ENV=live AND DHAN_ENABLE_LIVE_TRADING=true.
* ``--sim-account`` sizes against the SIMULATED_* balance and is refused with
  ``--submit``, so a practice balance can never size a real order.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional

from config.settings import get_settings, load_trading_rules
from src.analysis.analyst_agent import AnalystAgent, AnalystError
from src.analysis.options_agent import OptionsAnalystAgent
from src.execution.instruments import DhanInstruments, InstrumentError
from src.execution.order_executor import ExecutionError, OrderExecutor
from src.ingestion.data_fetcher import (
    DataFetchError,
    fetch_market_snapshot,
    to_yfinance_ticker,
)
from src.ingestion.fundamentals import FundamentalSnapshot, FundamentalsError, fetch_fundamentals
from src.ingestion.option_chain import OptionChainError, fetch_option_chain_snapshot
from src.risk.guardrails import (
    RiskInterceptor,
    RiskViolation,
    get_account_state,
    simulated_account_state,
)
from src.risk.options_guardrails import OptionRiskInterceptor


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _load_fundamentals(yf_ticker: str, current_price: float) -> Optional[FundamentalSnapshot]:
    """Layer 1c: company fundamentals. Best-effort - the run carries on without them."""
    log = logging.getLogger("main")
    try:
        return fetch_fundamentals(yf_ticker, current_price=current_price)
    except FundamentalsError as exc:
        log.warning("Fundamentals unavailable (%s); continuing with technicals only.", exc)
    except Exception as exc:  # noqa: BLE001 - never let an extra kill the run
        log.warning("Fundamentals failed (%s); continuing with technicals only.", exc)
    return None


def run_pipeline(
    symbol: str,
    *,
    exchange: str,
    submit: bool,
    proposed_qty: int | None,
    sim_account: bool = False,
) -> int:
    """Run Ingest -> Analyze -> Risk -> Execute for one Indian equity symbol."""
    settings = get_settings()
    log = logging.getLogger("main")

    dry_run = not submit
    log.info(
        "=== Swing Pipeline | %s (%s) | env=%s | dry_run=%s ===",
        symbol.upper(), exchange.upper(), settings.dhan_env, dry_run,
    )

    # --- Layer 1: Ingest (real prices via Yahoo Finance) ---
    yf_ticker = to_yfinance_ticker(symbol, exchange)
    try:
        snapshot = fetch_market_snapshot(yf_ticker)
    except DataFetchError as exc:
        log.error("Data ingestion failed: %s", exc)
        return 2
    fundamentals = _load_fundamentals(yf_ticker, snapshot.current_price)

    # --- Layer 2: Analyze ---
    try:
        decision = AnalystAgent(settings).analyze(snapshot, fundamentals)
    except AnalystError as exc:
        log.error("Analyst failed: %s", exc)
        return 3
    log.info("Reasoning: %s", decision.reasoning)

    # --- Layer 3: Risk intercept (hard safety) ---
    account = simulated_account_state(settings) if sim_account else get_account_state(settings)
    log.info("Account: equity=Rs %.2f cash=Rs %.2f (source=%s)",
             account.total_equity, account.cash_balance, account.source)
    try:
        order = RiskInterceptor().validate(
            decision,
            current_price=snapshot.current_price,
            account_state=account,
            proposed_quantity=proposed_qty,
            next_earnings_date=fundamentals.next_earnings_date if fundamentals else None,
        )
    except RiskViolation as exc:
        log.error("RISK VIOLATION - execution killed: %s", exc)
        return 4

    if not order.is_tradeable:
        log.info("No actionable order (%s). Pipeline complete.", order.action)
        return 0

    # --- Resolve Dhan securityId for the order payload ---
    exchange_segment = settings.exchange_segment(exchange)
    security_id = "<unresolved>"
    try:
        security_id = DhanInstruments().resolve(symbol, exchange)
    except InstrumentError as exc:
        if not dry_run:
            log.error("Cannot resolve securityId for a real order: %s", exc)
            return 5
        log.warning("securityId unresolved (%s); continuing dry-run with placeholder.", exc)

    # --- Layer 4: Execute ---
    try:
        result = OrderExecutor(settings, dry_run=dry_run).execute(
            order, security_id=security_id, exchange_segment=exchange_segment
        )
    except ExecutionError as exc:
        log.error("Execution failed: %s", exc)
        return 6

    log.info("Execution status: %s (submitted=%s, env=%s)",
             result.status, result.submitted, result.environment)
    log.info("=== Pipeline complete ===")
    return 0


def run_options_pipeline(
    symbol: str,
    *,
    exchange: str,
    submit: bool,
    proposed_lots: int | None,
    sim_account: bool = False,
) -> int:
    """Buy-only options: Ingest (stock + option chain) -> Analyze -> Risk -> Execute."""
    settings = get_settings()
    log = logging.getLogger("main")

    dry_run = not submit
    log.info(
        "=== Options Pipeline (buy-only) | %s (%s) | env=%s | dry_run=%s ===",
        symbol.upper(), exchange.upper(), settings.dhan_env, dry_run,
    )
    if exchange.strip().upper() != "NSE":
        log.error("Options mode supports NSE stock options only.")
        return 2
    rules = load_trading_rules()
    if rules.options is None or not rules.options.enabled:
        log.error("Options are disabled in config/trading_rules.json.")
        return 4
    if not settings.has_dhan:
        log.error(
            "Options mode needs Dhan API access: set DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN "
            "in .env (a live token with an active Data API plan)."
        )
        return 2

    # --- Layer 1a: the stock's technicals (Yahoo Finance) ---
    yf_ticker = to_yfinance_ticker(symbol, exchange)
    try:
        snapshot = fetch_market_snapshot(yf_ticker)
    except DataFetchError as exc:
        log.error("Data ingestion failed: %s", exc)
        return 2

    # --- Layer 1b: the option chain (Dhan Data API, read-only) ---
    try:
        chain = fetch_option_chain_snapshot(symbol, rules=rules.options, settings=settings)
    except OptionChainError as exc:
        log.error("Option chain failed: %s", exc)
        return 2

    # --- Layer 1c: company fundamentals (Yahoo Finance, best-effort) ---
    fundamentals = _load_fundamentals(yf_ticker, snapshot.current_price)

    # --- Layer 2: Analyze ---
    try:
        decision = OptionsAnalystAgent(settings).analyze(snapshot, chain, fundamentals)
    except AnalystError as exc:
        log.error("Options analyst failed: %s", exc)
        return 3
    log.info("Reasoning: %s", decision.reasoning)

    # --- Layer 3: Risk intercept (hard safety) ---
    account = simulated_account_state(settings) if sim_account else get_account_state(settings)
    log.info("Account: equity=Rs %.2f cash=Rs %.2f (source=%s)",
             account.total_equity, account.cash_balance, account.source)
    try:
        order = OptionRiskInterceptor(rules).validate(
            decision, chain, account, proposed_lots=proposed_lots,
            next_earnings_date=fundamentals.next_earnings_date if fundamentals else None,
        )
    except RiskViolation as exc:
        log.error("RISK VIOLATION - execution killed: %s", exc)
        return 4

    if not order.is_tradeable:
        log.info("No actionable order (%s). Pipeline complete.", order.action)
        return 0

    # --- Layer 4: Execute ---
    try:
        result = OrderExecutor(settings, dry_run=dry_run).execute_option(order)
    except ExecutionError as exc:
        log.error("Execution failed: %s", exc)
        return 6

    log.info("Execution status: %s (submitted=%s, env=%s)",
             result.status, result.submitted, result.environment)
    log.info("=== Pipeline complete ===")
    return 0


def parse_args(argv=None) -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="AI Swing Trading System orchestrator (India / Dhan).")
    parser.add_argument("--symbol", "-s", required=True, help="Plain symbol, e.g. RELIANCE")
    parser.add_argument("--exchange", "-e", default=settings.default_exchange,
                        choices=["NSE", "BSE"], help="Exchange (default from settings).")
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Send the order to Dhan (sandbox unless DHAN_ENV=live). Default is dry-run.",
    )
    parser.add_argument(
        "--qty",
        type=int,
        default=None,
        help="Optional proposed share quantity (risk layer will downsize if needed).",
    )
    parser.add_argument(
        "--options",
        action="store_true",
        help="Buy-only options mode: buy a call (bullish) or a put (bearish) instead of shares.",
    )
    parser.add_argument(
        "--lots",
        type=int,
        default=None,
        help="Options mode: optional proposed number of lots (risk layer will downsize if needed).",
    )
    parser.add_argument(
        "--sim-account",
        action="store_true",
        help="Size against the SIMULATED_* balance instead of your Dhan funds (dry-run only).",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    _configure_logging(args.verbose)
    log = logging.getLogger("main")
    if args.sim_account and args.submit:
        log.error("--sim-account is for dry runs only; it cannot be combined with --submit.")
        return 2
    if args.lots is not None and not args.options:
        log.error("--lots only applies with --options (use --qty for shares).")
        return 2
    if args.options:
        if args.qty is not None:
            log.error("--qty is for shares; use --lots with --options.")
            return 2
        return run_options_pipeline(
            args.symbol, exchange=args.exchange, submit=args.submit,
            proposed_lots=args.lots, sim_account=args.sim_account,
        )
    return run_pipeline(
        args.symbol, exchange=args.exchange, submit=args.submit, proposed_qty=args.qty,
        sim_account=args.sim_account,
    )


if __name__ == "__main__":
    sys.exit(main())
