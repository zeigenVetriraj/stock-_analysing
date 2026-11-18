"""Run the automated jobs (used by the GitHub Actions workflows).

    python automate.py daily     # weekdays: exits, screen Nifty 50, pick one, trade, WhatsApp report
    python automate.py penny     # weekly: screen liquid NSE stocks under Rs 50, pick one, trade, report
    python automate.py report    # holdings + bot positions + results dates only, no trading
    python automate.py intraday-open    # 09:45-11:30 IST: find setups, Claude picks at most one paper trade
    python automate.py intraday-check   # hourly: alert (and close) if the stop or target was hit
    python automate.py intraday-close   # ~15:20 IST: square off today's paper intraday trade
    python automate.py close-out        # close paper positions of strategies switched off in the rules
    python automate.py intraday-backtest  # save 5-minute bars and test every intraday setup model on them
    python automate.py telegram-chat-id   # after messaging your Telegram bot: shows your chat id

Add ``--no-whatsapp`` to print the report instead of sending it (Telegram or WhatsApp).
Add ``--quiet-unless-changed`` to send the report only if the run opened or closed
a position (used by GitHub's own late-running schedule, which is only a safety net).

Ledgers: runs inside GitHub Actions use state/ (committed back by the
workflow); runs anywhere else use state/local/ (git-ignored), so a test run on
your own computer never mixes with, or conflicts with, the bot's ledger.

Trading mode comes from AUTOTRADE_MODE: "paper" (default) records trades in
state/paper_ledger.json with the SIMULATED_* balance; "live" places real Dhan
orders only when DHAN_ENV=live, DHAN_ENABLE_LIVE_TRADING=true, Dhan credentials
are set and the job runs on a machine whose static IP is whitelisted with Dhan.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from config.settings import get_settings, load_trading_rules
from src.automation.jobs import JobDeps, run_job, today_ist
from src.automation.ledger import STATE_DIR
from src.automation.notify import WhatsAppNotifier, telegram_chat_ids
from src.automation.universe import load_universe


class _PrintNotifier:
    def send(self, text: str) -> bool:
        print(text)
        return True


class HoldNotifier:
    """Keeps the messages back; :meth:`release` sends them (only when the run changed something)."""

    def __init__(self, inner) -> None:
        self.inner, self.held = inner, []

    def send(self, text: str) -> bool:
        self.held.append(text)
        return True

    def release(self) -> None:
        for text in self.held:
            self.inner.send(text)
        self.held = []


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Automated swing / penny jobs.")
    parser.add_argument(
        "job",
        choices=["daily", "penny", "report", "intraday-open", "intraday-check", "intraday-close", "close-out",
                 "intraday-backtest", "telegram-chat-id"],
    )
    parser.add_argument("--no-whatsapp", action="store_true", help="Print the report instead of sending it.")
    parser.add_argument("--quiet-unless-changed", action="store_true",
                        help="Send the report only if the run opened or closed a position.")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # yfinance and urllib3 are chatty at INFO.
    for noisy in ("yfinance", "urllib3", "peewee", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    settings = get_settings()
    if args.job == "telegram-chat-id":
        return _show_telegram_chat_ids(settings)
    notifier = _PrintNotifier() if args.no_whatsapp else WhatsAppNotifier(settings)
    if args.job == "intraday-backtest":
        return _run_backtest(notifier)
    held = HoldNotifier(notifier) if args.quiet_unless_changed else None
    deps = JobDeps(
        settings=settings,
        rules=load_trading_rules(),
        universe=load_universe(),
        today=today_ist(),
        notifier=held or notifier,
        runner_environment=os.environ.get("RUNNER_ENVIRONMENT", ""),
        state_dir=STATE_DIR if os.environ.get("GITHUB_ACTIONS") == "true" else STATE_DIR / "local",
    )
    log = logging.getLogger("automate")
    log.info("=== %s job | mode=%s | %s ===", args.job, settings.autotrade_mode, deps.today)
    status = 0
    try:
        outcome = run_job(args.job, deps)
    except Exception as exc:  # noqa: BLE001 - tell the user, then fail the run
        log.exception("Job failed.")
        notifier.send(f"*{args.job} job failed* on {deps.today}: {type(exc).__name__}: {exc}")
        outcome, status = None, 1
    if outcome is not None:
        if held is not None:
            if outcome.traded or outcome.closed:
                held.release()
            else:
                log.info("Nothing changed; no message sent (--quiet-unless-changed).")
        log.info("=== done | traded=%s | closed=%d ===", outcome.traded, len(outcome.closed))
    # The penny and futures books run after the large-stock book, in the same run, and never affect it.
    if args.job in ("intraday-open", "intraday-check", "intraday-close"):
        from src.automation.futures_book import run_futures_intraday
        from src.automation.penny_book import run_penny_intraday

        for label, runner in (("Penny", run_penny_intraday), ("Futures", run_futures_intraday)):
            status = max(status, _run_book(label, runner, args.job, deps, notifier, args.quiet_unless_changed))
    # After the entry run: one table of the three books' picks (not from GitHub's late backup runs).
    if args.job == "intraday-open" and not args.quiet_unless_changed:
        try:
            from src.automation.daily_picks import picks_table

            notifier.send(picks_table(deps.state_dir, deps.rules, deps.today))
        except Exception:  # noqa: BLE001 - the summary must never fail the run
            log.exception("Could not send the picks table.")
    return status


def _run_book(label: str, runner, job: str, deps: JobDeps, notifier, quiet: bool) -> int:
    """Run one extra intraday book (penny or futures); its failure is reported and never spreads."""
    from dataclasses import replace

    log = logging.getLogger("automate")
    held = HoldNotifier(notifier) if quiet else None
    try:
        outcome = runner(job, replace(deps, notifier=held or notifier))
    except Exception as exc:  # noqa: BLE001 - tell the user; the other books have already run
        log.exception("%s book failed.", label)
        notifier.send(f"*{label} {job} failed* on {deps.today}: {type(exc).__name__}: {exc}")
        return 1
    if outcome is None:
        return 0
    if held is not None and (outcome.traded or outcome.closed):
        held.release()
    log.info("=== %s done | traded=%s | closed=%d ===", label.lower(), outcome.traded, len(outcome.closed))
    return 0


def _run_backtest(notifier) -> int:
    """Download 5-minute bars, replay the intraday rules on them and save the report (no trading)."""
    from src.automation import intraday_backtest, learning
    from src.automation.jobs import IST
    from src.ingestion.data_fetcher import to_yfinance_ticker

    log = logging.getLogger("automate")
    # The weekend profit and loss of all three books comes first: it needs no downloads.
    try:
        from src.automation.weekly_pnl import weekly_pnl

        notifier.send(weekly_pnl(STATE_DIR, load_trading_rules(), today_ist()))
    except Exception:  # noqa: BLE001 - the reviews still run
        log.exception("Could not send the weekly profit and loss.")
    tickers = [to_yfinance_ticker(symbol) for symbol in load_universe().large_caps]
    rules = load_trading_rules().intraday
    limits = {}
    if rules is not None:
        limits = {"cost_pct": rules.cost_pct, "max_stop_pct": rules.stop_loss_pct * 100,
                  "reward": rules.take_profit_pct / rules.stop_loss_pct}
    try:
        _, rows, flips = intraday_backtest.run(STATE_DIR / "backtests", tz=IST, tickers=tickers, fetch=True, **limits)
        cost = limits.get("cost_pct", intraday_backtest.DEFAULT_COST_PCT)
        playbook, _ = learning.build(rows, flips, STATE_DIR / "learning", cost_pct=cost)
    except Exception as exc:  # noqa: BLE001 - tell the user, then fail the run
        log.exception("Backtest failed.")
        notifier.send(f"*Intraday backtest failed*: {type(exc).__name__}: {exc}")
        return 1
    from src.automation import scorecard

    text = intraday_backtest.headline(rows, cost, coin_flip=flips)
    notifier.send(text + "\n\n*Weekly review*\n" + learning.lessons_summary(playbook)
                  + "\nFull review: state/learning/learning_report.md\n\n" + scorecard.weekly(STATE_DIR))
    log.info("=== backtest done | %d setups ===", len(rows))
    from src.automation.futures_book import run_futures_backtest
    from src.automation.penny_book import run_penny_backtest

    status = 0
    for label, runner in (("Penny", run_penny_backtest), ("Futures", run_futures_backtest)):
        try:
            runner(STATE_DIR, load_trading_rules(), notifier)
        except Exception as exc:  # noqa: BLE001 - tell the user, then fail the run
            log.exception("%s backtest failed.", label)
            notifier.send(f"*{label} backtest failed*: {type(exc).__name__}: {exc}")
            status = 1
    return status


def _show_telegram_chat_ids(settings) -> int:
    if not settings.telegram_bot_token:
        print("Set TELEGRAM_BOT_TOKEN in .env first (from @BotFather).")
        return 2
    try:
        chats = telegram_chat_ids(settings.telegram_bot_token)
    except Exception as exc:  # noqa: BLE001 - never echo the URL, it holds the token
        print(f"Could not reach Telegram ({type(exc).__name__}). Check TELEGRAM_BOT_TOKEN.")
        return 1
    if not chats:
        print("No messages yet. Open your bot in Telegram, press Start (or send 'hi'), then run this again.")
        return 1
    for chat_id, name in chats:
        print(f"TELEGRAM_CHAT_ID={chat_id}    ({name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
