"""Tests for the automated jobs: universes, screening, the picker's reply, the
ledger, WhatsApp messages and the daily / penny runs end to end.

Fakes only - NO network, NO LLM, NO orders. Run with:  ``pytest -q``
"""

from __future__ import annotations

import types
from datetime import date, datetime, timedelta

import pandas as pd
import pytest

from config.settings import Settings, load_trading_rules
from src.analysis.analyst_agent import AnalystError
from src.automation.jobs import (
    IST,
    JobDeps,
    disabled_strategies,
    in_market_hours,
    inr,
    is_market_open,
    live_blockers,
    penny_trading_rules,
    run_job,
)
from src.automation.ledger import Ledger, Position, exit_reason, ledger_path
from src.automation.notify import WhatsAppNotifier, split_message, telegram_chat_ids
from src.automation.picker import PickDecision, parse_pick
from src.automation.screener import rank, score_fundamentals, score_technicals
from src.automation.universe import (
    Universe,
    fetch_latest_bhavcopy,
    parse_bhavcopy,
    penny_candidates,
)
from src.ingestion.fundamentals import FundamentalSnapshot

TODAY = date(2026, 9, 30)  # a Wednesday
SHIPPED = load_trading_rules()
# The shipped rules currently switch swing and penny off (intraday only); these
# tests exercise every strategy, so they run with all of them switched on.
RULES = SHIPPED.model_copy(update={
    "swing_enabled": True,
    "penny": SHIPPED.penny.model_copy(update={"enabled": True}),
})
INTRADAY_ONLY = RULES.model_copy(update={
    "swing_enabled": False,
    "penny": RULES.penny.model_copy(update={"enabled": False}),
})


def bars(end: date = TODAY, n: int = 260, start: float = 100.0, step: float = 0.25,
         wiggle: float = 1.5, volume: float = 2e6) -> pd.DataFrame:
    """Business-day bars ending on ``end``: a trend plus an up/down wiggle."""
    index = pd.date_range(end=pd.Timestamp(end), periods=n, freq="B")
    close = [start + step * i + (wiggle if i % 2 else -wiggle) for i in range(n)]
    return pd.DataFrame(
        {"Open": close, "High": [c * 1.01 for c in close], "Low": [c * 0.99 for c in close],
         "Close": close, "Volume": [volume] * n},
        index=index,
    )


def settings(**overrides) -> Settings:
    return Settings(_env_file=None, ANTHROPIC_API_KEY="test-key", **overrides)


class Recorder:
    def __init__(self):
        self.messages = []

    def send(self, text):
        self.messages.append(text)
        return True


# --- small helpers ------------------------------------------------------------- #

def test_indian_rupee_formatting():
    assert inr(1234567.8) == "Rs 12,34,567.80"
    assert inr(1_000_000, 0) == "Rs 10,00,000"
    assert inr(-500) == "-Rs 500.00"
    assert inr(999) == "Rs 999.00"


def test_market_is_open_only_when_there_is_a_bar_for_today():
    assert is_market_open({"A.NS": bars()}, TODAY)
    assert not is_market_open({"A.NS": bars(end=TODAY - timedelta(days=1))}, TODAY)
    assert not is_market_open({}, TODAY)


def test_live_trading_needs_every_lock_and_a_static_ip():
    ready = settings(DHAN_ENV="live", DHAN_ENABLE_LIVE_TRADING="true",
                     DHAN_CLIENT_ID="1", DHAN_ACCESS_TOKEN="t", AUTOTRADE_MODE="live")
    assert live_blockers(ready, "self-hosted") == []
    assert any("static IP" in p for p in live_blockers(ready, "github-hosted"))
    assert len(live_blockers(settings(), "")) == 3


# --- universes ------------------------------------------------------------------ #

BHAVCOPY = """SYMBOL, SERIES, DATE1, CLOSE_PRICE, TTL_TRD_QNTY, TURNOVER_LACS, DELIV_PER
AAA, EQ, 29-Sep-2026, 12.50, 2000000, 250.00, 45.20
BBB, BE, 29-Sep-2026, 15.00, 2000000, 300.00, 90.00
CCC, EQ, 29-Sep-2026, 80.00, 2000000, 1600.00, 50.00
DDD, EQ, 29-Sep-2026, 1.50, 90000000, 1350.00, 20.00
EEE, EQ, 29-Sep-2026, 20.00, 150000, 30.00, 60.00
FFF, EQ, 29-Sep-2026, 30.00, 3000000, 900.00, 35.00
"""


def test_penny_candidates_keep_liquid_eq_stocks_in_the_price_band():
    found = penny_candidates(parse_bhavcopy(BHAVCOPY), RULES.penny)
    assert [c.symbol for c in found] == ["FFF", "AAA"]        # most traded first
    assert found[1].turnover_cr == 2.5 and found[1].delivery_pct == 45.2


def test_bhavcopy_download_skips_weekends_and_missing_days():
    calls = []

    def fake_get(url, headers, timeout):
        calls.append(url)
        ok = "25092026" in url
        return types.SimpleNamespace(status_code=200 if ok else 404, text=BHAVCOPY if ok else "")

    day, text = fetch_latest_bhavcopy(date(2026, 9, 28), get=fake_get)   # Monday
    assert day == date(2026, 9, 25) and text == BHAVCOPY
    assert len(calls) == 2                                              # Mon (404), Fri; weekend skipped


# --- screening ------------------------------------------------------------------ #

def test_uptrend_scores_higher_than_downtrend():
    up = score_technicals("UP.NS", bars(step=0.4))
    down = score_technicals("DOWN.NS", bars(start=200, step=-0.4))
    assert up.score > down.score
    assert "SMA50 over SMA200" in up.notes and up.avg_turnover_cr > 0
    assert score_technicals("SHORT.NS", bars(n=30)) is None


def test_fundamental_score_rewards_quality_and_punishes_losses():
    strong = FundamentalSnapshot(ticker="A", return_on_equity_pct=22, debt_to_equity=0.3,
                                 revenue_growth_yoy_pct=14, operating_profit_growth_yoy_pct=12)
    weak = FundamentalSnapshot(ticker="B", return_on_equity_pct=3, debt_to_equity=3.5,
                               revenue_growth_yoy_pct=-5, flags=["loss-making (negative earnings)"])
    assert score_fundamentals(None) == 50
    assert score_fundamentals(strong) == 95 and score_fundamentals(weak) == 0


def test_rank_combines_scores_and_keeps_top_n():
    techs = [score_technicals(f"S{i}.NS", bars(step=0.1 * i)) for i in range(1, 5)]
    good = FundamentalSnapshot(ticker="S1.NS", return_on_equity_pct=25, debt_to_equity=0.1)
    ranked = rank(techs, {"S1.NS": good}, top_n=2)
    assert len(ranked) == 2 and ranked[0].combined >= ranked[1].combined


# --- picker reply ---------------------------------------------------------------- #

def test_pick_maps_a_bare_symbol_to_the_candidate_ticker():
    text = '{"action": "BUY", "ticker": "INFY", "target_price": 1650, "reasoning": "r"}'
    assert parse_pick(text, ["TCS.NS", "INFY.NS"]).ticker == "INFY.NS"
    hold = parse_pick('{"action": "HOLD", "ticker": "TCS.NS", "target_price": 1, "reasoning": "r"}', ["TCS.NS"])
    assert hold.ticker is None and hold.target_price is None


def test_pick_outside_the_candidates_is_rejected():
    with pytest.raises(AnalystError):
        parse_pick('{"action": "BUY", "ticker": "ADANI", "target_price": 10, "reasoning": "r"}', ["TCS.NS"])


# --- ledger ---------------------------------------------------------------------- #

def position(**kw) -> Position:
    base = dict(symbol="AAA.NS", strategy="swing", quantity=10, entry_price=100.0, stop_loss=97.0,
                take_profit=106.0, opened_on=TODAY.isoformat(), max_hold_days=20)
    base.update(kw)
    return Position(**base)


def test_exit_rules():
    p = position()
    assert exit_reason(p, 96.5, TODAY) == "stop-loss hit"
    assert exit_reason(p, 106.0, TODAY) == "target hit"
    assert exit_reason(p, 101.0, TODAY) is None
    assert "time limit" in exit_reason(p, 101.0, TODAY + timedelta(days=20))


def test_paper_ledger_tracks_cash_and_round_trips(tmp_path):
    ledger = Ledger(mode="paper", starting_cash=10_000, cash=10_000)
    ledger.open(position())
    assert ledger.cash == 9_000 and ledger.equity({"AAA.NS": 110.0}) == 10_100
    trade = ledger.close(ledger.positions[0], 106.0, TODAY, "target hit")
    assert (trade.pnl, trade.pnl_pct, ledger.cash) == (60.0, 6.0, 10_060)
    path = tmp_path / "paper_ledger.json"
    ledger.save(path)
    assert Ledger.load(path, mode="paper", starting_cash=1).cash == 10_060
    with pytest.raises(ValueError):
        ledger.open(position(quantity=1000))


# --- WhatsApp -------------------------------------------------------------------- #

def test_long_reports_are_split_on_lines():
    text = "\n".join(f"line {i:03d} " + "x" * 40 for i in range(60))
    chunks = split_message(text, limit=500)
    assert all(len(c) <= 500 for c in chunks) and "\n".join(chunks) == text


def test_callmebot_send_uses_phone_text_and_key():
    seen = []

    def fake_get(url, params, timeout):
        seen.append((url, params))
        return types.SimpleNamespace(status_code=200)

    s = settings(WHATSAPP_PROVIDER="callmebot", WHATSAPP_PHONE="+919800000000", CALLMEBOT_API_KEY="123")
    assert WhatsAppNotifier(s, get=fake_get, pause=0).send("hello")
    assert seen[0][1] == {"phone": "+919800000000", "text": "hello", "apikey": "123"}
    assert WhatsAppNotifier(settings(), pause=0).send("only logged") is False


def test_telegram_send_posts_to_the_bot_and_falls_back_to_plain_text():
    calls = []

    def fake_post(url, json, timeout):
        calls.append((url, dict(json)))
        return types.SimpleNamespace(status_code=400 if len(calls) == 1 else 200)

    s = settings(WHATSAPP_PROVIDER="telegram", TELEGRAM_BOT_TOKEN="123:ABC", TELEGRAM_CHAT_ID="42")
    assert WhatsAppNotifier(s, post=fake_post, pause=0).send("*report* with odd_markdown")
    assert calls[0][0] == "https://api.telegram.org/bot123:ABC/sendMessage"
    assert calls[0][1]["chat_id"] == "42" and calls[0][1]["parse_mode"] == "Markdown"
    assert "parse_mode" not in calls[1][1]                        # retried as plain text
    assert not WhatsAppNotifier(settings(WHATSAPP_PROVIDER="telegram"), pause=0).configured


def test_telegram_chat_ids_are_read_from_recent_messages():
    updates = {"result": [
        {"message": {"chat": {"id": 42, "first_name": "Test", "last_name": "User"}}},
        {"message": {"chat": {"id": 42, "first_name": "Test", "last_name": "User"}}},
        {"edited_message": {"chat": {"id": 7, "username": "friend"}}},
    ]}

    def fake_get(url, timeout):
        assert url.endswith("/getUpdates")
        return types.SimpleNamespace(status_code=200, json=lambda: updates)

    assert telegram_chat_ids("t", get=fake_get) == [("42", "Test User"), ("7", "friend")]


# --- the jobs end to end ---------------------------------------------------------- #

class FakePicker:
    def __init__(self, decision):
        self.decision = decision
        self.seen = None

    def __call__(self, _settings):
        return self

    def pick(self, strategy, candidates, snapshots):
        self.seen = (strategy, [c.ticker for c in candidates])
        return self.decision


class FakeDebater:
    """Stands in for the bull vs bear debate: the judge gives each ticker the verdict in ``verdicts``
    (TRADE when not listed); ``fail`` makes the debate itself raise."""

    def __init__(self, verdicts=None, fail=False):
        self.verdicts, self.fail, self.seen = verdicts or {}, fail, []

    def __call__(self, _settings):
        return self

    def debate(self, packet, rounds=2):
        from src.automation.debate import DebateResult

        if self.fail:
            raise AnalystError("Anthropic API call failed: test")
        self.seen.append((packet["ticker"], rounds, packet))
        point = {"claim": "c", "field": "price", "value": packet["price"], "verified": True, "referee": "verified"}
        turns = [{"side": side, "round": r, "points": [point], "summary": f"{side} case"}
                 for r in range(1, rounds + 1) for side in ("bull", "bear")]
        return DebateResult(ticker=packet["ticker"], direction=packet["setup"], turns=turns,
                            decision=self.verdicts.get(packet["ticker"], "TRADE"), confidence=70,
                            reasoning=f"verdict on {packet['ticker']}")


MID_MORNING = datetime(2026, 9, 30, 10, 0, tzinfo=IST)
AFTER_CLOSE = datetime(2026, 9, 30, 16, 50, tzinfo=IST)


def deps(tmp_path, *, history, picker=None, s=None, universe=None, **extra) -> JobDeps:
    extra.setdefault("now", MID_MORNING)
    extra.setdefault("track_record", dict)        # no backtest file in tests unless one is given
    extra.setdefault("playbook", lambda: None)     # ... and no learned playbook
    extra.setdefault("debater_factory", FakeDebater())   # the judge approves Claude's pick unless told otherwise
    return JobDeps(
        settings=s or settings(),
        rules=extra.pop("rules", RULES),
        universe=universe or Universe(large_caps=["AAA", "BBB", "CCC"], watchlist=[], top_n=2,
                                      fundamentals_shortlist=3),
        today=TODAY,
        notifier=Recorder(),
        state_dir=tmp_path,
        bulk_history=lambda tickers: {t: f for t, f in history.items() if t in tickers},
        fundamentals=extra.pop("fundamentals", lambda ticker, price: None),
        snapshot=lambda ticker: None,
        picker_factory=picker or FakePicker(PickDecision(action="HOLD", reasoning="nothing clean")),
        **extra,
    )


HISTORY = {"AAA.NS": bars(step=0.5), "BBB.NS": bars(step=0.2), "CCC.NS": bars(start=300, step=-0.5)}


def test_daily_paper_run_screens_picks_and_paper_buys(tmp_path):
    picker = FakePicker(PickDecision(action="BUY", ticker="AAA.NS", target_price=260, reasoning="clean trend"))
    d = deps(tmp_path, history=HISTORY, picker=picker)
    outcome = run_job("daily", d)

    assert picker.seen[0] == "swing" and picker.seen[1][0] == "AAA.NS" and len(picker.seen[1]) == 2
    assert outcome.traded and outcome.opened.symbol == "AAA.NS"
    price = HISTORY["AAA.NS"]["Close"].iloc[-1]
    assert outcome.order.quantity == int(1_000_000 * RULES.max_position_pct // price)
    assert outcome.opened.stop_loss == round(price * 0.97, 2)
    saved = Ledger.load(ledger_path("paper", tmp_path), mode="paper", starting_cash=0)
    assert saved.holds("AAA.NS") and saved.cash < 1_000_000
    message = d.notifier.messages[0]
    assert "Paper-bought" in message and "Today's shortlist" in message and "not financial advice" in message


def test_no_trading_when_the_market_is_closed(tmp_path):
    closed = {t: bars(end=TODAY - timedelta(days=1)) for t in HISTORY}
    picker = FakePicker(PickDecision(action="BUY", ticker="AAA.NS", target_price=260, reasoning="r"))
    outcome = run_job("daily", deps(tmp_path, history=closed, picker=picker))
    assert not outcome.traded and picker.seen is None
    assert "did not trade today" in outcome.report.render()


def test_positions_are_closed_at_their_stop(tmp_path):
    ledger = Ledger(mode="paper", starting_cash=1_000_000, cash=990_000)
    ledger.positions.append(position(symbol="CCC.NS", entry_price=1_000.0, stop_loss=900.0,
                                     take_profit=1_100.0, quantity=10, opened_on="2026-09-20"))
    ledger.save(ledger_path("paper", tmp_path))
    outcome = run_job("daily", deps(tmp_path, history=HISTORY))
    assert [t.symbol for t in outcome.closed] == ["CCC.NS"] and outcome.closed[0].reason == "stop-loss hit"
    assert "Closed today" in outcome.report.render()


def test_live_mode_on_a_github_hosted_runner_never_trades(tmp_path):
    executor_calls = []
    s = settings(AUTOTRADE_MODE="live", DHAN_ENV="live", DHAN_ENABLE_LIVE_TRADING="true",
                 DHAN_CLIENT_ID="1", DHAN_ACCESS_TOKEN="t")
    picker = FakePicker(PickDecision(action="BUY", ticker="AAA.NS", target_price=260, reasoning="r"))
    outcome = run_job("daily", deps(tmp_path, history=HISTORY, picker=picker, s=s,
                                    runner_environment="github-hosted",
                                    holdings=lambda _s: [],
                                    executor_factory=lambda _s: executor_calls.append(1)))
    assert not outcome.traded and executor_calls == [] and picker.seen is None
    assert "Live trading is OFF" in outcome.report.render()


def test_live_mode_with_every_lock_places_the_order(tmp_path, monkeypatch):
    sent = []

    class FakeExecutor:
        def execute(self, order, security_id, exchange_segment):
            sent.append((order.side, order.quantity, security_id, exchange_segment))
            return types.SimpleNamespace(submitted=True, status="submitted_live")

    import src.automation.jobs as jobs
    from src.risk.guardrails import AccountState

    monkeypatch.setattr(jobs, "get_account_state",
                        lambda s: AccountState(total_equity=500_000, cash_balance=500_000, source="dhan-live"))
    s = settings(AUTOTRADE_MODE="live", DHAN_ENV="live", DHAN_ENABLE_LIVE_TRADING="true",
                 DHAN_CLIENT_ID="1", DHAN_ACCESS_TOKEN="t")
    picker = FakePicker(PickDecision(action="BUY", ticker="AAA.NS", target_price=260, reasoning="r"))
    outcome = run_job("daily", deps(tmp_path, history=HISTORY, picker=picker, s=s,
                                    runner_environment="self-hosted", holdings=lambda _s: [],
                                    resolve_security=lambda sym: "1234",
                                    executor_factory=lambda _s: FakeExecutor()))
    assert outcome.traded and sent == [("buy", outcome.order.quantity, "1234", "NSE_EQ")]
    assert Ledger.load(ledger_path("live", tmp_path), mode="live", starting_cash=0).holds("AAA.NS")


def test_penny_job_uses_the_stricter_rules(tmp_path):
    history = {"FFF.NS": bars(start=20, step=0.03, wiggle=0.3, volume=5e6),
               "AAA.NS": bars(start=8, step=0.01, wiggle=0.1, volume=3e6)}
    picker = FakePicker(PickDecision(action="BUY", ticker="FFF.NS", target_price=40, reasoning="r"))
    outcome = run_job("penny", deps(tmp_path, history=history, picker=picker,
                                    bhavcopy=lambda today: (TODAY, BHAVCOPY)))
    assert picker.seen[0] == "penny" and set(picker.seen[1]) <= {"FFF.NS", "AAA.NS"}
    price = history["FFF.NS"]["Close"].iloc[-1]
    assert outcome.order.quantity == int(10_000 // price)              # 1% of Rs 10 lakh
    assert outcome.opened.stop_loss == round(price * 0.92, 2)         # 8% stop
    assert outcome.opened.max_hold_days == 30
    assert penny_trading_rules(RULES).max_open_positions == 3


def test_report_lists_dhan_holdings_watchlist_and_results(tmp_path):
    rows = [{"tradingSymbol": "BBB", "totalQty": 10, "avgCostPrice": 100.0}]
    results_day = (TODAY + timedelta(days=16)).isoformat()

    def fundamentals(ticker, price):
        if ticker == "AAA.NS":
            return FundamentalSnapshot(ticker=ticker, next_earnings_date=results_day)
        return None

    s = settings(DHAN_CLIENT_ID="1", DHAN_ACCESS_TOKEN="t")
    d = deps(tmp_path, history=HISTORY, s=s, holdings=lambda _s: rows, fundamentals=fundamentals,
             universe=Universe(large_caps=[], watchlist=["AAA"]))
    outcome = run_job("report", d)
    text = outcome.report.render()
    assert "BBB 10 @ Rs 100.00" in text and "Watchlist" in text
    assert "AAA: 16 Oct (in 16 days" in text
    assert not outcome.traded


def test_expired_dhan_token_is_reported_not_fatal(tmp_path):
    def broken(_s):
        raise RuntimeError("401")

    s = settings(DHAN_CLIENT_ID="1", DHAN_ACCESS_TOKEN="expired")
    outcome = run_job("report", deps(tmp_path, history=HISTORY, s=s, holdings=broken))
    assert "update DHAN_ACCESS_TOKEN" in outcome.report.render()


# --- one entry a day, and market hours ---------------------------------------------- #

def test_market_hours_are_the_regular_nse_session():
    assert in_market_hours(MID_MORNING) and not in_market_hours(AFTER_CLOSE)
    assert not in_market_hours(datetime(2026, 9, 30, 9, 0, tzinfo=IST))
    assert in_market_hours(datetime(2026, 9, 30, 4, 30, tzinfo=IST.utc))      # 10:00 IST given in UTC


def test_a_second_daily_run_does_not_buy_again(tmp_path):
    first = FakePicker(PickDecision(action="BUY", ticker="AAA.NS", target_price=260, reasoning="r"))
    assert run_job("daily", deps(tmp_path, history=HISTORY, picker=first)).traded

    second = FakePicker(PickDecision(action="BUY", ticker="BBB.NS", target_price=200, reasoning="r"))
    outcome = run_job("daily", deps(tmp_path, history=HISTORY, picker=second, now=AFTER_CLOSE))
    assert not outcome.traded and second.seen is None
    assert "Already opened a swing position today (AAA)" in outcome.report.render()


def test_a_late_paper_run_still_trades_but_says_so(tmp_path):
    picker = FakePicker(PickDecision(action="BUY", ticker="AAA.NS", target_price=260, reasoning="r"))
    outcome = run_job("daily", deps(tmp_path, history=HISTORY, picker=picker, now=AFTER_CLOSE))
    assert outcome.traded
    assert "started at 16:50 IST, outside market hours" in outcome.report.render()


def test_a_late_live_run_opens_nothing(tmp_path):
    calls = []
    s = settings(AUTOTRADE_MODE="live", DHAN_ENV="live", DHAN_ENABLE_LIVE_TRADING="true",
                 DHAN_CLIENT_ID="1", DHAN_ACCESS_TOKEN="t")
    picker = FakePicker(PickDecision(action="BUY", ticker="AAA.NS", target_price=260, reasoning="r"))
    outcome = run_job("daily", deps(tmp_path, history=HISTORY, picker=picker, s=s, now=AFTER_CLOSE,
                                    runner_environment="self-hosted", holdings=lambda _s: [],
                                    executor_factory=lambda _s: calls.append(1)))
    assert not outcome.traded and calls == [] and picker.seen is None
    assert "outside market hours" in outcome.report.render()


# --- switching strategies off ---------------------------------------------------------- #

def test_a_switched_off_swing_job_screens_nothing(tmp_path):
    picker = FakePicker(PickDecision(action="BUY", ticker="AAA.NS", target_price=260, reasoning="r"))
    outcome = run_job("daily", deps(tmp_path, history=HISTORY, picker=picker, rules=INTRADAY_ONLY))
    assert not outcome.traded and picker.seen is None
    assert "swing job is switched off" in outcome.report.render()
    assert disabled_strategies(INTRADAY_ONLY) == ["swing", "penny"] and disabled_strategies(RULES) == []


def test_close_out_closes_only_the_switched_off_strategies(tmp_path):
    ledger = Ledger(mode="paper", starting_cash=1_000_000, cash=1_000_000)
    ledger.open(position(symbol="AAA.NS", strategy="swing", entry_price=200.0, stop_loss=190.0,
                         take_profit=230.0, quantity=10))
    ledger.open(position(symbol="BBB.NS", strategy="penny", entry_price=150.0, stop_loss=140.0,
                         take_profit=180.0, quantity=20))
    ledger.open(position(symbol="CCC.NS", strategy="intraday", entry_price=170.0, stop_loss=168.0,
                         take_profit=174.0, quantity=5))
    ledger.save(ledger_path("paper", tmp_path))

    d = deps(tmp_path, history=HISTORY, rules=INTRADAY_ONLY)
    outcome = run_job("close-out", d)
    assert sorted(t.symbol for t in outcome.closed) == ["AAA.NS", "BBB.NS"]
    assert all(t.reason.endswith("strategy switched off") for t in outcome.closed)
    aaa = next(t for t in outcome.closed if t.symbol == "AAA.NS")
    assert aaa.exit_price == round(float(HISTORY["AAA.NS"]["Close"].iloc[-1]), 2)
    saved = Ledger.load(ledger_path("paper", tmp_path), mode="paper", starting_cash=0)
    assert [p.symbol for p in saved.positions] == ["CCC.NS"]
    assert "Close-out" in d.notifier.messages[0]

    again = run_job("close-out", deps(tmp_path, history=HISTORY, rules=INTRADAY_ONLY))
    assert again.closed == [] and "No open positions belong to a switched-off strategy" in again.report.render()


def test_hold_notifier_keeps_messages_until_released():
    """GitHub's late safety-net runs only speak when they changed something."""
    from automate import HoldNotifier

    inner = Recorder()
    held = HoldNotifier(inner)
    assert held.send("one") and held.send("two") and inner.messages == []
    held.release()
    assert inner.messages == ["one", "two"] and held.held == []
