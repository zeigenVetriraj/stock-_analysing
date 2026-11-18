"""Tests for company fundamentals (Layer 1c), how they reach both analysts, and
the earnings blackout in both risk layers.

Fixtures and fakes only - NO network, NO LLM, NO orders. Run with:  ``pytest -q``
"""

from __future__ import annotations

import logging
import sys
import types
from datetime import date, datetime, timedelta

import pandas as pd
import pytest

import main
from config.settings import Settings, TradingRules
from src.analysis.analyst_agent import SYSTEM_PROMPT, AnalystAgent, AnalystDecision
from src.analysis.options_agent import OPTIONS_SYSTEM_PROMPT, OptionsAnalystAgent
from src.ingestion.fundamentals import (
    FundamentalsError,
    cagr_pct,
    fetch_fundamentals,
    margin_from_accounts,
    next_earnings_date,
    roe_from_accounts,
    statement_series,
    summarise_fundamentals,
    year_on_year_pct,
)
from src.risk.guardrails import AccountState, RiskInterceptor, RiskViolation
from src.risk.options_guardrails import OptionRiskInterceptor
from tests.test_analyst_replies import ScriptedClient
from tests.test_options import RULES as OPTION_TEST_RULES
from tests.test_options import SNAPSHOT
from tests.test_options import TODAY as CHAIN_DAY
from tests.test_options import decision as option_decision
from tests.test_options import make_chain

TODAY = date(2026, 9, 30)

INFO = {
    "longName": "Example Industries Limited",
    "sector": "Energy",
    "industry": "Oil & Gas Refining & Marketing",
    "currency": "INR",
    "marketCap": 1.6e13,
    "trailingPE": 24.456,
    "forwardPE": 20.1,
    "priceToBook": 2.1,
    "enterpriseToEbitda": 11.2,
    "returnOnEquity": 0.085,
    "profitMargins": 0.075,
    "operatingMargins": 0.12,
    "revenueGrowth": -0.02,
    "earningsGrowth": -0.12,
    "debtToEquity": 36.0,          # Yahoo reports this as a percentage
    "currentRatio": 1.1,
    "dividendRate": 5.5,
    "currentPrice": 1190.0,
    "trailingEps": 48.5,
    "fiftyTwoWeekHigh": 1600.0,
    "fiftyTwoWeekLow": 1150.0,
}

# Annual income statement, newest period first - the way yfinance returns it.
INCOME = pd.DataFrame(
    {
        pd.Timestamp("2026-03-31"): [133.1, 8.1, 16.0],
        pd.Timestamp("2025-03-31"): [121.0, 8.0, 15.0],
        pd.Timestamp("2024-03-31"): [110.0, 9.0, 14.0],
        pd.Timestamp("2023-03-31"): [100.0, 10.0, 13.0],
    },
    index=["Total Revenue", "Net Income", "Operating Income"],
)

# Annual balance sheet (shareholders' equity), newest first.
BALANCE = pd.DataFrame(
    {pd.Timestamp("2026-03-31"): [110.0], pd.Timestamp("2025-03-31"): [90.0]},
    index=["Stockholders Equity"],
)

# Quarterly income statement, newest first: the June quarter against a year ago.
QUARTERLY = pd.DataFrame(
    {
        pd.Timestamp("2026-06-30"): [311.0, 20.9, 30.0],
        pd.Timestamp("2026-03-31"): [290.0, 19.0, 28.0],
        pd.Timestamp("2025-12-31"): [280.0, 18.5, 27.5],
        pd.Timestamp("2025-09-30"): [260.0, 18.0, 26.0],
        pd.Timestamp("2025-06-30"): [248.0, 27.0, 27.3],
    },
    index=["Total Revenue", "Net Income", "Operating Income"],
)

# Yahoo's summary for a company like the one in the first live run: no ROE, and
# net profit down 22.4% against a year-ago quarter inflated by a one-off gain.
ONE_OFF_INFO = {"trailingPE": 21.69, "revenueGrowth": 0.297, "earningsGrowth": -0.224, "debtToEquity": 37.0}

CALENDAR = {"Earnings Date": [date(2026, 10, 17), date(2026, 10, 21)], "Ex-Dividend Date": date(2026, 8, 14)}


def rules(blackout_days: int = 5) -> TradingRules:
    return OPTION_TEST_RULES.model_copy(update={"earnings_blackout_days": blackout_days})


def settings() -> Settings:
    return Settings(_env_file=None, ANTHROPIC_API_KEY="test-key", DHAN_CLIENT_ID="1100000001",
                    DHAN_ACCESS_TOKEN="test-token", DHAN_ENV="sandbox")


# --- Layer 1c: turning Yahoo data into a snapshot ---------------------------- #

def test_snapshot_normalises_units_and_computes_metrics():
    s = summarise_fundamentals("example.ns", INFO, INCOME, CALENDAR, today=TODAY)
    assert s.ticker == "EXAMPLE.NS" and s.name == "Example Industries Limited"
    assert (s.trailing_pe, s.forward_pe, s.price_to_book) == (24.46, 20.1, 2.1)
    assert (s.return_on_equity_pct, s.profit_margin_pct, s.operating_margin_pct) == (8.5, 7.5, 12.0)
    assert (s.revenue_growth_yoy_pct, s.earnings_growth_yoy_pct) == (-2.0, -12.0)
    assert s.debt_to_equity == 0.36                          # 36% -> 0.36x
    assert s.dividend_yield_pct == pytest.approx(0.46)       # 5.5 / 1190
    assert s.market_cap_cr == 1_600_000                      # Rs 1.6 lakh crore... x10
    assert s.pct_below_52w_high == pytest.approx(25.625, abs=0.01)
    assert s.pct_above_52w_low == pytest.approx(3.48, abs=0.01)
    assert s.revenue_cagr_pct == pytest.approx(10.0)         # 100 -> 133.1 over 3 years
    assert s.net_income_cagr_pct == pytest.approx(-6.78, abs=0.01)
    assert s.cagr_years == 3
    assert (s.next_earnings_date, s.days_to_earnings) == ("2026-10-17", 17)
    assert "revenue shrinking (-2.0% year on year)" in s.flags
    assert "profits falling (-12.0% year on year)" in s.flags
    assert not any("results due" in f for f in s.flags)      # 17 days is outside the 14-day note


def test_flags_call_out_losses_debt_growth_valuation_and_results():
    info = {"trailingEps": -3.2, "profitMargins": -0.05, "debtToEquity": 250.0, "sector": "Industrials",
            "revenueGrowth": 0.22, "returnOnEquity": 0.18, "trailingPE": 65}
    s = summarise_fundamentals("X.NS", info, None, {"Earnings Date": [TODAY + timedelta(days=4)]}, today=TODAY)
    flags = " | ".join(s.flags)
    assert "loss-making" in flags
    assert "high debt (debt-to-equity 2.50x)" in flags
    assert "fast revenue growth (+22.0% year on year)" in flags
    assert "strong profitability (ROE 18.0%)" in flags
    assert "expensive (P/E 65.0)" in flags
    assert f"results due in 4 days ({(TODAY + timedelta(days=4)).isoformat()})" in flags


def test_banks_are_not_flagged_for_leverage():
    s = summarise_fundamentals("BANK.NS", {"sector": "Financial Services", "debtToEquity": 700.0}, today=TODAY)
    assert s.debt_to_equity == 7.0
    assert not any("high debt" in f for f in s.flags)


def test_cagr_is_only_computed_when_meaningful():
    assert cagr_pct(100, 121, 2) == pytest.approx(10.0)
    assert cagr_pct(-5, 10, 2) is None        # loss year at the start
    assert cagr_pct(100, 110, 0) is None
    assert cagr_pct(None, 1, 1) is None


def test_statement_series_sorts_oldest_first_and_skips_gaps():
    income = INCOME.copy()
    income.loc["Net Income", pd.Timestamp("2024-03-31")] = float("nan")
    points = statement_series(income, ("Net Income",))
    assert [p[0].year for p in points] == [2023, 2025, 2026]
    raw_names = INCOME.rename(index={"Total Revenue": "TotalRevenue"})
    assert statement_series(raw_names, ("Total Revenue", "TotalRevenue"))[0][1] == 100.0


def test_roe_and_margins_fall_back_to_the_annual_accounts():
    info = {k: v for k, v in INFO.items() if k not in ("returnOnEquity", "profitMargins", "operatingMargins")}
    s = summarise_fundamentals("X.NS", info, INCOME, balance_sheet=BALANCE, today=TODAY)
    assert s.return_on_equity_pct == pytest.approx(8.1)               # 8.1 / average(110, 90)
    assert s.profit_margin_pct == pytest.approx(6.09, abs=0.01)       # 8.1 / 133.1
    assert s.operating_margin_pct == pytest.approx(12.02, abs=0.01)   # 16 / 133.1
    # Yahoo's own figures win whenever they exist.
    s = summarise_fundamentals("X.NS", INFO, INCOME, balance_sheet=BALANCE, today=TODAY)
    assert (s.return_on_equity_pct, s.profit_margin_pct, s.operating_margin_pct) == (8.5, 7.5, 12.0)


def test_roe_from_accounts_needs_matching_positive_equity():
    profit = [(date(2025, 3, 31), 8.0), (date(2026, 3, 31), 9.0)]
    assert roe_from_accounts(profit, [(date(2026, 3, 31), 90.0)]) == 10.0     # closing equity only
    assert roe_from_accounts(profit, [(date(2026, 3, 31), -5.0)]) is None      # negative equity
    assert roe_from_accounts(profit, [(date(2024, 3, 31), 90.0)]) is None      # no equity for that year
    assert roe_from_accounts([], [(date(2026, 3, 31), 90.0)]) is None
    assert margin_from_accounts(profit, [(date(2026, 3, 31), 0.0)]) is None


def test_operating_profit_growth_compares_the_same_quarter_a_year_earlier():
    quarters = statement_series(QUARTERLY, ("Operating Income",))
    assert year_on_year_pct(quarters) == pytest.approx(9.89, abs=0.01)       # 30.0 vs 27.3
    assert year_on_year_pct(quarters[1:]) is None                            # no year-ago quarter
    assert year_on_year_pct([(date(2025, 6, 30), -2.0), (date(2026, 6, 30), 5.0)]) is None  # loss base


def test_a_one_off_profit_drop_is_not_flagged_as_weakness():
    s = summarise_fundamentals("X.NS", ONE_OFF_INFO, quarterly_income_statement=QUARTERLY, today=TODAY)
    assert s.operating_profit_growth_yoy_pct == pytest.approx(9.89, abs=0.01)
    assert ("net profit down 22.4% year on year but operating profit up 9.9%: likely one-off items "
            "(e.g. a gain in the year-ago quarter), not a weaker business") in s.flags
    assert not any(f.startswith("profits falling") for f in s.flags)


def test_a_profit_drop_with_growing_revenue_is_questioned_without_operating_data():
    s = summarise_fundamentals("X.NS", ONE_OFF_INFO, today=TODAY)
    assert ("profits falling (-22.4% year on year) despite revenue growth: "
            "check for one-off items in the year-ago quarter") in s.flags


def test_falling_operating_profit_is_flagged():
    quarterly = QUARTERLY.copy()
    quarterly.loc["Operating Income", pd.Timestamp("2026-06-30")] = 21.0      # vs 27.3 a year ago
    s = summarise_fundamentals("X.NS", {"earningsGrowth": 0.02}, quarterly_income_statement=quarterly,
                               today=TODAY)
    assert "operating profit falling (-23.1% year on year)" in s.flags


def test_next_results_date_ignores_the_past_and_accepts_any_date_type():
    assert next_earnings_date({"Earnings Date": [date(2026, 7, 20)]}, TODAY) is None
    mixed = {"Earnings Date": [date(2026, 7, 20), datetime(2026, 11, 3, 9, 0), "2026-10-25"]}
    assert next_earnings_date(mixed, TODAY) == date(2026, 10, 25)
    assert next_earnings_date({}, TODAY) is None


def test_missing_data_gives_a_valid_empty_snapshot():
    s = summarise_fundamentals("X.NS", {}, None, None, today=TODAY)
    assert s.trailing_pe is None and s.flags == []
    assert s.to_analyst_json().replace(" ", "").replace("\n", "") == '{"ticker":"X.NS","flags":[]}'


def _fake_yfinance(monkeypatch, *, info, income, calendar, balance=None, quarterly=None):
    class FakeTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        @property
        def info(self):
            return info

        @property
        def income_stmt(self):
            return income

        @property
        def calendar(self):
            if isinstance(calendar, Exception):
                raise calendar
            return calendar

        @property
        def balance_sheet(self):
            return balance

        @property
        def quarterly_income_stmt(self):
            return quarterly

    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(Ticker=FakeTicker))


def test_fetch_survives_a_failing_piece(monkeypatch):
    _fake_yfinance(monkeypatch, info=INFO, income=INCOME, calendar=RuntimeError("Yahoo hiccup"))
    s = fetch_fundamentals("example.ns", today=TODAY)
    assert s.trailing_pe == 24.46 and s.revenue_cagr_pct == pytest.approx(10.0)
    assert s.next_earnings_date is None


def test_fetch_ignores_pieces_in_an_unexpected_shape(monkeypatch):
    # Older yfinance versions returned the results calendar as a DataFrame.
    old_style_calendar = pd.DataFrame({"Value": [TODAY]}, index=["Earnings Date"])
    _fake_yfinance(monkeypatch, info=INFO, income="not a table", calendar=old_style_calendar)
    s = fetch_fundamentals("example.ns", today=TODAY)
    assert s.trailing_pe == 24.46
    assert s.revenue_cagr_pct is None and s.next_earnings_date is None


def _fundamentals_log_line(caplog) -> str:
    return next(r.getMessage() for r in caplog.records if r.getMessage().startswith("Fundamentals EXAMPLE.NS"))


def test_fetch_uses_the_accounts_and_logs_the_key_metrics(monkeypatch, caplog):
    _fake_yfinance(monkeypatch, info=ONE_OFF_INFO, income=INCOME, calendar=CALENDAR,
                   balance=BALANCE, quarterly=QUARTERLY)
    with caplog.at_level(logging.INFO):
        s = fetch_fundamentals("example.ns", today=TODAY)
    assert s.return_on_equity_pct == pytest.approx(8.1)
    assert s.operating_profit_growth_yoy_pct == pytest.approx(9.89, abs=0.01)
    line = _fundamentals_log_line(caplog)
    assert "ROE 8.1%" in line and "operating profit YoY +9.89%" in line and "net profit YoY -22.4%" in line
    assert "D/E 0.37x" in line and "None" not in line


def test_log_line_says_na_for_unknown_metrics(monkeypatch, caplog):
    _fake_yfinance(monkeypatch, info={"trailingPE": 21.69}, income=None, calendar={})
    with caplog.at_level(logging.INFO):
        fetch_fundamentals("example.ns", today=TODAY)
    line = _fundamentals_log_line(caplog)
    assert "P/E 21.69" in line and "ROE n/a" in line and "D/E n/a" in line and "None" not in line


def test_fetch_raises_when_yahoo_returns_nothing(monkeypatch):
    _fake_yfinance(monkeypatch, info={}, income=pd.DataFrame(), calendar={})
    with pytest.raises(FundamentalsError):
        fetch_fundamentals("NOTHING.NS", today=TODAY)


# --- Layer 2: both analysts see the fundamentals ----------------------------- #

def test_prompts_explain_how_to_use_fundamentals():
    assert "FUNDAMENTALS" in SYSTEM_PROMPT and "filter" in SYSTEM_PROMPT
    assert "FUNDAMENTALS" in OPTIONS_SYSTEM_PROMPT and "results" in OPTIONS_SYSTEM_PROMPT
    assert "one-off" in SYSTEM_PROMPT and "one-off" in OPTIONS_SYSTEM_PROMPT


def test_shares_analyst_sends_fundamentals_or_says_unavailable():
    fundamentals = summarise_fundamentals("RELIANCE.NS", INFO, INCOME, CALENDAR, today=TODAY)
    reply = '{"action": "HOLD", "ticker": "RELIANCE.NS", "target_price": 1193.4, "reasoning": "r"}'

    client = ScriptedClient(reply)
    AnalystAgent(settings(), client=client).analyze(SNAPSHOT, fundamentals)
    content = client.calls[0]["messages"][0]["content"]
    assert "FUNDAMENTALS:" in content and '"trailing_pe": 24.46' in content

    client = ScriptedClient(reply)
    AnalystAgent(settings(), client=client).analyze(SNAPSHOT)
    assert "FUNDAMENTALS:\nunavailable" in client.calls[0]["messages"][0]["content"]


def test_options_analyst_sends_fundamentals():
    fundamentals = summarise_fundamentals("RELIANCE.NS", INFO, INCOME, CALENDAR, today=TODAY)
    client = ScriptedClient('{"action": "HOLD", "ticker": "RELIANCE", "expiry": "2026-10-27", '
                            '"strike": null, "underlying_target": 1193.4, "reasoning": "r"}')
    OptionsAnalystAgent(settings(), client=client).analyze(SNAPSHOT, make_chain(), fundamentals)
    content = client.calls[0]["messages"][0]["content"]
    assert content.index("FUNDAMENTALS:") < content.index("OPTION_CHAIN:")
    assert '"next_earnings_date": "2026-10-17"' in content


# --- Layer 3: earnings blackout ------------------------------------------------ #

BUY = AnalystDecision(action="BUY", ticker="RELIANCE.NS", target_price=1250.0, reasoning="r")
ACCOUNT = AccountState(total_equity=1_000_000.0, cash_balance=1_000_000.0)


def _validate_shares(results_on, blackout=5, action=BUY):
    return RiskInterceptor(rules(blackout)).validate(
        action, current_price=100.0, account_state=ACCOUNT, next_earnings_date=results_on, today=TODAY
    )


def test_shares_blackout_blocks_new_buys_right_before_results():
    with pytest.raises(RiskViolation, match="earnings blackout"):
        _validate_shares(TODAY + timedelta(days=3))
    with pytest.raises(RiskViolation, match="earnings blackout"):
        _validate_shares((TODAY + timedelta(days=5)).isoformat())   # ISO strings work too


@pytest.mark.parametrize(
    "results_on, blackout",
    [
        (TODAY + timedelta(days=6), 5),   # just outside the window
        (None, 5),                        # unknown date never blocks
        ("not-a-date", 5),                # unreadable date never blocks
        (TODAY + timedelta(days=1), 0),   # rule switched off
        (TODAY - timedelta(days=1), 5),   # results already out
    ],
)
def test_shares_blackout_allows_everything_else(results_on, blackout):
    assert _validate_shares(results_on, blackout).is_tradeable


def test_hold_is_never_blocked_by_the_blackout():
    hold = AnalystDecision(action="HOLD", ticker="RELIANCE.NS", target_price=100.0, reasoning="r")
    assert _validate_shares(TODAY + timedelta(days=1), action=hold).is_tradeable is False


def test_options_blackout_blocks_new_calls_and_puts_before_results():
    interceptor = OptionRiskInterceptor(rules(5))
    with pytest.raises(RiskViolation, match="earnings blackout"):
        interceptor.validate(option_decision("BUY_PUT", 1200), make_chain(), ACCOUNT,
                             today=CHAIN_DAY, next_earnings_date=CHAIN_DAY + timedelta(days=2))
    order = interceptor.validate(option_decision("BUY_PUT", 1200), make_chain(), ACCOUNT,
                                 today=CHAIN_DAY, next_earnings_date=CHAIN_DAY + timedelta(days=20))
    assert order.is_tradeable


# --- Wiring: main.py ----------------------------------------------------------- #

def test_shares_pipeline_passes_fundamentals_and_enforces_the_blackout(monkeypatch, caplog):
    seen = {}
    soon = {"Earnings Date": [date.today() + timedelta(days=2)]}
    fundamentals = summarise_fundamentals("RELIANCE.NS", INFO, INCOME, soon)

    class FakeAgent:
        def __init__(self, _settings):
            pass

        def analyze(self, snapshot, fundamentals=None):
            seen["fundamentals"] = fundamentals
            return BUY

    monkeypatch.setattr(main, "get_settings", settings)
    monkeypatch.setattr(main, "fetch_market_snapshot", lambda ticker: SNAPSHOT)
    monkeypatch.setattr(main, "fetch_fundamentals", lambda ticker, **kw: fundamentals)
    monkeypatch.setattr(main, "AnalystAgent", FakeAgent)

    with caplog.at_level(logging.INFO):
        code = main.run_pipeline("RELIANCE", exchange="NSE", submit=False, proposed_qty=None, sim_account=True)
    assert seen["fundamentals"] is fundamentals
    assert code == 4 and "earnings blackout" in caplog.text     # shipped rules: 5-day blackout
