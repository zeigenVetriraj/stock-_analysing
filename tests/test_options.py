"""Tests for the buy-only options mode, layer by layer.

Everything here is offline: a fixture option chain in Dhan's response format,
a tiny scrip master CSV, a fake HTTP poster and a fake Claude client. NO network,
NO LLM, NO orders. Run with:  ``pytest -q``
"""

from __future__ import annotations

import logging
import types
from datetime import date, timedelta

import pytest

import main
from config.settings import OptionRules, Settings, TradingRules
from src.analysis.options_agent import (
    OPTIONS_SYSTEM_PROMPT,
    OptionDecision,
    OptionsAnalystAgent,
)
from src.execution.instruments import DhanInstruments, InstrumentError
from src.execution.order_executor import ExecutionError, OrderExecutor
from src.ingestion.data_fetcher import MarketSnapshot
from src.ingestion.fundamentals import FundamentalsError
from src.ingestion.option_chain import (
    DhanMarketData,
    OptionChainError,
    choose_expiry,
    fetch_option_chain_snapshot,
    parse_option_chain,
)
from src.risk.guardrails import AccountState, RiskViolation
from src.risk.options_guardrails import OptionRiskInterceptor

# --- Fixtures ---------------------------------------------------------------- #

TODAY = date(2026, 9, 29)
EXPIRY = "2026-10-27"  # 28 days after TODAY
LOT = 500

OPT_RULES = OptionRules(
    enabled=True, max_premium_pct=0.02, min_days_to_expiry=14, max_days_to_expiry=60,
    strikes_each_side=5, min_open_interest_lots=50, max_bid_ask_spread_pct=5.0,
    min_abs_delta=0.25, max_abs_delta=0.75, stop_loss_pct=0.30, take_profit_pct=0.60,
    max_lots=10,
)
RULES = TradingRules(
    max_position_pct=0.02, stop_loss_pct=0.03, take_profit_pct=0.06, max_open_positions=5,
    min_trade_notional=1.0, allowed_actions=("BUY", "SELL", "HOLD"),
    allow_short_selling=False, options=OPT_RULES,
)


def _leg(sec, ltp, bid, ask, oi_units, delta, iv=25.0):
    return {
        "security_id": sec, "last_price": ltp, "top_bid_price": bid, "top_ask_price": ask,
        "oi": oi_units, "volume": oi_units * 2, "implied_volatility": iv,
        "greeks": {"delta": delta, "theta": -0.9, "gamma": 0.002, "vega": 1.1},
    }


def chain_payload(**overrides):
    """A RELIANCE option chain in Dhan's /optionchain response format (spot 1193.40)."""
    oc = {
        "1140.000000": {"ce": _leg(1001, 66, 65.6, 66.4, 20_000, 0.78), "pe": _leg(2001, 12, 11.9, 12.1, 200_000, -0.22)},
        "1160.000000": {"ce": _leg(1002, 52, 51.8, 52.2, 50_000, 0.68), "pe": _leg(2002, 18, 17.9, 18.1, 300_000, -0.32)},
        "1180.000000": {"ce": _leg(1003, 40, 39.8, 40.2, 100_000, 0.58), "pe": _leg(2003, 26, 25.9, 26.1, 250_000, -0.42)},
        "1200.000000": {"ce": _leg(1004, 30, 29.9, 30.1, 200_000, 0.47), "pe": _leg(2004, 36.5, 36.4, 36.6, 150_000, -0.53)},
        "1220.000000": {"ce": _leg(1005, 22, 21.9, 22.1, 300_000, 0.36), "pe": _leg(2005, 48, 47.8, 48.2, 80_000, -0.64)},
        "1240.000000": {"ce": _leg(1006, 15, 14.9, 15.1, 450_000, 0.27), "pe": _leg(2006, 61, 60.6, 61.4, 40_000, -0.73)},
        "1260.000000": {"ce": _leg(1007, 10, 9.0, 11.0, 350_000, 0.19), "pe": _leg(2007, 76, 75.5, 76.5, 10_000, -0.81)},
    }
    for strike, sides in overrides.items():
        for side, fields in sides.items():
            oc[strike][side].update(fields)
    return {"data": {"last_price": 1193.4, "oc": oc}, "status": "success"}


def make_chain(payload=None, each_side=5, today=TODAY, expiry=EXPIRY):
    return parse_option_chain(
        payload or chain_payload(), underlying="reliance", underlying_security_id="2885",
        expiry=expiry, lot_size=LOT, strikes_each_side=each_side, today=today,
    )


def decision(action="BUY_PUT", strike=1200.0, expiry=EXPIRY, target=1150.0):
    return OptionDecision(
        action=action, ticker="RELIANCE", expiry=expiry, strike=strike,
        underlying_target=target, reasoning="test",
    )


def account(equity=1_000_000.0, cash=None, open_positions=0):
    return AccountState(total_equity=equity, cash_balance=equity if cash is None else cash,
                        open_positions=open_positions)


def settings(**overrides):
    base = {"DHAN_CLIENT_ID": "1100000001", "DHAN_ACCESS_TOKEN": "test-token",
            "DHAN_ENV": "sandbox", "DHAN_ENABLE_LIVE_TRADING": False}
    base.update(overrides)
    return Settings(_env_file=None, **base)


# --- Layer 1: option chain --------------------------------------------------- #

def test_choose_expiry_takes_the_nearest_inside_the_window():
    listed = ["2026-09-30", "2026-10-27", "2026-11-24", "2026-12-29"]
    assert choose_expiry(listed, TODAY, 14, 60) == "2026-10-27"   # 1 day is too close
    assert choose_expiry(listed, TODAY, 30, 60) == "2026-11-24"
    assert choose_expiry(listed, TODAY, 100, 200) is None


def test_parse_option_chain_builds_the_window_and_context():
    chain = make_chain(each_side=2)
    assert chain.underlying == "RELIANCE"
    assert chain.atm_strike == 1200.0                      # nearest to spot 1193.40
    assert [s.strike for s in chain.strikes] == [1160.0, 1180.0, 1200.0, 1220.0, 1240.0]
    assert chain.days_to_expiry == 28
    assert chain.put_call_oi_ratio == pytest.approx(0.70)  # 2,060 put lots / 2,940 call lots
    assert chain.max_call_oi_strike == 1240.0
    assert chain.max_put_oi_strike == 1160.0
    pe = chain.find(1200).pe
    assert (pe.security_id, pe.bid, pe.ask, pe.delta) == ("2004", 36.4, 36.6, -0.53)
    assert pe.open_interest_lots == 300.0                  # 150,000 units / 500


def test_parse_option_chain_rejects_empty_responses():
    with pytest.raises(OptionChainError):
        make_chain({"data": {"last_price": 1193.4, "oc": {}}, "status": "success"})
    with pytest.raises(OptionChainError):
        make_chain({"status": "success"})


class _FakeInstruments:
    def resolve(self, symbol, exchange):
        assert (symbol, exchange) == ("RELIANCE", "NSE")
        return "2885"

    def option_contract(self, security_id, exchange="NSE"):
        return types.SimpleNamespace(lot_size=LOT)


def test_fetch_option_chain_calls_the_live_data_api_politely():
    calls, sleeps = [], []

    def fake_post(url, headers, body):
        calls.append((url, headers, body))
        if url.endswith("/expirylist"):
            return {"data": ["2026-09-30", EXPIRY, "2026-11-24"], "status": "success"}
        return chain_payload()

    market = DhanMarketData(settings(), post=fake_post, sleep=sleeps.append)
    chain = fetch_option_chain_snapshot(
        "reliance", rules=OPT_RULES, settings=settings(), instruments=_FakeInstruments(),
        market_data=market, today=TODAY,
    )
    assert chain.expiry == EXPIRY and chain.lot_size == LOT
    (url1, headers, body1), (url2, _, body2) = calls
    assert url1 == "https://api.dhan.co/v2/optionchain/expirylist"   # live data API, even in sandbox mode
    assert url2 == "https://api.dhan.co/v2/optionchain"
    assert body1 == {"UnderlyingScrip": 2885, "UnderlyingSeg": "NSE_EQ"}
    assert body2 == {"UnderlyingScrip": 2885, "UnderlyingSeg": "NSE_EQ", "Expiry": EXPIRY}
    assert headers["client-id"] == "1100000001" and headers["access-token"] == "test-token"
    assert len(sleeps) == 1 and sleeps[0] > 3.0                       # Dhan's 3-second rate limit


def test_market_data_needs_dhan_credentials():
    market = DhanMarketData(settings(DHAN_CLIENT_ID="", DHAN_ACCESS_TOKEN=""), post=lambda *a: {})
    with pytest.raises(OptionChainError, match="DHAN_CLIENT_ID"):
        market.expiry_list(2885)


def test_option_contract_lot_size_comes_from_the_derivatives_segment(tmp_path):
    csv_path = tmp_path / "scrip.csv"
    csv_path.write_text(
        "SEM_EXM_EXCH_ID,SEM_SEGMENT,SEM_SMST_SECURITY_ID,SEM_INSTRUMENT_NAME,SEM_TRADING_SYMBOL,"
        "SEM_LOT_UNITS,SEM_CUSTOM_SYMBOL,SEM_EXPIRY_DATE,SEM_STRIKE_PRICE,SEM_OPTION_TYPE,SEM_SERIES\n"
        "NSE,E,2885,EQUITY,RELIANCE,1.0,Reliance Industries,,,,EQ\n"
        "NSE,D,2885,OPTSTK,TCS-Oct2026-4000-CE,175.0,TCS 27 OCT 4000 CALL,2026-10-27 14:30:00,4000.00000,CE,\n"
        "NSE,D,2004,OPTSTK,RELIANCE-Oct2026-1200-PE,500.0,RELIANCE 27 OCT 1200 PUT,2026-10-27 14:30:00,1200.00000,PE,\n",
        encoding="utf-8",
    )
    instruments = DhanInstruments(cache_path=csv_path)
    contract = instruments.option_contract("2004")
    assert (contract.lot_size, contract.option_type, contract.strike, contract.expiry) == (500, "PE", 1200.0, "2026-10-27")
    assert instruments.resolve("RELIANCE", "NSE") == "2885"
    assert instruments.option_contract(2885).lot_size == 175    # the D-segment row, not the equity one
    with pytest.raises(InstrumentError):
        instruments.option_contract("999999")


# --- Layer 2: options analyst ------------------------------------------------ #

def test_option_decision_parses_model_output_and_fixes_the_ticker():
    text = ('```json\n{"action": "BUY_PUT", "ticker": "RELIANCE.NS", "expiry": "2026-10-27", '
            '"strike": 1200, "underlying_target": 1150, "reasoning": "Bearish below SMA200."}\n```')
    d = OptionDecision.from_model_text(text, "RELIANCE")
    assert (d.action, d.ticker, d.strike, d.expiry) == ("BUY_PUT", "RELIANCE", 1200.0, "2026-10-27")


def test_option_decision_needs_a_strike_to_trade():
    with pytest.raises(ValueError):
        OptionDecision(action="BUY_CALL", ticker="X", expiry=EXPIRY, underlying_target=10, reasoning="r")
    hold = OptionDecision(action="HOLD", ticker="X", expiry=EXPIRY, underlying_target=10, reasoning="r")
    assert hold.strike is None


class _FakeClient:
    def __init__(self, text):
        self.calls = []
        outer = self

        class _Messages:
            def create(self, **kwargs):
                outer.calls.append(kwargs)
                return types.SimpleNamespace(content=[types.SimpleNamespace(type="text", text=text)])

        self.messages = _Messages()


SNAPSHOT = MarketSnapshot(
    ticker="RELIANCE.NS", as_of="2026-09-29T07:30:00+00:00", current_price=1193.4,
    rsi_14=31.0, sma_50=1284.1, sma_200=1359.9, trend_regime="bearish",
)


def test_options_analyst_sends_stock_and_chain_and_parses_the_reply():
    client = _FakeClient('{"action": "BUY_PUT", "ticker": "RELIANCE", "expiry": "2026-10-27", '
                         '"strike": 1200, "underlying_target": 1181.8, "reasoning": "r"}')
    d = OptionsAnalystAgent(settings(), client=client).analyze(SNAPSHOT, make_chain())
    assert d.action == "BUY_PUT" and d.strike == 1200.0
    call = client.calls[0]
    assert call["system"] == OPTIONS_SYSTEM_PROMPT
    content = call["messages"][0]["content"]
    assert "MARKET_SNAPSHOT" in content and "OPTION_CHAIN" in content and EXPIRY in content


def test_options_prompt_is_buy_only():
    assert "NEVER sell or write options" in OPTIONS_SYSTEM_PROMPT
    assert '"BUY_CALL" | "BUY_PUT" | "HOLD"' in OPTIONS_SYSTEM_PROMPT


# --- Layer 3: options risk interceptor --------------------------------------- #

def validate(dec, acct=None, rules=RULES, lots=None, today=TODAY, chain=None):
    return OptionRiskInterceptor(rules).validate(
        dec, chain or make_chain(), acct or account(), proposed_lots=lots, today=today
    )


def test_buy_put_is_sized_in_whole_lots_under_the_premium_cap():
    order = validate(decision("BUY_PUT", 1200))
    assert order.is_tradeable and order.option_type == "PE"
    assert order.security_id == "2004"
    assert order.limit_price == 36.6                      # pays the ask
    assert (order.lots, order.quantity) == (1, 500)       # 1 lot = Rs 18,300 <= 2% of Rs 10 lakh
    assert order.premium_outlay == pytest.approx(18_300.0)
    assert order.premium_outlay <= order.max_allowed_outlay
    assert order.stop_loss_premium == 25.6                # -30%, on the Rs 0.05 tick
    assert order.take_profit_premium == 58.55             # +60%
    assert order.risk_pct_of_equity == pytest.approx(1.83)


def test_buy_call_uses_the_call_leg_and_downsizes_big_requests():
    order = validate(decision("BUY_CALL", 1220), acct=account(5_000_000.0), lots=20)
    assert (order.option_type, order.security_id) == ("CE", "1005")
    assert order.lots == 9                                # floor(1,00,000 / 11,050)
    assert order.was_downsized is True
    assert order.quantity == 9 * LOT


def test_lots_are_capped_by_max_lots():
    order = validate(decision("BUY_CALL", 1220), acct=account(10_000_000.0))
    assert order.lots == OPT_RULES.max_lots


def test_hold_produces_no_order():
    order = validate(OptionDecision(action="HOLD", ticker="RELIANCE", expiry=EXPIRY,
                                    underlying_target=1193.4, reasoning="r"))
    assert order.is_tradeable is False and order.quantity == 0


@pytest.mark.parametrize(
    "dec, kwargs, message",
    [
        (decision("BUY_PUT", 1200), {"acct": account(500_000.0)}, "premium cap"),
        (decision("BUY_PUT", 1200), {"acct": account(1_000_000.0, cash=10_000.0)}, "premium cap"),
        (decision("BUY_PUT", 1210), {}, "not in the fetched"),
        (decision("BUY_PUT", 1200, expiry="2026-11-24"), {}, "not the fetched expiry"),
        (decision("BUY_PUT", 1200), {"today": date(2026, 10, 20)}, "days away"),
        (decision("BUY_CALL", 1140), {}, "open interest"),
        (decision("BUY_CALL", 1260), {}, "spread"),
        (decision("BUY_PUT", 1140), {}, "delta"),
        (decision("BUY_PUT", 1200), {"acct": account(open_positions=5)}, "Open positions"),
    ],
)
def test_unsafe_option_trades_are_killed(dec, kwargs, message):
    with pytest.raises(RiskViolation, match=message):
        validate(dec, **kwargs)


def test_missing_quote_is_killed():
    chain = make_chain(chain_payload(**{"1200.000000": {"pe": {"top_bid_price": 0}}}))
    with pytest.raises(RiskViolation, match="two-sided quote"):
        validate(decision("BUY_PUT", 1200), chain=chain)


def test_selling_options_is_impossible():
    bad = OptionDecision.model_construct(action="SELL_CALL", ticker="RELIANCE", expiry=EXPIRY,
                                         strike=1200.0, underlying_target=1100.0, reasoning="r")
    with pytest.raises(RiskViolation, match="Buy-only"):
        validate(bad)


def test_options_can_be_switched_off_in_the_rules():
    off = RULES.model_copy(update={"options": OPT_RULES.model_copy(update={"enabled": False})})
    with pytest.raises(RiskViolation, match="disabled"):
        validate(decision(), rules=off)
    with pytest.raises(RiskViolation, match="disabled"):
        validate(decision(), rules=RULES.model_copy(update={"options": None}))


# --- Layer 4: option orders -------------------------------------------------- #

def test_option_payload_is_an_nse_fno_limit_buy():
    order = validate(decision("BUY_PUT", 1200))
    payload = OrderExecutor(settings(), dry_run=True).build_option_payload(order)
    assert payload["transactionType"] == "BUY"
    assert payload["exchangeSegment"] == "NSE_FNO"
    assert payload["productType"] == "MARGIN"
    assert payload["orderType"] == "LIMIT" and payload["price"] == 36.6
    assert (payload["securityId"], payload["quantity"]) == ("2004", 500)


def test_option_dry_run_sends_nothing():
    result = OrderExecutor(settings(), dry_run=True).execute_option(validate(decision()))
    assert result.submitted is False and result.status == "dry_run_logged"
    assert result.risk_plan["max_loss"] == pytest.approx(18_300.0)


def test_live_lock_blocks_a_real_option_order():
    live = settings(DHAN_ENV="live", DHAN_ENABLE_LIVE_TRADING=False)
    with pytest.raises(ExecutionError, match="LIVE order blocked"):
        OrderExecutor(live, dry_run=False).execute_option(validate(decision()))


def test_partial_lots_are_refused_by_the_executor():
    order = validate(decision()).model_copy(update={"quantity": 250})
    with pytest.raises(ExecutionError, match="whole number of lots"):
        OrderExecutor(settings(), dry_run=True).execute_option(order)


# --- Wiring: main.py --------------------------------------------------------- #

def test_options_pipeline_runs_end_to_end_as_a_dry_run(monkeypatch, caplog):
    expiry = (date.today() + timedelta(days=28)).isoformat()
    chain = make_chain(today=date.today(), expiry=expiry)

    class FakeAgent:
        def __init__(self, _settings):
            pass

        def analyze(self, snapshot, option_chain, fundamentals=None):
            return decision("BUY_PUT", 1200, expiry=option_chain.expiry)

    def no_fundamentals(ticker, **kwargs):
        raise FundamentalsError("offline test")

    monkeypatch.setattr(main, "get_settings", lambda: settings())
    monkeypatch.setattr(main, "fetch_market_snapshot", lambda ticker: SNAPSHOT)
    monkeypatch.setattr(main, "fetch_option_chain_snapshot", lambda symbol, **kw: chain)
    monkeypatch.setattr(main, "fetch_fundamentals", no_fundamentals)  # never hit the network
    monkeypatch.setattr(main, "OptionsAnalystAgent", FakeAgent)

    with caplog.at_level(logging.INFO):
        code = main.run_options_pipeline(
            "RELIANCE", exchange="NSE", submit=False, proposed_lots=None, sim_account=True
        )
    assert code == 0
    assert "Fundamentals unavailable" in caplog.text          # degrades gracefully
    assert "[DRY-RUN] Would submit option order" in caplog.text


def test_cli_refuses_unsafe_flag_combinations():
    assert main.main(["--symbol", "RELIANCE", "--sim-account", "--submit"]) == 2
    assert main.main(["--symbol", "RELIANCE", "--lots", "2"]) == 2
    assert main.main(["--symbol", "RELIANCE", "--options", "--qty", "5"]) == 2
