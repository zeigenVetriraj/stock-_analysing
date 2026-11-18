"""Tests for the bull vs bear debate and its judge: the referee checks every cited number against
the real data, the judge only sees verified points, and the entry job trades the first setup the
judge approves (or none).

Fake Claude replies only - NO network, NO orders. Run with:  ``pytest -q``
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from src.automation import scorecard
from src.automation.daily_picks import picks_table
from src.automation.debate import DebateAgent, data_packet, flatten, verify_points
from src.automation.intraday import compute_features
from src.automation.jobs import IST, run_job
from src.automation.journal import load_day
from src.automation.ledger import Ledger, ledger_path
from src.automation.picker import PickDecision
from tests.test_automation import RULES, TODAY, FakeDebater, deps, settings
from tests.test_intraday import DAILY, INDEX, IntradayPicker
from tests.test_journal import AT, FULL, MORNING


def run(tmp_path, kind, picker, five_min, now, debater, rules=RULES):
    d = deps(tmp_path, history=DAILY, picker=picker, now=now, debater_factory=debater, rules=rules,
             intraday_history=lambda tickers: {t: f for t, f in five_min.items() if t in tickers})
    return run_job(kind, d), d


# --- the referee ----------------------------------------------------------------------- #
def test_the_referee_checks_every_cited_number():
    packet = {"rsi_5m": 61.27, "prev_day": {"high": 101.5}, "recent_bars": [{"close": 99.8}, {"close": 100.4}],
              "daily_trend": "down"}
    flat = flatten(packet)
    assert flat["prev_day.high"] == 101.5 and flat["recent_bars[1].close"] == 100.4
    checked = verify_points([
        {"claim": "momentum", "field": "rsi_5m", "value": 61.3},                    # rounding is fine
        {"claim": "resistance", "field": "prev_day.high", "value": 104},             # wrong number
        {"claim": "last close", "field": "recent_bars.1.close", "value": "100.4"},   # path style forgiven
        {"claim": "news", "field": "news_sentiment", "value": "positive"},           # not in the data
        {"claim": "a fifth point", "field": "rsi_5m", "value": 61.3},                # at most 4 are read
    ], flat)
    assert [p["verified"] for p in checked] == [True, False, True, False]
    assert checked[1]["referee"] == "value does not match (data: 101.5)"
    assert checked[3]["referee"] == "field not in the data"
    assert verify_points([{"claim": "trend", "field": "DATA.daily_trend", "value": "Down"}], flat)[0]["verified"]


def test_the_packet_carries_the_real_recent_candles():
    frame, index = MORNING["BBB.NS"], MORNING[INDEX]
    f = compute_features("BBB.NS", frame, TODAY, IST)
    setup = SimpleNamespace(features=f)
    packet = data_packet({"ticker": "BBB.NS", "price": f.price}, setup, frame, index, TODAY, IST)
    assert len(packet["recent_bars"]) == min(12, f.bars_today) == 8 and len(packet["nifty_recent_bars"]) == 6
    assert packet["recent_bars"][-1]["close"] == round(float(frame["Close"].iloc[-1]), 2)
    assert packet["day"]["high"] == round(f.day_high, 2)


# --- the agents ------------------------------------------------------------------------- #
class FakeClient:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []
        self.messages = self

    def create(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=self.replies.pop(0))])


def turn(*points, summary="s"):
    return json.dumps({"points": [{"claim": c, "field": f, "value": v} for c, f, v in points], "summary": summary})


def test_two_rounds_then_a_judge_who_sees_only_verified_points():
    packet = {"ticker": "BBB.NS", "setup": "short", "model": "breakout", "price": 98.9, "stop_price": 100.25,
              "target_price": 96.2, "stop_pct": 1.36, "target_pct": 2.72, "rsi_5m": 38.2, "vwap": 99.7}
    client = FakeClient([
        turn(("below VWAP", "vwap", 99.7), ("weak momentum", "rsi_5m", 38.2)),
        turn(("oversold soon", "rsi_5m", 38.2), ("invented support", "vwap", 95.0)),
        turn(("bear's support is not in the data", "vwap", 99.7)),
        turn(("stretched", "rsi_5m", 38.2)),
        json.dumps({"decision": "SKIP", "confidence": 64, "decisive_points": ["stretched"], "reasoning": "close call"}),
    ])
    result = DebateAgent(settings(), client=client).debate(packet, rounds=2)

    assert [t["side"] for t in result.turns] == ["bull", "bear", "bull", "bear"] and len(client.calls) == 5
    assert result.tally("bull") == (3, 3) and result.tally("bear") == (2, 3)
    assert (result.decision, result.confidence, result.reasoning) == ("SKIP", 64, "close call")
    third = client.calls[2]["messages"][0]["content"]
    assert "THROWN OUT (value does not match (data: 99.7))] invented support" in third     # the bull sees it
    judge = client.calls[4]["messages"][0]["content"]
    assert "invented support" not in judge and "1 point(s) thrown out by the referee" in judge
    assert client.calls[4]["system"].startswith("You are the JUDGE")
    lines = result.lines("BBB")
    assert lines[0] == "SHORT BBB:" and lines[1].startswith("🐂 Bull (3/3 points verified)")
    assert lines[3] == "⚖️ Judge: SKIP (64%) - close call"


# --- the entry job ------------------------------------------------------------------------- #
def test_the_judge_approves_claudes_pick(tmp_path):
    picker = IntradayPicker(PickDecision(action="SELL", ticker="BBB.NS", target_price=97.0, reasoning="clean drop"))
    debater = FakeDebater()
    outcome, d = run(tmp_path, "intraday-open", picker, MORNING, AT(9, 52), debater)
    assert outcome.traded and outcome.opened.symbol == "BBB.NS" and [s[0] for s in debater.seen] == ["BBB.NS"]
    assert debater.seen[0][1] == 2 and debater.seen[0][2]["recent_bars"]                 # 2 rounds, real candles
    text = d.notifier.messages[0]
    assert "*Debate (bull vs bear, then the judge)*" in text and "⚖️ Judge: TRADE (70%)" in text
    assert "Judge approved (70%): verdict on BBB.NS" in outcome.pick.reasoning
    (entry,) = load_day(tmp_path, TODAY)["entries"]
    assert [x["decision"] for x in entry["debates"]] == ["TRADE"]


def test_a_skipped_pick_hands_over_to_the_next_setup(tmp_path):
    picker = IntradayPicker(PickDecision(action="SELL", ticker="BBB.NS", target_price=97.0, reasoning="clean drop"))
    outcome, d = run(tmp_path, "intraday-open", picker, MORNING, AT(9, 52), FakeDebater({"BBB.NS": "SKIP"}))
    assert outcome.traded and outcome.opened.symbol == "AAA.NS" and outcome.opened.side == "long"
    assert outcome.pick.action == "BUY" and "first pick was skipped by the judge" in outcome.pick.reasoning
    (entry,) = load_day(tmp_path, TODAY)["entries"]
    assert [(x["ticker"], x["decision"]) for x in entry["debates"]] == [("BBB.NS", "SKIP"), ("AAA.NS", "TRADE")]
    assert entry["pick"]["ticker"] == "AAA.NS"

    outcome, d = run(tmp_path, "intraday-close", IntradayPicker(None), FULL, AT(15, 20), FakeDebater())
    review = load_day(tmp_path, TODAY)["review"]
    assert [(r["ticker"], r["decision"]) for r in review["judge"]] == [("BBB.NS", "SKIP"), ("AAA.NS", "TRADE")]
    bbb, aaa = review["judge"]
    assert aaa["ret"] > 0 > bbb["ret"]                                  # the judge was right this time
    assert "- Judge: SKIP BBB (would have ended" in d.notifier.messages[0]
    card = scorecard.build(Ledger.load(ledger_path("paper", tmp_path), mode="paper", starting_cash=0), tmp_path)
    assert card["judge"]["TRADE"]["n"] == 1 and card["judge"]["SKIP"]["won"] == 0
    assert "Judge (bull vs bear): approved 1 (1 won" in scorecard.text(card)


def test_no_trade_when_the_judge_skips_everything(tmp_path):
    picker = IntradayPicker(PickDecision(action="SELL", ticker="BBB.NS", target_price=97.0, reasoning="r"))
    outcome, d = run(tmp_path, "intraday-open", picker, MORNING, AT(9, 52),
                     FakeDebater({"BBB.NS": "SKIP", "AAA.NS": "SKIP"}))
    assert not outcome.traded and "The judge said SKIP to all 2 setup(s) debated; no trade today." in d.notifier.messages[0]
    (entry,) = load_day(tmp_path, TODAY)["entries"]
    assert entry["note"] == "Judge skipped every setup debated" and len(entry["debates"]) == 2
    assert "Large: no trade (the judge skipped every setup)" in picks_table(tmp_path, RULES, TODAY)


def test_a_failed_debate_does_not_cost_the_trade(tmp_path):
    picker = IntradayPicker(PickDecision(action="SELL", ticker="BBB.NS", target_price=97.0, reasoning="r"))
    outcome, d = run(tmp_path, "intraday-open", picker, MORNING, AT(9, 52), FakeDebater(fail=True))
    assert outcome.traded and outcome.opened.symbol == "BBB.NS"
    assert "The bull vs bear debate could not run" in d.notifier.messages[0]


def test_the_debate_can_be_switched_off(tmp_path):
    off = RULES.model_copy(update={"debate": RULES.debate.model_copy(update={"enabled": False})})
    picker = IntradayPicker(PickDecision(action="SELL", ticker="BBB.NS", target_price=97.0, reasoning="r"))
    debater = FakeDebater({"BBB.NS": "SKIP"})
    outcome, d = run(tmp_path, "intraday-open", picker, MORNING, AT(9, 52), debater, rules=off)
    assert outcome.traded and debater.seen == [] and "Debate" not in d.notifier.messages[0]
    assert RULES.debate.enabled and RULES.debate.rounds == 2 and RULES.debate.max_candidates == 3
