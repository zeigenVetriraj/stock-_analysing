"""Tests for the journal and the end-of-day review: what Claude saw and chose is
written down at the entry, and the square-off replays every shortlisted setup and
all of the day's setups, ranks Claude's pick and keeps running totals.

Fakes only - NO network, NO LLM, NO orders. Run with:  ``pytest -q``
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

from src.automation.jobs import IST, run_job
from src.automation.journal import day_path, load_day, running_totals, save_day
from src.automation.picker import PickDecision
from tests.test_automation import TODAY
from tests.test_intraday import DAILY, INDEX, OPENING, IntradayPicker, deps, two_days

# Full sessions (09:15-15:10): AAA breaks out and keeps rising; BBB breaks down, then rallies back.
AAA_DAY = OPENING + [101.2, 101.5] + [101.5 + 0.04 * i for i in range(1, 65)]
BBB_DAY = OPENING + [99.2, 98.9] + [98.9 + 0.03 * i for i in range(1, 65)]
FULL = {"AAA.NS": two_days(AAA_DAY), "BBB.NS": two_days(BBB_DAY), "CCC.NS": two_days(OPENING + [100.3] * 66),
        INDEX: two_days([c * 250 for c in AAA_DAY])}
MORNING = {t: f[f.index < datetime.combine(TODAY, datetime.min.time().replace(hour=9, minute=55), tzinfo=IST)]
           for t, f in FULL.items()}
AT = lambda h, m: datetime.combine(TODAY, datetime.min.time().replace(hour=h, minute=m), tzinfo=IST)  # noqa: E731


def run(tmp_path, kind, picker, five_min, now, playbook=None):
    d = deps(tmp_path, history=DAILY, picker=picker, now=now, playbook=lambda: playbook,
             intraday_history=lambda tickers: {t: f for t, f in five_min.items() if t in tickers})
    return run_job(kind, d), d


def test_the_entry_writes_what_claude_saw_and_chose(tmp_path):
    picker = IntradayPicker(PickDecision(action="SELL", ticker="BBB.NS", target_price=97.0, reasoning="clean drop"))
    outcome, _ = run(tmp_path, "intraday-open", picker, MORNING, AT(9, 52))
    assert outcome.traded
    data = load_day(tmp_path, TODAY)
    (entry,) = data["entries"]
    assert entry["scan"] == "09:52" and entry["note"] == "traded" and entry["pick"]["ticker"] == "BBB.NS"
    assert entry["practice"] is True                      # the shipped paper account is in practice mode
    assert {c["ticker"] for c in entry["shortlist"]} == {"AAA.NS", "BBB.NS"}
    short = next(c for c in entry["shortlist"] if c["ticker"] == "BBB.NS")
    assert short["model"] == "breakout" and short["direction"] == "short" and short["f_trend"] == 1
    assert short["stop_price"] > short["price"] > short["target_price"] and short["expected_pct"] is None
    assert entry["opened"]["symbol"] == "BBB.NS" and entry["opened"]["setup"] == "breakout"


def test_a_hold_is_written_down_too(tmp_path):
    picker = IntradayPicker(PickDecision(action="HOLD", reasoning="nothing clean"))
    run(tmp_path, "intraday-open", picker, MORNING, AT(9, 52))
    (entry,) = load_day(tmp_path, TODAY)["entries"]
    assert entry["note"] == "HOLD" and entry["pick"]["action"] == "HOLD" and entry["opened"] is None


def test_the_square_off_reviews_the_day(tmp_path):
    picker = IntradayPicker(PickDecision(action="BUY", ticker="AAA.NS", target_price=103.5, reasoning="trend"))
    run(tmp_path, "intraday-open", picker, MORNING, AT(9, 52))
    outcome, _ = run(tmp_path, "intraday-close", IntradayPicker(None), FULL, AT(15, 20))
    text = outcome.report.render()

    assert [t.symbol for t in outcome.closed] == ["AAA.NS"] and outcome.closed[0].pnl > 0
    assert "*Today's review*" in text and "*Experience so far*" in text
    assert "Claude's pick AAA (opening-range breakout) ranked 1 of 2 on the 09:52 shortlist" in text
    assert "All of today's setups (every stock, after costs): opening-range breakout" in text
    assert "Nifty 50 from 09:50 to 15:15: +" in text
    assert "1 trading day(s) reviewed, 1 paper trade(s), 1 winner(s)" in text
    assert "Claude's picks: +" in text and "- Lessons: none yet (the weekly review has not run)." in text

    review = load_day(tmp_path, TODAY)["review"]
    (shortlist,) = review["shortlists"]
    assert shortlist["pick_rank"] == 1 and [o["ticker"] for o in shortlist["outcomes"]] == ["AAA.NS", "BBB.NS"]
    aaa, bbb = shortlist["outcomes"]
    assert aaa["ret"] > 1 and aaa["picked"] and bbb["ret"] < -1 and bbb["reason"].startswith("stop-loss hit")
    assert review["setups"]["breakout"]["n"] >= 2 and review["rows"] and review["trades"][0]["symbol"] == "AAA.NS"


def test_a_day_without_an_entry_is_still_reviewed(tmp_path):
    """The bot learns from every setup it saw, even on a day it did not trade."""
    outcome, _ = run(tmp_path, "intraday-close", IntradayPicker(None), FULL, AT(15, 20))
    text = outcome.report.render()
    assert "No open intraday position" in text and "All of today's setups" in text
    assert load_day(tmp_path, TODAY)["review"]["shortlists"] == []


def test_the_playbook_reorders_the_shortlist(tmp_path):
    """A lesson in force (setups with the Nifty did worse, oddly) puts the setup against it first.

    The Nifty rises in this test, so the long AAA is with it and the short BBB is not. A lesson about
    a condition the bot no longer learns ("short") is ignored."""
    playbook = {
        "days": 30, "setups": 900, "baseline_pct": -0.12, "lessons": ["against the Nifty did better"],
        "watching": [], "models": {"breakout": {"label": "opening-range breakout", "adj_pct": 0.0}},
        "conditions": {"breakout": {
            "market": {"label": "with the Nifty's direction today", "status": "in force",
                       "share_with": 0.5, "diff_pct": -0.6, "shrink": 0.75},
            "short": {"label": "short trades (sell first)", "status": "in force",
                      "share_with": 0.5, "diff_pct": 5.0, "shrink": 1.0}}},
        "exits": {},
    }
    picker = IntradayPicker(PickDecision(action="HOLD", reasoning="r"))
    outcome, d = run(tmp_path, "intraday-open", picker, MORNING, AT(9, 52), playbook=playbook)
    assert [s["ticker"] for s in picker.seen] == ["BBB.NS", "AAA.NS"]
    first = picker.seen[0]["experience"]
    assert first["expected_net_pct"] == round(-0.12 + 0.5 * 0.6 * 0.75, 3) and first["coin_flip_pct"] == -0.12
    assert first["lessons_applied"] == ["with the Nifty's direction today: no (+0.22%)"]
    assert "expected +0.10%" in outcome.report.render()


def test_stocks_with_results_due_never_reach_claude(tmp_path):
    """Results inside the earnings blackout: the stock is left off the shortlist (not refused after the
    pick), and the next setup takes its place."""
    from src.ingestion.fundamentals import FundamentalSnapshot

    def fundamentals(ticker, price):
        due = TODAY.isoformat() if ticker == "BBB.NS" else None
        return FundamentalSnapshot(ticker=ticker, next_earnings_date=due)

    picker = IntradayPicker(PickDecision(action="BUY", ticker="AAA.NS", target_price=103.5, reasoning="r"))
    d = deps(tmp_path, history=DAILY, picker=picker, now=AT(9, 52), playbook=lambda: None,
             fundamentals=fundamentals,
             intraday_history=lambda tickers: {t: f for t, f in MORNING.items() if t in tickers})
    outcome = run_job("intraday-open", d)
    assert [s["ticker"] for s in picker.seen] == ["AAA.NS"] and outcome.traded
    assert "Left out (results due within 5 days): BBB." in outcome.report.render()


def test_a_learned_hold_exit_moves_the_target_out_of_reach(tmp_path):
    playbook = {"days": 30, "setups": 900, "baseline_pct": -0.12, "lessons": [], "watching": [], "models": {},
                "conditions": {}, "exits": {"breakout": {"use": "hold"}}}
    picker = IntradayPicker(PickDecision(action="BUY", ticker="AAA.NS", target_price=105.0, reasoning="r"))
    outcome, _ = run(tmp_path, "intraday-open", picker, MORNING, AT(9, 52), playbook=playbook)
    assert outcome.opened.take_profit == round(101.5 * 1.04, 2)
    assert "holding to 15:15 (learned" in outcome.report.render()


def test_running_totals_add_up_the_days(tmp_path):
    for offset, (pick_ret, other_ret) in enumerate([(1.0, -0.5), (-0.4, 0.2)]):
        day = TODAY - timedelta(days=offset)
        save_day(tmp_path, day, {"date": day.isoformat(), "entries": [], "review": {
            "shortlists": [{"outcomes": [{"ret": pick_ret, "picked": True}, {"ret": other_ret, "picked": False}]}],
            "rows": [{"model": "fib", "ret": 0.62}, {"model": "fib", "ret": -0.38}, {"model": "breakout", "ret": 1.12}],
        }})
    save_day(tmp_path, TODAY - timedelta(days=5), {"date": "x", "entries": []})          # not reviewed: ignored
    totals = running_totals(tmp_path, cost_pct=0.12)
    assert totals["days"] == 2 and totals["picks"] == 2 and totals["pick_avg_pct"] == 0.3
    assert totals["shortlist_avg_pct"] == round((0.25 + -0.1) / 2, 3)
    assert totals["models"]["fib"] == {"label": "Fibonacci pullback", "n": 4, "won": 2, "avg_pct": 0.0}
    assert json.loads(day_path(tmp_path, TODAY).read_text())["review"]
