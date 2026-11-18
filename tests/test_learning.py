"""Tests for the learning loop: lessons need real evidence, are measured on the
same days, change the expected result of a setup, and are checked walk-forward.

Synthetic setup rows only - NO network. Run with:  ``pytest -q``
"""

from __future__ import annotations

import random

import pytest

from src.automation import learning
from src.automation.learning import (
    CONDITIONS,
    exit_for,
    expected_pct,
    fit_playbook,
    lessons_summary,
    walk_forward,
)

COST = 0.12


def make_rows(days=40, per_day=30, *, market_effect=0.0, model_shift=0.0, hold_gain=0.0, seed=1):
    """Breakout setups (half of them 'with the Nifty'), a coin flip, and noise; effects are planted."""
    rng = random.Random(seed)
    rows, flips = [], []
    for d in range(days):
        day = f"2026-{7 + d // 28:02d}-{1 + d % 28:02d}"
        market_move = rng.gauss(0, 0.3)                     # the whole market's move that day
        for i in range(per_day):
            with_market = i % 2 == 0
            ret = market_move + rng.gauss(0, 0.5) + model_shift + (market_effect if with_market else 0.0)
            rows.append({
                "day": day, "scan": "09:50" if i % 3 else "10:50", "ticker": f"S{i}.NS", "model": "breakout",
                "direction": "long" if i % 4 else "short", "score": 70 + (i % 3) * 10, "first": 1,
                "relative_volume": 1.0 + (i % 5) / 10, "stop_pct": 1.0, "gap_pct": 0.2, "rsi": 55.0,
                "f_trend": i % 3 == 0, "f_market": int(with_market), "f_vwap": 1, "f_volume": rng.random() < 0.5,
                "f_bonus": 0, "ret": ret, "ret_stop_hold": ret + hold_gain + rng.gauss(0, 0.05),
            })
        for i in range(20):
            flips.append({"day": day, "ticker": f"S{i}.NS", "model": "coin_flip", "direction": "long",
                          "ret": market_move + rng.gauss(0, 0.5)})
    return rows, flips


def test_a_real_condition_becomes_a_lesson_and_noise_does_not():
    rows, flips = make_rows(market_effect=0.6)
    pb = fit_playbook(rows, flips, cost_pct=COST)
    market = pb["conditions"]["breakout"]["market"]
    assert market["status"] == "in force" and market["diff_pct"] == pytest.approx(0.6, abs=0.1)
    assert 0 < market["shrink"] < 1 and market["days"] == 40 and market["with"] == market["without"] == 600
    assert pb["conditions"]["breakout"]["volume"]["status"] in ("no evidence", "watching")
    assert any("with the Nifty's direction today: better by" in x for x in pb["lessons"])

    with_row, without_row = dict(rows[0], f_market=1), dict(rows[0], f_market=0)
    a, applied = expected_pct(with_row, pb)
    b, _ = expected_pct(without_row, pb)
    assert a - b == pytest.approx(market["diff_pct"] * market["shrink"], abs=1e-3)
    assert any(x.startswith("with the Nifty's direction today: yes") for x in applied)


def test_the_market_moving_all_day_cannot_fake_a_lesson():
    """On strong days most setups are 'with the Nifty' and on weak days few are; the result depends on
    the day alone. A plain comparison sees a big difference, the same-day comparison sees none."""
    rng = random.Random(11)
    base, _ = make_rows(days=1, per_day=1)
    rows, flips = [], []
    for d in range(40):
        day = f"2026-08-{1 + d:02d}" if d < 28 else f"2026-09-{d - 27:02d}"
        strong = d % 2 == 0
        move = 1.5 if strong else -1.5
        for i in range(30):
            ret = move + rng.gauss(0, 0.5)
            rows.append(dict(base[0], day=day, ticker=f"S{i}.NS", ret=ret, ret_stop_hold=ret,
                             f_market=int(i < 27 if strong else i < 3)))
        flips += [{"day": day, "ret": move + rng.gauss(0, 0.5)} for _ in range(20)]
    with_market = [r["ret"] for r in rows if r["f_market"]]
    without = [r["ret"] for r in rows if not r["f_market"]]
    assert sum(with_market) / len(with_market) - sum(without) / len(without) > 1.0     # looks like a big edge
    pb = fit_playbook(rows, flips, cost_pct=COST)
    market = pb["conditions"]["breakout"]["market"]
    assert market["status"] != "in force" and abs(market["diff_pct"]) < 0.3 and not pb["lessons"]


def test_models_are_judged_against_a_coin_flip_on_the_same_days():
    rows, flips = make_rows(model_shift=0.5, seed=5)
    pb = fit_playbook(rows, flips, cost_pct=COST)
    m = pb["models"]["breakout"]
    assert m["status"] == "in force" and m["vs_coin_flip_pct"] == pytest.approx(0.5, abs=0.12) and m["adj_pct"] > 0
    assert pb["baseline_pct"] == pytest.approx(-COST, abs=0.1)
    value, applied = expected_pct(rows[0], pb)
    assert value == pytest.approx(pb["baseline_pct"] + m["adj_pct"], abs=0.2) and applied

    plain = fit_playbook(*make_rows(seed=5), cost_pct=COST)
    assert plain["models"]["breakout"]["adj_pct"] == 0 and plain["lessons"] == []


def test_the_exit_is_learned_only_with_strong_evidence():
    assert exit_for("breakout", fit_playbook(*make_rows(hold_gain=0.3), cost_pct=COST)) == "hold"
    assert exit_for("breakout", fit_playbook(*make_rows(hold_gain=0.0), cost_pct=COST)) == "target"
    assert exit_for("breakout", None) == "target"


def test_too_little_data_means_no_lessons():
    pb = fit_playbook(*make_rows(days=10, market_effect=0.6), cost_pct=COST)
    assert pb["conditions"]["breakout"]["market"]["status"] == "too little data" and pb["lessons"] == []
    assert expected_pct(make_rows()[0][0], None) == (None, [])


def test_walk_forward_shows_whether_learning_helps():
    rows, flips = make_rows(days=45, market_effect=0.8, seed=7)
    for r in rows:                       # the plain score prefers the setups WITHOUT the helpful condition
        r["score"] = 60 if r["f_market"] else 90
    check = walk_forward(rows, flips, cost_pct=COST)
    assert check["days"] == 25 and check["days_with_lessons"] > 0
    assert check["learned_minus_plain"]["avg_pct"] > 0.3

    no_effect = walk_forward(*make_rows(days=30, seed=8), cost_pct=COST)
    assert no_effect["days"] == 10 and no_effect["days_with_lessons"] == 0
    assert no_effect["learned_minus_plain"]["avg_pct"] == 0


def test_reports_and_files(tmp_path):
    rows, flips = make_rows(market_effect=0.6)
    playbook, report = learning.build(rows, flips, tmp_path, cost_pct=COST)
    assert learning.load(tmp_path) == playbook and (tmp_path / learning.REPORT_FILE).read_text() == report
    for text in ("Lessons in force", "Watching", "against a coin flip", "Conditions", "Exits", "Does learning help"):
        assert text in report
    assert "with the Nifty's direction today" in lessons_summary(playbook)
    assert "none yet" in lessons_summary(fit_playbook(*make_rows(), cost_pct=COST))
    assert "has not run" in lessons_summary(None) and learning.load(tmp_path / "missing") is None
    assert set(CONDITIONS) >= {"trend", "market", "vwap", "volume", "early", "big_gap", "stretched"}
    assert "short" not in CONDITIONS          # shorts vs longs only measures whether the market fell
