"""What the intraday bot has learned from experience (paper trading only).

Experience = every setup the bot could have taken, replayed to the 15:15
square-off (the backtest over all saved 5-minute bars, which grows every week),
not just the one trade a day it actually takes: a handful of trades cannot tell
a good rule from luck, thousands of setups can start to.

:func:`fit_playbook` turns that experience into a playbook:

* each kind of setup (model) against a coin flip (random side, same widest stop
  and target) on the same days, after costs;
* for each model and each condition (with the daily trend, with the Nifty, on
  the trade's side of VWAP, busy volume, short, early in the day, a big opening
  gap, RSI already stretched, a tight stop, the model's bonus): setups WITH the
  condition against those WITHOUT it on the same days, so a market that rose or
  fell all day cannot fake a lesson;
* for each model: whether holding to 15:15 (keeping the stop) beat the target.

A difference becomes a lesson "in force" only when it is at least 3 times its
own uncertainty (t >= 3) over at least 20 days with at least 30 setups on each
side; between 2 and 3 it is only "watching". Lessons in force are shrunk towards
zero before use (by 1 - (2/t)^2), so a borderline lesson moves the ranking only a
little. Nothing here changes position size, the widest stop, or paper/live mode.

:func:`walk_forward` checks whether the playbook actually helps: for each day it
learns only from the days before, takes that day's top setup, and compares the
result with the plain ranking (score) and with the average setup.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from src.automation.intraday import MODEL_LABELS, MODELS

T_ADOPT = 3.0          # a difference this many times its uncertainty becomes a lesson in force
T_WATCH = 2.0          # ... and this many times is worth watching
MIN_DAYS = 20          # days with setups on both sides before anything is concluded
MIN_SIDE = 30          # setups needed on each side of a condition
HOLD_TARGET_PCT = 4.0  # the target used when the playbook says "hold to 15:15" (out of reach intraday)
PLAYBOOK_FILE, REPORT_FILE = "playbook.json", "learning_report.md"


def _stretched(row: dict) -> bool:
    rsi = row.get("rsi")
    if rsi is None or rsi == "":
        return False
    rsi = float(rsi)
    return rsi >= 70 if row.get("direction") == "long" else rsi <= 30


def _flag(name: str) -> Callable[[dict], bool]:
    return lambda row: bool(int(row.get(name) or 0))


# name -> (plain-language label, test on a setup row). "Short trades" is deliberately NOT a condition:
# over a few months shorts beat longs simply when the market fell, which says nothing about the next
# day. Direction is learned properly by "market" (trading with the Nifty's move that day).
CONDITIONS: Dict[str, Tuple[str, Callable[[dict], bool]]] = {
    "trend": ("with the daily trend", _flag("f_trend")),
    "market": ("with the Nifty's direction today", _flag("f_market")),
    "vwap": ("on the trade's side of VWAP", _flag("f_vwap")),
    "volume": ("on above-normal volume", _flag("f_volume")),
    "early": ("found before 10:30", lambda r: str(r.get("scan", "")) < "10:30"),
    "big_gap": ("the stock opened 1% or more away from yesterday's close",
                lambda r: abs(float(r.get("gap_pct") or 0)) >= 1.0),
    "stretched": ("RSI already stretched (70+ for a buy, 30 or less for a sell)", _stretched),
    "tight_stop": ("a tight stop (0.5% or less)", lambda r: float(r.get("stop_pct") or 0) <= 0.5),
    "bonus": ("the setup's quality bonus", _flag("f_bonus")),
}


# --------------------------------------------------------------------------- #
# Small statistics helpers (uncertainty is always measured across days)
# --------------------------------------------------------------------------- #
def _day_means(rows: Iterable[dict], value: Callable[[dict], float]) -> Dict[str, float]:
    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for r in rows:
        sums[r["day"]] = sums.get(r["day"], 0.0) + value(r)
        counts[r["day"]] = counts.get(r["day"], 0) + 1
    return {d: sums[d] / counts[d] for d in sums}


def _mean_se(values: Sequence[float]) -> Tuple[float, Optional[float], Optional[float]]:
    """(mean, standard error, t) of a list; se and t are None with fewer than 3 values or no spread."""
    n = len(values)
    if n == 0:
        return 0.0, None, None
    mean = sum(values) / n
    if n < 3:
        return mean, None, None
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    if variance <= 0:
        return mean, None, None
    se = math.sqrt(variance / n)
    return mean, se, mean / se


def _paired(a: Dict[str, float], b: Dict[str, float]) -> Tuple[float, Optional[float], Optional[float], int]:
    """Day-by-day difference a - b on the days both have: (mean, se, t, days)."""
    days = sorted(set(a) & set(b))
    mean, se, t = _mean_se([a[d] - b[d] for d in days])
    return mean, se, t, len(days)


def _status(t: Optional[float], days: int) -> str:
    if t is None or days < MIN_DAYS:
        return "too little data"
    if abs(t) >= T_ADOPT:
        return "in force"
    if abs(t) >= T_WATCH:
        return "watching"
    return "no evidence"


def _shrink(t: Optional[float]) -> float:
    """How much of a lesson to use: 0 unless in force, then 1 - (2/t)^2 (t=3: 0.56, t=4: 0.75)."""
    if t is None or abs(t) < T_ADOPT:
        return 0.0
    return max(0.0, 1.0 - (T_WATCH / t) ** 2)


def _r(value: Optional[float], digits: int = 3) -> Optional[float]:
    return None if value is None else round(float(value), digits)


# --------------------------------------------------------------------------- #
# The playbook
# --------------------------------------------------------------------------- #
def fit_playbook(rows: Sequence[dict], coin_flips: Sequence[dict], *, cost_pct: float) -> dict:
    """Learn from setup rows (the backtest's, first appearance per stock and day) and coin-flip rows."""
    firsts = [r for r in rows if int(r.get("first", 1))]
    days = sorted({r["day"] for r in firsts})
    flip_days = _day_means(coin_flips, lambda r: float(r["ret"]) - cost_pct)
    flip_all = [float(r["ret"]) - cost_pct for r in coin_flips]
    baseline = sum(flip_all) / len(flip_all) if flip_all else -cost_pct
    playbook: dict = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "period": [days[0], days[-1]] if days else [],
        "days": len(days), "setups": len(firsts), "cost_pct": cost_pct, "baseline_pct": round(baseline, 3),
        "thresholds": {"adopt_t": T_ADOPT, "watch_t": T_WATCH, "min_days": MIN_DAYS, "min_side": MIN_SIDE},
        "models": {}, "conditions": {}, "exits": {}, "lessons": [], "watching": [],
    }
    for model in MODELS:
        mine = [r for r in firsts if r["model"] == model]
        if not mine:
            continue
        label = MODEL_LABELS[model]
        net = [float(r["ret"]) - cost_pct for r in mine]
        mine_days = _day_means(mine, lambda r: float(r["ret"]) - cost_pct)
        diff, se, t, n_days = _paired(mine_days, flip_days)
        status = _status(t, n_days)
        playbook["models"][model] = {
            "label": label, "trades": len(mine), "avg_pct": _r(sum(net) / len(net)),
            "vs_coin_flip_pct": _r(diff), "se": _r(se), "t": _r(t, 2), "days": n_days,
            "status": status, "adj_pct": _r(diff * _shrink(t)),
        }
        _note(playbook, status, f"{label}s against a coin flip on the same days: {diff:+.2f}% a trade "
                                f"(±{se or 0:.2f}, {n_days} days)")

        conditions = {}
        for name, (cond_label, test) in CONDITIONS.items():
            with_rows = [r for r in mine if test(r)]
            without = [r for r in mine if not test(r)]
            if len(with_rows) < MIN_SIDE or len(without) < MIN_SIDE:
                continue
            d, s_e, tt, k = _paired(_day_means(with_rows, lambda r: float(r["ret"])),
                                    _day_means(without, lambda r: float(r["ret"])))
            st = _status(tt, k)
            conditions[name] = {
                "label": cond_label, "with": len(with_rows), "without": len(without),
                "share_with": _r(len(with_rows) / len(mine)), "diff_pct": _r(d), "se": _r(s_e), "t": _r(tt, 2),
                "days": k, "status": st, "shrink": _r(_shrink(tt)),
            }
            _note(playbook, st, f"{label}s, {cond_label}: {'better' if d > 0 else 'worse'} by {abs(d):.2f}% a trade "
                                f"than the rest (±{s_e or 0:.2f}; {len(with_rows)} vs {len(without)} setups over "
                                f"{k} days)")
        playbook["conditions"][model] = conditions

        if all("ret_stop_hold" in r and r["ret_stop_hold"] not in (None, "") for r in mine):
            d, s_e, tt, k = _paired(_day_means(mine, lambda r: float(r["ret_stop_hold"]) - float(r["ret"])),
                                    {day: 0.0 for day in mine_days})
            st = _status(tt, k)
            use = "hold" if st == "in force" and d > 0 else "target"
            playbook["exits"][model] = {"hold_minus_target_pct": _r(d), "se": _r(s_e), "t": _r(tt, 2),
                                        "days": k, "status": st, "use": use}
            _note(playbook, st, f"{label}s: holding to 15:15 (keeping the stop) instead of the target: "
                                f"{d:+.2f}% a trade (±{s_e or 0:.2f}, {k} days)")
    return playbook


def _note(playbook: dict, status: str, text: str) -> None:
    if status == "in force":
        playbook["lessons"].append(text)
    elif status == "watching":
        playbook["watching"].append(text)


def expected_pct(row: dict, playbook: Optional[dict]) -> Tuple[Optional[float], List[str]]:
    """Learned net % a trade for a setup row, and the lessons that moved it (None without a playbook)."""
    if not playbook or "baseline_pct" not in playbook:
        return None, []
    model = row.get("model")
    applied: List[str] = []
    value = float(playbook["baseline_pct"])
    info = playbook.get("models", {}).get(model, {})
    if info.get("adj_pct"):
        value += float(info["adj_pct"])
        applied.append(f"{info.get('label', model)}s vs a coin flip ({float(info['adj_pct']):+.2f}%)")
    for name, cond in playbook.get("conditions", {}).get(model, {}).items():
        if cond.get("status") != "in force" or name not in CONDITIONS:
            continue
        present = CONDITIONS[name][1](row)
        share = float(cond["share_with"])
        delta = ((1 - share) if present else -share) * float(cond["diff_pct"]) * float(cond["shrink"])
        value += delta
        applied.append(f"{cond['label']}: {'yes' if present else 'no'} ({delta:+.2f}%)")
    return round(value, 3), applied


def exit_for(model: str, playbook: Optional[dict]) -> str:
    """"hold" if the playbook learned that holding to 15:15 beats the target for this model, else "target"."""
    if not playbook:
        return "target"
    return playbook.get("exits", {}).get(model, {}).get("use", "target")


# --------------------------------------------------------------------------- #
# Does learning help? (walk-forward)
# --------------------------------------------------------------------------- #
def walk_forward(rows: Sequence[dict], coin_flips: Sequence[dict], *, cost_pct: float,
                 min_train_days: int = MIN_DAYS) -> dict:
    """Day by day, learn only from earlier days and compare the playbook's pick with the plain one."""
    firsts = [r for r in rows if int(r.get("first", 1))]
    days = sorted({r["day"] for r in firsts})
    per_day = []
    for i, day in enumerate(days):
        if i < min_train_days:
            continue
        playbook = fit_playbook([r for r in firsts if r["day"] < day], [r for r in coin_flips if r["day"] < day],
                                cost_pct=cost_pct)
        today = [r for r in firsts if r["day"] == day]
        first_scan = min(r["scan"] for r in today)
        candidates = [r for r in today if r["scan"] == first_scan]

        def learned_key(r):
            value, _ = expected_pct(r, playbook)
            return (value if value is not None else -1e9, float(r["score"]), float(r.get("relative_volume") or 0))

        learned = max(candidates, key=learned_key)
        plain = max(candidates, key=lambda r: (float(r["score"]), float(r.get("relative_volume") or 0)))
        per_day.append({
            "day": day, "learned": float(learned["ret"]) - cost_pct, "plain": float(plain["ret"]) - cost_pct,
            "average": sum(float(r["ret"]) for r in candidates) / len(candidates) - cost_pct,
            "lessons": len(playbook["lessons"]),
        })
    out = {"days": len(per_day), "per_day": per_day}
    if per_day:
        for key in ("learned", "plain", "average"):
            mean, se, _ = _mean_se([d[key] for d in per_day])
            out[key] = {"avg_pct": _r(mean), "se": _r(se)}
        diff, se, t = _mean_se([d["learned"] - d["plain"] for d in per_day])
        out["learned_minus_plain"] = {"avg_pct": _r(diff), "se": _r(se), "t": _r(t, 2)}
        out["days_with_lessons"] = sum(1 for d in per_day if d["lessons"])
    return out


# --------------------------------------------------------------------------- #
# Reports and files
# --------------------------------------------------------------------------- #
def _pm(se: Optional[float]) -> str:
    return "-" if se is None else f"±{se:.2f}"


def learning_report(playbook: dict, walk: Optional[dict] = None) -> str:
    if not playbook.get("days"):
        return "# What the bot has learned\n\nNo experience yet.\n"
    first, last = playbook["period"]
    out = [
        "# What the bot has learned",
        "",
        f"- Experience: {playbook['setups']} setups over {playbook['days']} trading days ({first} to {last}), "
        f"after {playbook['cost_pct']:.2f}% costs. A coin flip (random side, same stops) made "
        f"{playbook['baseline_pct']:+.2f}% a trade.",
        f"- A difference is used only when it is at least {T_ADOPT:g} times its own uncertainty (t >= {T_ADOPT:g}) "
        f"over at least {MIN_DAYS} days with {MIN_SIDE}+ setups on each side; between {T_WATCH:g} and {T_ADOPT:g} "
        "it is only watched. Conditions are compared on the same days, so a rising or falling market cannot fake a "
        "lesson.",
        "",
        "## Lessons in force (used when ranking setups)",
        "",
    ]
    out += [f"- {x}" for x in playbook["lessons"]] or ["- None yet: no difference is big enough to rule out luck."]
    out += ["", "## Watching (not used yet)", ""]
    out += [f"- {x}" for x in playbook["watching"]] or ["- Nothing."]
    out += ["", "## Each kind of setup against a coin flip (same days)", "",
            "| setup | setups | avg per trade | vs coin flip | ± | reading |", "|---|---|---|---|---|---|"]
    for m in playbook["models"].values():
        out.append(f"| {m['label']} | {m['trades']} | {m['avg_pct']:+.3f}% | {m['vs_coin_flip_pct']:+.3f}% | "
                   f"{_pm(m['se'])} | {m['status']} |")
    out += ["", "## Conditions: setups with the condition minus those without (same days)", "",
            "| setup | condition | with | without | difference | ± | days | reading |",
            "|---|---|---|---|---|---|---|---|"]
    for model, conditions in playbook["conditions"].items():
        for c in conditions.values():
            out.append(f"| {MODEL_LABELS[model]} | {c['label']} | {c['with']} | {c['without']} | {c['diff_pct']:+.3f}% | "
                       f"{_pm(c['se'])} | {c['days']} | {c['status']} |")
    if playbook["exits"]:
        out += ["", "## Exits: holding to 15:15 (keeping the stop) minus taking the target", "",
                "| setup | difference | ± | days | reading | exit used |", "|---|---|---|---|---|---|"]
        for model, e in playbook["exits"].items():
            out.append(f"| {MODEL_LABELS[model]} | {e['hold_minus_target_pct']:+.3f}% | {_pm(e['se'])} | {e['days']} | "
                       f"{e['status']} | {'hold to 15:15' if e['use'] == 'hold' else 'target'} |")
    if walk and walk.get("days"):
        lm = walk["learned_minus_plain"]
        out += ["", "## Does learning help? (each day learns only from the days before it)", "",
                "| one trade a day | days | avg per trade | ± |", "|---|---|---|---|",
                f"| top setup by the playbook | {walk['days']} | {walk['learned']['avg_pct']:+.3f}% | "
                f"{_pm(walk['learned']['se'])} |",
                f"| top setup by the plain score | {walk['days']} | {walk['plain']['avg_pct']:+.3f}% | "
                f"{_pm(walk['plain']['se'])} |",
                f"| the average setup | {walk['days']} | {walk['average']['avg_pct']:+.3f}% | "
                f"{_pm(walk['average']['se'])} |",
                "",
                f"Playbook minus plain score: {lm['avg_pct']:+.3f}% a day ({_pm(lm['se'])}); the playbook had at least "
                f"one lesson in force on {walk['days_with_lessons']} of {walk['days']} days."]
    out.append("")
    return "\n".join(out)


def lessons_summary(playbook: Optional[dict], limit: int = 3) -> str:
    """One or a few lines for Telegram."""
    if not playbook or not playbook.get("days"):
        return "Lessons: none yet (the weekly review has not run)."
    lessons = playbook.get("lessons", [])
    if not lessons:
        watching = len(playbook.get("watching", []))
        return (f"Lessons in force: none yet - no difference is big enough to rule out luck "
                f"({playbook['setups']} setups, {playbook['days']} days; {watching} being watched).")
    more = f" (+{len(lessons) - limit} more)" if len(lessons) > limit else ""
    return "Lessons in force:\n" + "\n".join(f"- {x}" for x in lessons[:limit]) + more


def save(playbook: dict, report: str, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / PLAYBOOK_FILE).write_text(json.dumps(playbook, indent=1) + "\n", encoding="utf-8")
    (folder / REPORT_FILE).write_text(report, encoding="utf-8")


def load(folder: Path) -> Optional[dict]:
    try:
        return json.loads((folder / PLAYBOOK_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def build(rows: Sequence[dict], coin_flips: Sequence[dict], folder: Path, *, cost_pct: float,
          walk: bool = True) -> Tuple[dict, str]:
    """Fit the playbook, run the walk-forward check, and save both files in ``folder``."""
    playbook = fit_playbook(rows, coin_flips, cost_pct=cost_pct)
    check = walk_forward(rows, coin_flips, cost_pct=cost_pct) if walk else None
    if check and check.get("days"):
        playbook["walk_forward"] = {k: v for k, v in check.items() if k != "per_day"}
    report = learning_report(playbook, check)
    save(playbook, report, folder)
    return playbook, report


__all__ = [
    "CONDITIONS",
    "HOLD_TARGET_PCT",
    "build",
    "exit_for",
    "expected_pct",
    "fit_playbook",
    "learning_report",
    "lessons_summary",
    "load",
    "save",
    "walk_forward",
]
