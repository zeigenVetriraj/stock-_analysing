"""Bull vs bear, then a judge: a structured debate about ONE intraday trade before it is taken.

For each setup debated (Claude's pick first, then the next ones on the shortlist):

1. Both sides get the same **data packet**: the setup (entry, stop, target, levels, indicators, volume,
   trend, the Nifty), its kind's backtest record and the bot's learned experience, and the last hour
   of 5-minute candles of the stock and of the Nifty.
2. **Bull** and **bear** argue in turns, ``rounds`` times each; from round 2 each must answer the
   other side's points.
3. Every point must quote one field of the packet and the value found there. A **referee** (plain
   code, :func:`verify_points`) checks each one; a point whose field does not exist or whose value
   does not match is marked unverified, and the judge is told to ignore it.
4. The **judge** reads the verified points and decides TRADE or SKIP for that exact trade (entry,
   stop and target are not up for debate; the risk layer still applies).

PAPER ONLY. The debate decides whether to take a setup; it never sizes or places anything itself.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from config.settings import Settings
from src.analysis.analyst_agent import (
    AnalystAgent,
    AnalystError,
    _extract_json_object,
    request_json_decision,
)

logger = logging.getLogger(__name__)

MAX_POINTS = 4
RECENT_BARS = 12            # the last hour of 5-minute candles
NIFTY_BARS = 6

_RULES = """\
Rules you MUST follow:
- Use ONLY the DATA packet. Every point quotes exactly one field from DATA by its path and the
  value found there, for example {{"field": "rsi_5m", "value": 61.3}} or
  {{"field": "recent_bars[11].close", "value": 1250.4}}. Paths use dots for nested keys and [n]
  for list items, exactly as in DATA.
- A referee checks every point against DATA before the judge sees it. A point whose field is not
  in DATA, or whose value does not match, is thrown out. Never invent news, results, order flow,
  sector views or anything else that is not in DATA.
- From round 2, answer the other side's strongest points directly: show with data why they are
  wrong, or concede them and explain why your conclusion still holds.
- Be precise and intellectually honest. The trade's direction, entry, stop and target are fixed:
  argue whether it will work, not how to change it.

Output format (CRITICAL): EXACTLY ONE valid JSON object and nothing else - no markdown. Never use
double quotes inside text values (use single quotes).
{{
  "points": [{{"claim": "<one sentence>", "field": "<path in DATA>", "value": <the value in DATA>}}],
  "summary": "<two sentences: your case>"
}}
Give 2 to {max_points} points.
"""

BULL_SYSTEM = (
    "You are the BULL in a structured debate about ONE intraday paper trade on the Indian market. "
    "Make the strongest honest case that this trade, exactly as specified, will reach its target "
    "before its stop-loss today.\n\n" + _RULES.format(max_points=MAX_POINTS)
)
BEAR_SYSTEM = (
    "You are the BEAR in a structured debate about ONE intraday paper trade on the Indian market. "
    "Make the strongest honest case that this trade, exactly as specified, will hit its stop-loss "
    "first or end the day at 15:15 without reaching its target.\n\n" + _RULES.format(max_points=MAX_POINTS)
)
JUDGE_SYSTEM = """\
You are the JUDGE of a structured debate about ONE intraday paper trade on the Indian market. A bull
argued that the trade will reach its target before its stop-loss; a bear argued that it will not.

Decide TRADE or SKIP for this exact trade (its direction, entry, stop and target are fixed).

Rules you MUST follow:
- Weigh ONLY the points marked VERIFIED. Points the referee threw out did not match the data:
  ignore them, and treat a side that made several as less careful.
- You may check any point against the DATA packet yourself.
- TRADE when the bull's verified evidence is clearly stronger: the setup is better than a typical
  one, with room to its target and the market not against it. SKIP when the bear shows a concrete,
  data-backed reason it is likely to fail (for example a level in the way of the target, a move
  already stretched, fading volume, the market moving against it). If the case is even, SKIP.
- Be decisive and brief.

Output format (CRITICAL): EXACTLY ONE valid JSON object and nothing else - no markdown. Never use
double quotes inside text values (use single quotes).
{
  "decision": "TRADE" | "SKIP",
  "confidence": <0-100>,
  "decisive_points": ["<the one to three points that decided it>"],
  "reasoning": "<two or three sentences>"
}
"""


# --------------------------------------------------------------------------- #
# The data packet and the referee
# --------------------------------------------------------------------------- #
def recent_bars(frame, day, tz, count: int) -> List[dict]:
    """The last ``count`` 5-minute candles of ``day`` (time, open, high, low, close, volume)."""
    if frame is None or getattr(frame, "empty", True):
        return []
    index = frame.index.tz_convert(tz) if frame.index.tz is not None else frame.index.tz_localize(tz)
    today = frame[index.date == day]
    out = []
    for when, row in today.tail(count).iterrows():
        when = when.tz_convert(tz) if when.tzinfo else when
        out.append({
            "time": when.strftime("%H:%M"), "open": round(float(row["Open"]), 2), "high": round(float(row["High"]), 2),
            "low": round(float(row["Low"]), 2), "close": round(float(row["Close"]), 2),
            "volume": int(row["Volume"]) if "Volume" in row and row["Volume"] == row["Volume"] else 0,
        })
    return out


def data_packet(candidate: dict, setup, stock_bars, index_bars, day, tz) -> dict:
    """Everything both sides may cite: the candidate as Claude saw it, plus recent candles."""
    packet = dict(candidate)
    f = setup.features
    if f is not None:
        packet["day"] = {"open": round(f.today_open, 2), "high": round(f.day_high, 2), "low": round(f.day_low, 2),
                         "bars_today": f.bars_today}
    packet["recent_bars"] = recent_bars(stock_bars, day, tz, RECENT_BARS)
    packet["nifty_recent_bars"] = [
        {"time": b["time"], "close": b["close"]} for b in recent_bars(index_bars, day, tz, NIFTY_BARS)
    ]
    return json.loads(json.dumps(packet, default=str))          # plain JSON values only


def flatten(data: Any, prefix: str = "") -> Dict[str, Any]:
    """Every leaf of ``data`` by its path (``a.b``, ``a[0].c``)."""
    out: Dict[str, Any] = {}
    if isinstance(data, dict):
        for key, value in data.items():
            out.update(flatten(value, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(data, list):
        for i, value in enumerate(data):
            out.update(flatten(value, f"{prefix}[{i}]"))
        if not data:
            out[prefix] = data
    else:
        out[prefix] = data
    return out


def _normalise(path: str) -> str:
    path = str(path).strip().strip("`'\" ")
    path = re.sub(r"^(DATA|data)\.", "", path)
    return re.sub(r"\.(\d+)(?=\.|$)", r"[\1]", path)


def _matches(actual: Any, cited: Any) -> bool:
    if isinstance(actual, bool) or isinstance(cited, bool) or actual is None or cited is None:
        return str(actual).strip().lower() == str(cited).strip().lower()
    try:
        a, c = float(actual), float(str(cited).replace(",", "").rstrip("%"))
        return abs(a - c) <= max(abs(a) * 0.005, 0.011)
    except (TypeError, ValueError):
        text = str(cited).strip().lower()
        return bool(text) and text in str(actual).strip().lower()


def verify_points(points: List[dict], flat: Dict[str, Any]) -> List[dict]:
    """The referee: mark each point verified only if its field exists and its value matches."""
    checked = []
    for p in points[:MAX_POINTS]:
        path = _normalise(p.get("field", ""))
        claim = str(p.get("claim", "")).strip()[:300]
        if path not in flat:
            why = "field not in the data"
        elif not _matches(flat[path], p.get("value")):
            why = f"value does not match (data: {flat[path]})"
        else:
            why = ""
        checked.append({"claim": claim, "field": path, "value": p.get("value"), "verified": not why,
                        "referee": why or "verified"})
    return checked


# --------------------------------------------------------------------------- #
# The debate
# --------------------------------------------------------------------------- #
@dataclass
class DebateResult:
    ticker: str
    direction: str
    turns: List[dict] = field(default_factory=list)   # {"side", "round", "points", "summary"}
    decision: str = "SKIP"
    confidence: int = 0
    reasoning: str = ""
    decisive_points: List[str] = field(default_factory=list)

    def tally(self, side: str) -> tuple:
        points = [p for t in self.turns if t["side"] == side for p in t["points"]]
        return sum(1 for p in points if p["verified"]), len(points)

    def best(self, side: str) -> str:
        """The side's first verified point of its last turn (else its summary)."""
        for turn in reversed(self.turns):
            if turn["side"] == side:
                verified = [p["claim"] for p in turn["points"] if p["verified"]]
                return verified[0] if verified else turn["summary"]
        return ""

    def lines(self, name: str) -> List[str]:
        """A short summary for the Telegram message."""
        out = [f"{self.direction.upper()} {name}:"]
        for side, mark in (("bull", "🐂 Bull"), ("bear", "🐻 Bear")):
            ok, total = self.tally(side)
            out.append(f"{mark} ({ok}/{total} points verified): {self.best(side)}")
        out.append(f"⚖️ Judge: {self.decision} ({self.confidence}%) - {self.reasoning}")
        return out

    def record(self) -> dict:
        return {"ticker": self.ticker, "direction": self.direction, "decision": self.decision,
                "confidence": self.confidence, "reasoning": self.reasoning,
                "decisive_points": self.decisive_points, "turns": self.turns}


def _parse_turn(text: str) -> dict:
    data = _extract_json_object(text)
    points = data.get("points")
    if not isinstance(points, list) or not points:
        raise AnalystError("the reply needs a non-empty 'points' list")
    for p in points:
        if not isinstance(p, dict) or not p.get("field") or "value" not in p or not p.get("claim"):
            raise AnalystError("every point needs 'claim', 'field' and 'value'")
    return {"points": points, "summary": str(data.get("summary", "")).strip()[:400]}


def _parse_verdict(text: str) -> dict:
    data = _extract_json_object(text)
    decision = str(data.get("decision", "")).strip().upper()
    if decision not in ("TRADE", "SKIP"):
        raise AnalystError("'decision' must be TRADE or SKIP")
    try:
        confidence = max(0, min(100, round(float(data.get("confidence", 50)))))
    except (TypeError, ValueError):
        confidence = 50
    decisive = data.get("decisive_points") or []
    return {"decision": decision, "confidence": confidence,
            "decisive_points": [str(x)[:300] for x in decisive][:3] if isinstance(decisive, list) else [],
            "reasoning": str(data.get("reasoning", "")).strip()[:600]}


def _trade_line(packet: dict) -> str:
    side = "BUY (long)" if packet.get("setup") == "long" else "SELL first (short)"
    return (f"{side} {packet.get('ticker')} - {packet.get('model')} setup: entry {packet.get('price')}, "
            f"stop-loss {packet.get('stop_price')} ({packet.get('stop_pct')}%), target {packet.get('target_price')} "
            f"({packet.get('target_pct')}%), squared off at 15:15 if neither is hit.")


def _transcript(turns: List[dict], *, for_judge: bool) -> str:
    if not turns:
        return "(nothing yet: you open the debate)"
    lines = []
    for t in turns:
        lines.append(f"Round {t['round']} - {t['side'].upper()}:")
        for p in t["points"]:
            if for_judge and not p["verified"]:
                continue
            mark = "VERIFIED" if p["verified"] else f"THROWN OUT ({p['referee']})"
            lines.append(f"  - [{mark}] {p['claim']} ({p['field']} = {json.dumps(p['value'])})")
        thrown = sum(1 for p in t["points"] if not p["verified"])
        if for_judge and thrown:
            lines.append(f"  ({thrown} point(s) thrown out by the referee)")
        lines.append(f"  Summary (not evidence): {t['summary']}")
    return "\n".join(lines)


class DebateAgent:
    """Runs bull, bear and judge with Claude (each call is one plain Messages API request)."""

    def __init__(self, settings: Settings, client=None) -> None:
        self.settings = settings
        self._base = AnalystAgent(settings, client=client)

    def _ask(self, system: str, user: str, parse, label: str):
        return request_json_decision(self._base._get_client(), model=self.settings.anthropic_model,
                                     system=system, user_content=user, parse=parse, label=label)

    def debate(self, packet: dict, rounds: int = 2) -> DebateResult:
        flat = flatten(packet)
        data = json.dumps(packet, indent=1)
        result = DebateResult(ticker=str(packet.get("ticker")), direction=str(packet.get("setup")))
        for round_no in range(1, rounds + 1):
            for side, system in (("bull", BULL_SYSTEM), ("bear", BEAR_SYSTEM)):
                user = (f"DATA:\n{data}\n\nTRADE UNDER DEBATE: {_trade_line(packet)}\n\n"
                        f"DEBATE SO FAR:\n{_transcript(result.turns, for_judge=False)}\n\n"
                        f"Your turn: {side.upper()}, round {round_no} of {rounds}.")
                reply = self._ask(system, user, _parse_turn, f"Debate {side}")
                result.turns.append({"side": side, "round": round_no,
                                     "points": verify_points(reply["points"], flat), "summary": reply["summary"]})
        user = (f"DATA:\n{data}\n\nTRADE UNDER DEBATE: {_trade_line(packet)}\n\n"
                f"THE DEBATE (verified points only):\n{_transcript(result.turns, for_judge=True)}\n\nYour verdict.")
        verdict = self._ask(JUDGE_SYSTEM, user, _parse_verdict, "Debate judge")
        result.decision, result.confidence = verdict["decision"], verdict["confidence"]
        result.reasoning, result.decisive_points = verdict["reasoning"], verdict["decisive_points"]
        logger.info("Debate on %s %s: %s (%d%%).", result.direction, result.ticker, result.decision, result.confidence)
        return result


def judge_summary(entries: List[dict]) -> Optional[dict]:
    """How the judge's TRADE and SKIP calls would have ended, from journal reviews' ``judge`` rows."""
    rows = [r for r in entries if r.get("ret") is not None]
    if not rows:
        return None
    out = {}
    for decision in ("TRADE", "SKIP"):
        rets = [r["ret"] for r in rows if r["decision"] == decision]
        out[decision] = {"n": len(rets), "won": sum(1 for x in rets if x > 0),
                         "avg_pct": round(sum(rets) / len(rets), 3) if rets else None}
    return out


__all__ = [
    "BEAR_SYSTEM",
    "BULL_SYSTEM",
    "JUDGE_SYSTEM",
    "DebateAgent",
    "DebateResult",
    "data_packet",
    "flatten",
    "judge_summary",
    "recent_bars",
    "verify_points",
]
