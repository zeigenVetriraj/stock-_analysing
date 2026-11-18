"""Layer 2c - The Picker: Claude compares a screened shortlist and picks at most one.

It sees each candidate's technicals (a compact market snapshot), fundamentals
and screening scores, and returns ONE long swing idea (BUY) or HOLD. As with
the other analysts it proposes an idea only: the ticker must be one of the
candidates, and Layer 3 still sizes the trade, sets the stop and target and
can refuse it.
"""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from config.settings import Settings, get_settings
from src.analysis.analyst_agent import (
    AnalystAgent,
    AnalystDecision,
    AnalystError,
    _extract_json_object,
    request_json_decision,
)
from src.automation.screener import RankedStock
from src.ingestion.data_fetcher import MarketSnapshot

logger = logging.getLogger(__name__)

Strategy = Literal["swing", "penny", "intraday"]

_COMMON_RULES = """\
Rules you MUST follow:
- Pick AT MOST ONE stock to BUY for a swing trade of days to a few weeks, or
  answer HOLD if no candidate has a clean setup. HOLD is a good answer.
- Long only: you cannot short. Do not pick a stock in a downtrend.
- Technicals time the entry: prefer an established uptrend (price above SMA50,
  SMA50 above SMA200) with RSI not overbought, or a pullback holding a
  Fibonacci level or a bullish fair value gap. Avoid buying straight into a
  bearish gap or just under a swing high.
- Fundamentals filter the pick: prefer profitable companies with manageable
  debt and growing revenue and operating profit. A single quarter's net profit
  can swing on one-off items; weigh operating profit and multi-year growth.
- If a company's results are due within about a week, prefer another candidate.
- Use ONLY the data given. Do NOT size the position or set stops: a separate
  hardcoded risk layer does that.
- target_price is your price objective for the move, above the current price.

Output format (CRITICAL):
- Respond with EXACTLY ONE valid JSON object and nothing else - no markdown, no
  commentary, no trailing commas.
  {
    "action": "BUY" | "HOLD",
    "ticker": "<one of the candidate tickers, or null for HOLD>",
    "target_price": <number, or null for HOLD>,
    "reasoning": "<two or three concise sentences: why this stock over the others, citing indicators and fundamentals>"
  }
"""

SWING_SYSTEM_PROMPT = (
    "You are a disciplined swing trader choosing today's single best long trade "
    "among pre-screened Indian large caps. You are not an investment adviser; you "
    "emit a machine-readable signal for a downstream automated risk system.\n\n"
    + _COMMON_RULES
)

PENNY_SYSTEM_PROMPT = (
    "You are a cautious trader choosing at most one low-priced (penny) Indian stock "
    "for a swing trade. Penny stocks are volatile, thinly traded and prone to "
    "manipulation, so you are sceptical by default. You are not an investment "
    "adviser; you emit a machine-readable signal for a downstream automated risk "
    "system.\n\n"
    "Extra penny-stock rules:\n"
    "- Strongly prefer companies that are profitable, have low debt and real "
    "revenue growth. Skip loss-making or highly indebted companies.\n"
    "- Be wary of pump patterns: a vertical price spike on huge volume, or RSI far "
    "above 75. Prefer steady uptrends with healthy delivery percentages.\n\n"
    + _COMMON_RULES
)

_INTRADAY_INTRO = """\
You are a disciplined intraday trader on the Indian cash market choosing at most \
ONE trade for today from pre-screened setups on 5-minute bars. Positions are \
closed the same day, so you may go long (BUY) or short (SELL). You are not an \
investment adviser; you emit a machine-readable signal for a downstream automated \
risk system.

Each candidate has a "model" (the kind of setup) and a "setup" side (long or short):
- "breakout": price is beyond the 09:15-09:45 opening range; trade with the break.
- "fvg": price has come back to an unfilled fair value gap made earlier today (a
  three-candle imbalance); trade in the gap's direction. "rests_on" is the far
  edge of the gap: if price trades through it the idea is wrong.
- "fib": price has given back 38.2-61.8% of today's swing; trade in the swing's
  direction. "rests_on" is the 78.6% retracement.
- "gap_fade": the stock with the day's biggest opening gap; trade AGAINST the gap
  (short a gap up, buy a gap down), expecting part of it to be given back.

Other fields: opening_range, vwap, ema9/ema20, rsi_5m, relative_volume (today's
volume against the same time on recent days), prev_day high/low/close, pivots
(classic floor pivots and the CPR), gap_pct, change_pct, market (the Nifty 50's
change today and the stock's strength against it), daily_trend, fair_value_gaps
(nearby unfilled gaps on 5m and 15m bars: bullish gaps below price act as
support, bearish gaps above as resistance), todays_swing with its Fibonacci
levels, and notes. "score" only counts how many things line up (daily trend, the
Nifty, VWAP, volume); it is NOT a prediction. "backtest" (when present) is how
that model did on recent months of data after costs: number of trades, share of
winners and the average % per trade. "experience" (when present) is what the
bot has learned from every setup it has seen: the expected net % per trade for a
setup like this one, a coin flip's result for comparison, and the lessons that
moved the estimate. Candidates are listed best expected first.

"""

_INTRADAY_RULES = """\
Rules you MUST follow:
- BUY only a "long" setup and SELL only a "short" one.
- Weigh the whole picture: is there room to the next obstacle (yesterday's
  high/low, a pivot, an opposing fair value gap) before the target, is the market
  helping or fighting the trade, is volume behind the move, is the stop resting on
  a real level?
- Be sceptical. In backtests on these large stocks none of these setups was
  reliably profitable on its own after costs: early breakouts often faded and
  big day-over-day moves tended to reverse. Give weight to each candidate's
  "backtest" record, and take a trade only when several independent things
  agree; a high score alone is not a reason.
- Answer HOLD if nothing is clean. HOLD is a good answer.
- Use ONLY the data given. Do NOT size the position or set stops: each setup
  already carries its stop_price and target_price, enforced by a separate
  hardcoded risk layer with a same-day square-off.
- target_price: repeat the candidate's target_price.

"""

# Practice mode (paper only): the same briefing, but Claude must trade every day unless something
# is clearly wrong, so the paper account gathers experience while no setup has a proven edge.
_INTRADAY_PRACTICE_RULES = """\
Rules you MUST follow:
- This is a PAPER PRACTICE account: no real money is at risk, and it exists to
  build experience about which setups work and which do not. So pick ONE setup
  to trade every day. Expected results that are negative or close to zero after
  costs are normal at this stage and are NOT a reason to hold: choose the
  candidate you judge most likely to work.
- BUY only a "long" setup and SELL only a "short" one.
- Choose by weighing the whole picture: is there room to the next obstacle
  (yesterday's high/low, a pivot, an opposing fair value gap) before the target,
  is the market helping or fighting the trade, is volume behind the move, is the
  stop resting on a real level? Compare the candidates' "backtest" and
  "experience" data too.
- Answer HOLD ONLY if something is clearly wrong with every candidate: the data
  look broken or stale, the price is already at or beyond its target or stop, or
  the stock is in an extreme move (for example near a circuit limit). If you
  hold, name the problem in the reasoning.
- In the reasoning, say why you chose this candidate over the others.
- Use ONLY the data given. Do NOT size the position or set stops: each setup
  already carries its stop_price and target_price, enforced by a separate
  hardcoded risk layer with a same-day square-off.
- target_price: repeat the candidate's target_price.

"""

_INTRADAY_OUTPUT = """\
Output format (CRITICAL):
- Respond with EXACTLY ONE valid JSON object and nothing else - no markdown, no
  commentary, no trailing commas.
  {
    "action": "BUY" | "SELL" | "HOLD",
    "ticker": "<one of the candidate tickers, or null for HOLD>",
    "target_price": <number, or null for HOLD>,
    "reasoning": "<two concise sentences citing the setup data>"
  }
- Inside "reasoning" never use double quotes; use single quotes to quote a phrase.
"""

USER_TEMPLATE = """\
Here are today's {count} screened candidates, best combined screening score first.
Choose at most one to BUY, per your instructions.

CANDIDATES:
{candidates_json}
"""

INTRADAY_SYSTEM_PROMPT = _INTRADAY_INTRO + _INTRADAY_RULES + _INTRADAY_OUTPUT
INTRADAY_PRACTICE_PROMPT = _INTRADAY_INTRO + _INTRADAY_PRACTICE_RULES + _INTRADAY_OUTPUT

INTRADAY_USER_TEMPLATE = """\
Here are {count} intraday setups found just now, highest score first.
Choose at most one to trade (BUY a long setup or SELL a short one), or HOLD.

CANDIDATES:
{candidates_json}
"""

INTRADAY_PRACTICE_USER_TEMPLATE = """\
Here are {count} intraday setups found just now, best expected result first.
This is the paper practice account: pick the one you judge best (BUY a long
setup or SELL a short one). HOLD only if something is clearly wrong with every
candidate.

CANDIDATES:
{candidates_json}
"""


class PickDecision(BaseModel):
    action: Literal["BUY", "SELL", "HOLD"]
    ticker: Optional[str] = None
    target_price: Optional[float] = Field(default=None, gt=0)
    reasoning: str

    @model_validator(mode="after")
    def _buy_needs_ticker_and_target(self) -> "PickDecision":
        if self.action != "HOLD" and (not self.ticker or self.target_price is None):
            raise ValueError(f"{self.action} needs a ticker and a target_price")
        return self

    def to_analyst_decision(self, price: float) -> AnalystDecision:
        """The shape the shares risk layer validates."""
        return AnalystDecision(
            action=self.action,
            ticker=self.ticker or "",
            target_price=self.target_price or price,
            reasoning=self.reasoning,
        )


def candidate_payload(stock: RankedStock, snapshot: Optional[MarketSnapshot]) -> Dict:
    """Compact per-candidate data for the prompt."""
    item: Dict = {
        "ticker": stock.ticker,
        "price": stock.price,
        "screening": {
            "combined": stock.combined,
            "technical": stock.technical,
            "fundamental": stock.fundamental,
            "notes": stock.notes,
        },
    }
    if snapshot is not None:
        snap = json.loads(snapshot.model_dump_json())
        snap["recent_4h_candles"] = snap.get("recent_4h_candles", [])[-6:]
        snap.pop("as_of", None)
        item["technicals"] = snap
    if stock.fundamentals is not None:
        item["fundamentals"] = json.loads(stock.fundamentals.model_dump_json(exclude_none=True))
    return item


def parse_pick(text: str, allowed: List[str], *, allow_sell: bool = False) -> PickDecision:
    decision = PickDecision.model_validate(_extract_json_object(text))
    if decision.action == "SELL" and not allow_sell:
        raise AnalystError("SELL is not allowed here: this job only buys.")
    if decision.action == "HOLD":
        return decision.model_copy(update={"ticker": None, "target_price": None})
    wanted = (decision.ticker or "").strip().upper()
    match = next((t for t in allowed if t.upper() == wanted or t.upper().split(".")[0] == wanted), None)
    if match is None:
        raise AnalystError(f"Picked ticker {decision.ticker!r} is not one of the candidates {allowed}.")
    return decision.model_copy(update={"ticker": match})


class PickerAgent:
    def __init__(self, settings: Optional[Settings] = None, client=None) -> None:
        self.settings = settings or get_settings()
        self._base = AnalystAgent(self.settings, client=client)

    def pick(
        self,
        strategy: Strategy,
        candidates: List[RankedStock],
        snapshots: Dict[str, MarketSnapshot],
    ) -> PickDecision:
        if not candidates:
            return PickDecision(action="HOLD", reasoning="No candidates passed the screen.")
        client = self._base._get_client()
        payload = [candidate_payload(c, snapshots.get(c.ticker)) for c in candidates]
        allowed = [c.ticker for c in candidates]
        logger.info("Asking Claude to pick from %d %s candidates.", len(candidates), strategy)
        decision = request_json_decision(
            client,
            model=self.settings.anthropic_model,
            system=PENNY_SYSTEM_PROMPT if strategy == "penny" else SWING_SYSTEM_PROMPT,
            user_content=USER_TEMPLATE.format(
                count=len(payload), candidates_json=json.dumps(payload, indent=1)
            ),
            parse=lambda text: parse_pick(text, allowed),
            label="Picker",
        )
        logger.info("Pick: %s %s target %s", decision.action, decision.ticker or "-", decision.target_price)
        return decision

    def pick_intraday(self, setups: List[Dict], *, practice: bool = False) -> PickDecision:
        """``setups``: IntradaySetup.as_prompt() dicts, best first.

        ``practice``: the paper practice account, which takes the best setup every day and holds only when
        something is clearly wrong with every candidate.
        """
        if not setups:
            return PickDecision(action="HOLD", reasoning="No intraday setups right now.")
        client = self._base._get_client()
        allowed = [s["ticker"] for s in setups]
        directions = {s["ticker"]: s.get("setup") for s in setups}

        def parse(text: str) -> PickDecision:
            decision = parse_pick(text, allowed, allow_sell=True)
            wanted = {"BUY": "long", "SELL": "short"}.get(decision.action)
            if wanted and directions.get(decision.ticker) != wanted:
                raise AnalystError(
                    f"{decision.action} {decision.ticker} goes against its {directions.get(decision.ticker)} setup."
                )
            return decision

        logger.info("Asking Claude to pick from %d intraday setups%s.", len(setups),
                    " (practice: trade the best one)" if practice else "")
        template = INTRADAY_PRACTICE_USER_TEMPLATE if practice else INTRADAY_USER_TEMPLATE
        decision = request_json_decision(
            client,
            model=self.settings.anthropic_model,
            system=INTRADAY_PRACTICE_PROMPT if practice else INTRADAY_SYSTEM_PROMPT,
            user_content=template.format(count=len(setups), candidates_json=json.dumps(setups, indent=1)),
            parse=parse,
            label="Intraday picker",
        )
        logger.info("Intraday pick: %s %s target %s", decision.action, decision.ticker or "-", decision.target_price)
        return decision


__all__ = [
    "INTRADAY_PRACTICE_PROMPT",
    "INTRADAY_SYSTEM_PROMPT",
    "PENNY_SYSTEM_PROMPT",
    "SWING_SYSTEM_PROMPT",
    "PickDecision",
    "PickerAgent",
    "candidate_payload",
    "parse_pick",
]
