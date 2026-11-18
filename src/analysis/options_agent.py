"""Layer 2b - The Options Analyst (buy-only).

Reads the stock's :class:`MarketSnapshot` (technicals, Fibonacci levels, fair
value gaps), its :class:`FundamentalSnapshot` (when available) and an
:class:`OptionChainSnapshot`, and asks Claude for ONE strictly-typed
:class:`OptionDecision`: buy a call, buy a put, or hold.

Like the equity analyst it proposes an idea only. It never sees the account,
cannot size the position or set stops, and cannot send orders - Layer 3
(``src/risk/options_guardrails.py``) re-checks every field against the option
chain and the immutable option rules before anything reaches the broker.
"""

from __future__ import annotations

import logging
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

from config.settings import Settings, get_settings
from src.analysis.analyst_agent import (
    AnalystAgent,
    _extract_json_object,
    fundamentals_json,
    request_json_decision,
)
from src.ingestion.data_fetcher import MarketSnapshot
from src.ingestion.fundamentals import FundamentalSnapshot
from src.ingestion.option_chain import OptionChainSnapshot

logger = logging.getLogger(__name__)

OptionAction = Literal["BUY_CALL", "BUY_PUT", "HOLD"]

OPTIONS_SYSTEM_PROMPT = """\
You are a disciplined technical swing trader who expresses views ONLY by BUYING \
options (calls or puts) on Indian stocks, and who uses the company's \
fundamentals as a filter. You are not an investment adviser and you do not give \
financial advice to humans - you emit a single machine-readable signal for a \
downstream automated risk system.

You receive:
- MARKET_SNAPSHOT: the stock's technicals (RSI, SMA50/SMA200, trend regime, price
  vs SMA50, Fibonacci retracements, fair value gaps, recent 4h candles).
- FUNDAMENTALS: the company's valuation, profitability, growth (the latest
  quarter against a year ago for revenue, operating profit and net profit, plus
  multi-year CAGR), debt, dividend yield, next results date and pre-computed
  "flags" (or a note that they are unavailable).
- OPTION_CHAIN: one expiry of the stock's option chain around the money. For each
  strike, "ce" is the call and "pe" the put: premium (last_price, bid, ask), open
  interest and volume in lots, implied volatility and Greeks. It also gives the
  lot size, days to expiry, the put/call open-interest ratio and the strikes with
  the most call and put open interest.

Rules you MUST follow:
- Decide direction from the stock's technicals like a swing trader: a bullish
  setup -> BUY_CALL; a bearish setup -> BUY_PUT; no clear edge -> HOLD.
  Fibonacci levels, fair value gaps and the strikes with the most open interest
  act as support/resistance: use them for the target, and avoid buying straight
  into one.
- Use FUNDAMENTALS as a filter: prefer calls on profitable, growing companies
  with manageable debt; prefer HOLD over a call on loss-making, shrinking or
  highly indebted ones unless the setup is exceptionally strong; weak and
  deteriorating fundamentals strengthen a put. A single quarter's net profit
  can swing on one-off items: when revenue and operating profit are growing, a
  fall in net profit is not weakness, and a one-off jump is not strength.
- Quarterly results: option premiums usually carry extra implied volatility
  before results and lose it right after. If the next results date falls before
  the option's expiry, say so in the reasoning and prefer HOLD unless the setup
  is strong.
- Pick exactly ONE strike that is listed in OPTION_CHAIN and has the chosen leg.
  Prefer |delta| between 0.35 and 0.60, meaningful open interest and a tight
  bid/ask spread. Avoid a strike whose implied volatility is far above its
  neighbours.
- Use the expiry exactly as given in OPTION_CHAIN.
- NEVER sell or write options. Do NOT size the position or set stops - a separate
  hardcoded risk layer does that. Do NOT invent data you were not given; missing
  fundamentals are unknown.
- underlying_target is your price objective for the STOCK (not the option
  premium). For HOLD, set strike to null and underlying_target to the current
  stock price.

Output format (CRITICAL):
- Respond with EXACTLY ONE valid JSON object and nothing else.
- No markdown, no code fences, no commentary before or after, no trailing commas.
- Keys, in this exact shape:
  {
    "action": "BUY_CALL" | "BUY_PUT" | "HOLD",
    "ticker": "<stock symbol>",
    "expiry": "<YYYY-MM-DD from OPTION_CHAIN>",
    "strike": <number or null>,
    "underlying_target": <number>,
    "reasoning": "<one or two concise sentences citing the indicators, levels, fundamentals and option data>"
  }
"""

OPTIONS_USER_TEMPLATE = """\
Analyse the stock, its fundamentals and its option chain below and return your \
signal as a single JSON object per your instructions.

MARKET_SNAPSHOT:
{snapshot_json}

FUNDAMENTALS:
{fundamentals_json}

OPTION_CHAIN:
{chain_json}
"""


class OptionDecision(BaseModel):
    """Validated options idea emitted by the options analyst."""

    action: OptionAction
    ticker: str
    expiry: str
    strike: Optional[float] = Field(default=None, gt=0)
    underlying_target: float = Field(gt=0)
    reasoning: str

    @model_validator(mode="after")
    def _strike_needed_to_trade(self) -> "OptionDecision":
        if self.action != "HOLD" and self.strike is None:
            raise ValueError("strike is required for BUY_CALL and BUY_PUT")
        return self

    @classmethod
    def from_model_text(cls, text: str, expected_ticker: str) -> "OptionDecision":
        """Parse and validate the raw model text into a decision."""
        decision = cls.model_validate(_extract_json_object(text))
        expected = expected_ticker.strip().upper()
        if decision.ticker.strip().upper() != expected:
            logger.warning(
                "Options analyst returned ticker %r; overriding with requested %r.",
                decision.ticker, expected,
            )
        return decision.model_copy(update={"ticker": expected})


class OptionsAnalystAgent:
    """Wraps the Anthropic client to turn a snapshot + option chain into an idea."""

    def __init__(self, settings: Optional[Settings] = None, client=None) -> None:
        self.settings = settings or get_settings()
        # Reuse the equity analyst's key checks and SDK wiring.
        self._base = AnalystAgent(self.settings, client=client)

    def analyze(
        self,
        snapshot: MarketSnapshot,
        chain: OptionChainSnapshot,
        fundamentals: Optional[FundamentalSnapshot] = None,
    ) -> OptionDecision:
        """Return a validated BUY_CALL / BUY_PUT / HOLD decision."""
        client = self._base._get_client()
        user_content = OPTIONS_USER_TEMPLATE.format(
            snapshot_json=snapshot.to_analyst_json(),
            fundamentals_json=fundamentals_json(fundamentals),
            chain_json=chain.to_analyst_json(),
        )

        logger.info(
            "Requesting options decision for %s, expiry %s (model=%s).",
            chain.underlying, chain.expiry, self.settings.anthropic_model,
        )
        decision = request_json_decision(
            client,
            model=self.settings.anthropic_model,
            system=OPTIONS_SYSTEM_PROMPT,
            user_content=user_content,
            parse=lambda text: OptionDecision.from_model_text(text, chain.underlying),
            label="Options analyst",
        )

        logger.info(
            "Options decision: %s %s strike %s expiry %s, stock target %.2f",
            decision.action, decision.ticker, decision.strike, decision.expiry,
            decision.underlying_target,
        )
        return decision


__all__ = [
    "OPTIONS_SYSTEM_PROMPT",
    "OptionAction",
    "OptionDecision",
    "OptionsAnalystAgent",
]
