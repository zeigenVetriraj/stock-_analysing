"""Layer 2 - The Analyst Agent.

A thin, well-guarded interface over the Anthropic (Claude) API. The agent reads
the Layer 1 :class:`MarketSnapshot` and returns a strictly-typed
:class:`AnalystDecision`.

Design notes
------------
* The system prompt forces Claude to behave as a disciplined *technical swing
  trader* and to answer with a single JSON object - no markdown, no prose.
* The response is defensively parsed (markdown fences, trailing commas and
  stray Unicode spaces are forgiven) and validated with pydantic; an unusable
  reply gets one corrective retry and then fails loudly instead of leaking
  garbage into the risk layer.
* This layer proposes a trade *idea only*. It has NO authority over position
  size, stops, or whether the order is actually sent - that is Layer 3's job.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Literal, Optional

from pydantic import BaseModel, Field, ValidationError

from config.settings import Settings, get_settings
from src.ingestion.data_fetcher import MarketSnapshot
from src.ingestion.fundamentals import FundamentalSnapshot

logger = logging.getLogger(__name__)

Action = Literal["BUY", "SELL", "HOLD"]

SYSTEM_PROMPT = """\
You are a disciplined technical swing trader. You trade multi-day to multi-week \
moves timed by price action and technical indicators, and you use the company's \
fundamentals as a filter on those trades. You are not an investment adviser and \
you do not give financial advice to humans - you emit a single machine-readable \
trade signal for a downstream automated risk system.

Rules you MUST follow:
- Reason ONLY from the data provided: the technicals in MARKET_SNAPSHOT (RSI,
  SMA50, SMA200, trend regime, price relative to SMA50, Fibonacci retracements,
  fair value gaps, recent 4h candles) and the company data in FUNDAMENTALS.
- Prefer BUY when the regime is bullish (SMA50 >= SMA200) and momentum supports a
  continuation or a pullback entry; prefer SELL when the regime is bearish and
  momentum confirms weakness; otherwise HOLD.
- "fibonacci" holds the 23.6-78.6% retracement levels of the recent daily swing.
  In an up-swing, a pullback holding the 0.382-0.618 zone is a potential
  continuation entry; in a down-swing, a bounce stalling in that zone is
  potential resistance. A close beyond 0.786 weakens the swing.
- "fair_value_gaps" lists unfilled three-candle imbalances (4h and 1d). A bullish
  gap below price is potential support; a bearish gap above price is potential
  resistance; unfilled gaps also act as magnets price may return to. Weigh gaps
  that are nearer, larger and less filled more heavily; 1d gaps outrank 4h gaps.
- FUNDAMENTALS describes the company: valuation (P/E, price-to-book, EV/EBITDA),
  profitability (ROE, margins), growth (the latest quarter against the same
  quarter a year ago for revenue, operating profit and net profit, plus the
  multi-year CAGR), debt-to-equity, dividend yield, the next results date and
  pre-computed "flags". Technicals time the trade; fundamentals filter it: be
  more willing to BUY profitable, growing companies with manageable debt, and
  prefer HOLD over BUY for loss-making, shrinking or highly indebted ones unless
  the technical setup is exceptionally strong. Weak and deteriorating
  fundamentals strengthen a bearish case. A single quarter's net profit can
  swing on one-off items (such as a gain on selling investments a year
  earlier): when revenue and operating profit are growing, do not treat a fall
  in net profit as weakness, nor a one-off jump as strength. If results are due
  within about two weeks, mention that event risk in the reasoning.
- Prefer a target_price that lines up with a Fibonacci level, a fair value gap or
  a recent swing high/low, and mention the level you used in the reasoning.
- Do NOT invent data you were not given; if "fibonacci" is null,
  "fair_value_gaps" is empty, FUNDAMENTALS is unavailable or a fundamental field
  is missing, treat it as unknown and decide from the rest. Do NOT size the
  position or set stops - a separate hardcoded risk layer does that.
- target_price is your technical price objective for the move (a positive number).
  For HOLD, set target_price to the current price.

Output format (CRITICAL):
- Respond with EXACTLY ONE valid JSON object and nothing else.
- No markdown, no code fences, no commentary before or after, no trailing commas.
- Keys, in this exact shape:
  {
    "action": "BUY" | "SELL" | "HOLD",
    "ticker": "<symbol>",
    "target_price": <number>,
    "reasoning": "<one or two concise sentences citing the indicators, levels and fundamentals>"
  }
"""

USER_TEMPLATE = """\
Analyse the following market snapshot and company fundamentals and return your \
trade signal as a single JSON object per your instructions.

MARKET_SNAPSHOT:
{snapshot_json}

FUNDAMENTALS:
{fundamentals_json}
"""

FUNDAMENTALS_UNAVAILABLE = "unavailable (decide from the technicals alone)"


def fundamentals_json(fundamentals: Optional[FundamentalSnapshot]) -> str:
    """The FUNDAMENTALS block for a prompt, or a plain 'unavailable' note."""
    return fundamentals.to_analyst_json() if fundamentals is not None else FUNDAMENTALS_UNAVAILABLE


class AnalystDecision(BaseModel):
    """Validated trade idea emitted by the analyst agent."""

    action: Action
    ticker: str
    target_price: float = Field(gt=0)
    reasoning: str

    @classmethod
    def from_model_text(cls, text: str, expected_ticker: str) -> "AnalystDecision":
        """Parse and validate the raw model text into a decision."""
        payload = _extract_json_object(text)
        decision = cls.model_validate(payload)
        # Normalise / defend the ticker so a hallucinated symbol can't slip through.
        if decision.ticker.strip().upper() != expected_ticker.strip().upper():
            logger.warning(
                "Analyst returned ticker %r; overriding with requested %r.",
                decision.ticker, expected_ticker,
            )
            decision = decision.model_copy(update={"ticker": expected_ticker.upper()})
        else:
            decision = decision.model_copy(update={"ticker": expected_ticker.upper()})
        return decision


class AnalystError(RuntimeError):
    """Raised when the analyst cannot produce a valid decision."""


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


def _extract_json_object(text: str) -> dict:
    """Best-effort extraction of a single JSON object from model output."""
    cleaned = _FENCE_RE.sub("", text).strip()
    candidates = [cleaned]
    # Fall back to the first {...} span if the model added stray characters.
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        candidates.append(cleaned[start : end + 1])

    last_error: Optional[json.JSONDecodeError] = None
    for candidate in candidates:
        # Also forgive small slips such as trailing commas ("a": 1,}).
        for variant in (candidate, _repair_json(candidate)):
            try:
                # strict=False: a raw line break or tab inside a text value (3 Nov 2026) is accepted.
                parsed = json.loads(variant, strict=False)
            except json.JSONDecodeError as exc:
                last_error = exc
                continue
            if isinstance(parsed, dict):
                return parsed
    if last_error is not None:
        raise AnalystError(f"Could not parse JSON from analyst output: {last_error}") from last_error
    raise AnalystError("Analyst output contained no JSON object.")


_JSON_WHITESPACE = " \t\n\r"
_INVISIBLE = {"​", "‌", "‍", "⁠", "﻿"}  # zero-width characters, BOM


def _repair_json(text: str) -> str:
    """Fix common model slips outside strings; text inside strings is untouched.

    * escapes a stray double quote inside a string (one not followed by ``,``, ``}``, ``]`` or ``:``),
      such as a reasoning that quotes a phrase;
    * drops commas that directly precede ``}`` or ``]`` (trailing commas);
    * drops zero-width characters and turns other Unicode spaces (such as a
      non-breaking space) into plain spaces, since JSON only allows ASCII
      whitespace between tokens.
    """
    out = []
    in_string = escaped = False
    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                # A real closing quote is followed by , } ] or : - any other quote is a stray one
                # inside the text (the reasoning quoting a phrase), so it is escaped instead.
                j = i + 1
                while j < len(text) and text[j] in _JSON_WHITESPACE:
                    j += 1
                if j < len(text) and text[j] not in ",}]:":
                    out.append('\\"')
                    continue
                in_string = False
            out.append(ch)
            continue
        if ch == '"':
            in_string = True
        elif ch in _INVISIBLE:
            continue
        elif ch.isspace() and ch not in _JSON_WHITESPACE:
            ch = " "
        elif ch == ",":
            j = i + 1
            while j < len(text) and (text[j].isspace() or text[j] in _INVISIBLE):
                j += 1
            if text[j:j + 1] in ("}", "]"):
                continue  # trailing comma: skip it
        out.append(ch)
    return "".join(out)


def _json_error_excerpt(exc: BaseException, width: int = 25) -> str:
    """The reply text around a JSON syntax error, so odd characters show in the log."""
    cause = exc.__cause__
    if isinstance(cause, json.JSONDecodeError):
        return cause.doc[max(0, cause.pos - width): cause.pos + width]
    return ""


RETRY_TEMPLATE = """\
Your previous reply could not be used: {error}
Reply again with ONLY the corrected JSON object, exactly in the required shape - \
no markdown, no commentary, no trailing commas, and no double quotes inside the text \
values (use single quotes there)."""


def request_json_decision(client, *, model: str, system: str, user_content: str, parse, label: str):
    """Ask Claude for a JSON decision and parse it with ``parse``.

    If the reply is unusable (no text, invalid JSON or failed validation) Claude is
    asked ONCE more, with the error, to correct it. A second failure raises
    :class:`AnalystError`. Any API/SDK failure raises immediately.
    """
    messages = [{"role": "user", "content": user_content}]
    error = ""
    for attempt in (1, 2):
        try:
            response = client.messages.create(
                model=model, max_tokens=1024, system=system, messages=messages,
            )
        except Exception as exc:  # noqa: BLE001 - surface any SDK/API failure
            raise AnalystError(f"Anthropic API call failed: {exc}") from exc

        text = _first_text_block(response)
        where = ""
        if not text:
            error = "the reply contained no text"
        else:
            try:
                return parse(text)
            except (AnalystError, ValidationError) as exc:
                error = str(exc)
                where = _json_error_excerpt(exc)

        if attempt == 1:
            logger.warning(
                "%s reply was unusable (%s%s); asking once more.",
                label, error.splitlines()[0], f", near {where!r}" if where else "",
            )
            messages = messages + [
                {"role": "assistant", "content": text or "(no text)"},
                {"role": "user", "content": RETRY_TEMPLATE.format(error=error.splitlines()[0])},
            ]
    raise AnalystError(f"{label} reply was still unusable after one retry: {error}")


class AnalystAgent:
    """Wraps the Anthropic client to turn a snapshot into a trade idea."""

    def __init__(self, settings: Optional[Settings] = None, client=None) -> None:
        self.settings = settings or get_settings()
        self._client = client  # allow dependency injection for tests

    def _get_client(self):
        if self._client is not None:
            return self._client
        if not self.settings.has_anthropic:
            raise AnalystError(
                "ANTHROPIC_API_KEY is not set. Add it to your .env to run the analyst."
            )
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - environment guard
            raise AnalystError(
                "anthropic SDK not installed. Run `pip install -r requirements.txt`."
            ) from exc
        self._client = anthropic.Anthropic(api_key=self.settings.anthropic_api_key)
        return self._client

    def analyze(
        self, snapshot: MarketSnapshot, fundamentals: Optional[FundamentalSnapshot] = None
    ) -> AnalystDecision:
        """Run the analyst on a market snapshot (+ fundamentals) and return a decision."""
        client = self._get_client()
        user_content = USER_TEMPLATE.format(
            snapshot_json=snapshot.to_analyst_json(),
            fundamentals_json=fundamentals_json(fundamentals),
        )

        logger.info("Requesting analyst decision for %s (model=%s).",
                    snapshot.ticker, self.settings.anthropic_model)
        decision = request_json_decision(
            client,
            model=self.settings.anthropic_model,
            system=SYSTEM_PROMPT,
            user_content=user_content,
            parse=lambda text: AnalystDecision.from_model_text(text, snapshot.ticker),
            label="Analyst",
        )

        logger.info("Analyst decision: %s %s @ target %.2f",
                    decision.action, decision.ticker, decision.target_price)
        return decision


def _first_text_block(response) -> str:
    """Extract the first text block from an Anthropic response (skips thinking)."""
    content = getattr(response, "content", None)
    if not content:
        return ""
    for block in content:
        if getattr(block, "type", None) == "text":
            return block.text
    return ""


__all__ = [
    "Action",
    "AnalystDecision",
    "AnalystAgent",
    "AnalystError",
    "SYSTEM_PROMPT",
    "request_json_decision",
]
