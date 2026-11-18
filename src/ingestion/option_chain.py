"""Layer 1b - Option chain ingestion (Dhan Data API).

Fetches one expiry of a stock's option chain from Dhan and condenses it into an
:class:`OptionChainSnapshot` for the options analyst (Layer 2) and the options
risk layer (Layer 3).

* Expiry: the nearest listed expiry at least ``min_days_to_expiry`` and at most
  ``max_days_to_expiry`` days away (``config/trading_rules.json`` -> ``options``).
* Strikes: the at-the-money strike plus ``strikes_each_side`` strikes above and
  below it, each with the call ("ce") and put ("pe") premium, bid/ask, open
  interest, volume, implied volatility and Greeks.
* Context: the put/call open-interest ratio and the strikes holding the most
  call and put open interest across the whole expiry.

Dhan reports open interest and volume in units (shares); they are converted to
lots here so they are comparable across stocks.

Dhan serves market data only from its live API with an active Data API plan
(the sandbox has no quotes), so these calls always go to ``DHAN_DATA_BASE_URL``
whatever ``DHAN_ENV`` says. They are read-only. The parsing helpers are pure
functions so they can be tested offline.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import date
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

from config.settings import (
    DHAN_DATA_BASE_URL,
    NSE_FNO_SEGMENT,
    OptionRules,
    Settings,
    get_settings,
)

logger = logging.getLogger(__name__)

# Dhan's exchange segment for a stock (cash market) underlying.
UNDERLYING_SEGMENT_EQUITY = "NSE_EQ"
# Dhan allows one unique option-chain request every 3 seconds.
RATE_LIMIT_SECONDS = 3.1

PostFn = Callable[[str, Dict[str, str], Dict[str, Any]], Dict[str, Any]]


class OptionChainError(RuntimeError):
    """Raised when the option chain cannot be fetched or is unusable."""


class OptionLeg(BaseModel):
    """One side (call or put) of a strike."""

    security_id: str
    last_price: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    open_interest_lots: float = 0.0
    volume_lots: float = 0.0
    implied_volatility: Optional[float] = None
    delta: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None


class OptionStrike(BaseModel):
    """A strike with its call and put legs."""

    strike: float
    ce: Optional[OptionLeg] = None
    pe: Optional[OptionLeg] = None


class OptionChainSnapshot(BaseModel):
    """One expiry of a stock's option chain, trimmed to strikes near the money."""

    underlying: str
    underlying_security_id: str
    exchange_segment: str = NSE_FNO_SEGMENT
    underlying_price: float
    expiry: str = Field(description="YYYY-MM-DD")
    days_to_expiry: int
    lot_size: int = Field(gt=0)
    atm_strike: float
    put_call_oi_ratio: Optional[float] = None
    max_call_oi_strike: Optional[float] = Field(
        default=None, description="Strike with the most call open interest (often resistance)"
    )
    max_put_oi_strike: Optional[float] = Field(
        default=None, description="Strike with the most put open interest (often support)"
    )
    strikes: List[OptionStrike] = Field(default_factory=list)

    def to_analyst_json(self) -> str:
        """Compact JSON payload handed to the options analyst."""
        return self.model_dump_json(indent=2)

    def find(self, strike: float) -> Optional[OptionStrike]:
        """Return the listed strike equal to ``strike``, or None."""
        for row in self.strikes:
            if abs(row.strike - float(strike)) < 1e-6:
                return row
        return None


# --------------------------------------------------------------------------- #
# Pure helpers (offline-testable)
# --------------------------------------------------------------------------- #
def _num(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _opt_round(value: Any, digits: int) -> Optional[float]:
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def choose_expiry(
    expiries: List[str], today: date, min_days: int, max_days: int
) -> Optional[str]:
    """Nearest expiry (YYYY-MM-DD) that is ``min_days``-``max_days`` days away."""
    candidates = []
    for raw in expiries:
        text = str(raw).strip()[:10]
        try:
            expiry = date.fromisoformat(text)
        except ValueError:
            continue
        days = (expiry - today).days
        if min_days <= days <= max_days:
            candidates.append((expiry, text))
    return min(candidates)[1] if candidates else None


def _leg(raw: Any, lot_size: int) -> Optional[OptionLeg]:
    if not isinstance(raw, dict) or raw.get("security_id") in (None, "", 0, "0"):
        return None
    greeks = raw.get("greeks") or {}
    return OptionLeg(
        security_id=str(raw.get("security_id")),
        last_price=round(_num(raw.get("last_price")), 2),
        bid=round(_num(raw.get("top_bid_price")), 2),
        ask=round(_num(raw.get("top_ask_price")), 2),
        open_interest_lots=round(_num(raw.get("oi")) / lot_size, 1),
        volume_lots=round(_num(raw.get("volume")) / lot_size, 1),
        implied_volatility=_opt_round(raw.get("implied_volatility"), 2),
        delta=_opt_round(greeks.get("delta"), 3),
        theta=_opt_round(greeks.get("theta"), 2),
        vega=_opt_round(greeks.get("vega"), 2),
    )


def first_leg_security_id(payload: Dict[str, Any]) -> Optional[str]:
    """Security id of any contract in a raw option chain response (for lot-size lookup)."""
    data = payload.get("data") if isinstance(payload, dict) else None
    for sides in ((data or {}).get("oc") or {}).values():
        for side in ("ce", "pe"):
            leg = (sides or {}).get(side)
            if isinstance(leg, dict) and leg.get("security_id") not in (None, "", 0, "0"):
                return str(leg["security_id"])
    return None


def parse_option_chain(
    payload: Dict[str, Any],
    *,
    underlying: str,
    underlying_security_id: str,
    expiry: str,
    lot_size: int,
    strikes_each_side: int,
    today: date,
) -> OptionChainSnapshot:
    """Turn a raw Dhan ``/optionchain`` response into an :class:`OptionChainSnapshot`."""
    if lot_size < 1:
        raise OptionChainError(f"Invalid lot size {lot_size}.")
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, dict):
        raise OptionChainError("Option chain response has no 'data' object.")
    spot = _num(data.get("last_price"))
    if spot <= 0:
        raise OptionChainError("Option chain response has no underlying price.")

    rows: List[OptionStrike] = []
    for strike_key, sides in (data.get("oc") or {}).items():
        try:
            strike = round(float(strike_key), 2)
        except (TypeError, ValueError):
            continue
        sides = sides or {}
        row = OptionStrike(strike=strike, ce=_leg(sides.get("ce"), lot_size), pe=_leg(sides.get("pe"), lot_size))
        if row.ce or row.pe:
            rows.append(row)
    if not rows:
        raise OptionChainError(f"No tradable strikes in the {underlying} {expiry} option chain.")
    rows.sort(key=lambda r: r.strike)

    atm = min(range(len(rows)), key=lambda i: abs(rows[i].strike - spot))
    window = rows[max(0, atm - strikes_each_side): atm + strikes_each_side + 1]

    call_oi = sum(r.ce.open_interest_lots for r in rows if r.ce)
    put_oi = sum(r.pe.open_interest_lots for r in rows if r.pe)
    top_call = max((r for r in rows if r.ce), key=lambda r: r.ce.open_interest_lots, default=None)
    top_put = max((r for r in rows if r.pe), key=lambda r: r.pe.open_interest_lots, default=None)

    return OptionChainSnapshot(
        underlying=underlying.strip().upper(),
        underlying_security_id=str(underlying_security_id),
        underlying_price=round(spot, 2),
        expiry=expiry,
        days_to_expiry=(date.fromisoformat(expiry) - today).days,
        lot_size=lot_size,
        atm_strike=rows[atm].strike,
        put_call_oi_ratio=round(put_oi / call_oi, 2) if call_oi > 0 else None,
        max_call_oi_strike=top_call.strike if top_call and top_call.ce.open_interest_lots > 0 else None,
        max_put_oi_strike=top_put.strike if top_put and top_put.pe.open_interest_lots > 0 else None,
        strikes=window,
    )


# --------------------------------------------------------------------------- #
# Network client (read-only)
# --------------------------------------------------------------------------- #
def _requests_post(url: str, headers: Dict[str, str], body: Dict[str, Any]) -> Dict[str, Any]:
    import requests

    resp = requests.post(url, headers=headers, data=json.dumps(body), timeout=20)
    if resp.status_code >= 400:
        try:
            detail: Any = resp.json()
        except Exception:  # noqa: BLE001
            detail = resp.text
        raise OptionChainError(
            f"Dhan data API error {resp.status_code}: {detail}. Options mode needs a LIVE "
            "access token and an active Data API plan (sandbox tokens have no market data)."
        )
    return resp.json()


class DhanMarketData:
    """Read-only client for Dhan's option chain endpoints (live Data API)."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        post: Optional[PostFn] = None,
        sleep: Callable[[float], None] = time.sleep,
        base_url: str = DHAN_DATA_BASE_URL,
    ) -> None:
        self.settings = settings or get_settings()
        self._post = post or _requests_post
        self._sleep = sleep
        self.base_url = base_url.rstrip("/")
        self._last_call: Optional[float] = None

    def expiry_list(self, underlying_id: int, segment: str = UNDERLYING_SEGMENT_EQUITY) -> List[str]:
        """POST /optionchain/expirylist -> listed expiries (YYYY-MM-DD)."""
        payload = self._call(
            "/optionchain/expirylist",
            {"UnderlyingScrip": int(underlying_id), "UnderlyingSeg": segment},
        )
        data = payload.get("data")
        if not isinstance(data, list):
            raise OptionChainError(f"Unexpected expiry list response: {payload}")
        return [str(item) for item in data]

    def option_chain(
        self, underlying_id: int, expiry: str, segment: str = UNDERLYING_SEGMENT_EQUITY
    ) -> Dict[str, Any]:
        """POST /optionchain -> raw option chain for one expiry."""
        return self._call(
            "/optionchain",
            {"UnderlyingScrip": int(underlying_id), "UnderlyingSeg": segment, "Expiry": expiry},
        )

    def _call(self, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        if not self.settings.has_dhan:
            raise OptionChainError(
                "Options mode needs DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN in .env - a live "
                "access token with an active Data API plan."
            )
        if self._last_call is not None:
            wait = RATE_LIMIT_SECONDS - (time.monotonic() - self._last_call)
            if wait > 0:
                self._sleep(wait)
        try:
            payload = self._post(f"{self.base_url}{path}", self.settings.data_headers(), body)
        finally:
            self._last_call = time.monotonic()
        if not isinstance(payload, dict):
            raise OptionChainError(f"Unexpected response from {path}: {payload!r}")
        status = payload.get("status")
        if status is not None and str(status).lower() != "success":
            raise OptionChainError(f"Dhan {path} call failed: {payload}")
        return payload


def fetch_option_chain_snapshot(
    symbol: str,
    *,
    rules: OptionRules,
    settings: Optional[Settings] = None,
    instruments: Any = None,
    market_data: Optional[DhanMarketData] = None,
    today: Optional[date] = None,
) -> OptionChainSnapshot:
    """Fetch and condense one expiry of ``symbol``'s NSE option chain."""
    settings = settings or get_settings()
    today = today or date.today()
    sym = symbol.strip().upper()
    if instruments is None:
        from src.execution.instruments import DhanInstruments

        instruments = DhanInstruments()
    market_data = market_data or DhanMarketData(settings)

    try:
        underlying_id = instruments.resolve(sym, "NSE")
    except Exception as exc:  # noqa: BLE001
        raise OptionChainError(f"Could not resolve {sym} on NSE: {exc}") from exc

    expiries = market_data.expiry_list(int(underlying_id))
    expiry = choose_expiry(expiries, today, rules.min_days_to_expiry, rules.max_days_to_expiry)
    if expiry is None:
        listed = ", ".join(expiries[:6]) or "none"
        raise OptionChainError(
            f"No {sym} option expiry between {rules.min_days_to_expiry} and "
            f"{rules.max_days_to_expiry} days away (listed: {listed}). "
            f"Is {sym} in the F&O segment?"
        )

    payload = market_data.option_chain(int(underlying_id), expiry)
    leg_id = first_leg_security_id(payload)
    if leg_id is None:
        raise OptionChainError(f"The {sym} {expiry} option chain came back empty.")
    try:
        lot_size = instruments.option_contract(leg_id).lot_size
    except Exception as exc:  # noqa: BLE001
        raise OptionChainError(f"Could not find the lot size for {sym} options: {exc}") from exc

    chain = parse_option_chain(
        payload,
        underlying=sym,
        underlying_security_id=underlying_id,
        expiry=expiry,
        lot_size=lot_size,
        strikes_each_side=rules.strikes_each_side,
        today=today,
    )
    logger.info(
        "Option chain %s: expiry %s (%d days), lot %d, spot %.2f, ATM %.2f, "
        "PCR %s, most call OI at %s, most put OI at %s",
        sym, chain.expiry, chain.days_to_expiry, chain.lot_size, chain.underlying_price,
        chain.atm_strike, chain.put_call_oi_ratio, chain.max_call_oi_strike,
        chain.max_put_oi_strike,
    )
    return chain


__all__ = [
    "DhanMarketData",
    "OptionChainError",
    "OptionChainSnapshot",
    "OptionLeg",
    "OptionStrike",
    "choose_expiry",
    "fetch_option_chain_snapshot",
    "first_leg_security_id",
    "parse_option_chain",
]
