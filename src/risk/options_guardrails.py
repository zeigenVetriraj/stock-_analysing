"""Layer 3b - The Options Risk Interceptor  (CRITICAL SAFETY LAYER, buy-only).

This layer is **100% hardcoded, deterministic Python** with NO LLM calls. Every
idea from the options analyst passes through here before Layer 4, and it raises
a hard :class:`RiskViolation` (killing execution) unless ALL of these hold:

1. Options are enabled in ``config/trading_rules.json`` and the action is
   BUY_CALL, BUY_PUT or HOLD. There is no way to sell or write an option.
2. The company's results are not due within ``earnings_blackout_days``
   (shared rule with the shares risk layer; an unknown date never blocks).
3. The expiry is the one Layer 1 fetched, and it is between
   ``min_days_to_expiry`` and ``max_days_to_expiry`` days away.
4. The strike is listed in the fetched chain and has the chosen leg (CE / PE).
5. The contract is liquid: a two-sided quote, a bid/ask spread within
   ``max_bid_ask_spread_pct`` and open interest of at least
   ``min_open_interest_lots`` lots.
6. |delta| is inside [``min_abs_delta``, ``max_abs_delta``]: no far
   out-of-the-money lottery tickets, no deep in-the-money capital sinks.
7. **Premium cap.** A bought option can lose at most its premium, so the premium
   outlay is capped at ``max_premium_pct`` of equity (and by free cash), in whole
   lots, and at ``max_lots``. If a single lot does not fit, the trade dies.

Every tradeable order also carries a stop-loss and take-profit on the premium
(``stop_loss_pct`` / ``take_profit_pct``). The numbers come from the immutable
rules file; the enforcement logic lives here.
"""

from __future__ import annotations

import logging
import math
from datetime import date
from typing import Literal, Optional, Union

from pydantic import BaseModel

from config.settings import TradingRules, load_trading_rules
from src.analysis.options_agent import OptionDecision
from src.ingestion.option_chain import OptionChainSnapshot
from src.risk.guardrails import AccountState, RiskViolation, check_earnings_blackout

logger = logging.getLogger(__name__)

TICK_SIZE = 0.05  # NSE option premium tick size (Rs)
ALLOWED_OPTION_ACTIONS = ("BUY_CALL", "BUY_PUT", "HOLD")

OptionType = Literal["CE", "PE"]


def _to_tick(price: float) -> float:
    """Round a premium to the exchange tick, never below one tick."""
    return max(TICK_SIZE, round(round(price / TICK_SIZE) * TICK_SIZE, 2))


class ValidatedOptionOrder(BaseModel):
    """A fully risk-checked BUY order for one option contract (or a HOLD).

    For HOLD ``is_tradeable`` is False and the quantity is zero; Layer 4 must
    skip these.
    """

    underlying: str
    action: str
    is_tradeable: bool
    option_type: Optional[OptionType] = None
    strike: Optional[float] = None
    expiry: str = ""
    days_to_expiry: int = 0
    security_id: str = ""
    exchange_segment: str = ""
    lot_size: int = 0
    lots: int = 0
    quantity: int = 0
    limit_price: float = 0.0
    premium_outlay: float = 0.0
    max_allowed_outlay: float = 0.0
    stop_loss_premium: Optional[float] = None
    take_profit_premium: Optional[float] = None
    risk_pct_of_equity: float = 0.0
    was_downsized: bool = False
    underlying_price: float = 0.0
    underlying_target: float = 0.0
    delta: Optional[float] = None
    reasoning: str = ""
    notes: str = ""


class OptionRiskInterceptor:
    """Deterministic validator for options ideas (buy-only)."""

    def __init__(self, rules: Optional[TradingRules] = None) -> None:
        self.rules = rules or load_trading_rules()

    def validate(
        self,
        decision: OptionDecision,
        chain: OptionChainSnapshot,
        account_state: AccountState,
        proposed_lots: Optional[int] = None,
        today: Optional[date] = None,
        next_earnings_date: Optional[Union[date, str]] = None,
    ) -> ValidatedOptionOrder:
        """Validate and size an options idea, or raise :class:`RiskViolation`."""
        opt = self.rules.options
        if opt is None or not opt.enabled:
            raise RiskViolation("Options are disabled in trading rules.")

        action = str(decision.action).upper()
        if action not in ALLOWED_OPTION_ACTIONS:
            raise RiskViolation(
                f"Option action '{action}' is not allowed. Buy-only: {ALLOWED_OPTION_ACTIONS}."
            )

        if action == "HOLD":
            return ValidatedOptionOrder(
                underlying=chain.underlying,
                action="HOLD",
                is_tradeable=False,
                underlying_price=chain.underlying_price,
                underlying_target=decision.underlying_target,
                reasoning=decision.reasoning,
                notes="HOLD signal - no order generated.",
            )

        sym = chain.underlying

        # --- Earnings blackout: no new position right before results ---
        check_earnings_blackout(self.rules, next_earnings_date, today)

        # --- Expiry: must be the fetched one and inside the allowed window ---
        if decision.expiry != chain.expiry:
            raise RiskViolation(
                f"Expiry {decision.expiry} is not the fetched expiry {chain.expiry}."
            )
        try:
            days = (date.fromisoformat(chain.expiry) - (today or date.today())).days
        except ValueError as exc:
            raise RiskViolation(f"Unreadable expiry {chain.expiry!r}.") from exc
        if not opt.min_days_to_expiry <= days <= opt.max_days_to_expiry:
            raise RiskViolation(
                f"Expiry {chain.expiry} is {days} days away; allowed window is "
                f"{opt.min_days_to_expiry}-{opt.max_days_to_expiry} days."
            )

        # --- Strike and leg must exist in the fetched chain ---
        if decision.strike is None:
            raise RiskViolation("No strike given for an options trade.")
        row = chain.find(decision.strike)
        if row is None:
            raise RiskViolation(f"Strike {decision.strike:g} is not in the fetched {sym} chain.")
        option_type: OptionType = "CE" if action == "BUY_CALL" else "PE"
        leg = row.ce if option_type == "CE" else row.pe
        label = f"{sym} {row.strike:g} {option_type} {chain.expiry}"
        if leg is None:
            raise RiskViolation(f"{label} is not listed.")

        # --- Liquidity ---
        if leg.bid <= 0 or leg.ask <= 0:
            raise RiskViolation(f"{label} has no two-sided quote (bid {leg.bid}, ask {leg.ask}).")
        if leg.ask < leg.bid:
            raise RiskViolation(f"{label} has a crossed quote (bid {leg.bid} > ask {leg.ask}).")
        spread_pct = (leg.ask - leg.bid) / leg.ask * 100
        if spread_pct > opt.max_bid_ask_spread_pct:
            raise RiskViolation(
                f"{label} bid/ask spread is {spread_pct:.1f}%, above the "
                f"{opt.max_bid_ask_spread_pct:.1f}% limit."
            )
        if leg.open_interest_lots < opt.min_open_interest_lots:
            raise RiskViolation(
                f"{label} open interest is {leg.open_interest_lots:g} lots, below the "
                f"{opt.min_open_interest_lots:g}-lot minimum."
            )

        # --- Delta band ---
        if leg.delta is None:
            raise RiskViolation(f"{label} has no delta; cannot check moneyness.")
        if not opt.min_abs_delta <= abs(leg.delta) <= opt.max_abs_delta:
            raise RiskViolation(
                f"{label} delta {leg.delta:+.2f} is outside the allowed "
                f"{opt.min_abs_delta:.2f}-{opt.max_abs_delta:.2f} band."
            )

        # --- Account guards ---
        if account_state.total_equity <= 0:
            raise RiskViolation("Account total equity is non-positive; refusing to trade.")
        if account_state.open_positions >= self.rules.max_open_positions:
            raise RiskViolation(
                f"Open positions ({account_state.open_positions}) at or above cap "
                f"({self.rules.max_open_positions})."
            )

        # --- Premium cap, in whole lots ---
        premium = leg.ask
        lot_cost = premium * chain.lot_size
        max_outlay = account_state.total_equity * opt.max_premium_pct
        fundable = min(max_outlay, account_state.cash_balance)
        cap_lots = min(math.floor(fundable / lot_cost), opt.max_lots)
        if cap_lots < 1:
            raise RiskViolation(
                f"One lot of {label} costs Rs {lot_cost:,.2f} ({chain.lot_size} x Rs {premium:,.2f}), "
                f"above the premium cap of Rs {max_outlay:,.2f} ({opt.max_premium_pct:.0%} of "
                f"equity; cash Rs {account_state.cash_balance:,.2f})."
            )
        if proposed_lots is None:
            lots, was_downsized = cap_lots, False
        else:
            if proposed_lots < 1:
                raise RiskViolation(f"Proposed lots {proposed_lots} is below 1.")
            lots = min(proposed_lots, cap_lots)
            was_downsized = lots < proposed_lots

        quantity = lots * chain.lot_size
        outlay = round(quantity * premium, 2)

        order = ValidatedOptionOrder(
            underlying=sym,
            action=action,
            is_tradeable=True,
            option_type=option_type,
            strike=row.strike,
            expiry=chain.expiry,
            days_to_expiry=days,
            security_id=leg.security_id,
            exchange_segment=chain.exchange_segment,
            lot_size=chain.lot_size,
            lots=lots,
            quantity=quantity,
            limit_price=round(premium, 2),
            premium_outlay=outlay,
            max_allowed_outlay=round(max_outlay, 2),
            stop_loss_premium=_to_tick(premium * (1 - opt.stop_loss_pct)),
            take_profit_premium=_to_tick(premium * (1 + opt.take_profit_pct)),
            risk_pct_of_equity=round(outlay / account_state.total_equity * 100, 4),
            was_downsized=was_downsized,
            underlying_price=chain.underlying_price,
            underlying_target=decision.underlying_target,
            delta=leg.delta,
            reasoning=decision.reasoning,
            notes=(
                f"Downsized to the premium cap ({opt.max_premium_pct:.0%} of equity, "
                f"max {opt.max_lots} lots)."
                if was_downsized else "Within the premium cap."
            ),
        )
        logger.info(
            "Validated BUY %s x%d lot(s) = %d units @ Rs %.2f -> premium Rs %.2f "
            "(%.2f%% of equity, max loss) SL=%.2f TP=%.2f%s",
            label, order.lots, order.quantity, order.limit_price, order.premium_outlay,
            order.risk_pct_of_equity, order.stop_loss_premium, order.take_profit_premium,
            " [DOWNSIZED]" if was_downsized else "",
        )
        return order


__all__ = [
    "ALLOWED_OPTION_ACTIONS",
    "OptionRiskInterceptor",
    "ValidatedOptionOrder",
]
