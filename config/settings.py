"""Configuration and environment variables.

This module centralises two things:

1. ``Settings`` - runtime/environment configuration loaded from a ``.env`` file
   or the process environment (Anthropic + Dhan credentials, environment
   selection, dry-run flag, ...).
2. ``TradingRules`` - the *immutable* risk parameters loaded from
   ``config/trading_rules.json``. These are frozen so no other layer can mutate
   them at runtime; Layer 3 (``guardrails.py``) reads them but the enforcement
   logic itself is hardcoded in Python.

Broker: Dhan (India). Amounts are in INR. The default environment is the Dhan
**sandbox** (mock) API, so the full pipeline can run end-to-end with no real
money. Switching to live requires deliberate, separate opt-in flags.

Nothing here talks to the network - it only resolves configuration.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
CONFIG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CONFIG_DIR.parent
TRADING_RULES_PATH = CONFIG_DIR / "trading_rules.json"
ENV_PATH = PROJECT_ROOT / ".env"
DATA_DIR = PROJECT_ROOT / "data"

# Dhan API base URLs (verified against DhanHQ v2 docs).
DHAN_SANDBOX_BASE_URL = "https://sandbox.dhan.co/v2"
DHAN_LIVE_BASE_URL = "https://api.dhan.co/v2"
# Market data (e.g. the option chain) only exists on the live API - the sandbox
# has no quotes - so data calls always go here, whatever DHAN_ENV says. They are
# read-only and need a live access token with an active Data API plan.
DHAN_DATA_BASE_URL = DHAN_LIVE_BASE_URL

# Map a plain exchange to Dhan's exchangeSegment enum (equity cash).
EXCHANGE_SEGMENTS: Dict[str, str] = {
    "NSE": "NSE_EQ",
    "BSE": "BSE_EQ",
}

# Exchange segment for NSE futures & options orders.
NSE_FNO_SEGMENT = "NSE_FNO"


# --------------------------------------------------------------------------- #
# Environment settings
# --------------------------------------------------------------------------- #
class Settings(BaseSettings):
    """Environment-driven configuration.

    Values are read from ``.env`` (if present) and the process environment.
    Field names are case-insensitive against env var names.
    """

    model_config = SettingsConfigDict(
        env_file=str(ENV_PATH),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Anthropic (Layer 2) ---
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    anthropic_model: str = Field(default="claude-sonnet-5", alias="ANTHROPIC_MODEL")

    # --- Dhan broker (Layer 4) ---
    dhan_client_id: str = Field(default="", alias="DHAN_CLIENT_ID")
    dhan_access_token: str = Field(default="", alias="DHAN_ACCESS_TOKEN")
    # "sandbox" (mock, default) or "live" (real money).
    dhan_env: str = Field(default="sandbox", alias="DHAN_ENV")
    # Second lock for real trading: even with dry_run False and dhan_env live,
    # nothing is sent unless this is explicitly True.
    dhan_enable_live_trading: bool = Field(
        default=False, alias="DHAN_ENABLE_LIVE_TRADING"
    )

    # --- Market defaults ---
    default_exchange: str = Field(default="NSE", alias="DEFAULT_EXCHANGE")
    # CNC = delivery (multi-day swing). INTRADAY/MARGIN/MTF/BO also valid on Dhan.
    product_type: str = Field(default="CNC", alias="PRODUCT_TYPE")
    # Product for option orders: MARGIN = carry forward (NRML, multi-day swing),
    # INTRADAY = squared off the same day. CNC does not apply to F&O.
    options_product_type: str = Field(default="MARGIN", alias="OPTIONS_PRODUCT_TYPE")

    # --- Safety switch ---
    dry_run: bool = Field(default=True, alias="DRY_RUN")

    # --- Simulation fallback account (used when no Dhan creds are present) ---
    # Defaults mirror the Dhan sandbox's daily virtual capital of Rs 10,00,000.
    simulated_total_equity: float = Field(
        default=1_000_000.0, alias="SIMULATED_TOTAL_EQUITY"
    )
    simulated_cash_balance: float = Field(
        default=1_000_000.0, alias="SIMULATED_CASH_BALANCE"
    )

    # --- Automation (automate.py / GitHub Actions) ---
    # "paper" (default) tracks trades in state/paper_ledger.json with the
    # SIMULATED_* balance; "live" sends real orders and additionally needs
    # DHAN_ENV=live, DHAN_ENABLE_LIVE_TRADING=true and a whitelisted static IP.
    autotrade_mode: str = Field(default="paper", alias="AUTOTRADE_MODE")

    # --- WhatsApp notifications ---
    # "telegram" (free, recommended), "callmebot" (free, personal use), "twilio",
    # or empty to only log the message. (Named WHATSAPP_* for history.)
    whatsapp_provider: str = Field(default="", alias="WHATSAPP_PROVIDER")
    whatsapp_phone: str = Field(default="", alias="WHATSAPP_PHONE")  # e.g. +919876543210
    callmebot_api_key: str = Field(default="", alias="CALLMEBOT_API_KEY")
    twilio_account_sid: str = Field(default="", alias="TWILIO_ACCOUNT_SID")
    twilio_auth_token: str = Field(default="", alias="TWILIO_AUTH_TOKEN")
    twilio_whatsapp_from: str = Field(default="", alias="TWILIO_WHATSAPP_FROM")
    # Telegram (free, official Bot API): a token from @BotFather and your chat id.
    telegram_bot_token: str = Field(default="", alias="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: str = Field(default="", alias="TELEGRAM_CHAT_ID")

    @field_validator("dhan_env")
    @classmethod
    def _normalise_env(cls, v: str) -> str:
        v = (v or "sandbox").strip().lower()
        if v not in ("sandbox", "live"):
            raise ValueError("DHAN_ENV must be 'sandbox' or 'live'.")
        return v

    @field_validator("default_exchange")
    @classmethod
    def _normalise_exchange(cls, v: str) -> str:
        v = (v or "NSE").strip().upper()
        if v not in EXCHANGE_SEGMENTS:
            raise ValueError(f"DEFAULT_EXCHANGE must be one of {list(EXCHANGE_SEGMENTS)}.")
        return v

    @field_validator("autotrade_mode")
    @classmethod
    def _normalise_autotrade_mode(cls, v: str) -> str:
        v = (v or "paper").strip().lower()
        if v not in ("paper", "live"):
            raise ValueError("AUTOTRADE_MODE must be 'paper' or 'live'.")
        return v

    @field_validator("whatsapp_provider")
    @classmethod
    def _normalise_whatsapp_provider(cls, v: str) -> str:
        v = (v or "").strip().lower()
        if v not in ("", "none", "callmebot", "twilio", "telegram"):
            raise ValueError("WHATSAPP_PROVIDER must be 'telegram', 'callmebot', 'twilio' or empty.")
        return "" if v == "none" else v

    @field_validator("options_product_type")
    @classmethod
    def _normalise_options_product(cls, v: str) -> str:
        v = (v or "MARGIN").strip().upper()
        if v not in ("MARGIN", "INTRADAY"):
            raise ValueError("OPTIONS_PRODUCT_TYPE must be 'MARGIN' or 'INTRADAY'.")
        return v

    @property
    def has_anthropic(self) -> bool:
        return bool(self.anthropic_api_key)

    @property
    def has_dhan(self) -> bool:
        return bool(self.dhan_client_id and self.dhan_access_token)

    @property
    def is_sandbox(self) -> bool:
        return self.dhan_env == "sandbox"

    @property
    def base_url(self) -> str:
        return DHAN_SANDBOX_BASE_URL if self.is_sandbox else DHAN_LIVE_BASE_URL

    def auth_headers(self) -> Dict[str, str]:
        """Headers for Dhan v2 REST calls."""
        return {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "access-token": self.dhan_access_token,
        }

    def data_headers(self) -> Dict[str, str]:
        """Headers for Dhan Data APIs (option chain), which also need the client id."""
        return {**self.auth_headers(), "client-id": self.dhan_client_id}

    def exchange_segment(self, exchange: str | None = None) -> str:
        ex = (exchange or self.default_exchange).strip().upper()
        if ex not in EXCHANGE_SEGMENTS:
            raise ValueError(f"Unknown exchange '{ex}'. Expected one of {list(EXCHANGE_SEGMENTS)}.")
        return EXCHANGE_SEGMENTS[ex]


# --------------------------------------------------------------------------- #
# Immutable trading rules
# --------------------------------------------------------------------------- #
class OptionRules(BaseModel):
    """Frozen risk parameters for BUY-ONLY options (calls and puts).

    A bought option can lose at most its premium, so the premium outlay is what
    gets capped. Writing (selling) options is not supported anywhere.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    enabled: bool = False
    max_premium_pct: float = Field(gt=0, le=1)
    min_days_to_expiry: int = Field(ge=0)
    max_days_to_expiry: int = Field(gt=0)
    strikes_each_side: int = Field(gt=0, le=20)
    min_open_interest_lots: float = Field(ge=0)
    max_bid_ask_spread_pct: float = Field(gt=0)
    min_abs_delta: float = Field(ge=0, le=1)
    max_abs_delta: float = Field(gt=0, le=1)
    stop_loss_pct: float = Field(gt=0, lt=1)
    take_profit_pct: float = Field(gt=0)
    max_lots: int = Field(gt=0)


class PennyRules(BaseModel):
    """Frozen, stricter parameters for the weekly penny-stock job."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    enabled: bool = False
    min_price: float = Field(ge=0)
    max_price: float = Field(gt=0)
    min_avg_turnover_cr: float = Field(ge=0)
    max_position_pct: float = Field(gt=0, le=1)
    stop_loss_pct: float = Field(gt=0, lt=1)
    take_profit_pct: float = Field(gt=0)
    max_open_positions: int = Field(gt=0)
    max_hold_days: int = Field(gt=0, le=365)


INTRADAY_MODELS = ("breakout", "fvg", "fib", "gap_fade")


class IntradayRules(BaseModel):
    """Frozen parameters for the paper intraday job (entries and square-off in IST)."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    enabled: bool = False
    max_position_pct: float = Field(gt=0, le=1)
    # The widest stop a setup may use; each setup places its own stop where its
    # idea is proven wrong, and the target is take_profit_pct / stop_loss_pct
    # times that stop.
    stop_loss_pct: float = Field(gt=0, lt=1)
    take_profit_pct: float = Field(gt=0, lt=1)
    max_trades_per_day: int = Field(default=1, gt=0, le=10)
    entry_after: str = "09:45"
    entry_cutoff: str = "11:30"
    square_off: str = "15:15"
    # Kinds of setup the job may trade (see src/automation/intraday.py).
    models: Tuple[str, ...] = INTRADAY_MODELS
    # Estimated round-trip charges, % of the position (brokerage, STT, exchange
    # fees, GST, stamp duty, slippage), taken off every paper intraday trade.
    cost_pct: float = Field(default=0.12, ge=0, le=2)
    # Paper practice: Claude must take the best setup every day and may hold only when something is
    # clearly wrong, so the paper account builds experience while no setup has a proven edge.
    practice: bool = False

    @field_validator("models")
    @classmethod
    def _known_models(cls, v: Tuple[str, ...]) -> Tuple[str, ...]:
        unknown = sorted(set(v) - set(INTRADAY_MODELS))
        if unknown:
            raise ValueError(f"unknown intraday model(s) {unknown}; choose from {list(INTRADAY_MODELS)}")
        return v

    @field_validator("entry_after", "entry_cutoff", "square_off")
    @classmethod
    def _hh_mm(cls, v: str) -> str:
        hours, minutes = (int(x) for x in v.split(":"))
        if not (0 <= hours < 24 and 0 <= minutes < 60):
            raise ValueError("times must be HH:MM")
        return f"{hours:02d}:{minutes:02d}"


class PennyIntradayRules(IntradayRules):
    """The same intraday job run on penny stocks, with its own paper account and learning (state/penny/).

    The list of stocks is rebuilt every trading day from NSE's bhavcopy: EQ series only (the
    trade-to-trade series cannot be traded intraday), priced min_price-max_price, the max_stocks most
    traded with at least min_avg_turnover_cr (Rs crore) traded a day on average over 20 days.
    """

    min_price: float = Field(default=2.0, ge=0)
    max_price: float = Field(default=50.0, gt=0)
    min_avg_turnover_cr: float = Field(default=1.0, ge=0)
    max_stocks: int = Field(default=50, gt=0, le=200)
    # A setup is skipped unless its entry, stop and target all sit at least this far (% of price)
    # inside the stock's daily circuit limits, so the paper account never pretends it could get out
    # of a frozen stock. default_band_pct is assumed when NSE's list of price bands cannot be read.
    circuit_margin_pct: float = Field(default=0.5, ge=0, le=5)
    default_band_pct: float = Field(default=5.0, gt=0, le=20)


class FuturesIntradayRules(IntradayRules):
    """The same intraday job on Nifty, Bank Nifty, Fin Nifty and Midcap Nifty futures (state/futures/).

    Positions are sized in lots: enough lots that the loss at the stop is about risk_per_trade_pct of
    the account (one lot is allowed up to twice that), at most max_lots, and the margin (margin_pct of
    the contract value) at most max_margin_pct of the account. Only the margin is blocked from cash.
    lot_sizes are used when NSE's list of lot sizes cannot be read.
    """

    lot_sizes: Dict[str, int] = Field(
        default_factory=lambda: {"NIFTY": 65, "BANKNIFTY": 30, "FINNIFTY": 60, "MIDCPNIFTY": 120})
    margin_pct: float = Field(default=12.0, gt=0, le=100)
    risk_per_trade_pct: float = Field(default=1.0, gt=0, le=5)
    max_lots: int = Field(default=3, gt=0, le=50)
    max_margin_pct: float = Field(default=60.0, gt=0, le=100)


class DebateRules(BaseModel):
    """Bull vs bear, then a judge, before every intraday trade (all three books).

    Claude's pick is debated first; if the judge says SKIP, the next setups on the shortlist are
    debated, up to max_candidates, and the first one the judge approves is traded. rounds is how many
    turns each side gets.
    """

    model_config = ConfigDict(frozen=True)

    enabled: bool = False
    rounds: int = Field(default=2, ge=1, le=4)
    max_candidates: int = Field(default=3, ge=1, le=5)


class TradingRules(BaseModel):
    """Frozen risk parameters (INR). Attempting to set an attribute raises."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    max_position_pct: float = Field(gt=0, le=1)
    stop_loss_pct: float = Field(gt=0, lt=1)
    take_profit_pct: float = Field(gt=0, lt=1)
    max_open_positions: int = Field(gt=0)
    min_trade_notional: float = Field(ge=0)
    allowed_actions: Tuple[str, ...]
    allow_short_selling: bool = False
    # Refuse NEW positions when the company's results are due within this many
    # days (0 turns the rule off). Unknown results dates never block.
    earnings_blackout_days: int = Field(default=0, ge=0, le=60)
    # Automated swing positions are closed after this many calendar days if
    # neither the stop nor the target has been hit.
    max_hold_days: int = Field(default=20, gt=0, le=365)
    # Switches the automated daily swing job on or off (automate.py daily).
    swing_enabled: bool = True
    options: Optional[OptionRules] = None
    penny: Optional[PennyRules] = None
    intraday: Optional[IntradayRules] = None
    penny_intraday: Optional[PennyIntradayRules] = None
    futures_intraday: Optional[FuturesIntradayRules] = None
    debate: Optional[DebateRules] = None


@lru_cache(maxsize=1)
def load_trading_rules(path: Path = TRADING_RULES_PATH) -> TradingRules:
    """Load and validate the immutable trading rules (cached)."""
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    raw.pop("_comment", None)  # strip the human notes before validation
    for block in ("options", "penny", "intraday", "penny_intraday", "futures_intraday", "debate"):
        if isinstance(raw.get(block), dict):
            raw[block].pop("_comment", None)
    return TradingRules(**raw)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()


__all__ = [
    "Settings",
    "OptionRules",
    "PennyRules",
    "IntradayRules",
    "INTRADAY_MODELS",
    "TradingRules",
    "get_settings",
    "load_trading_rules",
    "PROJECT_ROOT",
    "DATA_DIR",
    "TRADING_RULES_PATH",
    "EXCHANGE_SEGMENTS",
    "NSE_FNO_SEGMENT",
    "DHAN_SANDBOX_BASE_URL",
    "DHAN_LIVE_BASE_URL",
    "DHAN_DATA_BASE_URL",
]
