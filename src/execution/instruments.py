"""Dhan instrument master resolver.

Dhan orders reference a numeric ``securityId``, not the trading symbol. This
module downloads Dhan's public scrip-master CSV (no auth required), caches it
locally, and resolves ``(symbol, exchange) -> securityId`` for equity cash
instruments. It also looks up option contracts by ``securityId`` to get their
lot size (the option chain API returns contract ids but not lot sizes).

Scrip master columns (compact file) include:
    SEM_EXM_EXCH_ID (NSE/BSE), SEM_SEGMENT (E equity / D derivatives),
    SEM_SMST_SECURITY_ID, SEM_INSTRUMENT_NAME (EQUITY / OPTSTK / OPTIDX ...),
    SEM_TRADING_SYMBOL (e.g. RELIANCE), SEM_SERIES (EQ), SEM_LOT_UNITS (lot size),
    SEM_CUSTOM_SYMBOL, SEM_EXPIRY_DATE, SEM_STRIKE_PRICE, SEM_OPTION_TYPE (CE/PE).
"""

from __future__ import annotations

import csv
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from config.settings import DATA_DIR

logger = logging.getLogger(__name__)

SCRIP_MASTER_URL = "https://images.dhan.co/api-data/api-scrip-master.csv"
CACHE_PATH = DATA_DIR / "dhan_scrip_master.csv"
CACHE_MAX_AGE_SECONDS = 24 * 60 * 60  # refresh daily


class InstrumentError(RuntimeError):
    """Raised when an instrument cannot be resolved."""


@dataclass(frozen=True)
class OptionContract:
    """An exchange-traded option contract from the scrip master."""

    security_id: str
    lot_size: int
    trading_symbol: str
    display_name: str
    expiry: str
    strike: float
    option_type: str


class DhanInstruments:
    """Loads the Dhan scrip master and resolves equity security ids."""

    def __init__(self, cache_path: Path = CACHE_PATH, url: str = SCRIP_MASTER_URL) -> None:
        self.cache_path = cache_path
        self.url = url
        self._index: Optional[Dict[str, str]] = None  # "EXCHANGE:SYMBOL" -> securityId
        self._contracts: Dict[str, OptionContract] = {}  # securityId -> contract

    # ---- public API ---- #
    def resolve(self, symbol: str, exchange: str = "NSE") -> str:
        """Return the securityId for an equity ``symbol`` on ``exchange``."""
        if self._index is None:
            self._load()
        key = f"{exchange.strip().upper()}:{symbol.strip().upper()}"
        security_id = (self._index or {}).get(key)
        if not security_id:
            raise InstrumentError(
                f"Could not resolve securityId for {key}. "
                "Check the symbol, or delete the cached scrip master to refresh."
            )
        return security_id

    def option_contract(self, security_id: str | int, exchange: str = "NSE") -> OptionContract:
        """Return the option contract (lot size, symbol...) for a derivative ``security_id``.

        Security ids are only unique within an exchange segment, so the lookup is
        restricted to ``exchange``'s derivatives segment (``SEM_SEGMENT == D``).
        """
        key = str(security_id).strip()
        if key in self._contracts:
            return self._contracts[key]
        self._ensure_cache()

        exch = exchange.strip().upper()
        with open(self.cache_path, "r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                if (row.get("SEM_SMST_SECURITY_ID") or "").strip() != key:
                    continue
                if (row.get("SEM_EXM_EXCH_ID") or "").strip().upper() != exch:
                    continue
                if (row.get("SEM_SEGMENT") or "").strip().upper() != "D":
                    continue
                try:
                    lot_size = int(float(row.get("SEM_LOT_UNITS") or 0))
                    strike = float(row.get("SEM_STRIKE_PRICE") or 0)
                except ValueError as exc:
                    raise InstrumentError(f"Unreadable contract row for securityId {key}: {exc}") from exc
                if lot_size < 1:
                    raise InstrumentError(f"Contract {key} has no usable lot size in the scrip master.")
                contract = OptionContract(
                    security_id=key,
                    lot_size=lot_size,
                    trading_symbol=(row.get("SEM_TRADING_SYMBOL") or "").strip(),
                    display_name=(row.get("SEM_CUSTOM_SYMBOL") or "").strip(),
                    expiry=(row.get("SEM_EXPIRY_DATE") or "").strip()[:10],
                    strike=strike,
                    option_type=(row.get("SEM_OPTION_TYPE") or "").strip().upper(),
                )
                self._contracts[key] = contract
                return contract
        raise InstrumentError(
            f"Could not find derivative contract {exch}:{key} in the scrip master. "
            "Delete the cached scrip master to refresh it."
        )

    # ---- internals ---- #
    def _ensure_cache(self) -> None:
        if not self._is_cache_fresh():
            self._download()

    def _is_cache_fresh(self) -> bool:
        return (
            self.cache_path.exists()
            and (time.time() - self.cache_path.stat().st_mtime) < CACHE_MAX_AGE_SECONDS
            and self.cache_path.stat().st_size > 0
        )

    def _download(self) -> None:
        try:
            import requests
        except ImportError as exc:  # pragma: no cover
            raise InstrumentError("requests not installed; cannot download scrip master.") from exc

        logger.info("Downloading Dhan scrip master from %s ...", self.url)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(self.url, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            with open(self.cache_path, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 16):
                    if chunk:
                        fh.write(chunk)
        logger.info("Scrip master cached at %s", self.cache_path)

    def _load(self) -> None:
        self._ensure_cache()

        index: Dict[str, str] = {}
        with open(self.cache_path, "r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if row.get("SEM_INSTRUMENT_NAME") != "EQUITY":
                    continue
                if row.get("SEM_SERIES") not in ("EQ", "BE", ""):
                    continue
                exch = (row.get("SEM_EXM_EXCH_ID") or "").strip().upper()
                sym = (row.get("SEM_TRADING_SYMBOL") or "").strip().upper()
                sec_id = (row.get("SEM_SMST_SECURITY_ID") or "").strip()
                if exch and sym and sec_id:
                    index.setdefault(f"{exch}:{sym}", sec_id)
        if not index:
            raise InstrumentError("Scrip master parsed but no equity instruments were found.")
        self._index = index
        logger.info("Loaded %d equity instruments from scrip master.", len(index))


__all__ = ["DhanInstruments", "InstrumentError", "OptionContract", "SCRIP_MASTER_URL", "CACHE_PATH"]
