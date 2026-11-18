"""Which stocks the automated jobs look at.

* Daily swing job: the ``large_caps`` list in ``config/universe.json`` (roughly
  the Nifty 50, editable).
* Weekly penny job: every NSE stock in the ``EQ`` series from NSE's daily
  "bhavcopy" (one public CSV with every stock's price and traded value),
  filtered to the price band and liquidity floor in the ``penny`` rules. The
  trade-to-trade segments (``BE``/``BZ``) are excluded outright.
* Penny intraday book: the same bhavcopy filter, then the most traded stocks by
  their 20-day average traded value (:func:`penny_intraday_list`), with each
  stock's daily circuit limit from NSE's list of price bands.

Parsing and filtering are pure functions so they can be tested offline.
"""

from __future__ import annotations

import csv
import io
import json
import logging
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from config.settings import CONFIG_DIR, PennyRules

logger = logging.getLogger(__name__)

UNIVERSE_PATH = CONFIG_DIR / "universe.json"
BHAVCOPY_URL = "https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{day}.csv"
# Every listed security with its price band (the daily circuit limit, % from the previous close).
PRICE_BANDS_URL = "https://nsearchives.nseindia.com/content/equities/sec_list.csv"
# NSE's archive refuses requests that do not look like a browser.
BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    ),
    "Accept": "text/csv,text/plain,*/*",
}
LAKH_PER_CRORE = 100.0


class UniverseError(RuntimeError):
    """Raised when a stock universe cannot be built."""


class Universe(BaseModel):
    large_caps: List[str]
    watchlist: List[str] = Field(default_factory=list)
    top_n: int = Field(default=10, gt=0, le=30)
    fundamentals_shortlist: int = Field(default=20, gt=0, le=60)
    penny_max_candidates: int = Field(default=150, gt=0, le=500)


def load_universe(path: Path = UNIVERSE_PATH) -> Universe:
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    raw.pop("_comment", None)
    return Universe(**raw)


@dataclass(frozen=True)
class PennyCandidate:
    symbol: str
    close: float
    turnover_cr: float          # that day's traded value, Rs crore
    delivery_pct: Optional[float]


def _number(value: Optional[str]) -> Optional[float]:
    try:
        return float(str(value).strip().replace(",", ""))
    except (TypeError, ValueError):
        return None


def parse_bhavcopy(text: str) -> List[dict]:
    """Rows of NSE's ``sec_bhavdata_full`` CSV with keys and values stripped."""
    reader = csv.DictReader(io.StringIO(text))
    rows = []
    for row in reader:
        rows.append({(k or "").strip(): (v or "").strip() for k, v in row.items()})
    return rows


def penny_candidates(rows: Iterable[dict], rules: PennyRules, limit: int = 150) -> List[PennyCandidate]:
    """EQ-series stocks inside the penny price band, most traded first.

    Uses a lenient single-day liquidity pre-filter (half the average-turnover
    floor); the screener later checks the 20-day average properly.
    """
    picked = []
    for row in rows:
        if row.get("SERIES", "").upper() != "EQ":
            continue
        close = _number(row.get("CLOSE_PRICE"))
        turnover_lacs = _number(row.get("TURNOVER_LACS"))
        if close is None or turnover_lacs is None:
            continue
        if not rules.min_price <= close <= rules.max_price:
            continue
        turnover_cr = turnover_lacs / LAKH_PER_CRORE
        if turnover_cr < rules.min_avg_turnover_cr / 2:
            continue
        picked.append(
            PennyCandidate(
                symbol=row.get("SYMBOL", "").upper(),
                close=close,
                turnover_cr=round(turnover_cr, 2),
                delivery_pct=_number(row.get("DELIV_PER")),
            )
        )
    picked.sort(key=lambda c: c.turnover_cr, reverse=True)
    return picked[:limit]


def fetch_latest_bhavcopy(
    today: date, *, days_back: int = 7, get: Optional[Callable] = None
) -> Tuple[date, str]:
    """Download the most recent NSE bhavcopy on or before ``today``."""
    if get is None:
        import requests

        get = requests.get
    for offset in range(days_back + 1):
        day = today - timedelta(days=offset)
        if day.weekday() >= 5:
            continue
        url = BHAVCOPY_URL.format(day=day.strftime("%d%m%Y"))
        try:
            resp = get(url, headers=BROWSER_HEADERS, timeout=30)
        except Exception as exc:  # noqa: BLE001 - try the previous day
            logger.debug("Bhavcopy %s failed: %s", day, exc)
            continue
        if getattr(resp, "status_code", 0) == 200 and "SYMBOL" in resp.text[:200]:
            logger.info("Using NSE bhavcopy for %s.", day.isoformat())
            return day, resp.text
    raise UniverseError(f"Could not download an NSE bhavcopy for the {days_back} days before {today}.")


def parse_price_bands(text: str) -> Dict[str, Optional[float]]:
    """EQ-series symbol -> price band in % (None for "No Band", e.g. stocks with derivatives)."""
    bands: Dict[str, Optional[float]] = {}
    for row in parse_bhavcopy(text):
        keys = {k.lower(): k for k in row}
        symbol = row.get(keys.get("symbol", ""), "").upper()
        series = row.get(keys.get("series", ""), "").upper()
        band_key = next((k for low, k in keys.items() if "band" in low), None)
        if not symbol or series != "EQ" or band_key is None:
            continue
        bands[symbol] = _number(row.get(band_key))
    return bands


def fetch_price_bands(*, get: Optional[Callable] = None) -> Dict[str, Optional[float]]:
    """NSE's price bands ({} if the list cannot be downloaded: callers then assume a default band)."""
    if get is None:
        import requests

        get = requests.get
    try:
        resp = get(PRICE_BANDS_URL, headers=BROWSER_HEADERS, timeout=30)
    except Exception as exc:  # noqa: BLE001 - best effort
        logger.warning("Could not download NSE's price bands (%s).", exc)
        return {}
    if getattr(resp, "status_code", 0) != 200:
        logger.warning("NSE's price bands: HTTP %s.", getattr(resp, "status_code", "?"))
        return {}
    return parse_price_bands(resp.text)


def average_turnover_cr(frame, days: int = 20) -> Optional[float]:
    """Average daily traded value over the last ``days`` sessions, Rs crore (close x volume)."""
    if frame is None or getattr(frame, "empty", True) or not {"Close", "Volume"} <= set(frame.columns):
        return None
    recent = frame.dropna(subset=["Close", "Volume"]).tail(days)
    if recent.empty:
        return None
    return float((recent["Close"] * recent["Volume"]).mean()) / 1e7


def penny_intraday_list(candidates: Sequence[PennyCandidate], daily: Dict[str, object], rules,
                        to_ticker: Callable[[str], str]) -> List[Tuple[str, float]]:
    """(symbol, 20-day average traded value in Rs crore) for the penny intraday book, most traded first.

    ``candidates`` come from :func:`penny_candidates`; ``daily`` maps Yahoo tickers to daily bars;
    ``rules`` gives ``min_avg_turnover_cr`` and ``max_stocks``.
    """
    ranked = []
    for c in candidates:
        avg = average_turnover_cr(daily.get(to_ticker(c.symbol)))
        if avg is not None and avg >= rules.min_avg_turnover_cr:
            ranked.append((c.symbol, round(avg, 2)))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked[: rules.max_stocks]


__all__ = [
    "PRICE_BANDS_URL",
    "BHAVCOPY_URL",
    "PennyCandidate",
    "Universe",
    "UniverseError",
    "average_turnover_cr",
    "fetch_latest_bhavcopy",
    "fetch_price_bands",
    "load_universe",
    "parse_bhavcopy",
    "parse_price_bands",
    "penny_candidates",
    "penny_intraday_list",
]
