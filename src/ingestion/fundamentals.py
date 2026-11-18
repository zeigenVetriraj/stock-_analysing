"""Layer 1c - Fundamentals (Yahoo Finance via ``yfinance``).

Technicals time a swing trade; fundamentals say what kind of company is being
traded. This module collects a compact, analyst-friendly picture of the company
behind a ticker and hands it to Layer 2 as a :class:`FundamentalSnapshot`:

* Valuation     - trailing / forward P/E, price-to-book, EV/EBITDA
* Profitability - return on equity, profit and operating margins (worked out
                  from the annual accounts when Yahoo's summary lacks them)
* Growth        - latest quarter vs the same quarter a year ago (revenue,
                  operating profit, net profit) and the multi-year revenue /
                  net-income CAGR from the annual accounts
* Balance sheet - debt-to-equity (as a ratio, e.g. 0.36x), current ratio
* Shareholders  - dividend yield, trailing EPS, market cap (in Rs crore)
* Price context - distance from the 52-week high and low
* Events        - the next quarterly results date and days until it
* Flags         - plain-language observations computed in Python, e.g.
                  "loss-making", "high debt", "results due in 4 days", or a
                  net-profit drop that looks like a one-off (operating profit
                  still growing)

Yahoo covers large and mid caps well and small caps patchily, so every field is
optional and the whole step is best-effort: if it fails, the pipeline carries on
with technicals only. Neither Dhan's API nor TradingView offers fundamentals to
scripts, which is why Yahoo (already the price source) is used.

Units: Yahoo reports ratios such as ROE and margins as fractions and
debt-to-equity as a percentage; they are normalised here (percent / ratio) so
the analyst never has to guess.
"""

from __future__ import annotations

import logging
import math
from datetime import date, datetime
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

RS_PER_CRORE = 1e7

# Thresholds for the plain-language flags. These are analysis heuristics for the
# model's context, NOT risk limits (those live in config/trading_rules.json).
HIGH_DEBT_TO_EQUITY = 2.0            # x
EXPENSIVE_PE = 50.0
STRONG_ROE_PCT = 15.0
FAST_GROWTH_PCT = 15.0
FALLING_PROFITS_PCT = -10.0
RESULTS_SOON_DAYS = 14
LEVERAGE_EXEMPT_SECTORS = {"Financial Services"}  # banks/NBFCs run on leverage by design

# Statement rows, as yfinance labels them (pretty names first, raw keys as backup).
REVENUE_ROWS = ("Total Revenue", "TotalRevenue", "Operating Revenue", "OperatingRevenue")
NET_INCOME_ROWS = ("Net Income", "NetIncome", "Net Income Common Stockholders", "NetIncomeCommonStockholders")
OPERATING_INCOME_ROWS = ("Operating Income", "OperatingIncome")  # excludes other income / one-off gains
EQUITY_ROWS = ("Stockholders Equity", "StockholdersEquity", "Common Stock Equity", "CommonStockEquity")

SAME_PERIOD_DAYS = 10           # statement dates this close count as the same period end
YEAR_AGO_DAYS = (350, 380)      # "the same quarter a year earlier"


class FundamentalsError(RuntimeError):
    """Raised when no fundamental data at all could be retrieved."""


class FundamentalSnapshot(BaseModel):
    """Company fundamentals for one ticker. Every metric is optional."""

    ticker: str
    name: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    currency: Optional[str] = None
    market_cap_cr: Optional[float] = Field(default=None, description="Market cap in Rs crore (INR only)")

    trailing_pe: Optional[float] = None
    forward_pe: Optional[float] = None
    price_to_book: Optional[float] = None
    ev_to_ebitda: Optional[float] = None

    return_on_equity_pct: Optional[float] = None
    profit_margin_pct: Optional[float] = None
    operating_margin_pct: Optional[float] = None

    revenue_growth_yoy_pct: Optional[float] = Field(
        default=None, description="Latest quarter's revenue vs the same quarter a year ago"
    )
    operating_profit_growth_yoy_pct: Optional[float] = Field(
        default=None,
        description="Latest quarter's operating profit vs the same quarter a year ago "
                    "(leaves out other income and one-off gains)",
    )
    earnings_growth_yoy_pct: Optional[float] = Field(
        default=None, description="Latest quarter's net profit vs the same quarter a year ago"
    )
    revenue_cagr_pct: Optional[float] = None
    net_income_cagr_pct: Optional[float] = None
    cagr_years: Optional[int] = Field(default=None, description="Years covered by the CAGR figures")

    debt_to_equity: Optional[float] = Field(default=None, description="Ratio, e.g. 0.36 means 0.36x")
    current_ratio: Optional[float] = None

    dividend_yield_pct: Optional[float] = None
    eps_ttm: Optional[float] = None

    pct_below_52w_high: Optional[float] = None
    pct_above_52w_low: Optional[float] = None

    next_earnings_date: Optional[str] = Field(default=None, description="YYYY-MM-DD")
    days_to_earnings: Optional[int] = None

    flags: List[str] = Field(default_factory=list)

    def to_analyst_json(self) -> str:
        """Compact JSON payload handed to the analysts (empty fields dropped)."""
        return self.model_dump_json(indent=2, exclude_none=True)

    def next_earnings(self) -> Optional[date]:
        """The next results date as a ``date``, if known."""
        return date.fromisoformat(self.next_earnings_date) if self.next_earnings_date else None


Series = List[Tuple[date, float]]


# --------------------------------------------------------------------------- #
# Pure helpers (offline-testable)
# --------------------------------------------------------------------------- #
def _num(value: Any) -> Optional[float]:
    """A finite float, or None for missing / non-numeric / NaN / inf values."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _pct(fraction: Any) -> Optional[float]:
    number = _num(fraction)
    return round(number * 100, 2) if number is not None else None


def _round(value: Any, digits: int = 2) -> Optional[float]:
    number = _num(value)
    return round(number, digits) if number is not None else None


def _first(info: Dict[str, Any], *keys: str) -> Optional[float]:
    """First key in ``info`` holding a usable number."""
    for key in keys:
        number = _num(info.get(key))
        if number is not None:
            return number
    return None


def _show(value: Optional[float], unit: str = "", signed: bool = False) -> str:
    """A metric for the log line: 'n/a' when unknown."""
    if value is None:
        return "n/a"
    return f"{value:+g}{unit}" if signed else f"{value:g}{unit}"


def cagr_pct(first: Optional[float], last: Optional[float], years: int) -> Optional[float]:
    """Compound annual growth rate in %, or None when it is not meaningful.

    Undefined when either end is missing or not positive (e.g. a loss year), or
    when fewer than one year separates them.
    """
    if first is None or last is None or years < 1 or first <= 0 or last <= 0:
        return None
    return round(((last / first) ** (1 / years) - 1) * 100, 2)


def statement_series(statement: Optional[pd.DataFrame], row_names: Iterable[str]) -> Series:
    """``(period_end, value)`` pairs, oldest first, for the first matching row.

    Works for any yfinance statement (annual or quarterly income statement,
    balance sheet).
    """
    if statement is None or getattr(statement, "empty", True):
        return []
    for name in row_names:
        if name in statement.index:
            row = statement.loc[name]
            if isinstance(row, pd.DataFrame):  # duplicated label: take the first
                row = row.iloc[0]
            points = []
            for period, value in row.items():
                number = _num(value)
                if number is None:
                    continue
                try:
                    points.append((pd.Timestamp(period).date(), number))
                except (TypeError, ValueError):
                    continue
            return sorted(points)
    return []


def _value_on(series: Series, period: date) -> Optional[float]:
    """The series value for the period ending on (or within a few days of) ``period``."""
    return next((value for day, value in series if abs((day - period).days) <= SAME_PERIOD_DAYS), None)


def year_on_year_pct(quarters: Series) -> Optional[float]:
    """Latest quarter vs the same quarter a year earlier, in %.

    None when no quarter sits about a year before the latest one, or when the
    year-ago figure is not positive (growth from a loss is not meaningful).
    """
    if len(quarters) < 2:
        return None
    latest_day, latest = quarters[-1]
    low, high = YEAR_AGO_DAYS
    for day, value in reversed(quarters[:-1]):
        if low <= (latest_day - day).days <= high:
            return round((latest / value - 1) * 100, 2) if value > 0 else None
    return None


def roe_from_accounts(net_income: Series, equity: Series) -> Optional[float]:
    """Return on equity in %: the latest annual net profit over average equity.

    Averages the opening and closing shareholders' equity when the previous
    year-end is available, otherwise uses the closing figure. None when equity
    for that year is missing or not positive.
    """
    if not net_income or not equity:
        return None
    period, profit = net_income[-1]
    closing = _value_on(equity, period)
    if closing is None or closing <= 0:
        return None
    opening = next(
        (value for day, value in reversed(equity)
         if day < period and SAME_PERIOD_DAYS < (period - day).days <= YEAR_AGO_DAYS[1]),
        None,
    )
    base = (opening + closing) / 2 if opening is not None and opening > 0 else closing
    return round(profit / base * 100, 2)


def margin_from_accounts(profit: Series, revenue: Series) -> Optional[float]:
    """The latest annual profit figure as a % of that year's revenue."""
    if not profit or not revenue:
        return None
    period, amount = profit[-1]
    sales = _value_on(revenue, period)
    if sales is None or sales <= 0:
        return None
    return round(amount / sales * 100, 2)


def next_earnings_date(calendar: Optional[Dict[str, Any]], today: date) -> Optional[date]:
    """Earliest listed results date on or after ``today`` (Yahoo lists a range)."""
    raw = (calendar or {}).get("Earnings Date") or []
    if not isinstance(raw, (list, tuple)):
        raw = [raw]
    upcoming = []
    for item in raw:
        if isinstance(item, datetime):
            item = item.date()
        elif not isinstance(item, date):
            try:
                item = pd.Timestamp(item).date()
            except (TypeError, ValueError):
                continue
        if item >= today:
            upcoming.append(item)
    return min(upcoming) if upcoming else None


def _profit_flags(s: FundamentalSnapshot) -> List[str]:
    """Net vs operating profit: separate a weaker business from one-off items."""
    net, operating = s.earnings_growth_yoy_pct, s.operating_profit_growth_yoy_pct
    if net is not None and net <= FALLING_PROFITS_PCT:
        if operating is not None and operating >= 0:
            return [
                f"net profit down {-net:.1f}% year on year but operating profit up {operating:.1f}%: "
                "likely one-off items (e.g. a gain in the year-ago quarter), not a weaker business"
            ]
        if operating is None and (s.revenue_growth_yoy_pct or 0) > 0:
            return [
                f"profits falling ({net:+.1f}% year on year) despite revenue growth: "
                "check for one-off items in the year-ago quarter"
            ]
        return [f"profits falling ({net:+.1f}% year on year)"]
    if operating is not None and operating <= FALLING_PROFITS_PCT:
        return [f"operating profit falling ({operating:+.1f}% year on year)"]
    return []


def _flags(s: FundamentalSnapshot) -> List[str]:
    flags: List[str] = []
    if (s.eps_ttm is not None and s.eps_ttm < 0) or (s.profit_margin_pct is not None and s.profit_margin_pct < 0):
        flags.append("loss-making (negative earnings over the last 12 months)")
    if (
        s.debt_to_equity is not None
        and s.debt_to_equity > HIGH_DEBT_TO_EQUITY
        and s.sector not in LEVERAGE_EXEMPT_SECTORS
    ):
        flags.append(f"high debt (debt-to-equity {s.debt_to_equity:.2f}x)")
    if s.revenue_growth_yoy_pct is not None:
        if s.revenue_growth_yoy_pct < 0:
            flags.append(f"revenue shrinking ({s.revenue_growth_yoy_pct:+.1f}% year on year)")
        elif s.revenue_growth_yoy_pct >= FAST_GROWTH_PCT:
            flags.append(f"fast revenue growth ({s.revenue_growth_yoy_pct:+.1f}% year on year)")
    flags.extend(_profit_flags(s))
    if s.return_on_equity_pct is not None and s.return_on_equity_pct >= STRONG_ROE_PCT:
        flags.append(f"strong profitability (ROE {s.return_on_equity_pct:.1f}%)")
    if s.trailing_pe is not None and s.trailing_pe > EXPENSIVE_PE:
        flags.append(f"expensive (P/E {s.trailing_pe:.1f})")
    if s.days_to_earnings is not None and 0 <= s.days_to_earnings <= RESULTS_SOON_DAYS:
        flags.append(f"results due in {s.days_to_earnings} days ({s.next_earnings_date})")
    return flags


def summarise_fundamentals(
    ticker: str,
    info: Optional[Dict[str, Any]],
    income_statement: Optional[pd.DataFrame] = None,
    calendar: Optional[Dict[str, Any]] = None,
    *,
    balance_sheet: Optional[pd.DataFrame] = None,
    quarterly_income_statement: Optional[pd.DataFrame] = None,
    today: Optional[date] = None,
    current_price: Optional[float] = None,
) -> FundamentalSnapshot:
    """Turn raw yfinance outputs into a :class:`FundamentalSnapshot`.

    Yahoo's summary figures come first; the annual accounts fill in ROE and the
    margins when the summary lacks them, and the quarterly income statement
    supplies operating-profit growth.
    """
    info = info or {}
    today = today or date.today()

    price = _num(current_price) or _first(info, "currentPrice", "regularMarketPrice")
    currency = info.get("currency") or info.get("financialCurrency")
    market_cap = _num(info.get("marketCap"))

    debt_to_equity = _num(info.get("debtToEquity"))  # Yahoo: a percentage
    dividend_rate = _first(info, "dividendRate", "trailingAnnualDividendRate")
    high, low = _num(info.get("fiftyTwoWeekHigh")), _num(info.get("fiftyTwoWeekLow"))

    revenue = statement_series(income_statement, REVENUE_ROWS)
    net_income = statement_series(income_statement, NET_INCOME_ROWS)
    operating_income = statement_series(income_statement, OPERATING_INCOME_ROWS)
    equity = statement_series(balance_sheet, EQUITY_ROWS)
    quarterly_operating = statement_series(quarterly_income_statement, OPERATING_INCOME_ROWS)

    years = None
    revenue_cagr = net_income_cagr = None
    if len(revenue) >= 2:
        years = len(revenue) - 1
        revenue_cagr = cagr_pct(revenue[0][1], revenue[-1][1], years)
    if len(net_income) >= 2:
        net_income_cagr = cagr_pct(net_income[0][1], net_income[-1][1], len(net_income) - 1)
        years = years or len(net_income) - 1

    roe = _pct(info.get("returnOnEquity"))
    if roe is None:
        roe = roe_from_accounts(net_income, equity)
    profit_margin = _pct(info.get("profitMargins"))
    if profit_margin is None:
        profit_margin = margin_from_accounts(net_income, revenue)
    operating_margin = _pct(info.get("operatingMargins"))
    if operating_margin is None:
        operating_margin = margin_from_accounts(operating_income, revenue)

    earnings_day = next_earnings_date(calendar, today)

    snapshot = FundamentalSnapshot(
        ticker=ticker.strip().upper(),
        name=info.get("longName") or info.get("shortName"),
        sector=info.get("sector"),
        industry=info.get("industry"),
        currency=currency,
        market_cap_cr=round(market_cap / RS_PER_CRORE, 0) if market_cap and currency == "INR" else None,
        trailing_pe=_round(info.get("trailingPE")),
        forward_pe=_round(info.get("forwardPE")),
        price_to_book=_round(info.get("priceToBook")),
        ev_to_ebitda=_round(info.get("enterpriseToEbitda")),
        return_on_equity_pct=roe,
        profit_margin_pct=profit_margin,
        operating_margin_pct=operating_margin,
        revenue_growth_yoy_pct=_pct(info.get("revenueGrowth")),
        operating_profit_growth_yoy_pct=year_on_year_pct(quarterly_operating),
        earnings_growth_yoy_pct=_pct(_first(info, "earningsGrowth", "earningsQuarterlyGrowth")),
        revenue_cagr_pct=revenue_cagr,
        net_income_cagr_pct=net_income_cagr,
        cagr_years=years,
        debt_to_equity=round(debt_to_equity / 100, 2) if debt_to_equity is not None else None,
        current_ratio=_round(info.get("currentRatio")),
        dividend_yield_pct=round(dividend_rate / price * 100, 2) if dividend_rate and price else None,
        eps_ttm=_round(info.get("trailingEps")),
        pct_below_52w_high=round((high - price) / high * 100, 2) if high and price else None,
        pct_above_52w_low=round((price - low) / low * 100, 2) if low and price else None,
        next_earnings_date=earnings_day.isoformat() if earnings_day else None,
        days_to_earnings=(earnings_day - today).days if earnings_day else None,
    )
    return snapshot.model_copy(update={"flags": _flags(snapshot)})


# --------------------------------------------------------------------------- #
# Network fetch (best-effort)
# --------------------------------------------------------------------------- #
def _attempt(label: str, getter: Callable[[], Any]) -> Any:
    try:
        return getter()
    except Exception as exc:  # noqa: BLE001 - each piece is optional
        logger.debug("Fundamentals: %s unavailable (%s).", label, exc)
        return None


def _table(value: Any) -> Optional[pd.DataFrame]:
    """A usable statement table, or None."""
    return value if isinstance(value, pd.DataFrame) and not value.empty else None


def fetch_fundamentals(
    ticker: str, *, current_price: Optional[float] = None, today: Optional[date] = None
) -> FundamentalSnapshot:
    """Fetch fundamentals for a Yahoo ticker such as ``RELIANCE.NS``.

    Raises :class:`FundamentalsError` only when nothing at all came back.
    """
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - environment guard
        raise FundamentalsError("yfinance is not installed.") from exc

    symbol = ticker.strip().upper()
    tkr = yf.Ticker(symbol)
    # Anything in an unexpected shape (older yfinance versions returned the
    # calendar as a DataFrame, for example) is ignored rather than trusted.
    info = _attempt("company info", lambda: tkr.info)
    info = info if isinstance(info, dict) else {}
    income = _table(_attempt("income statement", lambda: tkr.income_stmt))
    quarterly = _table(_attempt("quarterly income statement", lambda: tkr.quarterly_income_stmt))
    balance = _table(_attempt("balance sheet", lambda: tkr.balance_sheet))
    calendar = _attempt("results calendar", lambda: tkr.calendar)
    calendar = calendar if isinstance(calendar, dict) else None

    has_info = any(
        info.get(key) is not None
        for key in ("trailingPE", "priceToBook", "returnOnEquity", "profitMargins", "marketCap", "trailingEps")
    )
    if not (has_info or income is not None or quarterly is not None or calendar):
        raise FundamentalsError(f"Yahoo Finance returned no fundamentals for {symbol}.")

    s = summarise_fundamentals(
        symbol, info, income, calendar,
        balance_sheet=balance, quarterly_income_statement=quarterly,
        today=today, current_price=current_price,
    )
    logger.info(
        "Fundamentals %s: P/E %s | ROE %s | revenue YoY %s | operating profit YoY %s | "
        "net profit YoY %s | D/E %s | next results %s%s",
        symbol, _show(s.trailing_pe), _show(s.return_on_equity_pct, "%"),
        _show(s.revenue_growth_yoy_pct, "%", signed=True),
        _show(s.operating_profit_growth_yoy_pct, "%", signed=True),
        _show(s.earnings_growth_yoy_pct, "%", signed=True),
        _show(s.debt_to_equity, "x"), s.next_earnings_date or "unknown",
        f" | flags: {'; '.join(s.flags)}" if s.flags else "",
    )
    return s


__all__ = [
    "FundamentalSnapshot",
    "FundamentalsError",
    "cagr_pct",
    "fetch_fundamentals",
    "margin_from_accounts",
    "next_earnings_date",
    "roe_from_accounts",
    "statement_series",
    "summarise_fundamentals",
    "year_on_year_pct",
]
