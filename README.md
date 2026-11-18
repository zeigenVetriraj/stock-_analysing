# AI Swing Trading System (India / Dhan)

> **New here? Start with [SETUP.md](SETUP.md)**: what you need, how to set it up
> on GitHub in about 30 minutes, and how to run it. Paper trading only.

A modular, **safety-first** scaffold for an autonomous multi-agent swing-trading
pipeline for **Indian equities (NSE/BSE)**. It separates concerns into four
isolated layers, with a hardcoded risk interceptor sitting between the LLM's
suggestion and any order that reaches the broker.

> ⚠️ **Not financial advice. Educational scaffold only.** It defaults to
> **dry-run**, and when it does send orders it targets the **Dhan sandbox**
> (mock money) unless you deliberately switch to live. Automated trading can
> lose money quickly.

---

## Architecture

```
Ingest (L1) ─▶ Analyze (L2) ─▶ Risk Intercept / Validate (L3) ─▶ Execute (L4)
 yfinance       Claude LLM        hardcoded Python (no AI)          Dhan (sandbox)
 real prices +                    5% cap · 3% SL · 6% TP            mock by default
 fundamentals                     no new trades just before results
```

| Layer | File | Role | AI? |
|-------|------|------|-----|
| **1. Ingestion** | `src/ingestion/data_fetcher.py`, `src/ingestion/fundamentals.py` | Real prices (Yahoo Finance, `.NS`/`.BO`) + RSI, SMA50, SMA200, Fibonacci retracements, fair value gaps; company fundamentals and the next results date | No |
| **2. Analyst** | `src/analysis/analyst_agent.py` | Claude reads the technicals and fundamentals, returns a strict JSON trade signal | **Yes** |
| **3. Risk interceptor** | `src/risk/guardrails.py` | Enforce the 5% max position, append 3% stop / 6% target, block new trades when results are 5 days away or less, or kill the trade | **No — deterministic** |
| **4. Executor** | `src/execution/order_executor.py` | Build the Dhan payload; dry-run by default, sandbox before live | No |

Supporting modules: `src/execution/dhan_client.py` (Dhan REST calls) and
`src/execution/instruments.py` (resolves a symbol to Dhan's `securityId`).

**The safety guarantee:** Layer 3 is 100% hardcoded Python. It makes no model
calls. Every order is capped at 5% of equity (`max_position_pct`) and carries a mandatory
stop-loss/take-profit, or it is rejected with a hard error.

### Fibonacci levels and fair value gaps

Layer 1 also hands the analyst two sets of price levels. Both are computed in
plain Python (no AI) and covered by `tests/test_indicators.py`:

- **Fibonacci retracements**: the 23.6 / 38.2 / 50 / 61.8 / 78.6% levels of the
  swing between the highest high and lowest low of the last 120 daily sessions
  (`fib_lookback`), plus the swing direction, how much of it price has given
  back, and the nearest level.
- **Fair value gaps**: three-candle imbalances on 4h and daily bars (candle 3's
  low above candle 1's high, or its high below candle 1's low) that price has
  not fully traded back through. The three nearest per timeframe are sent, with
  how much of each gap is already filled; gaps under 0.1% of price are ignored.

Claude is told to treat them as support, resistance and target zones. They never
change position size or stops: Layer 3 still enforces those.

### Fundamentals

Technicals time the trade; fundamentals say what kind of company is being
traded. `src/ingestion/fundamentals.py` pulls them from Yahoo Finance (already
the price source) and hands Claude a compact summary next to the technicals:

| Group | What is sent |
|-------|--------------|
| Valuation | trailing and forward P/E, price-to-book, EV/EBITDA |
| Profitability | return on equity, profit margin, operating margin (worked out from the annual accounts when Yahoo's summary lacks them) |
| Growth | latest quarter vs the same quarter a year ago (revenue, operating profit, net profit); multi-year revenue and net-profit CAGR from the annual accounts |
| Balance sheet | debt-to-equity (as a ratio, e.g. 0.36x), current ratio |
| Shareholders | dividend yield, trailing EPS, market cap in ₹ crore |
| Price context | distance from the 52-week high and low |
| Events | the next quarterly results date |

Python also turns these into plain-language flags, such as "loss-making",
"high debt" (financial companies such as banks and NBFCs are exempt, since they
run on leverage), "revenue shrinking", "profits falling", "strong
profitability", "expensive" or "results due in 4 days", so the model is not
left to interpret raw ratios.

Net profit for a single quarter can swing on one-off items. When net profit
falls but operating profit (which leaves out other income and one-off gains)
still grows, the flag says so: "net profit down 22.4% year on year but operating
profit up 9.9%: likely one-off items", and both analysts are told not to read
such a drop as weakness, nor a one-off jump as strength.

How they are used:

- **As a filter, not a trigger.** Claude still times entries from the
  technicals. For shares it prefers HOLD over BUY on a loss-making, shrinking or
  heavily indebted company unless the setup is exceptionally strong. In options
  mode, calls are for healthy companies, and weak fundamentals strengthen the
  case for a put.
- **Results blackout (a hard rule in Layer 3).** No new position, shares or
  options, when the company's quarterly results are due within
  `earnings_blackout_days` (5 in `config/trading_rules.json`; 0 turns it off).
  Results can gap a stock either way, and option premiums usually drop straight
  after the announcement as implied volatility falls. An unknown results date
  never blocks a trade, and HOLD is never blocked.
- **Best-effort.** Yahoo covers large and mid caps well and small caps
  patchily. Every metric is optional, and if nothing comes back the run logs
  "Fundamentals unavailable" and carries on with technicals only.

Dhan's API has no fundamentals endpoints, and TradingView (whose charts Dhan
uses) has no official data API for scripts, which is why Yahoo is used.

### Why Dhan sandbox?

Dhan is one of the few Indian brokers with a **mock API** ([developer.dhanhq.co](https://developer.dhanhq.co)):
you get ₹10,00,000 of virtual capital (resets daily) and orders are simulated,
never routed to the exchange. That lets the **whole pipeline run end-to-end —
including the real order API call — with zero money at risk.** Note the sandbox
fills every order at ₹100 and has no live quotes, so it validates *plumbing*, not
strategy P&L (that's why prices/analysis still come from Yahoo Finance).

---

## Setup

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate   |   macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # Windows: copy .env.example .env   — then edit
```

### `.env` variables

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `ANTHROPIC_API_KEY` | for Layer 2 | — | Claude API key |
| `ANTHROPIC_MODEL` | no | `claude-sonnet-5` | Analyst model |
| `DHAN_CLIENT_ID` / `DHAN_ACCESS_TOKEN` | to submit orders | — | Dhan credentials (sandbox token works) |
| `DHAN_ENV` | no | `sandbox` | `sandbox` (mock) or `live` (real money) |
| `DHAN_ENABLE_LIVE_TRADING` | no | `false` | Second lock; must be `true` for a live order |
| `DEFAULT_EXCHANGE` | no | `NSE` | `NSE` or `BSE` |
| `PRODUCT_TYPE` | no | `CNC` | `CNC` (delivery/swing), `INTRADAY`, `BO`, ... |
| `OPTIONS_PRODUCT_TYPE` | no | `MARGIN` | Option orders: `MARGIN` (carry forward) or `INTRADAY` |
| `DRY_RUN` | no | `true` | Print the payload instead of sending |
| `SIMULATED_TOTAL_EQUITY` / `SIMULATED_CASH_BALANCE` | no | `1000000` | Equity used when no Dhan creds are set |

---

## Running

**Dry-run (default — no order sent, prints payload + risk plan):**

```bash
python main.py --symbol RELIANCE
```

Works with **no Dhan account at all** — analysis uses real Yahoo Finance prices
and the risk layer falls back to the `SIMULATED_*` equity. Only Layer 2 needs an
`ANTHROPIC_API_KEY`.

**Propose a size and watch the risk layer cap it to 5%:**

```bash
python main.py --symbol TCS --qty 100000
```

**Submit to the Dhan sandbox (mock money, real API round-trip):**

```bash
# needs DHAN_CLIENT_ID / DHAN_ACCESS_TOKEN (sandbox token) and DHAN_ENV=sandbox
python main.py --symbol INFY --submit
```

**Going live (real money) — deliberately gated:** set `DHAN_ENV=live` **and**
`DHAN_ENABLE_LIVE_TRADING=true`, then `--submit`. If either is missing the
executor refuses. Test everything in sandbox first.

**Size against the practice balance instead of your Dhan funds (dry-run only):**

```bash
python main.py --symbol TCS --sim-account
```

`--sim-account` uses `SIMULATED_TOTAL_EQUITY` / `SIMULATED_CASH_BALANCE` and is
refused together with `--submit`, so a practice balance can never size a real order.

---

## Options mode (buy-only)

`--options` swaps shares for **bought** options on NSE stocks: a call for a
bullish view, a put for a bearish one. Options are never sold or written, so the
most any trade can lose is the premium paid.

```bash
python main.py --symbol RELIANCE --options                 # dry-run: prints the order, sends nothing
python main.py --symbol RELIANCE --options --lots 2        # propose 2 lots; Layer 3 downsizes if needed
python main.py --symbol RELIANCE --options --sim-account   # size against the practice balance
```

| Layer | File | Role |
|-------|------|------|
| **1b. Option chain** | `src/ingestion/option_chain.py` | Dhan option chain for the nearest expiry 14-60 days out: the at-the-money strike ± 5 strikes with premium, bid/ask, open interest, IV and Greeks, plus the put/call OI ratio and the strikes with the most call and put OI. Lot size comes from the scrip master. |
| **2b. Options analyst** | `src/analysis/options_agent.py` | Claude reads the stock's technicals (incl. Fibonacci and fair value gaps), its fundamentals and the chain, then picks BUY_CALL / BUY_PUT / HOLD and one listed strike. If results fall before expiry it must say so, and it prefers HOLD unless the setup is strong. |
| **3b. Options risk** | `src/risk/options_guardrails.py` | Hardcoded, no AI: no new trade when results are 5 days away or less, premium capped at 2% of equity in whole lots (max 10), expiry inside the window, strike must exist, two-sided quote, spread ≤ 5%, open interest ≥ 50 lots, abs(delta) 0.25-0.75, stop-loss -30% / target +60% on the premium. |
| **4. Executor** | `src/execution/order_executor.py` | `execute_option`: NSE_FNO LIMIT BUY at the ask, `OPTIONS_PRODUCT_TYPE` (default MARGIN). Same dry-run default and live locks as shares. |

All the limits live in the `options` block of `config/trading_rules.json`
(set `"enabled": false` there to switch options off).

**What it needs from Dhan**

- A **live** access token (web.dhan.co -> My Profile -> Access DhanHQ APIs; valid
  24 hours) in `DHAN_CLIENT_ID` / `DHAN_ACCESS_TOKEN`. The sandbox has no market
  data, so the option chain always comes from the live API. Those calls are read-only.
- An active **Data API** plan (₹499 + GST a month) for the option chain.
- `DHAN_ENV=live` lets the risk layer size against your real balance (read-only);
  keep `DHAN_ENABLE_LIVE_TRADING=false` so nothing can be sent. Or use `--sim-account`.
- Placing real orders through the API also needs a static IP whitelisted with
  Dhan (a SEBI rule). Data calls and dry runs do not.

Options are high-risk: SEBI found that over 91% of individual traders lost money
in equity F&O in FY25.

---

## Automation: daily swing, weekly penny, WhatsApp report

> **Current setup: intraday only.** The automated swing (`daily`) and penny jobs
> are switched off in `config/trading_rules.json` (`"swing_enabled": false`, and
> `"enabled": false` in the `penny` block), and the workflow only schedules the
> intraday jobs. Set them back to `true` (and add their schedule lines back) to
> use them again. `python automate.py close-out` closes any paper positions that
> belong to a switched-off strategy at the current price.

`automate.py` runs the whole pipeline without you naming a stock:

```bash
python automate.py daily     # exits, screen ~Nifty 50, Claude picks one, risk check, trade, report
python automate.py penny     # weekly: screen liquid NSE stocks under Rs 50, pick one, trade, report
python automate.py report    # holdings, bot positions, watchlist and results dates only
python automate.py intraday-open    # 09:45-11:30 IST: find setups, Claude picks at most one, long or short (paper)
python automate.py intraday-check   # hourly 11:03-14:03 IST: alert and close if the stop or target was hit
python automate.py intraday-close   # ~15:20 IST: square off today's intraday trade (paper)
python automate.py intraday-backtest  # weekly: save 5-minute bars, test every setup model, write a report
python automate.py daily --no-whatsapp   # print the report instead of sending it
```

What a run does:

1. **Exits first.** Every open bot position is checked against its stop-loss,
   target and time limit (`max_hold_days`, 20 days for swing, 30 for penny) and
   closed if one is hit. Prices are checked once per run.
2. **Screen (no AI).** `src/automation/screener.py` scores every stock in the
   universe on technicals (trend, RSI, 20-day momentum, distance from the 52-week
   high, volume), fetches fundamentals for the best 20 and ranks them 60%
   technical / 40% fundamental. The top 10 go to Claude.
3. **Pick (Claude).** `src/automation/picker.py` compares the shortlist and picks
   at most one stock to buy, or HOLD. It must pick one of the candidates.
4. **Risk (no AI).** The same hardcoded risk layer sizes the trade and sets the
   stop and target; the earnings blackout and position caps still apply.
5. **Trade and report.** Paper mode records the trade; live mode sends the order.
   A Telegram or WhatsApp report lists your Dhan holdings with profit/loss, the bot's
   positions, closed trades, the shortlist, the pick and upcoming results dates.
   If NSE did not trade that day (weekend or holiday), nothing is traded.

| File | Role |
|------|------|
| `config/universe.json` | The daily universe (about the Nifty 50, edit freely), your watchlist and the shortlist sizes |
| `config/trading_rules.json` | `max_hold_days`, and the stricter `penny` block: price Rs 2-50, average traded value at least Rs 1 crore a day, 1% of equity per trade, 8% stop, 20% target, at most 3 penny positions |
| `state/paper_ledger.json` | The practice account (created on the first run) |

**Penny universe.** The penny job reads NSE's daily "bhavcopy" (one public
file with every stock's price and traded value), keeps `EQ`-series stocks only
(the trade-to-trade `BE`/`BZ` segments are excluded) inside the price band, and
checks the 20-day average traded value before screening. Penny stocks are
volatile, thinly traded and prone to manipulation; the picker is told to be
sceptical and HOLD is a normal answer.

**Intraday (paper only).** `intraday-open` measures every large cap on 5-minute
bars (`src/automation/intraday.py`): the 09:15-09:45 opening range, VWAP, EMA
9/20, RSI, yesterday's high / low / close, pivot levels and the CPR, relative
volume, fair value gaps on 5- and 15-minute bars, today's swing with its
Fibonacci levels, and the stock against the Nifty 50. It then looks for four
kinds of setup ("models"), long or short (shorting is fine intraday because the
position closes the same day):

| Model | The idea | Stop |
|-------|----------|------|
| `breakout` | Price is beyond the opening range: trade with the break | Back at the middle of the range |
| `fvg` | Price has come back to an unfilled fair value gap made earlier today: trade in the gap's direction | Just beyond the far side of the gap |
| `fib` | Price has given back 38.2-61.8% of today's swing (at least 1%): trade in the swing's direction | Beyond the 78.6% level |
| `gap_fade` | The stock with the day's biggest opening gap (at least 0.5%): trade against the gap | The widest stop allowed |

The best few of each model go to Claude, which picks at most one. The paper
account is in **practice mode** (`"practice": true`): Claude must take the best
setup every day and may hold only when something is clearly wrong with every
candidate (broken or stale data, a price already past its target or stop, an
extreme move), so the bot builds experience while no setup has a proven edge.
Set `"practice": false` to trade only clean setups, where HOLD is a normal
answer. Each journal entry records whether it was a practice pick. Stocks whose
results are due inside the earnings blackout (5 days) are left off the
shortlist, so Claude only chooses among setups it is allowed to trade. The
`intraday` block of `config/trading_rules.json` sets the
limits: 10% of equity with no leverage, one trade a day, entries only between
09:45 and 11:30, the widest stop a setup may use (`stop_loss_pct`, 1%; no stop
is tighter than 0.3%), and a target at `take_profit_pct / stop_loss_pct` (2x)
the stop. `models` lists the models that may trade; remove a name to switch one
off. `cost_pct` (0.12% of the position) is taken off every paper intraday trade
as estimated charges (brokerage, STT, exchange fees, GST, stamp duty, slippage),
so paper results are not flattered. The entry is refused if the latest
5-minute bar is more than 20 minutes old.

`intraday-close` replays the 5-minute bars after the entry to see whether the
stop or the target was hit first (a bar touching both counts as the stop) and
otherwise squares off at 15:15. In between, `intraday-check` runs hourly: it
replays the bars so far and, if the stop or the target has been touched, closes
the paper trade and sends an alert; otherwise it stays silent. Live intraday is
refused for now: it needs broker-side stop orders and a static IP.

**Does any of it work? The backtest.** `python automate.py intraday-backtest`
downloads Yahoo's 5-minute bars (about 60 days, its limit), saves them month by
month under `state/backtests/` (so the history grows with every run), and
replays each model with the same code the live job uses: at 09:50, 10:20, 10:50
and 11:20 it rebuilds only what was known at that moment, enters at the next
bar's open, and applies the setup's stop and target, the 15:15 square-off and
the costs. The report is `state/backtests/intraday_backtest.md`; each model's
record is also shown to Claude and printed under every pick. The report includes
a coin flip for comparison (every stock, a random side, the same stop and
target): a model is only worth trading if it clearly beats that line.

First run (17 Jul - 5 Oct 2026, 55 trading days, a period in which the Nifty
fell about 6%): **no model had an edge.** Breakouts, fair value gap retests and
Fibonacci pullbacks each lost roughly the trading costs per trade, the same as
the coin flip, and a higher score did not mean a better trade. Only the gap
reversal was not negative, and by less than its own uncertainty.
Treat the intraday job as an experiment in paper mode, not as a strategy that is
known to make money; re-read the report as the history grows.

**How the bot learns (the reason for the paper account).** One trade a day is
far too few to tell a good rule from luck, so the bot learns from every setup it
sees, not only the one it trades:

1. **At the entry** it writes the shortlist Claude saw, each setup's facts and
   Claude's decision to `state/journal/YYYY-MM-DD.json`.
2. **At the square-off** (`intraday-close`) it replays every shortlisted setup and
   all of the day's setups for every stock (the same replay as the backtest),
   records where Claude's pick ranked, and sends an **end-of-day review** on
   Telegram: today's trade, the shortlist's results, how each kind of setup did,
   the Nifty's move, and running totals (paper P&L, Claude's picks against the
   rest of the shortlist, each kind of setup since the journal began).
3. **Every week** (`intraday-backtest`) it rebuilds `state/learning/playbook.json`
   and `state/learning/learning_report.md` from all the saved history
   (`src/automation/learning.py`): each kind of setup against a coin flip on the
   same days; for each kind and each condition (with the daily trend, with the
   Nifty, VWAP side, volume, short, early, a big opening gap, RSI stretched, a
   tight stop, the setup's bonus) the setups with the condition against those
   without it on the same days; and whether holding to 15:15 beat the target.
4. **Lessons are adopted only on strong evidence:** a difference must be at least
   3 times its own uncertainty over 20+ days with 30+ setups on each side (2-3 is
   "watching"), and it is shrunk before use. Lessons in force re-rank tomorrow's
   setups (best expected result first), are shown to Claude, and can switch a kind
   of setup to holding until 15:15. They never change position size, the widest
   stop, or paper/live mode.
5. **A walk-forward check** in the weekly report learns each day only from the
   days before it and compares the playbook's pick with the plain ranking, so it
   is visible whether the learning actually helps.

**GitHub's schedule is not punctual.** GitHub starts scheduled workflows on a
best-effort basis; on this repository they have been starting 6-9 hours late.
The intraday entry must run between 09:45 and 11:30 IST, so start the jobs from
something that keeps time: an external scheduler that calls the workflow's
"Run workflow" API (`workflow_dispatch` with `job=intraday-open`, etc.), Windows
Task Scheduler with `gh workflow run`, or a small always-on server. The schedule
lines in the workflow stay as a silent safety net: a run started by GitHub's own
schedule passes `--quiet-unless-changed` and sends a message only if it actually
opened or closed a position (for example squaring off a trade the punctual run
missed).

**Paper vs live** (`AUTOTRADE_MODE`). Paper (default) keeps a practice account in
`state/paper_ledger.json`, starting from `SIMULATED_TOTAL_EQUITY`. Live places
real orders and needs every lock: `AUTOTRADE_MODE=live`, `DHAN_ENV=live`,
`DHAN_ENABLE_LIVE_TRADING=true`, Dhan credentials, and a machine whose **static
IP is whitelisted with Dhan** (a SEBI rule; Dhan rejects API orders from any
other IP). A live run missing any of these reports but does not trade. In live
mode the bot only ever sells positions it bought itself (`state/live_ledger.json`),
never shares you bought.

**Notifications.** `WHATSAPP_PROVIDER=telegram` (recommended, free, Telegram's
official Bot API): message @BotFather, send `/newbot`, and put the token it
gives you in `TELEGRAM_BOT_TOKEN`; open your new bot and press Start; then run
`python automate.py telegram-chat-id` and put the id it prints in
`TELEGRAM_CHAT_ID`. WhatsApp works too: `callmebot` (free, personal use; save
+34 611 021 695, send it "I allow callmebot to send me messages", and set
`WHATSAPP_PHONE` and `CALLMEBOT_API_KEY` from its reply) or `twilio`.

**Bull vs bear, then a judge (all three books).** Before each intraday trade,
Claude's pick is debated (`src/automation/debate.py`, settings in the `debate`
block of `config/trading_rules.json`): a bull Claude argues that the trade will
reach its target before its stop, a bear Claude that it will not, two rounds each,
and from round 2 each answers the other's points. Both get the same data packet
(the setup with its levels, indicators, volume, trend and the Nifty, its kind's
backtest record, the bot's learned experience, and the last hour of 5-minute
candles of the stock and the Nifty). Every point must quote one field of that
packet and its value, and a referee in code throws out any point whose field does
not exist or whose number does not match. A judge Claude then decides TRADE or
SKIP from the verified points only. On SKIP the next setups on the shortlist are
debated, up to three; if the judge approves none, there is no trade that day.
Entry, stop, target and the risk layer are unchanged, and if the debate itself
cannot run, Claude's pick is traded without it (with a warning). The 3:20 review
records how each approved and skipped setup would have ended, and the Saturday
scorecard reports the judge's record. Telegram shows each side's best verified
point and the verdict.

**Penny intraday (paper only, its own account).** The same intraday job also
runs on penny stocks (`src/automation/penny_book.py`), in the same 09:50 / hourly /
15:20 runs, with separate Telegram messages titled "Penny". Everything it keeps
is under `state/penny/`: its own paper account (`paper_ledger.json`, the same
starting cash), today's stock list (`stocks.json`), journal, backtests and
learned lessons, so it never mixes with the large-stock book. The list is
rebuilt each trading day from NSE's bhavcopy: EQ series only (the
trade-to-trade series cannot be traded intraday), priced Rs 2-50, the 50 most
traded by their 20-day average traded value (at least Rs 1 crore a day). The
`penny_intraday` block of `config/trading_rules.json` holds its settings: stops
up to 2% (penny stocks swing more), the target at 2x the stop, 0.30% charges per
trade (wider buy/sell spreads), and practice mode. A setup is skipped unless its
entry, stop and target sit inside the stock's daily circuit limits (from NSE's
list of price bands, 5% assumed if it cannot be read), because a stock frozen at
its limit cannot be sold. The Saturday job also re-tests the penny setups and
rebuilds the penny lessons (`state/penny/learning/`). A failure in the penny
book never affects the large-stock book.

**Index futures (paper only, its own account).** A third book runs the same
intraday job on **Nifty, Bank Nifty, Fin Nifty and Midcap Nifty Select futures**
(`src/automation/futures_book.py`),
in the same runs, with Telegram messages titled "Futures" and everything under
`state/futures/` (its own paper ledger, journal, backtests and lessons). Prices
are the index levels from Yahoo (a future trades very close to its index during
the day; Yahoo has no Indian futures prices). Trades are sized in **lots**: enough
lots that the loss at the stop is about 1% of the account (one lot is allowed up
to 2%), at most 3 lots, with the margin (12% of the contract value) capped at 60%
of the account. Lot sizes are read from NSE's list each entry run (65 for Nifty,
30 for Bank Nifty, 60 for Fin Nifty and 120 for Midcap Nifty are used if it cannot
be read). Only the margin is blocked
from the paper cash, while profit or loss is the whole contract's move: that is
the leverage, so a 0.5% index move on one Nifty lot is about Rs 8,000. Charges are
0.04% of the contract value. The settings are in the `futures_intraday` block of
`config/trading_rules.json`. Options are not traded.

**Daily picks table.** After each punctual 9:50 entry run, one more Telegram
message lists the three books' picks in a table (`src/automation/daily_picks.py`):
the pick (book, buy or sell, name, quantity and price), its stop-loss, its target,
and the profit if the target is hit after estimated charges. A book that did not
trade says why. GitHub's late backup runs do not send it.

**Weekly profit and loss.** The Saturday job first sends one table
(`src/automation/weekly_pnl.py`): each book's trades and profit or loss this week
(Monday to Saturday) and since its practice began, after estimated charges, with
the total of all three.

**Weekly scorecard.** Each book's Saturday message ends with a scorecard of its
own practice trades (`src/automation/scorecard.py`, saved as
`learning/scorecard.json`): four checks, each "met" or "not yet" - at least 50
trades; the average result after charges above zero by at least twice its
uncertainty; Claude's picks beating the average shortlisted setup the same way
(over 30+ picks); and the worst drop at most 10% of the account - plus about how
many more trades the current average would need. It only reports; it never
changes how the bot trades.

**Dhan token.** Holdings are read from Dhan (read-only) when `DHAN_ACCESS_TOKEN`
is valid. The token lasts 24 hours; when it has expired the report says so and
the run carries on without holdings.

---

## Tests

```bash
pytest -q
```

No network, no LLM. `tests/test_guardrails.py` asserts positions never exceed
the position cap, oversized proposals are downsized, stop-loss/take-profit are appended
correctly, HOLD produces no order, and unsafe inputs raise a hard `RiskViolation`.
`tests/test_indicators.py` checks the Fibonacci and fair value gap maths on
synthetic candles. `tests/test_options.py` covers options mode layer by layer
(option chain parsing, the analyst's output, every options risk rule, the
NSE_FNO payload and the live lock) using a fixture chain and fake clients.
`tests/test_analyst_replies.py` checks that small JSON slips in Claude's reply
(such as trailing commas or stray invisible characters) are forgiven, and that
an unusable reply gets exactly one corrective retry before the run stops.
`tests/test_fundamentals.py` checks the fundamentals maths and units (CAGR,
debt-to-equity, dividend yield, ROE from the annual accounts, operating-profit
growth, the flags including one-off detection) on a fixture company, that a
failing Yahoo call never stops a run, that both analysts receive the
fundamentals, and that the results blackout blocks new shares and options
trades (but not HOLD).
`tests/test_automation.py` covers the automation end to end with fakes: the
penny universe, screening scores, the picker's reply, the ledger's exits and
cash, WhatsApp messages, paper and live runs (including that a live run on a
machine without a static IP never trades) and the report.
`tests/test_intraday.py` covers the intraday measurements, the four setup models
(breakout, fair value gap retest, Fibonacci pullback, gap reversal) with their
stops, the bar-by-bar exit replay (long and short), short positions and charges
in the ledger and the intraday runs.
`tests/test_intraday_backtest.py` covers the backtest: the trade replay, the
statistics (uncertainty is measured on daily averages), the saved data files and
that the backtest never uses a bar from after the decision time.

---

## Notes specific to the Indian market

- **CNC (delivery) can't be shorted overnight**, so `allow_short_selling` is
  `false` by default — a `SELL` signal is rejected by the risk layer. To act on a
  bearish view, use options mode (`--options`), which buys a put instead.
- **SL/TP for CNC** are provided as a *risk plan* (reference/stop/target prices)
  printed with every order. Placing them automatically as GTT/exit orders is a
  documented extension point. For the `BO` product they attach to the order
  directly (`boStopLossValue` / `boProfitValue`), but `BO` is intraday only.
- **Dhan needs a `securityId`**, resolved automatically from Dhan's public scrip
  master (cached under `data/`, refreshed daily).

## Safe-deployment playbook

1. **Stay in dry-run** until you've read the payloads and agree with them.
2. **Sandbox next.** Submit to Dhan sandbox and confirm the round-trip before
   ever touching live.
3. **Treat `trading_rules.json` as immutable.** Change limits only via reviewed
   commits; never let the model rewrite it.
4. **The risk layer is the boundary.** Keep Layer 3 free of any AI calls.
5. **Add monitoring & kill-switches** before live: position reconciliation,
   daily loss limits, a manual halt. These are *not* included here.
6. **Understand the risk.** This is a learning scaffold, not a production trading
   system, and comes with no warranty.
