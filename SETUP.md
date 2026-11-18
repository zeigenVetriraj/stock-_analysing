# Getting started

This bot **paper-trades** Indian markets every weekday: large stocks, penny stocks and index
futures (Nifty, Bank Nifty, Fin Nifty, Midcap Nifty), one practice trade a day in each, with its
own ₹10 lakh virtual account per section. Claude picks a setup, a bull Claude and a bear Claude
debate it, and a judge Claude decides whether to take it. Every evening it reviews the day, and
every Saturday it re-tests its setups and learns. Reports arrive on Telegram.

**It never places real orders.** Paper mode is fixed in the GitHub workflow. Nothing here is
financial advice.

Setting it up on GitHub takes about 30 minutes. You do not need a server, and you do not need a
broker account.

---

## 1. Prerequisites

### Accounts you need

| What | Why | Cost |
|---|---|---|
| **GitHub account** | The bot runs on GitHub Actions, on a schedule | Free. A private repo gets about 2,000 Actions minutes a month; the bot uses roughly 500-800 |
| **Anthropic API key** ([console.anthropic.com](https://console.anthropic.com)) | Claude makes the daily pick and runs the bull vs bear debate | Pay as you go. Each weekday morning makes roughly 10-50 Claude calls across the three sections (more when the judge skips and the next setup is debated). Set a monthly spend limit in the console |
| **Telegram** and a bot from **@BotFather** | Where the reports are sent | Free |
| **cron-job.org account** (recommended) | Starts the runs on time. GitHub's own schedule is often hours late | Free |

### Not needed

- **A broker account (Dhan or any other).** Prices come from Yahoo Finance, and trades are kept in a
  paper ledger inside the repo. The Dhan settings in `.env.example` are only for the manual
  command-line tools in `main.py`.
- **A computer that stays on.** Everything runs on GitHub.

### Only for running it on your own computer (optional)

- **Python 3.11 or newer** (GitHub uses 3.13).
- The packages in `requirements.txt`: `pip install -r requirements.txt`.

---

## 2. Set it up on GitHub

### Step 1: make your own copy

Create **your own private repository** from this one: on its GitHub page, click
**Use this template → Create a new repository** and choose **Private**. Or clone it and push it to
a new private repo of yours.

Use your own copy, because the bot commits its paper ledger, journal and lessons to the repo it
runs in.

### Step 2: create the Telegram bot

1. In Telegram, open **@BotFather**, send `/newbot` and follow the steps. It gives you a **bot
   token**. Keep it secret.
2. Open your new bot and press **Start** (send it any message).
3. Find your **chat id** in either of these ways:
   - on your computer, after step 4 of section 3: `python automate.py telegram-chat-id`;
   - or open `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates` in a browser and copy
     `"chat":{"id": ...}`.

### Step 3: add the secrets

In your repo: **Settings → Secrets and variables → Actions → Secrets → New repository secret**.

| Secret | Value |
|---|---|
| `ANTHROPIC_API_KEY` | Your Anthropic API key |
| `TELEGRAM_BOT_TOKEN` | The token from @BotFather |
| `TELEGRAM_CHAT_ID` | Your chat id |

Never put keys or tokens in any file in the repo. They belong in these secrets only.

### Step 4: switch the bot on

1. **Settings → Secrets and variables → Actions → Variables → New repository variable**:
   name `BOT_ENABLED`, value `true`. Until this is set, the workflow does nothing.
2. **Settings → Actions → General**: allow GitHub Actions. Under **Workflow permissions**, choose
   **Read and write permissions**, so the bot can save its ledger.

### Step 5: test it

**Actions → Trading bot (paper) → Run workflow**, job `intraday-check`. It is safe at any time:
with no open trade it only checks. The run should finish green. Then try `report` to get a first
Telegram message.

Do **not** start `intraday-close` by hand during market hours: it squares off the day's trades.

### Step 6: run on time with cron-job.org (recommended)

GitHub's own schedule in the workflow is only a backup, because it often starts hours late.

1. **Create a GitHub token** for cron-job.org: GitHub → **Settings → Developer settings → Personal
   access tokens → Fine-grained tokens → Generate new token**. Repository access: **only your copy
   of this repo**. Permissions: **Actions: Read and write**. Nothing else.
2. In cron-job.org, create **three jobs**. Everything is the same except the title, the schedule
   and the request body:

   - **URL** (replace `OWNER/REPO` with your repo):
     `https://api.github.com/repos/OWNER/REPO/actions/workflows/trading-bot.yml/dispatches`
   - **Request method:** `POST`
   - **Time zone:** `Asia/Kolkata`
   - **Headers:**

     | Key | Value |
     |---|---|
     | `Authorization` | `Bearer ` followed by your token |
     | `Accept` | `application/vnd.github+json` |
     | `X-GitHub-Api-Version` | `2022-11-28` |
     | `Content-Type` | `application/json` |

   | Title | Crontab expression | Request body |
   |---|---|---|
   | Bot entry | `50 9 * * 1-5` | `{"ref":"main","inputs":{"job":"intraday-open"}}` |
   | Bot checks | `3 11-14 * * 1-5` | `{"ref":"main","inputs":{"job":"intraday-check"}}` |
   | Bot close | `20 15 * * 1-5` | `{"ref":"main","inputs":{"job":"intraday-close"}}` |

3. Make sure each job's **Enable job** switch is on. A test run of *Bot checks* should answer
   **204**. A **401** means the token is wrong or expired.

The Saturday review (`intraday-backtest`) can stay on GitHub's own schedule; its timing does not
matter.

---

## 3. Run it on your own computer (optional)

```bash
git clone <your copy>
cd <repo>
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # then fill in ANTHROPIC_API_KEY and the Telegram values
python automate.py telegram-chat-id
python -m pytest -q                # the tests use fake data: no network, no Claude calls
python automate.py intraday-check  # any job name from the table above
```

Set `WHATSAPP_PROVIDER=telegram` in `.env` to get messages. Local runs keep their ledger in
`state/local/`, which is never committed.

---

## 4. What happens every day

| When (IST) | What |
|---|---|
| 9:50 | Each section scans for setups. Claude picks one, bull and bear debate it, the judge decides. Then a summary table of the three picks |
| 11:03-14:03, hourly | Checks open trades; a message only if a stop-loss or target is hit |
| 15:20 | Squares off at the 15:15 price, replays every setup of the day, and sends the day's review |
| Saturday | Weekly profit and loss, re-test of all setups on about 3 months of data, new lessons, and a scorecard per section |

Everything the bot keeps is in `state/` (created on the first runs): `state/` for large stocks,
`state/penny/` and `state/futures/`, each with its paper ledger, daily journal, backtests and
lessons.

---

## 5. Settings you may want to change

All in `config/trading_rules.json`. Each block has a `_comment` that explains it.

| Block | What it controls |
|---|---|
| `intraday` | Large stocks: stop and target, entry window, charges, practice mode |
| `penny_intraday` | Penny stocks: price range, number of stocks, stop size, circuit check |
| `futures_intraday` | Index futures: lot sizes (read from NSE each day), margin, risk per trade, maximum lots |
| `debate` | Bull vs bear: `enabled`, `rounds` (2), `max_candidates` (3) |

Set `"enabled": false` in a block to switch that section off. `config/universe.json` lists the
large stocks.

---

## 6. Safety

- **Paper only.** The workflow fixes `AUTOTRADE_MODE: paper` and `DHAN_ENABLE_LIVE_TRADING: "false"`.
  Keep it that way.
- **Never commit `.env`** or any key. It is listed in `.gitignore`.
- Give the cron-job.org token access to **one repo** and **Actions only**. If it leaks, delete it on
  GitHub and make a new one.
- Results so far are early and unproven. Read the weekly scorecard before drawing conclusions.
