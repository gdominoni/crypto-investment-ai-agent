# How to Run This Project on Your Own Computer

A step-by-step guide with no assumed prior knowledge of this codebase. If you can open a terminal and copy-paste a command, you can follow this. Every step says exactly what to type and what you should see if it worked.

This project never opens a funded position — nothing in these steps risks any money. Nothing here touches an exchange account, a wallet, or a broker.

---

## What you'll need before starting

- **A computer** running macOS, Linux, or Windows (with [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) — plain Windows terminal is not tested).
- **Python 3.11 or newer.** Check what you have with `python3 --version`. If that fails or shows an older version, install Python from [python.org](https://www.python.org/downloads/).
- **Git.** Check with `git --version`. If missing, install from [git-scm.com](https://git-scm.com/downloads).
- **An Anthropic API key** (for Claude Haiku/Sonnet) — free to create, pay-as-you-go, get one at [console.anthropic.com](https://console.anthropic.com/). A few dollars of credit is enough to explore this project for weeks (see [`PROJECT_MAP.md`](PROJECT_MAP.md)'s "Cost Optimization" section for real measured numbers).
- **A Telegram account** and 10 minutes to create a bot (instructions below — no coding involved, just chatting with another bot).
- **~15 minutes** and about 500 MB of free disk space (mostly for one optional dependency, Freqtrade — see Step 3).

You do **not** need a server, a cloud account, or a credit card beyond the Anthropic key above. Everything runs on your own machine.

---

## Step 1 — Get the code

```bash
git clone https://github.com/gdominoni/crypto-investment-ai-agent.git
cd crypto-investment-ai-agent
```

The clone lands on `main`, which is this project. (An earlier version of these
instructions asked you to `git checkout sentiment-agent-rebuild`; that branch was
renamed to `main` on 2026-09-05 and the checkout now fails. The multi-module
system that used to occupy `main` is preserved on `multi-module-agent`.)

**Check it worked:** running `ls` should show, among others, `README.md`, `candidates/`, `telegram/`, `scheduler/`.

---

## Step 2 — Create an isolated Python environment

This keeps this project's dependencies separate from anything else Python-related on your computer. It's optional but strongly recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate        # on Windows (WSL): same command
```

**Check it worked:** your terminal prompt should now start with `(.venv)`. You'll need to re-run the `source` command every time you open a new terminal window to work on this project — that's normal.

---

## Step 3 — Install the dependencies

```bash
pip install -r requirements.txt
```

This installs everything the project needs: `pandas`, `numpy`, `anthropic` (Claude's SDK), `ccxt` (exchange data), `python-dotenv`, and `freqtrade` (used only for the optional hyperopt cross-check described in Phase 3 of the README — it's the slowest part of this install, several minutes and a few hundred MB, but nothing else in this project depends on it, so it's safe to let it run in the background).

**Check it worked:** `python3 -c "import pandas, anthropic, ccxt; print('ok')"` should print `ok` with no errors.

---

## Step 4 — Get your API keys

You need three pieces of information. None of them cost anything to create.

### 4a. Anthropic API key
1. Go to [console.anthropic.com](https://console.anthropic.com/), sign up or log in.
2. Add a small amount of credit (a few dollars is plenty to start — see the cost note in Step 1's prerequisites).
3. Go to **API Keys** → **Create Key**. Copy the value starting with `sk-ant-...`.

### 4b. A Telegram bot token
1. Open Telegram (app or web) and search for the user **@BotFather** (this is Telegram's own official bot for creating other bots).
2. Send it `/newbot` and follow the prompts (pick any name and a unique username ending in `bot`).
3. BotFather replies with a token that looks like `123456789:AAExxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`. Copy it.

### 4c. Your Telegram chat ID
This tells the bot which conversation to send messages to (yours).
1. In Telegram, search for the bot you just created (by the username you picked) and send it any message, e.g. `hello`.
2. In your browser, open this URL, replacing `<TOKEN>` with the token from step 4b:
   `https://api.telegram.org/bot<TOKEN>/getUpdates`
3. You'll see a block of JSON text. Find `"chat":{"id":` — the number right after it is your chat ID (it may be negative, e.g. `-123456789` — copy it exactly, including the minus sign if present).

If the page is empty (`{"ok":true,"result":[]}`), you sent the message to the wrong bot or too long ago — send it another message and reload the URL.

*(There used to be a fourth item here, a CryptoCompare key for the news feed.
The headline-screening layer it fed was removed — see "Scope, and one component
removed for it" in the README — so there is nothing left to configure.)*

---

## Step 5 — Configure your `.env` file

```bash
cp .env.example .env
```

Open the new `.env` file in any text editor and fill in the values you collected in Step 4:

```
ANTHROPIC_API_KEY=sk-ant-...
TELEGRAM_BOT_TOKEN=123456789:AAE...
TELEGRAM_CHAT_ID=-123456789
FREQTRADE_DB_PATH=execution/tradesv3.sqlite
```

Leave `FREQTRADE_DB_PATH` exactly as shown. **Never share this file or commit it to git** — it's already listed in `.gitignore` so a normal `git add`/`git commit` won't pick it up.

---

## Step 6 — Verify the setup works

```bash
python3 -m pytest tests/ -q
```

**Check it worked:** you should see a line like `253 passed in 40s` with no `FAILED` or `ERROR` lines. This runs entirely offline — it does not call Anthropic or Telegram, so it can't cost anything or send you a message. If anything fails here, stop and re-check Steps 2–3 before continuing (a failure this early is almost always a missing dependency, not a real code bug).

---

## Step 7 — Bring market data up to date

The repository already ships with years of historical price and macro data (nothing to download from scratch). This step just catches it up to today:

```bash
python3 -m data_ingestion.market_data.binance_fetcher
```

**Check it worked:** you'll see one line per coin (BTC, ETH, BNB, XRP, DOGE, ADA, LTC), each ending in something like `+3 daily candle(s), +72 funding entry(ies)`. This calls Binance's public API only — no account or key needed, and it's free.

---

## Step 8 — Run it

There are two parts. **Part 2 is the actual project** — Part 1 is optional and you can skip straight to Part 2 if you'd rather start from a clean slate and just watch it run forward from today.

### Part 1 (optional) — Historical replay: backfill years of results in minutes

This replays years of real historical data (from 2017 onward) day by day, as if it were happening live, running the exact same statistical/LLM pipeline Part 2 uses and sending the same kind of Telegram messages the live system would have sent at the time. The point of running this first is speed: instead of waiting hourly/weekly for real conditions to trigger and results to accumulate one at a time, this gives you years of tracked patterns, tested conditions, and resolved live tests in one sitting — a fully populated `/replay_summary` almost immediately, instead of an empty one you'd otherwise have to wait months to fill in for real.

```bash
python3 -m replay.orchestrator
```

**Check it worked:** the terminal prints one line per simulated step, and your Telegram bot starts receiving messages (proposed conditions awaiting your Test It / Don't Test It, a monthly digest of what the live tests did, confirmation checkpoints, and periodic check-ins). Individual live tests are deliberately NOT sent one by one — a full replay opens over 20,000 of them. This does call the real Anthropic API repeatedly, so it does cost real (small) amounts of credit — see [`PROJECT_MAP.md`](PROJECT_MAP.md)'s "Cost Optimization" section for real measured per-call costs. Stop it any time with `Ctrl+C`; it checkpoints its progress and resumes from where it stopped the next time you run it. This never touches live state — it's entirely isolated (`/replay_summary` vs. `/summary` on Telegram), so running it has no effect on Part 2 below, whether you run it before, after, or never.

### Part 2 — Go live (the real thing)

This is the one command that runs the whole system: the Telegram bot, the hourly scans, and the weekly re-validation, all in a single process, with no separate cron job or scheduler to configure.

```bash
python3 -m scheduler.live_daemon
```

**Check it worked:** within a few seconds you should receive a Telegram message from your bot saying *"Live daemon started."* From here on, open your bot's chat and send `/help` to see every available command (including `/summary`, a plain-language snapshot of every pattern currently being tracked). Leave this terminal window open (or run it in the background, e.g. inside `tmux`/`screen`, or via `nohup python3 -m scheduler.live_daemon &`) — it needs to keep running to keep scanning and to answer you on Telegram. Stop it any time with `Ctrl+C`; restarting it later picks up exactly where it left off (it remembers when each job last ran).

---

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| The test `test_messages_create_accepts_every_kwarg_the_project_passes` fails | Your installed `anthropic` SDK does not accept an argument this project passes. The project deliberately passes nothing version-specific — `temperature` was removed after the API began rejecting it, and the code is verified against both 0.x and 1.x — so this most likely means the virtual environment from Step 2 is not active and a different copy of the library is being picked up. Activate it and re-run Step 3. |
| `ModuleNotFoundError: No module named 'anthropic'` (or similar) | Your virtual environment isn't active. Run `source .venv/bin/activate` again (Step 2), or re-run Step 3. |
| `KeyError: 'ANTHROPIC_API_KEY'` or similar at startup | `.env` is missing, misnamed, or in the wrong folder. It must be named exactly `.env` and sit in the project's root folder (Step 5). |
| No message ever arrives on Telegram | Double-check `TELEGRAM_CHAT_ID` (Step 4c) — a wrong ID fails silently from Telegram's side. Also confirm you sent your bot at least one message before running `getUpdates`. |
| `anthropic.BadRequestError: ... credit balance is too low` | Your Anthropic account needs more credit — add some at [console.anthropic.com](https://console.anthropic.com/) (Step 4a). No progress is lost; both run modes checkpoint and resume. |
| Freqtrade install fails or hangs in Step 3 | It's only needed for the optional hyperopt cross-check (see README's Phase 3). Both `Part 1` and `Part 2` above work without it — you can install everything else first (`pip install -r requirements.txt --no-deps` is not recommended; instead just let it finish, or remove the `freqtrade[hyperopt]` line from `requirements.txt` if you don't need that feature). |
| Tests fail in Step 6 | Almost always a dependency issue, not a real bug — re-check Steps 2 and 3 (right Python version, virtual environment active, `pip install` completed without errors). |

If something goes wrong that isn't listed here, [`PROJECT_MAP.md`](PROJECT_MAP.md) has a "Partial Failures & Crashes" section describing exactly how this system is designed to fail loudly (a Telegram alert) rather than silently — check your bot's chat for an alert message before assuming something is broken.
