# Putting the agent on a server

The live daemon has to keep running to be worth anything: it scans hourly, and a laptop that closes its lid stops scanning. This is how it moves to a machine that does not.

Nothing here is exotic. One small Linux box, one systemd service, one file of secrets. Budget about half an hour.

---

## What the host actually has to do

Less than you would guess, because the expensive work does not live here.

| | |
|---|---|
| **Runs on the host** | The Telegram bot, the hourly market-data refresh and scans, the daily parked-proposal re-check, the weekly re-validation, the monthly digest. |
| **Stays on your own machine** | The Freqtrade hyperopt cross-check — the one genuinely heavy computation in the project. See [HOW_TO_RUN.md](../HOW_TO_RUN.md)'s Part 3. |

That split is what keeps the host small. `freqtrade` and `scipy` are in `requirements.txt` but **nothing the daemon runs imports either of them** — they belong to the local cross-check and to the offline analyses in `forecast/`. Installing them anyway costs several hundred MB of disk and buys the host nothing.

### Sizing

Measured, not estimated. The heaviest thing the host ever does is the weekly battery — it loads daily *and* hourly history for all 7 coins and scores every tracked condition — so that run was profiled directly: **159 candidates, peak RSS 140 MB, about 20 minutes.**

| Resource | Needed | Why |
|---|---|---|
| RAM | **1 GB** | The measured peak is 140 MB, most of it the Python + pandas baseline rather than the data. 512 MB genuinely works; 1 GB costs a euro or two more and removes the question. |
| Disk | **5 GB** | Repo, data and state are ~1 GB together. The rest is headroom for the venv, the journal, and years of slow market-data growth. |
| CPU | 1 shared vCPU | The daemon sleeps through most of every hour. The weekly battery is the only sustained burn — ~20 min on a laptop, so budget up to an hour on a small shared vCPU. Once a week, and nothing waits on it. |
| Bandwidth | Negligible | Incremental Binance fetches and Telegram long-polls. |

Any provider's smallest tier meets this comfortably. The daemon is not latency-sensitive and holds no exchange connection, so the region only matters for your own SSH comfort.

> **On cost:** the server is the *predictable* expense; the Anthropic API is the variable one. See `PROJECT_MAP.md`'s Cost Optimization for measured per-call figures — calls happen only during compression episodes, so quiet weeks cost nothing.

---

## 1. Create the box

Any provider, smallest tier, **Ubuntu 24.04 LTS**. Add your SSH key during creation rather than using a root password.

Then, as root:

```bash
adduser --disabled-password --gecos "" agent
rsync --archive --chown=agent:agent ~/.ssh /home/agent/
apt update && apt install -y python3-venv python3-pip git
```

The daemon runs as `agent`, not root. It holds two API credentials and polls a public network service; there is no reason for it to be able to touch the rest of the system.

---

## 2. Get the code and its dependencies

As `agent` (`ssh agent@<host>`):

```bash
git clone https://github.com/gdominoni/crypto-investment-ai-agent.git crypto-agent
cd crypto-agent
python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy pyarrow requests anthropic python-dotenv ccxt
```

That is the deliberate install — the lean one, not `pip install -r requirements.txt`. It omits `freqtrade` and `scipy` for the reason above. If you would rather not think about it, installing everything works fine and just wastes disk.

---

## 3. Secrets

```bash
nano .env
```

The same three keys as on your laptop (`HOW_TO_RUN.md` Step 5 explains where each comes from):

```
ANTHROPIC_API_KEY=sk-ant-...
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
```

```bash
chmod 600 .env
```

`.env` is gitignored and must never be committed. Type it on the host; do not paste it through anything that keeps a log.

---

## 4. Move the history across

Skip this only if you want the host to start from nothing.

The runtime state is deliberately **not** in git — it is a live, continuously-rewritten record, not a repo artefact — so `git clone` gives you the code with an empty history. From your laptop:

```bash
scp candidates/dynamic_candidates.json candidates/status_history.json \
    agent@<host>:crypto-agent/candidates/
scp execution/live_tests.json execution/live_battery_state.json \
    execution/confirmation_priors.json execution/parked_proposals.json \
    execution/horizons.json \
    agent@<host>:crypto-agent/execution/
scp scheduler/previous_status.json agent@<host>:crypto-agent/scheduler/
scp llm_pipeline/pending_test.json agent@<host>:crypto-agent/llm_pipeline/
scp compression_escalated.json llm_usage.json agent@<host>:crypto-agent/
```

Around 12 MB in total, most of it `live_tests.json`. A file that does not exist on your laptop simply has nothing to send yet — `scp` will say so, and that is not an error.

Two of these look skippable and are not:

- **`previous_status.json`** is the baseline the weekly keep-or-drop review diffs against. Leave it behind and the first weekly cycle on the host reports *every* tracked condition as a fresh status change.
- **`horizons.json`** is each condition's evaluation horizon. It gets re-derived on the next battery run, but until then the host is reading horizons it does not have.

Deliberately **not** copied: `scheduler/live_daemon_state.json`. It records when each job last ran, and it belongs to the machine that ran them. Letting the host start with a clean schedule is the intent.

---

## 5. Check before you start

This is the step worth not skipping. A misconfigured daemon starts perfectly, announces itself on Telegram, and does nothing useful — stale market data and un-migrated state both look exactly like a healthy system.

```bash
source .venv/bin/activate
python3 -m data_ingestion.market_data.binance_fetcher   # catch the data up first
python3 -m deploy.preflight
```

`preflight` reads only. It verifies Python's version, the packages, the three secrets, that the Telegram token really works, that both the daily *and hourly* candle files are current, that the state actually arrived, the system clock, and disk. Every line must say `PASS` before you continue.

---

## 6. Install the service

```bash
sudo cp deploy/crypto-agent.service /etc/systemd/system/
sudo nano /etc/systemd/system/crypto-agent.service   # check WorkingDirectory and ExecStart paths
sudo systemctl daemon-reload
sudo systemctl enable --now crypto-agent
```

`enable` is what survives a reboot; `--now` starts it immediately. Within seconds Telegram should say **"Live daemon started."**

```bash
systemctl status crypto-agent
journalctl -u crypto-agent -f      # follow the log; Ctrl+C stops following, not the daemon
```

---

## Living with it

| | |
|---|---|
| Follow the log | `journalctl -u crypto-agent -f` |
| Look back | `journalctl -u crypto-agent --since "2 hours ago"` |
| Restart | `sudo systemctl restart crypto-agent` |
| Stop | `sudo systemctl stop crypto-agent` |
| Update the code | `git pull && sudo systemctl restart crypto-agent` |
| Re-check health | `python3 -m deploy.preflight` |

**Restarting is safe.** Last-run timestamps persist in `scheduler/live_daemon_state.json`, so a restart does not re-fire jobs that already ran, and does not lose the schedule.

**You do not need to watch it.** Every scheduled job is isolated: one failing sends a Telegram alert naming it and is retried on its next normal cycle, without taking the daemon down. If the *process* dies, systemd restarts it after 30 seconds — and gives up after 5 failures in 10 minutes, because a daemon crash-looping on a bad config will not fix itself by trying harder. Silence from the bot means nothing is wrong; that is the design.

**Once a month** it may ask you to run the hyperopt cross-check on your own machine, with the command already filled in. That is the only thing it ever needs a human for.
