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

Measured, not estimated — and measured twice, because the two numbers disagree in a way worth knowing about. The weekly battery profiled on its own (159 candidates, every write redirected to a temp dir) peaks at **140 MB RSS** over about 20 minutes. The full daemon on a real host reported peaks of **350–410 MB** to systemd across its first few hourly cycles. The gap is everything the isolated run does not carry: the long-poll loop, the API clients, and the loaded frames held between jobs.

Trust the larger one. It is the figure from the thing you are actually going to run.

| Resource | Needed | Why |
|---|---|---|
| RAM | **1 GB** | 350–410 MB observed on a live host during ordinary hourly cycles, before the weekly battery has even run. 512 MB is not the safe choice it looks like; 1 GB costs a euro or two more and removes the question. |
| Disk | **5 GB** | Repo, data and state are ~1 GB together. The rest is headroom for the venv, the journal, and years of slow market-data growth. |
| CPU | 1 shared vCPU | The daemon sleeps through most of every hour. The weekly battery is the only sustained burn — ~20 min on a laptop, so budget up to an hour on a small shared vCPU. Once a week, and nothing waits on it. |
| Bandwidth | Negligible | Incremental Binance fetches and Telegram long-polls. |

Any provider's smallest tier meets this comfortably. The daemon is not latency-sensitive and holds no exchange connection, so the region only matters for your own SSH comfort.

> **On cost:** the server is the *predictable* expense; the Anthropic API is the variable one. See `PROJECT_MAP.md`'s Cost Optimization for measured per-call figures — calls happen only during compression episodes, so quiet weeks cost nothing.

---

## 0. If you are using Oracle Cloud, read this first

Oracle's Always Free tier is genuinely free and generously sized, but it has one policy that lands directly on a system shaped like this one.

**Oracle reclaims idle Always Free instances.** An instance counts as idle if, over a rolling 7-day window, CPU utilisation at the 95th percentile is under 20%, network utilisation is under 20%, and — on Ampere A1 shapes — memory utilisation is under 20%. This daemon sleeps between hourly scans and burns real CPU for about twenty minutes a week. It is exactly the profile that policy is written to catch.

**The fix is to upgrade the account to Pay As You Go.** Always Free allowances remain free after upgrading — you are charged only for usage beyond them — and instances are no longer subject to idle reclamation. It requires a payment method on file. Set a budget alert at a low threshold if that makes you more comfortable; nothing this project runs touches a paid resource.

Whatever you decide, **configure the heartbeat in step 3**. Reclamation is not the only way a host can die, and this daemon cannot tell you it has stopped — see that step for why silence is the one failure it cannot report.

Two smaller things:

- **Shape.** `VM.Standard.A1.Flex` (Ampere, ARM) is the better free shape by a wide margin — 4 OCPU and 24 GB against the AMD micro's single core and 1 GB. ARM is fine here: pandas, numpy and pyarrow all ship `aarch64` wheels, and everything else is pure Python. Ask for a small slice (1 OCPU, 6 GB is already luxurious against a 140 MB peak). If you hit **"Out of host capacity"**, that is a well-known A1 shortage in busy regions, not a mistake on your part — try another availability domain, another region, or retry later.
- **Networking.** Nothing needs to reach this host from outside except your own SSH. The daemon only makes outbound connections (Telegram, Binance, Anthropic), so you can leave the default security list alone and skip opening any port.

The default login user on Oracle's Ubuntu images is `ubuntu`, not `root`.

---

## 1. Create the box

Any provider, smallest tier, **Ubuntu 24.04 LTS**. Add your SSH key during creation rather than using a password.

The daemon runs as your ordinary login user, not as root. On a machine that exists only to run this, a dedicated account buys nothing over the unprivileged user the image already gives you — and the systemd unit narrows it further, allowing writes to the project directory and nowhere else.

---

## 2. Set the host up

SSH in, then run this once:

```bash
curl -fsSL https://raw.githubusercontent.com/gdominoni/crypto-investment-ai-agent/main/deploy/bootstrap.sh | bash
```

It installs the system packages, clones the repo into `~/crypto-agent`, builds a virtualenv, installs the dependencies, verifies the daemon actually imports, and writes a systemd unit with this machine's real user and paths already filled in. It is safe to re-run, and it deliberately does not start anything — steps 3 and 4 still have to happen first.

Piping a script into `bash` deserves a look before you run it, so: [`deploy/bootstrap.sh`](bootstrap.sh), or `curl -fsSLO <url>` and read it locally.

<details>
<summary>Or do it by hand</summary>

```bash
sudo apt update && sudo apt install -y python3-venv python3-pip git
git clone https://github.com/gdominoni/crypto-investment-ai-agent.git ~/crypto-agent
cd ~/crypto-agent
python3 -m venv .venv
.venv/bin/pip install pandas numpy pyarrow requests anthropic python-dotenv ccxt
```

Then edit `deploy/crypto-agent.service` — `User`, `WorkingDirectory`, `ExecStart`, `ReadWritePaths` — and copy it to `/etc/systemd/system/`. Generating it is what the script does; a wrong `WorkingDirectory` fails in a way that reads like a code bug.
</details>

That package list is the deliberate one — the lean install, not `pip install -r requirements.txt`. It omits `freqtrade` and `scipy` for the reason above. Installing everything also works and just wastes a few hundred MB.

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

Then one optional fourth line, which is the only thing that can tell you the host has died:

```
HEARTBEAT_URL=https://hc-ping.com/<your-check-uuid>
```

Everything else in this system reports its own failures, and that works because something is still alive to do the reporting. The host dying is the one case that breaks — it takes the messenger along with the message. The resulting silence is indistinguishable from a quiet week, which here is the normal, healthy state, so you would not notice for a long time.

Only an outside observer can catch that, and it has to work by *expecting* a signal rather than watching for one. Create a check on any dead-man's-switch service ([healthchecks.io](https://healthchecks.io) has a free tier; several others do too), paste its ping URL here, and set it to alert you if it goes quiet for a few hours. The daemon pings it after each completed hourly cycle — after the real work, so it vouches for a cycle that actually finished rather than merely for a process that is running.

Leave the line out and nothing happens: the system runs exactly the same, with no third-party account required. On Oracle Cloud in particular, do not leave it out.

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
    ubuntu@<host>:crypto-agent/candidates/
scp execution/live_tests.json execution/live_battery_state.json \
    execution/confirmation_priors.json execution/parked_proposals.json \
    execution/horizons.json \
    ubuntu@<host>:crypto-agent/execution/
scp scheduler/previous_status.json ubuntu@<host>:crypto-agent/scheduler/
scp llm_pipeline/pending_test.json ubuntu@<host>:crypto-agent/llm_pipeline/
scp compression_escalated.json llm_usage.json ubuntu@<host>:crypto-agent/
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

## 6. Start it

The unit is already installed — step 2 wrote it with this machine's real paths. All that is left:

```bash
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
| Check for a second bot | `journalctl -u crypto-agent \| grep 409` |
| Stop | `sudo systemctl stop crypto-agent` |
| Update the code | `git pull && sudo systemctl restart crypto-agent` |
| Re-check health | `python3 -m deploy.preflight` |

**Restarting is safe.** Last-run timestamps persist in `scheduler/live_daemon_state.json`, so a restart does not re-fire jobs that already ran, and does not lose the schedule.

**Only one machine may run the bot.** Telegram allows a single poller per token: a second one — the copy still running on the laptop you deployed from, typically — makes both fail with `409 Conflict`, and incoming commands are then split arbitrarily between them, with the stale copy acting on state it no longer owns. The daemon logs the conflict and keeps retrying rather than dying, so the symptom is a bot that answers only sometimes. Stop the other one; `journalctl -u crypto-agent | grep 409` confirms which situation you are in.

**You do not need to watch it.** Every scheduled job is isolated: one failing sends a Telegram alert naming it and is retried on its next normal cycle, without taking the daemon down. If the *process* dies, systemd restarts it after 30 seconds — and gives up after 5 failures in 10 minutes, because a daemon crash-looping on a bad config will not fix itself by trying harder.

**Silence from the bot means nothing is wrong — with one exception.** Quiet is the normal, healthy state here: the message set was deliberately trimmed to the few things worth interrupting you for. The exception is that a dead *host* is also silent, and looks identical. That is the entire job of the `HEARTBEAT_URL` from step 3, and the reason to bother setting it up: it converts an absence, which humans do not notice, into an alert, which they do.

**Once a month** it may ask you to run the hyperopt cross-check on your own machine, with the command already filled in. That is the only thing it ever needs a human for.
