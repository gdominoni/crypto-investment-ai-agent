# Crypto Pattern-Discovery Agent: Can Macro Events and Real-Time Market Conditions Reveal Statistically Real, Repeatable Patterns in Crypto Prices?

![Python](https://img.shields.io/badge/Python-3.11+-blue) ![Claude](https://img.shields.io/badge/Claude-Sonnet-6B4FBB) ![Telegram](https://img.shields.io/badge/Telegram-human--in--the--loop-26A5E4) ![Freqtrade](https://img.shields.io/badge/Freqtrade-hyperopt%20cross--check-orange) ![Status](https://img.shields.io/badge/Status-Case%20Study-blue)

<div align="center">
<table width="82%">
<tr>
<td align="center">
<br>
<b>TL;DR — Key Engineering Highlights</b>
<br><br>
<sub><b>ARCHITECTURE</b><br>
Autonomous Claude Sonnet agent that proposes and tests market hypotheses,<br>
integrated with a Telegram bot interface &amp; human-in-the-loop gating.<br>
<i>A Claude Haiku news-screening layer was built, measured, and then removed —<br>
see "Scope, and one component removed for it" below.</i></sub>
<br><br>
<sub><b>STATISTICAL RIGOR</b><br>
Block-bootstrap significance testing, Benjamini–Hochberg FDR control,<br>
and walk-forward validation with strict causality-lag controls.</sub>
<br><br>
<sub><b>PRODUCTION &amp; DEVOPS</b><br>
Python 3.11+, crash-safe atomic JSON state (fsync + atomic replace),<br>
a synchronous single-process Telegram daemon, and a zero-funded-position<br>
observational engine running via a single daemon scheduler.</sub>
<br><br>
<sub><b>RESULT — one full replay, 2017-08-26 → 2026-09-05</b><br>
Nine years of real market data walked day by day, causality-safe:<br>
<b>159</b> conditions tracked, <b>23,495</b> observational live tests, <b>0</b> funded positions.<br>
<b>2</b> conditions ended <code>accepted</code>; <b>1</b> was also <code>CONFIRMED</code> at a checkpoint —<br>
p = 0.001, N = 896, MFE/MAE 1.31, no coin above 26% and no year above 44%.<br>
It clears Benjamini–Hochberg on a family of 101, and its <b>106 independent<br>
episodes</b> put it past the <b>96</b> its own power calculation asks for.<br>
<i>A candidate that cleared every gate this system has — which is not the same<br>
claim as a demonstrated edge. Full numbers and caveats below.</i></sub>
<br><br>
</td>
</tr>
</table>
</div>

## What This Project Tests

Whether an LLM-driven agent — analyzing real-time macro releases and market dynamics — can formulate and statistically validate genuine, repeatable patterns in crypto prices.

Rather than relying on unvalidated LLM assertions or curve-fitted backtests, this system acts as an **autonomous quantitative researcher**: every hypothesis proposed by Claude Sonnet is rigorously evaluated against historical baselines using non-parametric bootstrap testing, multiplicity control (FDR), and prospective live tracking.

**This project never opens a funded position.** It is an AI-driven quantitative data analytics engine. Every "trade" is an observational test designed to measure price behavior, forward returns, and adverse excursions (MFE/MAE) without risking financial capital.

## Executive Summary

Static, rule-based indicators often fail to maintain an edge across changing market regimes. This project implements an adaptive pipeline: a statistical engine that continuously evaluates baseline distributions, a Claude Sonnet strategist that proposes structured market hypotheses, and a human supervisor overseeing hypothesis gating via Telegram.

**How this project checks its own instruments.** A system that reports "no pattern found" has an obvious failure mode: a detector that never fires looks identical to a detector that is broken. Before trusting any null result, this project plants a synthetic signal it already knows the answer to and confirms the pipeline finds it — and confirms a pure-noise arm stays silent.

### Key Results (2017–2026 Full Market Replay)

The historical replay evaluated 9 years of market data day-by-day, enforcing strict point-in-time data isolation:

| Metric | Result |
|---|---|
| Simulated Span | **2017-08-26 → 2026-09-05** (9 years) |
| Hypotheses Proposed & Tracked | **159** |
| Observational Tests Evaluated | **23,495** |
| Final Accepted Candidates | **2** |
| Statistically Confirmed Candidates | **1** still accepted (2 ever reached a checkpoint) |

### Prime Candidate Case Study: `44fb`

The top-performing hypothesis combines labor market surprises with volume expansion:

* **Logic:** Jobless claims print >0.3 SD below its own recent prints (a strong economic reading) followed within 7 days by a 30-day Volume Z-Score > 1.0 → Long (3-day hold).
* **Statistical Power:** p = 0.001 (vs. alpha = 0.100 threshold). N = 896 historical occurrences; **183 confirmations postdating the hypothesis, comprising 106 independent episodes** — exceeding the 96 required for 80% statistical power.
* **Risk Profile:** Favorable MFE/MAE ratio of **1.31** (favorable price excursion dominates drawdown).
* **Robustness:** Survives Benjamini–Hochberg False Discovery Rate (FDR) control across a 101-hypothesis family, with zero single-coin (26% ≤ 60%) or single-year (44% ≤ 60%) over-concentration.

**One condition cleared every gate this system has, which is not the same claim as a demonstrated edge.** Of the 101 hypotheses with a p-value, 19 sit under the raw threshold where chance alone predicts ~10 — which is why the FDR pass is not optional. And `CONFIRMED` means persistence re-earned at each checkpoint, not proof: the other candidate that once earned it is `rejected` today. Full reasoning, including how redundancy is measured differently across time and across coins, is in [`methodology-decisions.md`](docs/case_study/methodology-decisions.md).

### Scope, and one component removed for it

This system tests **market conditions combined with macro events** — CPI, Fed funds and jobless-claims releases, dated by real publication time and graded as a surprise against that series' own recent behaviour. It does **not** test news-headline sentiment, and the reason is a measurement rather than a preference.

An early Claude Haiku layer screened live news headlines. Before keeping it, the question was made concrete: *how good would a news feed have to be before this pipeline could detect anything in it?* Modelled across a range of feed qualities, the answer was that a feed of realistic quality produces results **indistinguishable from pure noise** — 3 accepted conditions out of 57, against 2 out of 57 for a feed containing no information at all. Detection only appears at a signal strength several times better than published work reports for news sentiment.

So the layer was removed rather than kept for the badge. The honest scope is macro events plus market state, which is what the results above actually rest on. Full numbers: [`methodology-decisions.md`](docs/case_study/methodology-decisions.md).

### Screenshots

<table>
<tr>
<td width="33%" align="center"><img src="docs/case_study/assets/telegram_novel_condition_proposal.png" alt="Sonnet proposing a novel condition, with Test It / Don't Test It buttons" width="280"><br><sub>Sonnet proposes a novel condition — human approves with a button, never free text</sub></td>
<td width="33%" align="center"><img src="docs/case_study/assets/telegram_live_test_resolved.png" alt="A live test resolved, with real forward return, best and worst point reached" width="280"><br><sub>A live test resolves — real forward return, best/worst point reached, no TP/SL</sub></td>
<td width="33%" align="center"><img src="docs/case_study/assets/telegram_prune_decision.png" alt="A keep-or-drop decision after 2+ years untested" width="280"><br><sub>2+ years untested — the human decides Keep or Drop <i>(now delivered as one periodic digest, computed offline)</i></sub></td>
</tr>
<tr>
<td width="33%" align="center"><img src="docs/case_study/assets/telegram_replay_summary.png" alt="/replay_summary grouping every tracked candidate by status" width="280"><br><sub><code>/replay_summary</code> — every tracked candidate, grouped by status, recomputed fresh <i>(verdicts shown are pre-audit)</i></sub></td>
<td width="33%" align="center"><img src="docs/case_study/assets/telegram_replay_details.png" alt="/replay_details showing the full numeric breakdown for one candidate" width="280"><br><sub><code>/replay_details</code> — every number behind one candidate's verdict <i>(pre-audit figures)</i></sub></td>
<td width="33%" align="center"><img src="docs/case_study/assets/telegram_help_pinned.png" alt="The pinned /help reference listing every standard command" width="280"><br><sub>The pinned <code>/help</code> reference — every command, always one scroll away</sub></td>
</tr>
</table>

## How It Works

<p align="center">
  <img src="docs/case_study/assets/architecture_diagram.svg" alt="System architecture: a statistical baseline continuously re-validated and checked against an adaptive Claude Sonnet discovery layer, both feeding live testing and Telegram" width="900">
</p>

The static baseline (a fixed set of rule-based triggers, tested once under full walk-forward validation) found no persistent edge on its own — the finding this project's adaptive layer is built to test against, not one it re-litigates by re-running the same rules hoping for a different answer. Every non-obvious methodology or design decision — why each threshold is what it is, why a fixed horizon instead of a TP/SL ladder, why no funded position is ever opened — is logged with its own stated reasoning in [`docs/case_study/methodology-decisions.md`](docs/case_study/methodology-decisions.md). Step-by-step walk-through of what happens on one simulated day, which triggers consult the LLM and which never do: [`docs/case_study/how-the-replay-runs.md`](docs/case_study/how-the-replay-runs.md). File-by-file guide to the whole codebase: [`PROJECT_MAP.md`](PROJECT_MAP.md). Want to run this yourself, step by step, no prior knowledge of the code assumed? See [`HOW_TO_RUN.md`](HOW_TO_RUN.md).

---

## The Telegram Interface

Two interaction modes, kept structurally apart: **free-text conversation never generates a financial number itself** — every figure a message cites comes from a real computation, never invented by the model. **Structured commands and buttons never touch the language model at all** — a fixed set of valid answers is always presented as buttons, never left to free-text guessing.

The messages below are illustrative — real message formats, example figures, not a claim about the current battery's exact state, which `/summary` always reports fresh.

### A pair of ideas, proposed and approved

Bitcoin's volatility has just come out of an unusually quiet stretch. Sonnet is shown what happened during that quiet spell — the real macro releases, dated and graded as a surprise, not just "something came out" — and proposes up to two different, specific ideas at once, sharing one approval:

```
🤖 Agent: COMPRESSION RESOLVED: BTCUSDT

          Quiet since: 2024-02-01 (12 days, volatility 1.7 sd below
          this coin's own normal)
          Broke out: 2024-02-13 (-3.10% that day)

          Assessment: Two ideas worth testing from this squeeze —
          jobless claims worsened twice during it, and the market
          was already stretched two different ways.

          This needs your input.

          1. "weak_claims_then_oversold"

          (jobless-claims surprise worse than expected, within the
          last 7 days, AND 14-day RSI below 30) → long

          2. "weak_claims_then_volume_dryup"

          (jobless-claims surprise worse than expected, within the
          last 7 days, AND 30-day volume z-score below -1.0) → long

          Test It runs a real walk-forward backtest of each condition
          before they are tracked as live tests (no real money is
          ever placed). Don't Test It dismisses them.

          [ Test It ]  [ Don't Test It ]

You:     [taps "Test It"]

🤖 Agent: Historical backtest -- weak_claims_then_oversold

          (jobless-claims surprise worse than expected, within the
          last 7 days, AND 14-day RSI below 30 → long)

          Pattern signal: excess return +2.10% vs. this coin's own
          baseline over the same period, p=0.031, N=134 occurrences
          across 41 separate episodes.
          Risk profile: MFE/MAE ratio=2.05 (favorable -- the typical
          run in your favor exceeds the typical run against you).

          Verdict: ACCEPTED

          This is a historical screening result, not a live track
          record -- the real test is ongoing: added to the battery
          now, re-tested every week.
```

Two ideas rather than one deeper combination is a deliberate design choice, not generosity: each added condition divides how often it has actually happened by roughly eight, so a single three-part idea is usually untestable where two separate two-part ideas both are — and if only one survives, that's a finding a single combined idea would have hidden. If the two ideas turn out to fire on nearly the same days, the second is dropped automatically before it ever reaches a human, since it would just be one idea counted twice.

### A live test, opened and resolved

Once tracked, a candidate's own trigger opens and resolves a live occurrence with no human or model involved — mechanical, hourly detection, no TP/SL:

```
🤖 Agent: 2024-03-04

          Live test opened -- LONG BTCUSDT

          (candidate weak_claims_then_oversold: jobless-claims surprise
          worse than expected AND RSI below 30)

          Held for 7d, no TP/SL.
```

```
🤖 Agent: 2024-03-11

          Live test resolved -- LONG BTCUSDT

          (candidate weak_claims_then_oversold, held 7d, opened
          2024-03-04)

          Forward return: +3.10%
          Best point reached: +4.20%
          Worst point reached: -1.80%

          Confirmation record -- weak_claims_then_oversold
          Occurrence 23 of 307 (needed for a 5% effect at 80% power
          over 7d)
          Trend materialised: 61% of 23 resolved
          Mean best point +3.40%, mean worst -2.10% -- MFE/MAE 1.62
          TP/SL: pending hyperopt cross-check.
```

That second number in "Occurrence 23 of 307" is deliberate, not an oversight: at this project's horizons, proving an effect with real statistical confidence needs occurrences in the hundreds, and no realistic tracking window reaches that. So the system never claims proof — only that a pattern has kept happening and still holds up on the larger sample. That's why the word here is **confirmed**, not "validated": persistence, honestly labeled as persistence.

**A confirmed candidate's own aggregate is, by design, resistant to a short losing streak — a separate, fast alert covers the gap that creates.** A well-established candidate can absorb 20-30 consecutive worst-case losses before its own significance test would ever flip status — correct behavior against noise, but too slow to flag a genuine regime change on its own. So immediately after any live test resolves, if a confirmed candidate's last 2+ resolved live tests came back negative in a row, it fires a purely informational alert — never a status change, long before the aggregate statistics would say anything.

### Natural-language conversation

```
You:     How is the market going today?

🤖 Agent: BTC +1.8% (24h), ETH +2.1%, broad risk-on tone. No macro
          releases in the last 10 days. Nothing currently accepted;
          5 on watch for concentration or risk-path reasons, 1
          rejected on significance. No live tests open.
```

### Structured commands — no LLM involved

`/summary`, `/details`, and `/help` never touch the language model — a command recomputes the real battery fresh (or, for `/help`, prints a static reference) and renders it directly.

```
You:     /summary

🤖 Agent: Still under test

          Watch (4)
          a real pattern signal, but fails a robustness check
          (concentration or an unfavorable risk profile), or too
          little data for the risk check yet
            weak_claims_then_oversold -- N=134, p=0.031, MFE/MAE=1.62
            c1_long (control) -- N=325, p=0.010, MFE/MAE=0.74
            ...

You:     /details weak_claims_then_oversold

🤖 Agent: weak_claims_then_oversold
          What triggers it: jobless-claims surprise worse than
          expected within the last 7 days, AND 14-day RSI below 30.
          Status: watch -- a real pattern signal, but fails a
          robustness check (direction: long)
          Held for: 7d (empirically-derived, re-checked weekly)

          • Historical occurrences (N): 134, across 41 independent
            episodes
          • Statistical significance: significant (p=0.031), excess
            return vs. this coin's own baseline: +2.10%
          • Risk path (mean favorable / mean adverse excursion): 1.62
            (favorable if > 1.0)
          • Coin concentration: 38% from a single coin -- inside the
            60% limit
          • Year concentration: 64% from a single year -- flagged
            above 60%

          Why not accepted: a statistically significant pattern, but
          too much of it comes from one year to trust as general.
```

`/summary` is deliberately terse — a status line answers "what's the verdict" but not "what does 'elevated concentration' actually mean in numbers." `/details <name>` exists for exactly that: the trigger's own exact numeric definition, plus every number behind its current classification, one candidate at a time.

---

## Repository Structure & Build Plan

```
crypto-sentiment-trading-agent/
├── candidates/                  # Statistical methodology + battery: methodology.py, definitions.py, run_battery.py
├── execution/                   # Live test engine (live_testing.py), local-only hyperopt cross-check (hyperopt_runner.py)
├── llm_pipeline/                # Sonnet judgment on compression exits, live context builder, novel-condition tester, compression_detector.py
├── telegram/                    # Both interaction modes -- free text and structured commands
├── scheduler/                   # live_daemon.py (the one command that runs everything), weekly_revalidation.py
├── data_ingestion/               # market_data/binance_fetcher.py (keeps data/ current, from-scratch backfill capable)
├── data/                        # Historical + periodically-refreshed market/macro data
├── replay/                      # Historical walk-forward simulation used to validate the system and build an initial live-test track record before going live -- see PROJECT_MAP.md
├── forecast/                    # Offline experiments that test the SYSTEM rather than the market: does the pipeline detect a signal known to be there, which gate is too tight, would better news data help. No API calls -- see PROJECT_MAP.md
├── docs/case_study/             # methodology-decisions.md, this project's build log
├── .github/workflows/           # tests.yml -- the tests below run automatically on every push
└── tests/                       # candidates/methodology.py, status_history.py, novel_condition_tester.py, run_battery.py
```

**Running this live is one command:** `python3 -m scheduler.live_daemon`. This project is built to be operated as an agent, not maintained as infrastructure — one process owns the Telegram bot, the hourly scans, and the weekly re-validation, so there's no separate cron job to configure or forget. It picks up right where it left off after a restart.

**Safety & guardrails:** this project never opens a funded position, at any point, in any phase — the single guarantee everything else is built around, not a configuration flag that could be toggled off. Every "trade" described above is an observational live test: a real, dated occurrence tracked and measured, never capital at risk. The human gate sits on exactly one decision — whether a genuinely new, LLM-proposed condition is worth testing at all; everything downstream is deterministic and code-driven, with no further human input needed.

**Observation & reporting:** the candidate battery re-validates weekly against live data; a live test's outcome, once resolved, is never retroactively re-tuned. `/summary` and the full decision log are the report at any point — including if the honest result is "no better than the static baseline already found."

File-by-file guide to what every module does, including cost-optimization details: [`PROJECT_MAP.md`](PROJECT_MAP.md). Every non-obvious methodology or design decision, with its own reasoning: [`docs/case_study/methodology-decisions.md`](docs/case_study/methodology-decisions.md).

---

## About the Author

**Giovanni Dominoni** — Riga, Latvia
[giovanni.dominoni@gmail.com](mailto:giovanni.dominoni@gmail.com) · [LinkedIn](https://www.linkedin.com/in/giovannidominoni/)
