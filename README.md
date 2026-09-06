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

Every message below is a real render from the actual code — either built from a genuine historical episode run through the live pipeline, or, where noted, the literal output the system sent after the full replay finished. None of it is mocked up. `/summary` and `/replay_summary` always report the current battery fresh, which will differ from any specific numbers quoted here as the system keeps running.

### A pair of ideas, proposed and approved

Bitcoin's volatility has just come out of a 12-day quiet stretch. Sonnet is shown what happened during that squeeze — the real macro releases, dated and graded as a surprise, not just "something came out" — and proposes up to two different, specific ideas at once, sharing one approval. This is a real render of the actual message format, built from a real historical episode (BTC, March–April 2022):

```
🤖 Agent: EVENT ALERT: VOLATILITY COMPRESSION RESOLVED
          Asset: BTCUSDT
          Period: 2022-03-30 to 2022-04-11 (12 Days Coiling)

          --- SQUEEZE METRICS ---
          • Start Volatility: 1.65 SD below normal
          • Mid-Squeeze Drift: -16.01%
          • Exit Bar Move (2022-04-11): -6.23%
          • Post-Squeeze State: Neutral (14-day RSI: 35.37 |
            30-day volume z-score: 1.37)

          --- MACRO CONTEXT DURING SQUEEZE ---
            2022-03-31 Initial Jobless Claims: 202k, change vs
            prior +14k, surprise +0.6 sd
            2022-04-07 Initial Jobless Claims: 166k, change vs
            prior -5,000, surprise -1.4 sd

          --- ASSESSMENT ---
          Jobless claims improved twice during the squeeze (a
          hawkish signal), and the market was already stretched
          two different ways going into the breakout.
          Sonnet generated 2 testable hypotheses.

          PROPOSAL 1: hawkish_claims_then_oversold_short
          Status: PROPOSED (Awaiting Human Gating)

          --- CONDITION & PARAMETERS ---
          • Logic: jobless-claims surprise at most -1.0 (within
            the last 7 days) AND 14-day RSI at most 40 → SHORT

          PROPOSAL 2: hawkish_claims_then_volume_spike_short
          Status: PROPOSED (Awaiting Human Gating)

          --- CONDITION & PARAMETERS ---
          • Logic: jobless-claims surprise at most -1.0 (within
            the last 7 days) AND 30-day volume z-score at least
            1.0 → SHORT

          --- ACTION REQUIRED (HUMAN-IN-THE-LOOP) ---
          [ Test It ] → Runs the full walk-forward backtest on
          history up to this date, registers both conditions,
          and tracks them forward as observational live tests.
          No capital, ever.
          [ Don't Test It ] → Dismisses them untested. Nothing
          is recorded, so the same idea can surface again later.

You:     [taps "Test It"]

🤖 Agent: HISTORICAL BACKTEST REPORT
          Candidate: hawkish_claims_then_oversold_short  eb5a
          Status: WATCH -- real signal, fails a robustness check

          --- CONDITION & PARAMETERS ---
          • Logic: jobless-claims surprise at most -1.0 (within
            7 days) AND 14-day RSI at most 40 → short

          --- STATISTICAL VERDICT ---
          • Pattern Significance: NOT SIGNIFICANT (p = 0.516 |
            Target: p < 0.100)
          • Sample Size: N = 65 (Control Sample: N = 509)
          • Excess Return vs incremental: -0.05%
            ⚠️ WARNING: Effect runs OPPOSITE to trade direction.

          --- RISK PROFILE (RAW PATH) ---
          • MFE / MAE Ratio: 0.45 (adverse excursion dominates)
          • Raw Sortino Ratio: -2.17 (No fees / No TP/SL)

          --- HISTORICAL TP/SL REFERENCE (INFORMATIONAL) ---
          • Trades: N = 65 | Win Rate: 43.8% | Sortino: -3.54
          Does not affect the verdict above -- this project
          accepts on pattern significance, not on a barrier
          structure's P&L.

          ---
          STATUS DEFINITION [WATCH]
          A real pattern signal that fails a robustness check.
          It trades zero real capital while accumulating new
          occurrences toward the sample a null result would
          need to be informative. Re-evaluated automatically
          every 7 simulated days.
```

This is the honest outcome for most proposals, and it is a better illustration than a cherry-picked accepted one: a real hypothesis, real numbers, real reason for the verdict — including the wrong-direction warning, which is exactly the check that once let four of six static candidates be called "significant" while doing this. The 4-character id (`eb5a`) is what a reader types into `/replay_details` to pull this candidate up again later.

Two ideas rather than one deeper combination is a deliberate design choice, not generosity: each added condition divides how often it has actually happened by roughly eight, so a single three-part idea is usually untestable where two separate two-part ideas both are — and if only one survives, that's a finding a single combined idea would have hidden. If the two ideas turn out to fire on nearly the same days, the second is dropped automatically before it ever reaches a human, since it would just be one idea counted twice.

### What replaced one message per live test

Earlier versions of this system sent a Telegram message every time a tracked candidate's trigger opened a live test, and another when it resolved. Over the full nine-year replay that was **15,500 of 16,363 messages — 95% of all traffic** — and Telegram answered the volume with a rate limit long enough to stall the run outright. The mechanical fix (queue and retry) was the smaller half of the problem: 7,800 notifications are not a history anyone reads. Both per-test messages were removed. The dated record survives in full in the trade log and is reachable through `/replay_details <name or id>`, which prints a candidate's last occurrences with their outcomes on demand.

What replaced them is a bounded monthly digest and, separately, a message every time a candidate crosses a confirmation checkpoint. Both below are real renders from the completed replay's actual final state — nothing invented, no placeholder numbers:

```
🤖 Agent: ━━━ MONTHLY DIGEST -- July 2026 ━━━

          Live tests  389 opened - 463 resolved - 44 still open
          This month  243/463 positive (52%) - mean -0.02% -
          MFE/MAE 0.88
          All time  23451 resolved - 50% positive - mean -0.48%
          Events assessed this month: 2

          Confirmation progress (none of these is a result --
          the denominator is the point)
            c2_short  b02d  confirmed 1459 -- powered (needed
            417)  -  trend 51%  -  watch
            strong_labor_print_then_volume_blo  9300  confirmed
            1103 -- powered (needed 672)  -  trend 49%  -  watch
            jobless_claims_beat_volume_surge_l  72f4  confirmed
            1030 -- powered (needed 554)  -  trend 52%  -  rejected
            ... and 93 more -- /replay_summary for all of them
            5 of the rows above are past their power threshold:
            for those, 'no effect found' is a measurement, not
            a missing answer.

          Battery  159 tracked - 103 active - 126 reached a
          checkpoint - 2 currently CONFIRMED - 75 parked

          Individual live tests are no longer sent one by one.
          Every figure above comes from the full trade log --
          /replay_summary for the table, /replay_details <name
          or id> for one trigger with its last dated occurrences.
```

Rows are ranked by how far a candidate is toward its confirmation count, **never by how well it has done** — ranking by success rate would put the luckiest small sample on top, and on a real run that meant a candidate at n=6 with a 100% hit rate whose own backtest status was `rejected`.

```
🤖 Agent: 2026-08-24

          STATUS UPDATE: ACCEPTED
          Candidate: hawkish_claims_surprise_then_volume_
          spike_capitulation  44fb
          (jobless-claims surprise at most -0.3 (within 7 days)
          AND 30-day volume z-score at least 1.0 → long)

          --- VERDICT & POWER ---
          • Pattern Significance: SIGNIFICANT (p = 0.001 |
            Target: p < 0.100)
          • Power Progress: 180 / 96 occurrences (SAMPLE
            SUFFICIENT -- a null here is a measurement)

          --- PERFORMANCE & EXCURSION ---
          • Trend Realized: 58.5%
          • Market-Adjusted Excess: +0.56% per occurrence vs
            universe baseline (56% positive after adjustment)
          • MFE / MAE: +8.90% / -7.12% (Ratio: 1.25)

          ---
          Next Checkpoint: 200 occurrences (re-evaluated fresh
          each time, never a permanent verdict)
          CONFIRMED at this checkpoint. Confirmed, not validated:
          persistence on an enlarged sample, not proof -- a
          conclusive test needs the occurrence count shown above.
```

`hawkish_claims_surprise_then_volume_spike_capitulation` (`44fb`) is the one candidate this project's full replay produced that cleared every gate it has — see the result summary near the top of this README for the full statistical picture, including the multiplicity control it was checked against. That second number, "Power Progress: 180 / 96", is deliberate, not an oversight: at this project's horizons, proving an effect with real statistical confidence needs occurrences in the hundreds, and most tracked candidates never get there. So the system never claims proof — only, once the sample is large enough, that a pattern has kept happening and still holds up. That's why the word here is **CONFIRMED**, not "validated": persistence, honestly labeled as persistence, re-earned fresh at every checkpoint rather than awarded once.

**A confirmed candidate's own aggregate is, by design, resistant to a short losing streak — a separate, fast alert covers the gap that creates.** A well-established candidate can absorb 20-30 consecutive worst-case losses before its own significance test would ever flip status — correct behavior against noise, but too slow to flag a genuine regime change on its own. So immediately after any live test resolves, if a confirmed candidate's last 2+ resolved live tests came back negative in a row, it fires a purely informational alert — never a status change, long before the aggregate statistics would say anything.

### Natural-language conversation

This is a real answer, sent by the actual system after the replay finished — not a mocked-up illustration:

```
You:     Tell me the confirmed candidates up to now

🤖 Agent: As of 2026-09-05, two candidates in the battery have
          hit a "confirmed" checkpoint (meaning the pattern kept
          occurring after it was flagged and still passed its
          checks when it reached 20 such occurrences -- that's
          a persistence check, not proof, since a real
          demonstration needs occurrences in the hundreds):

          - hawkish_claims_surprise_then_volume_spike_
          capitulation (44fb) -- currently accepted (its trigger
          is live-testing automatically) and confirmed at its
          checkpoint. This is the one case where both labels
          currently apply together.

          - claims_surprise_then_funding_stretched_reversion
          (37fa) -- was confirmed at its checkpoint, but its
          current status is rejected (no statistically
          significant edge found against the coin's own baseline
          going forward). This shows why the two words aren't
          interchangeable: it persisted long enough to earn
          "confirmed" once, but that isn't a permanent badge,
          and it has since failed the significance test.

          No other candidate in the list carries a "CONFIRMED"
          marker as of this date.
```

Every candidate the model names carries its 4-character id in parentheses — what a reader types into `/replay_details` for the full picture — and it never writes "validated": that word is banned both in the prompt and, more to the point, in the data the model is actually shown, after an earlier version of this project caught the state summary itself handing the model the retired word and watched it repeat what it was given.

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
