# Rebuild Plan (Draft) — AI Agent on Freqtrade: Testing Whether Market Sentiment and Financial News Can Create Predictable, Tradeable Patterns in Crypto

Status: **live — this plan's phases (§6) are built and verified**, tracked here as the record of what was built and why, not a forward-looking to-do list. See `README.md` for the live system's own description and current battery results.

---

## 0. What this project is, honestly

Everything built and tested in the current repository — a deterministic macro-event audit across 10 candidate variants and multiple asset classes, run with full walk-forward validation, leakage detection, and honest reporting — did not find a retail-executable statistical edge. That result stands; it is not being re-litigated here.

This project is a deliberate next step anyway: **not a bet that a fixed rule beats the market, but a test of whether an adaptive, LLM-mediated judgment layer — continuously re-validated, escalating genuinely novel conditions to a human instead of guessing — behaves differently from a static rule set over a live, forward-looking window.** The honest prior is that it probably doesn't outperform either. The point of building it is to find out for real, on live data the system hasn't seen, rather than assume the answer from backtests alone. Every report this project produces — while running and at the end — states results plainly, including if the answer turns out to be "no better than the static rules already ruled out."

**Success is not "the agent makes money."** Success is "the agent runs live for the observation period, produces an honest, complete, inspectable record of every decision and why, and the project reports the real outcome, positive or negative, without moving the goalposts."

---

## 1. Non-negotiable methodology, fixed before any code is written

These are lessons paid for expensively in the prior repository. They are inputs to this plan, not aspirations to revisit later:

1. **No trade signal is used unless its trigger condition is knowable at decision time.** If a condition depends on how a bar/day closes, the earliest legal action is at the *next* bar's open — never the same bar's close. This is enforced structurally (see §3), not by convention alone.
2. **One barrier methodology, used everywhere, never mixed mid-comparison.** Take-profit/stop-loss levels are always duration-bucketed anchors — mean Maximum Favorable/Adverse Excursion per time horizon, fit fresh on a train set, applied via two free multipliers (TP mult, SL mult) chosen by walk-forward grid search. No candidate is ever screened on a different, ad-hoc barrier (a flat %, a hand-picked ATR multiple) and then compared to one screened this way.
3. **Every reported win rate is shown with its strict win rate (timeouts counted as non-wins) and its Sortino ratio, never alone.** A high win rate sitting on a large timeout fraction is not reported as good news until the strict number and Sortino are checked too.
4. **Every candidate that looks positive gets a concentration check before being trusted**: per-coin breakdown (is one coin carrying the whole pooled result?) and per-year breakdown (is one lucky year carrying the whole result?). A candidate that fails either check is not promoted, regardless of its pooled headline number.
5. **Walk-forward only, expanding window, never a random train/test split.** Anchors and multipliers are refit each fold using only prior data.
6. **Re-validation is scheduled, not one-and-done.** Market structure drifts (proven directly in this project's own prior work: a hand-tuned strategy that looked excellent on 2025-2026 was net-negative in 7 of the 8 years before it). Every candidate in the live battery gets its anchors and multipliers refit on a fixed cadence (weekly — see §6), not fit once at launch and left alone.
7. **Extreme shocks are statistically isolated from the static battery, not blended into it.** A z-scored, causally-safe measure of short-term realized volatility against its own trailing baseline flags extreme events; the static battery's fitting and trading uses only non-extreme events. Extreme events aren't discarded -- they route to a dedicated live detection path (§4.4) instead, tested with the same walk-forward rigor on their own excluded population, not assumed away or blended into "normal" anchors.
8. **Fees are modeled honestly; slippage and funding carry are not conflated with them.** Every backtested trade nets a fixed 0.20% round-trip transaction cost. Slippage is not modeled historically (no order book to backtest against) but is read directly from live execution instead. Funding/holding cost during a multi-day hold is not subtracted in the historical battery (funding is used only as one candidate's signal, `c1`) -- backtested short-side results are a directional validation on spot price data, not a simulation of real futures carry cost. That distinction only becomes load-bearing once a position is actually held live in futures mode (§2).

---

## 2. Architecture

Three modules, unchanged in their division of responsibility from the prior project, rebuilt clean:

| Module | Role |
|---|---|
| **A — Safety Circuit Breaker** | Deterministic, non-LLM kill switch: forces flat on a volatility spike or ahead of a high-impact macro release; hardcoded leverage/size ceilings no AI component can raise. Ported forward essentially unchanged, and wired directly into the execution engine's entry/exit callbacks (§2.B) so it's checked first, ahead of every other rule -- not merely present in the repository. |
| **B — Execution Engine (Freqtrade)** | Places and manages trades on a duration-bucketed, anchor-based TP/SL ladder (§1.2). Runs `trading_mode: "futures"` / `margin_mode: "isolated"` so both long and short can actually execute (spot alone can't hold a short). Entries can come from two sources: (a) the C-candidate battery, when one is live-flagged as validated, or (b) an explicit Sonnet Strategist recommendation -- including a real-time shock/crash detection (§4.4) -- escalated from Haiku and approved by an explicit human confirmation (never a blind LLM trade — see §4). |
| **C — Signal Layer: candidate battery + Haiku/Sonnet judgment** | Two parts, working together, described in §3 and §4. |

---

## 3. The C-candidate battery: continuously tested, not fixed at launch

A candidate battery (the "C1...C6"-style deterministic triggers from the prior research, re-derived clean rather than copy-pasted with their old bugs) is the *baseline* signal source — even though none currently validates, the battery itself is the mechanism Haiku/Sonnet are meant to supplement and check against, and it is the thing that gets re-tested on a schedule rather than assumed static:

- **At launch:** every candidate is re-built from scratch against the fixed methodology in §1 — no candidate is carried over from the old repository's code, only the validated *idea* of what to test. Each is walk-forward validated, concentration-checked, and explicitly labeled with a live status: `validated` (clears every check), `watch` (positive but thin/unconcentrated-unverified), or `rejected` (fails walk-forward or concentration). Only `validated` candidates are allowed to place live trades unattended; `watch` candidates are visible to Sonnet as context but never trigger a trade on their own.
- **On a weekly cadence:** every candidate — `validated`, `watch`, and `rejected` alike — is automatically re-run against the latest data (fresh walk-forward fold, fresh anchors, fresh concentration check). A candidate's status can move in either direction: `validated` can degrade to `watch` if a new fold breaks its OOS performance; `rejected` is re-checked too, cheaply, in case conditions genuinely changed (this is a real, not theoretical, risk in crypto specifically) — not to fish for a different answer on the same data, but because new data each week is a genuinely new test, not a re-run of the old one.
- **On demand, via the novel-condition escalation loop (§4.3):** a human-approved ad-hoc test of a condition Haiku/Sonnet flagged as unfamiliar. If it validates, it's added to the battery as a new candidate under the same weekly-refresh regime; if not, it's logged as tested-and-rejected so it isn't proposed again without new justification.

---

## 4. Haiku Scout → Sonnet Strategist: the adaptive layer

### 4.1 Haiku Scout (always-on, cheap)
Polls news/market data continuously. For each item: extracts asset, sentiment, magnitude (1-5), event type, and — new in this rebuild — **checks whether the current market state matches any known C-candidate's trigger definition, any of the explicitly-logged "already tested and rejected" conditions, or neither.** Only genuinely unmatched conditions plus magnitude ≥ threshold escalate to Sonnet; routine matches to an existing `validated`/`watch`/`rejected` candidate are logged, not escalated (Sonnet's time/cost is reserved for judgment calls, not routine classification).

### 4.2 Sonnet Strategist (escalated, judgment)
Given a Haiku escalation, a live technical snapshot (real open-trade state read directly from Freqtrade's own database, not a static placeholder — a known gap in the prior build, fixed here from day one), and a curated context file stating current candidate statuses and their confidence levels as load-bearing facts, Sonnet proposes one of: `no_action`, `watch` (with a concrete, checkable condition), `enter_long`/`enter_short` (with reasoning and the specific TP/SL bucket ladder to apply, drawn from the anchor methodology in §1.2 — never a freehand number), or `exit_now` (with reasoning) for an open position. Sonnet's own reasoning, and the exact data it was given, are always logged in full — every trade the system places must be traceable back to the specific signal, snapshot, and context that justified it.

### 4.3 Novel-condition escalation and human-approved testing
When Haiku/Sonnet jointly determine a condition doesn't match anything in the current battery *and* isn't already logged as tested-and-rejected, it's flagged to the human via Telegram (mockup: README §6) rather than acted on unilaterally. A human reply of "test it" (or the structured equivalent) triggers an ad-hoc walk-forward backtest of that specific condition against historical data, using the exact §1 methodology — no shortcuts because it was proposed live. The result (validated / watch / rejected) is reported back and, if validated, folded into the weekly-refreshed battery.

### 4.4 Real-time shock/crash detection ("Mode B: Live Reactive Entry")
A distinct, market-data-driven escalation path, independent of news headlines: live volatility is scanned for the same statistical shock signature §1.7 excludes from the static battery, and Sonnet is escalated directly when a coin crosses that threshold. Sonnet's only available response is `propose_novel_test` using the whitelisted `shock_zscore` indicator, run on that specific coin's own historical shock-regime events -- the exact population §1.7 excluded from the static battery, now given its own dedicated, walk-forward-validated test rather than being discarded. If a human approves and it validates, the resulting anchors trade the live occurrence immediately, tagged `shock_reactive` so its outcome is measured separately from routine trades (§5.2) -- a direct test of whether the LLM layer can recognize and react to a crash or a bull surge as it happens, not just process routine headlines.

**A second mode was designed and deliberately dropped.** "Mode A" would have simulated an idealized entry at the exact catalyst candle, in hindsight, to benchmark the live system's real-time reaction against a theoretical best case. Rejected before being built: identifying *which* candle was "the" catalyst is only possible after the fact, which would have reintroduced the same lookahead problem this project's entire methodology exists to avoid, dressed up as a benchmark rather than a trading rule. Only Mode B -- live, reactive, no hindsight -- was built.

---

## 5. Telegram Human-AI Interface

Two distinct interaction modes, kept structurally separate so free-text LLM output is never the source of a financial figure:

1. **Natural-language conversation** — market checks, trade-trigger explanations, post-mortems. Sonnet-generated prose, but every number it cites (a price, a TP level, a win rate) is interpolated from a direct database/computation call, never generated freehand by the model.
2. **Structured commands / inline keyboards** — KPI breakdowns (win rate, net profit, max drawdown, Sharpe, Sortino), filterable by coin, by signal, and by `signal_class` (`battery` / `manual` / `shock_reactive` — §4.4's tag), so the LLM's real-time judgment on critical conditions can be measured against the routine, statistically-validated path, not just lumped in with it. Output is template-rendered from a real query against the trade database — no LLM in this path at all, by design, so numeric reporting can never hallucinate.
3. **Automated post-mortems** — fired automatically when a stop-loss is hit or a position is closed, explaining the likely structural cause (the market condition, the signal that opened it, what changed) — Sonnet-generated narrative, same "cite only computed numbers" rule as (1), auto-appended to that trade's permanent history record.

Full mockup conversation: README §6.

---

## 6. Build phases

1. ✅ **Clean-room candidate re-derivation.** Rebuilt against §1's methodology (including §1.7's shock isolation), on the live coin universe, both directions per candidate. No code ported from the old repository — only the validated ideas carried over. `validated`/`watch`/`rejected` labeling and KPI tables: `README.md` §3.
2. ✅ **Execution engine.** Freqtrade strategy implementing the duration-bucketed anchor ladder as a reusable, candidate-agnostic exit mechanism, `trading_mode: "futures"`; entry sourced from `validated` battery candidates or a Sonnet-approved manual/shock signal, handled through a pending/active signal-store split (`execution/signal_store.py`) so an approval fires exactly once and its anchors are recoverable at exit time, days later.
3. ✅ **Haiku Scout + Sonnet Strategist pipeline**, wired to Freqtrade's real open-trade state (not a placeholder), the live-generated context (not a hand-maintained file), the novel-condition-match/escalation logic (§4.3), and the real-time shock-detection path (§4.4).
4. ✅ **Telegram interface**, both modes from §5, the automated post-mortem trigger, and the `signal_class` KPI breakdown.
5. ✅ **Weekly re-validation cron**: re-runs every candidate's walk-forward fold, concentration check, and shock-regime exclusion; updates `validated`/`watch`/`rejected` statuses; notifies the human of any status change.
6. **Dry-run launch.** Live market data, live Haiku/Sonnet pipeline, paper trading only, for a burn-in period before any live-capital discussion — matching this project's existing default-to-dry-run safety convention.
7. **Observation window.** Let it run untouched except for the scheduled weekly refresh and any human-approved novel-condition or shock tests, for the period the user sets (a few months, per direction). No retroactive re-tuning of the observation-window results themselves — the weekly refresh updates the *live* battery going forward, it does not rewrite what already happened.
8. **Report, honestly, either way.** At the end of the observation window (or at any checkpoint along the way), the KPI dashboard and the full decision log are the report. If the result is "no better than a coin flip, consistent with the prior research," that is reported as plainly as a positive result would be.

---

## 7. Documentation policy for this rebuild (a deliberate change from the prior repository's style)

The prior repository's case study documented process exhaustively — every bug, every wrong turn, in a running decisions log. This rebuild's code ships as **final, clean Python files only** — no bug-by-bug commentary embedded in the codebase, no half-fixed intermediate versions kept around. The README states plainly, once, that the methodology in §1 is the product of a rigorous data-leakage and metric-reporting cleanup process (referencing the prior repository's case study for anyone who wants the full trace) — without re-narrating that process inside this repository. This repository's own case study log, if kept, documents *this* project's build and its live results, not a re-telling of the audit that preceded it.

---

## 8. Migration

This plan does not touch the current repository. When this plan and the README draft are approved and the rebuild is actually built and validated per §6, the current repository's contents are replaced with the new one, as an explicit, separate, confirmed step — not an implied side effect of writing this plan.
