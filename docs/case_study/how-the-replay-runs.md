# How the replay runs, step by step

What actually happens on one simulated day, which events reach Sonnet and which
never do, and who decides. Everything below is read off `replay/engine.py`,
`replay/judgment.py` and `llm_pipeline/novel_condition_tester.py` rather than
described from memory; file and function names are given so each claim can be
checked against the code.

## The short answer to three questions

**Which triggers consult Sonnet?** Exactly two: a **real macro release** (CPI, Fed
Funds Rate, Initial Jobless Claims) and a **volatility-shock transition** on one
coin. Nothing else in the replay ever calls a model.

**Who decides?** Deterministic code, not a model. `replay/engine.py::advance()`
walks the calendar and fires on dated facts — a release exists in the ALFRED
vintage record for that day, or a coin's shock z-score crossed 2.0 having been
below it yesterday. Sonnet is never asked *whether* to look; it is asked what it
makes of something the code already decided was worth looking at.

**Is Haiku still used?** Not in the replay. `HAIKU_MODEL` appears only in
`llm_pipeline/haiku_sonnet_pipeline.py::haiku_scout`, whose job is to read live
news headlines cheaply and forward only the ones worth Sonnet's attention. The
replay has no headline feed — historical news, dated and attributable to a coin,
is precisely what this project does not have — so there is nothing for Haiku to
filter and it is never called. The replay's only model calls are
`replay/judgment.py:222` (event judgment) and `:379` (ad-hoc market questions).

## One simulated day, in order

Five things happen per day, in this sequence. The first three cost nothing and
run unattended; only the last two involve a model or a human.

### 1. Resolve anything due — no model

`_check_live_tests(d)` runs **every day**, not only on event days, because a live
test opened on any earlier day can come due today. A test is held for exactly the
horizon `pattern_significance` selected for that candidate, then resolved by
recording forward return, MFE and MAE — the same measure the backtest used, not a
different TP/SL strategy inspired by it.

`state.due_reveals(d)` surfaces any result that was deliberately held back to its
scheduled reveal date.

### 2. Scan mechanical triggers — no model

`_scan_mechanical_triggers` checks every tracked candidate's trigger against
today's hourly bars and opens a live test the moment one fires. This is the bulk
of the system's activity and it is entirely unattended: a pattern that has already
been accepted does not need re-judging each time it recurs, it needs recording.

### 3. Weekly: re-validate the battery — no model

Every 7 simulated days, `run_replay_battery(d)` re-runs the acceptance test on
as-of data only, then:

- `_check_prune_decisions` emits **one keep-or-drop digest per year**, splitting
  candidates into recommended-to-keep and recommended-to-drop, with N, p, MFE/MAE
  and the reason for each. This used to be one Sonnet call per candidate — 665 of
  1203 calls over a 5.5-year replay, 55% of the run's spend — for advice formed
  entirely from numbers already in the message. It is now computed by
  `methodology.prune_recommendation`, which adds the one thing the opinion could
  not: whether there was statistical **power** to detect an effect, separating
  "tested and found nothing" from "never actually asked".
- `_check_n50_milestones` fires each time a candidate crosses a new multiple of
  50 resolved live occurrences. This is the only place the word *validated* is
  used, and it is re-earned at every checkpoint rather than awarded once.

Neither function calls a model any more.

### 4. Macro releases — **Sonnet trigger #1**

For each of the three series in `candidates.macro_vintage.MACRO_SERIES`:

| key | label |
|---|---|
| `cpi` | CPI |
| `fed_funds_rate` | Fed Funds Rate |
| `initial_jobless_claims` | Initial Jobless Claims |

the day fires if `release_dates(series_key, d, d)` is non-empty — that is, if the
series was **actually published on this date**, taken from ALFRED's
`realtime_start` (the real publication date) rather than approximated. An earlier
version assumed CPI lands on the 13th; only 21% of releases matched, with a mean
error of 2.16 days, which puts the event on the wrong side of a price move often
enough to matter.

Sonnet receives, via `judge_event`:

- the release itself, with its change from the prior print
  (`format_macro_event`);
- an **indicator snapshot for every tracked coin** — a macro release belongs to no
  single coin, so restricting it to one would be arbitrary;
- real macro releases from the last 10 simulated days;
- the current battery state and the replay's own history so far.

All of it is time-sandboxed to `as_of`.

**Worked example.** On 12 March, CPI prints. The code knows this is a genuine
release date. Sonnet sees the print, the change from the prior one, and that BTC's
RSI is 28 while it is down 9% over five days. It returns a
`propose_novel_test` with a specific spec: *cool CPI surprise arriving on an
oversold market, long, 1/3/7/14/21-day horizons*. That proposal now goes through
step 5.

### 5. Volatility shocks — **Sonnet trigger #2**

For each coin, `_shock_transition` compares the shock z-score today against
yesterday, both computed on strictly backward-looking windows. It fires **only on
the transition into** shock regime — crossing `SHOCK_ZSCORE_THRESHOLD = 2.0`
having been below it the day before. Without the transition check, a five-day
shock would bill five near-identical Sonnet calls to say the same thing.

A shock is already about one specific coin, so Sonnet gets that coin's snapshot
plus its lead-up (`build_indicator_leadup`), which is the more targeted read.

**Worked example.** ETH's realised volatility crosses z=2.4 on 19 May, having sat
at 1.6 on the 18th. The transition fires once. Sonnet sees ETH's snapshot and the
days leading into it, plus any macro release in the previous 10 days — which is
how "a shock that *follows* a CPI print" becomes expressible at all.

## What happens to a proposal (`_handle_assessment`)

A proposal is not sent to the human as-is. Three deterministic checks run first,
in order, and each can end it:

**1. Is it on-thesis?** `spec_from_proposal` rejects any spec with no news or
macro event clause. This project tests market conditions **combined with a
real-world event**; the event term is a necessary condition, not an option.
`shock_zscore` is a market event, not news, and does not satisfy this alone. A
proposal that is purely technical is refused here and never reaches the phone.

**2. Is it measurable?** `count_occurrences(spec, COINS, as_of=d)` counts how often
the condition has occurred **up to the simulated date**. Roughly 0.25s of local
computation, no API call. Below `MIN_HISTORICAL_OCCURRENCES = 35`, a walk-forward
test would burn a full run and return `insufficient_data` — where 193 of 234
candidates ended in a real run.

**3. If too rare, can it be rescued?** `relax_to_testable` searches for the
smallest loosening of the thresholds that reaches the floor: 10%, then 25%, then
50%. On the 118 candidates the replay had accumulated, 17 sat below the floor and
**all 17 were recoverable**, six of them at 10%.

The search criterion is the occurrence count and nothing else — no return, no
p-value, no outcome is visible to it. That is what separates a **power
calculation** from p-hacking, and the separation is structural here rather than a
matter of discipline. Thresholds also move toward each indicator's neutral point
and never across it: `RELAXATION_NEUTRAL` puts RSI's at 50 and consensus at 0, so
"overbought, RSI ≥ 70" relaxes to 60 rather than to the 52.5 a naive
percentage-of-magnitude step produces. When a relaxation is applied, the Telegram
message says so before the human presses anything — the condition being tested is
not the one proposed, and an approval that hid that would be an approval of
nothing in particular.

**Then, and only then, the replay stops.** The proposal is sent with *Test It* /
*Don't Test It* buttons and the checkpoint is saved as `waiting_for_human`. This
is the single decision a person makes. There is no free-text channel back to the
model, so it cannot be talked into running its own idea.

## What never reaches Sonnet

Worth stating explicitly, because it is most of the system:

- every mechanical trigger firing on an already-tracked candidate;
- every live test resolution;
- the weekly battery re-validation;
- the annual keep-or-drop digest and the N=50 milestone reports;
- any day with no macro release and no shock transition — the large majority.

## Cost consequence

Two triggers, both rare and both deterministic, are what make a 5.5-year replay
affordable. Sonnet is consulted on dated macro releases (three series) and on
shock *transitions* (not shock days), and every proposal is filtered by two free
local checks before a human is ever interrupted. The API is asked for judgment,
never for arithmetic it has no advantage at.
