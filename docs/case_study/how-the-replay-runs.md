# How the replay runs, step by step

What actually happens on one simulated day, which events reach Sonnet and which
never do, and who decides. Everything below is read off `replay/engine.py`,
`replay/judgment.py` and `llm_pipeline/novel_condition_tester.py` rather than
described from memory; file and function names are given so each claim can be
checked against the code.

## The short answer to three questions

**Which triggers consult Sonnet?** Exactly one: a **confirmed exit from a
volatility-compression episode** on one coin. Macro releases and volatility
shocks were both removed — a macro release is one of the causes being sought, a
shock is the outcome, and each was measured not to select for what this project
looks for. Nothing else in the replay ever calls a model.

**Who decides?** Deterministic code, not a model. `replay/engine.py::advance()`
walks the calendar and fires on a dated fact — a coin's volatility compression
ended five days ago and has not resumed since. Sonnet is never asked *whether* to look; it is asked what it
makes of something the code already decided was worth looking at.

**Is Haiku still used?** Not in the replay. `HAIKU_MODEL` appears only in
`llm_pipeline/haiku_sonnet_pipeline.py::haiku_scout`, whose job is to read live
news headlines cheaply and forward only the ones worth Sonnet's attention. The
replay has no headline feed — historical news, dated and attributable to a coin,
is precisely what this project does not have — so there is nothing for Haiku to
filter and it is never called. The replay's only model calls are
`replay/judgment.py:222` (event judgment) and `:379` (ad-hoc market questions).

## One simulated day, in order

Four things happen per day, in this sequence. The first three cost nothing and
run unattended; only the last involves a model or a human.

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

### 4. The one trigger — a confirmed exit from volatility compression

`_compression_exit` fires when a coin's `vol_compression_zscore` was at or above
1.25 (unusually quiet for that coin), then dropped below it, and **did not go
back above for five days**. Nothing else calls a model.

Both previous triggers were removed, on one principle and each with its own
measurement (`forecast/trigger_value.py`):

| removed trigger | why | measured |
|---|---|---|
| macro release | it is one of the **causes being sought** | indistinguishable from ordinary days at every horizon, all three series |
| volatility shock | it is the **outcome**, and not a trend | *anti*-precursor: 8.8% trend rate vs an 11.8% baseline |

Compression is a precursor instead (16.1% vs 11.1%) and is the right shape
besides: it says a directional move is brewing **without saying which way**,
leaving the direction to be explained by the macro context and the market state.

**Why the exit and not the state.** Compression is a state, not an event —
episodes run a median of 4 days and up to 38. Triggering on the state would ask
the same question up to 38 times about the same market: a measured 6.7×
duplication.

**Why the five-day confirmation.** An exit followed by re-compression is a
flicker inside the same lull, and is followed by a defined trend 15.2% of the
time against 23.8% for confirmed exits.

**The confirmation decides *whether* to ask, never *what* is shown.** The replay
physically stands at point C, but everything handed to Sonnet — and `as_of` for
the backtest — is dated to point B, the exit itself.

Sonnet receives the episode as a three-phase story:

- **Phase A** — when the compression began, and how quiet it got;
- **Phase middle** — how long it lasted, how price drifted across it, and every
  macro release published while it lasted, each as a **change from the prior
  print** and a **surprise in standard deviations** of that series' usual move,
  never as a bare level;
- **Phase B** — the day it ended.

The exit day's own direction is reported but explicitly flagged as carrying no
information: measured over 214 episodes, the direction of the first bar out of
compression has no relationship to where price is two weeks later (ρ = −0.051,
p = 0.46).

**Worked example.** BTC compresses on 2 January 2022 and stays quiet for 19 days.
Jobless claims deteriorate steadily through the squeeze — surprises of +0.8, then
+1.2, then +2.3 standard deviations. Compression ends on 21 January. Sonnet is
asked on 26 January, sees everything as of the 21st, and must say whether that
macro deterioration and the market state it landed in explain what follows.

**Cost.** 217 triggers across the entire replay, about **$3.32** — against roughly
1,200 calls and $18 before, of which 650 went to macro releases that selected
nothing.

## What happens to a proposal (`_handle_assessment`)

A proposal is not sent to the human as-is. Three deterministic checks run first,
in order, and each can end it:

A proposal now arrives as a SET of one or two conditions, and each is prepared
separately before the set is offered for one human decision.

**1. Is it on-thesis, and buildable?** `spec_from_proposal` rejects any spec with
no news or macro event clause — this project tests market conditions **combined
with a real-world event**, and the event term is a necessary condition, not an
option. It also rejects a spec with more than **two clauses**, and any spec using
an indicator in `NON_PROPOSABLE_INDICATORS`: `shock_zscore` (a price outcome, not
a market state), `vol_compression_zscore` (the trigger itself, fixed at every
proposal by construction), `is_macro_day` (records that a release was scheduled,
never what it said) and raw `daily_range_pct` (non-stationary — a fixed threshold
on it selects a year rather than a market state).

**2. Is it measurable?** `is_testable(spec, COINS, as_of=d)` applies two floors,
both counted **up to the simulated date**, in about a second of local computation
and no API call:

- **120 raw occurrences**, because below that a walk-forward test cannot run —
  measured, 90% of conditions with 35-60 occurrences returned `insufficient_data`;
- **40 separate occasions**, because `build_events` makes one row per triggered
  bar, so a condition true for several consecutive days produces several events
  whose outcome windows overlap almost entirely. Those are one piece of evidence
  counted many times. A seven-day lookback yields roughly eight times the firings
  for 1.7 times the independent evidence.

The two bind in different regimes, which is why neither replaces the other.

**3. If too rare, can it be rescued?** `relax_to_testable` searches for the
smallest loosening that reaches BOTH floors: 10%, then 25%, then 50%. Each
threshold moves toward its indicator's neutral point and never across it —
`RELAXATION_NEUTRAL` puts RSI's at 50 and consensus at 0, so "overbought, RSI ≥
70" relaxes to 60 rather than to the 52.5 a naive percentage-of-magnitude step
produces, and "cool CPI, surprise ≤ 0" cannot flip to matching hot prints.

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

**4. Are the two actually two?** `filter_redundant_proposals` drops a second
proposal that fires on the same days as the first — judged on behaviour
(`behavioural_agreement`, the Jaccard overlap of their firing days), never on
shared clauses, since the intended pattern is precisely two proposals sharing
their news term and differing in the market term. Measured, that pattern's
overlap is 0.000 while a near-duplicate scores 0.84.

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
