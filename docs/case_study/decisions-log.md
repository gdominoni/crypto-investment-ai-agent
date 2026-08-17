# Technical Decisions Log

Running log of non-obvious technical decisions, in chronological order. Each entry: what was decided, why, and what it affects.

---

### 2026-08-17 — Multi-factor confluence replaces isolated single-family testing (Module B)

**Context:** After reviewing Phase 1's coarse-grid results (216 backtests, zero genuine OOS wins across three isolated strategy families), the human director identified the actual methodological gap: the families were never tested *in isolation* by design intent, but the Phase 1 build had done exactly that -- each family alone, never combined. A breakout entry with nothing confirming real volume behind it can't distinguish a genuine move from chop; a mean-reversion entry with nothing confirming the reversal is real can't distinguish a bounce from a falling knife.

**Decision:** replaced isolated testing with `MultiFactorConfluence`, a single strategy combining all three families on every candle in asymmetric roles -- Family 2 (breakout) as the timing trigger, Family 1 (RSI) and Family 3 (volume) as confirming filters -- rather than a flat AND of all three families' original full conditions (which would multiply 3+3+3 sub-conditions together and likely never fire at a statistically meaningful rate, especially at 4h/1d where Phase 1 already showed thin trade counts). This is a judgment call about *how* to combine, not fully specified by the instruction to "cross" the three families, and is presented for approval alongside the grid itself rather than assumed correct.

**Also decided in the same redesign:**
- 15m dropped entirely (Phase 1: uniformly worst timeframe across every family, consistent with fee-drag from high trade frequency).
- Exit grid narrowed from 4 presets to 2 (null + one wide SL+TP pair) -- Phase 1's finding that the null/indicator-only exit outperformed every fixed-percentage exit on average directly motivated both the narrowing and weighting the new strategy's own exit logic toward indicator/trend-reversal signals.
- Added an RSI-Bollinger-Bands variant (Bollinger Bands applied to the RSI series itself, not price) as a second Family-1 option, per explicit instruction not to test only fixed-threshold RSI. Verified `pandas_ta.bbands` works correctly on an arbitrary series before committing it to strategy code.
- Per-family option counts shrunk from 3 (Phase 1) to 2, specifically because this grid multiplies three dimensions together (Cartesian product) rather than testing them separately -- keeping the combined grid a genuine coarse screen rather than letting the product size balloon.

**Why the old section stays in the README, marked superseded rather than removed:** the negative result Phase 1 found (no isolated family shows OOS edge) is still true and is the direct cause of this redesign -- a case study should show the wrong turn, not just the corrected one.

**Impact:** `modules/module_b_trend_following/multi_factor_grid.py`, `user_data/strategies/multi_factor_confluence.py`. Not yet run -- explicitly deferred pending approval of the grid itself, printed via `print_multi_factor_matrix.py` (pure enumeration, no Docker execution).

---

### 2026-08-17 — Freqtrade resolves a strategy's parameter file from its .py path, not the class name

**Context:** Building Module B's coarse-grid sweep (Phase 9, Phase 1 of the new strategy-family screening), the orchestration script wrote each combination's parameters to `<StrategyClassName>.json` (e.g. `MeanReversionBBRSI.json`), mirroring what looked like the pattern from hyperopt's own auto-exported `trend_ema_adx.json`.

**What went wrong:** every combination silently ran on the strategy's class-level defaults, because Freqtrade actually resolves the parameter file as `Path(self.__file__).with_suffix(".json")` -- the strategy's own **file** path (`mean_reversion_bb_rsi.json`, matching the .py filename), not the class name. `trend_ema_adx.json` had looked like a class-name match purely because that strategy's filename and class name both reduce to the same snake_case/PascalCase pair by coincidence.

**How it was caught:** two deliberately different exit presets (a permissive "null" stoploss/ROI and a tight one) produced bit-for-bit identical backtest results. That has no innocent explanation, so it was investigated immediately rather than accepted -- confirmed by reading `freqtrade/strategy/hyper.py`'s `load_params_from_file()` directly rather than guessing again.

**Decision:** added an explicit `STRATEGY_FILENAMES` mapping (class name -> file basename) in `run_coarse_grid.py`, rather than trying to derive one from the other programmatically -- explicit and easy to verify beats a naming-convention assumption that already failed once.

**Impact:** `modules/module_b_trend_following/run_coarse_grid.py`. Worth remembering for any future script that writes a Freqtrade strategy parameter file programmatically.

---

### 2026-08-17 — Double-backgrounding orphaned a long-running sweep from its own tracking

**Context:** Launching the ~90-minute, 216-backtest coarse-grid sweep, the first attempt combined a shell `&` (to background the python process within the command) with the tool's own background-execution flag (also meant to background the whole command).

**What went wrong:** the tool's tracking followed the *outer* command, which returned almost immediately once the `&`-backgrounded process was launched and its PID echoed -- so the harness reported the task "completed" within seconds, while the actual 216-backtest sweep kept running, undetected and unmonitored, as an orphaned process outside the tracked job.

**How it was caught:** the "completed" task's output log had only one progress line where ~216 were expected, and a process listing showed the real python process still alive well past when the tracked task had supposedly finished.

**Decision:** killed the orphan, discarded its partial (1-row) output, and relaunched using only the tool's own background flag, with no shell `&` inside the command -- letting the harness track the actual long-running process directly, which then correctly reported completion at the real end of the sweep.

**Impact:** operational only, no code changed. Worth remembering for any future long-running background command: background it exactly one way, not two.

---

### 2026-08-17 — Custom Freqtrade hyperopt loss forgot the project's own significance filter

**Context:** Building Module B's hyperopt (Phase 9), wrote a custom `IHyperOptLoss` (`project_hierarchy_loss.py`) so the search would optimize the project's actual Win Rate -> Sortino -> Net Profit hierarchy instead of one of Freqtrade's single-metric built-ins. First 100-epoch run's reported "best" result had exactly 1 trade -- a single lucky win claiming a perfect 100% win rate, which the composite score (win rate dominates by scale) rated above every larger, more realistic sample.

**Root cause:** the loss function scored every candidate on the hierarchy directly, with no floor on sample size -- exactly the overfitting failure mode `candidate_ranking.py`'s `dynamic_min_trade_count` filter was built in Phase 4 to catch during candidate *selection*, now silently reproduced inside the *search* itself, one layer earlier than where the project had already solved it once.

**Decision:** ported the same significance-floor formula into the loss function (duplicated rather than imported, since this file runs inside the Freqtrade Docker container which only has `user_data/` mounted -- the identical boundary that led to `data_ingestion/macro_data/loaders.py` existing separately from Module C's `freqai_utils.py`). Any candidate below the dynamic trade-count floor now scores as the worst possible outcome, same treatment as zero trades.

**Why this is logged:** a concrete instance of a rule existing in one place in a codebase not automatically applying itself everywhere it's needed -- the project had already solved "don't trust a thin sample" once, and it had to be solved again, explicitly, at a different layer. Worth remembering whenever a new consumer of backtest-style results gets added: check whether it needs its own copy of the significance filter, don't assume filtering happens downstream.

**Impact:** `modules/module_b_trend_following/user_data/hyperopts/project_hierarchy_loss.py`.

---

### 2026-08-17 — Module C moved off FreqAI entirely (Phase 9)

**Context:** The human director specified a significantly more rigorous ML methodology for Module C: dynamic per-fold threshold calibration (not a fixed 0.5 cutoff), purged expanding-window walk-forward CV, Triple Barrier labeling, class-imbalance handling, and SHAP-based feature selection. This is real, well-regarded quant ML practice (Lopez de Prado's purging and Triple Barrier method), but none of it fits FreqAI's walk-forward retraining without heavy, undocumented subclassing of `IFreqaiModel`/DataKitchen internals.

**Decision:** Build the ML methodology as a bespoke local Python pipeline (plain pandas/lightgbm/shap, no Freqtrade dependency at all), and keep a thin Freqtrade strategy only to consume the resulting per-candle signal for realistic trade simulation (fees, ROI, stoploss) -- not to reinvent Freqtrade's backtester. This was proposed explicitly and confirmed by the human director before any code was written, precisely because it reverses the Phase 2 instruction that named FreqAI for Module C.

**Also decided in the same conversation:** the target changed from Phase 6's continuous forward-volatility regression to a binary "high-risk" classification, since the new spec's vocabulary (probability threshold, precision-recall curve, F-beta, `scale_pos_weight`) is classification-specific -- confirmed with the human director before building, since it's a different model class and target definition, not an incremental tweak.

**Why this is logged as a reconsideration, not hidden as if bespoke were the plan all along:** the original FreqAI choice was reasoned and correct for Phase 6's simpler requirements; the new requirements outgrew it. Both READMEs (Module C's, this log) keep the FreqAI-based Phase 6 code and its rationale intact rather than deleting it, specifically so a reader can see the requirements evolve rather than assume more foresight than there was.

**Impact:** `modules/module_c_volatility_ml/` gained `labeling.py`, `threshold_calibration.py`, `feature_selection.py`, `walk_forward.py`, `train.py`, and a new thin strategy (`volatility_gate_signal.py`) alongside the untouched Phase 6 FreqAI strategy.

---

### 2026-08-17 — Triple Barrier calibration bug: 86% base rate from "textbook" defaults

**Context:** First run of the new labeling pipeline, with a common default (2.0x ATR barriers, 24-candle/1-day vertical barrier), produced an 86% "high-risk" label rate -- the gate was flagging almost every candle.

**Root cause:** first-passage-time math for a diffusive price process: a narrow barrier (2x ATR) is touched almost certainly given a long enough lookforward window (24 candles). The vertical barrier being "wide" relative to the horizontal barriers' width made the label degenerate into an almost-always-on flag, independent of anything the model could learn.

**Decision:** swept the ATR multiple empirically (2.0 through 5.0) while holding the vertical barrier fixed at the previously-agreed 24 candles (since that value was specifically tied to the purge-buffer sizing agreement), and landed on 4.5x ATR, giving a ~41% base rate.

**Why this is logged:** caught by actually inspecting the label distribution before running the full (expensive) walk-forward pipeline on it, not after. Worth remembering for any future barrier-based labeling: the ATR multiple and the vertical barrier length aren't independent choices, and a "reasonable-sounding" default for one can silently break the other.

**Impact:** `modules/module_c_volatility_ml/train.py` (`BARRIER_ATR_MULTIPLE`).

---

### 2026-08-17 — Real classifier signal, but a naive directional strategy built on it loses money -- and those are two different findings

**Context:** After fixing the base-rate bug, the full walk-forward pipeline produced a genuinely informative result: 0.51 precision against a 0.41 base rate, out-of-sample, across 57 folds -- modest but real predictive power. Translating that signal into a long-only Freqtrade strategy ("enter when calm, exit when high-risk") then lost 54% over a real, statistically meaningful 493-trade backtest.

**Why this isn't a contradiction:** the classifier predicts *whether a large move is coming*, symmetrically in either direction -- it was never a directional signal. "Calm" doesn't mean "price will rise." Forcing it into "enter long when calm" is an arbitrary translation, done only so the result could be reported on the project's Win Rate -> Sortino -> Net Profit hierarchy like every other module, and the loss reflects the weakness of *that translation*, not proof the classifier has no value.

**Decision:** report both findings honestly and separately in the README rather than letting the trade-level loss overshadow the classifier's real (if modest) signal, or letting the classifier's decent PR-curve numbers spin the trading loss as a near-miss. This also reinforces, empirically, the module's own stated purpose: Module C's value is as a *gate* on other modules' position-taking, not a standalone directional trader -- a conclusion this result actively supports.

**Impact:** `modules/module_c_volatility_ml/README.md`. Also affects Phase 7's capital allocator: Module C is now excluded for non-positive net profit (a real economic judgment) rather than insufficient sample size (a statistical technicality) -- a qualitatively better rejection reason than before.

---

### 2026-08-17 — News sentiment source: CryptoCompare News API + RSS, not CryptoPanic

**Context:** The original spec named CryptoPanic's free tier as the news source for Haiku-driven sentiment extraction. By the time Phase 0 started, CryptoPanic had moved its API behind a paid plan.

**Decision:** Use the CryptoCompare News API (`min-api.cryptocompare.com/data/v2/news/`) as the primary source — free, no API key required for basic use (a free key raises the rate limit), and returns structured JSON (title, body, source, tags, timestamp) already aggregated from ~50+ outlets. Supplement with a small, curated RSS feed set (CoinDesk, Cointelegraph, The Block, Decrypt) parsed via `feedparser`, deduplicated against the CryptoCompare stream by URL/title similarity, as a redundancy layer in case CryptoCompare has an outage or misses a story.

**Why this and not alternatives:**
- NewsAPI.org / GNews: free tiers are ~100 requests/day and not crypto-specific — too restrictive for continuous polling.
- CoinGecko: doesn't offer a real news-article endpoint, only asset "status updates" — not a fit for general market sentiment.
- Pure RSS-only: workable but noisier (raw HTML, inconsistent formatting across publishers) and requires more normalization before it reaches Haiku.

**Impact:** Confined entirely to `data_ingestion/news_sentiment/`. The Haiku prompt/interface downstream (headline + summary in → `{asset, sentiment, magnitude, event_type}` JSON out) is unaffected — this is a pure source-adapter swap.

---

### 2026-08-17 — Model IDs updated from spec

**Context:** The original spec referenced `claude-3-5-haiku` for the server daemon and "local Sonnet" for backtesting assistance — an older naming convention.

**Decision:** Use `claude-haiku-4-5` (server-side, always-on) and `claude-sonnet-5` (local, dev-time only). Same cost-tiering intent as the original spec — cheap model for continuous/high-frequency calls, capable model only for periodic local work — just updated to current model IDs.

**Impact:** Affects any code that instantiates the Anthropic client; no architectural change.

---

### 2026-08-17 — httpx's default logging leaked the live Telegram bot token

**Context:** After Phase 8 shipped, the human director tried `/status` against the bot and nothing happened -- because the bot had never actually been run as a standing polling process, only exercised via one-off scripts. Starting it for the first time (`python -m orchestrator.telegram_bot`, redirected to a log file for inspection) surfaced a more serious problem: `telegram_bot.py`'s `logging.basicConfig(level=logging.INFO)` also raised the `httpx` library's logger to INFO, and httpx logs the full URL of every HTTP request it makes. The Telegram Bot API embeds the bot token directly in the URL path (`api.telegram.org/bot<TOKEN>/<method>`), so every single API call -- `getMe`, `getUpdates`, `sendMessage` -- logged the live token in plaintext. That log file was then displayed while diagnosing the original issue, putting the token in this chat transcript.

**Immediate response:** stopped the process, deleted the local log file, and flagged the exposure to the human director directly rather than quietly patching around it -- a leaked credential is the user's to decide how to handle (rotate or accept the risk), not something to paper over.

**First fix, and why it wasn't quite right either:** set `logging.getLogger("httpx").setLevel(logging.WARNING)` to silence the token-bearing request logs. This worked for the token, but silenced *all* httpx-level visibility, including the routine `getUpdates` polling that would have shown whether a message was even received. The next verification attempt produced an empty log with no way to distinguish "no message arrived" from "arrived but now invisible."

**Second, complete fix:** added explicit `logger.info(...)` calls inside each command handler itself (`start`, `status`, `dry_run`, `handle_text`) -- logging the chat_id and, for free-text messages, only the *length* of the text (not its content, since that's exactly the channel the live-mode confirmation phrase travels over). This restored real observability without reintroducing the token leak, and immediately confirmed the fix worked: the next `/status` attempt showed `Received /status from chat_id=...` and `Replied to /status` in the log.

**Recommendation given to the user:** rotate the bot token via BotFather's `/token` command as a precaution, since it appeared in plaintext in a conversation transcript. Low severity (bot-control only, no funds or exchange access), but cheap to rotate and the right default reaction to any credential that leaks outside its intended storage, regardless of severity.

**Why this whole sequence is logged, not just the final state:** the first fix looked complete and wasn't -- it traded one real problem (secret exposure) for a different real problem (no observability), which only surfaced on the very next verification attempt. Worth remembering for any future "silence this logger" fix: check what else that logger was the only source of visibility into before calling it done.

**Impact:** `orchestrator/telegram_bot.py`.

---

### 2026-08-17 — No open-ended Haiku chat fallback in the Telegram bot

**Context:** Building the Telegram bot (Phase 8), the natural design for an "unrecognized message" fallback would route it to Haiku for a conversational reply -- friendlier than a static error message.

**Decision:** Didn't build it. Haiku's only role in `orchestrator/` is formatting already-computed, already-safe structured data (`status_formatter.py`) into text -- it never receives free-form user text and produces a response that could be mistaken for, or accidentally influence, a system decision. Unrecognized messages get a static instruction string instead.

**Why:** this is the same "architectural isolation over access control" principle the Phase 3 safety kernel established (the LLM has no code path to a risk parameter, not just an instruction not to touch one) applied one layer up, at the bot itself. An open-ended chat handler sitting next to the exact-phrase live-mode check would be a standing invitation to eventually let Haiku's interpretation influence that check "just this once" -- easier to never build the temptation than to police it later.

**Impact:** `orchestrator/telegram_bot.py`'s `handle_text()`. Locked in by `tests/test_telegram_bot.py`.

---

### 2026-08-17 — A cost-model test's own docstring tripped its own check

**Context:** Wrote `test_orchestrator_never_references_sonnet_or_opus` (mirroring `test_safety_isolation.py`'s pattern) to enforce the Haiku-only cost model concretely. It failed immediately on the first run.

**What actually happened:** the failure wasn't a real Sonnet/Opus usage -- it was `status_formatter.py`'s own module docstring, which *explains* the Haiku-only policy in prose ("never Sonnet/Opus"), tripping a naive substring search for the bare words "sonnet"/"opus".

**Decision:** Narrowed the check to the actual model-id prefixes that would indicate real usage (`claude-sonnet`, `claude-opus`) instead of the bare English words, which legitimately appear in policy-explaining comments without being a violation of that policy.

**Why this is logged:** a small, concrete example of a broader pattern worth remembering for any future "scan the code for forbidden string X" test: forbidding the *word* often isn't the same as forbidding the *usage*, especially once the code has grown enough documentation explaining why the word matters.

**Impact:** `tests/test_orchestrator_cost_model.py`.

---

### 2026-08-17 — Capital allocation weighted by rank, not raw Sortino magnitude

**Context:** Building the cross-module capital allocator (Phase 7), the initial plan was to weight each module's capital share proportionally to its raw Sortino ratio -- a reasonable-sounding idea, since Sortino is already a risk-adjusted-return measure.

**What real data caught before that plan was implemented:** adding a Sortino calculation to Module A's funding-yield analysis (needed so Module A could be ranked on the same hierarchy as B/C) and running it against real funding-rate history produced Sortino values of 103-136 -- because funding-rate "returns" are tiny, low-variance, near-fixed-yield numbers, and annualizing a ratio of mean/downside-deviation blows up fast at that scale. Module B and C's Sortino, computed on trade-level P&L, sit in the -2 to +2 range typical of a trading strategy. Seeing the real 103-136 numbers before writing the allocator's weighting formula made it obvious that weighting by raw magnitude would let Module A swamp everything else by two orders of magnitude -- not because it's genuinely that much better, but because the two Sortino calculations aren't measuring comparable things.

**Decision:** Weight by rank position within the eligible, hierarchy-sorted list (Win Rate -> Sortino -> Net Profit) instead of by raw metric magnitude. This only ever asks "who's better, by the spec's own ordering," never "how many times better" -- sidestepping the cross-scale comparability problem entirely rather than trying to normalize incompatible units.

**Why this is logged:** it's a real instance of a common multi-strategy portfolio construction pitfall (comparing Sharpe/Sortino ratios computed on structurally different return series without normalization) -- avoided here specifically *because* real numbers were checked before finalizing the design, not because it was anticipated in the abstract.

**Impact:** `orchestrator/capital_allocator.py`. Applies to any future module whose "returns" aren't directly comparable in kind to a trading strategy's per-trade P&L.

---

### 2026-08-17 — FreqAI needs a different Docker image tag than plain Freqtrade

**Context:** Module B's Freqtrade Docker setup uses `freqtradeorg/freqtrade:stable`. Trying to use FreqAI (Module C) against that same image failed immediately: `list-freqaimodels` raised `ModuleNotFoundError: No module named 'datasieve'` — the base image doesn't bundle FreqAI's ML dependencies (scikit-learn, LightGBM, datasieve) at all.

**Decision:** Module C uses `freqtradeorg/freqtrade:stable_freqai` instead — a separate image tag Freqtrade maintains specifically for FreqAI, confirmed via `list-freqaimodels` to have `LightGBMRegressor` and several other models working (PyTorch/RL-based models fail to load in this tag, which is expected and fine -- they need a further `stable_freqaitorch`/`stable_freqairl` tag this project doesn't need).

**Impact:** `modules/module_c_volatility_ml/docker-compose.yml`. Worth remembering if any future module or a rebuild of Module B ever needs FreqAI-adjacent functionality.

---

### 2026-08-17 — Hummingbot's `CONFIG_PASSWORD` env var breaks first-time setup

**Context:** After the human director set `HUMMINGBOT_CONFIG_PASSWORD` in `.env` and ran the documented one-time interactive setup command, the container exited immediately with no visible output — even with `-it`. The default Docker image entrypoint (`docker-entrypoint.sh`) redirects the app's stderr into `logs/errors.log`, which is host-mounted, so the real error was recoverable: `FileNotFoundError` on `conf/.password_verification`, from `Security.login() -> validate_password()`.

**Root cause, found by reading `bin/hummingbot_quickstart.py` directly:** `main()` checks for a `CONFIG_PASSWORD` env var *before* deciding whether to show the interactive password-creation screen (`login_prompt()`) or skip straight to validating a password that's assumed to already exist. Since `docker-compose.yml` always injects `CONFIG_PASSWORD` from `.env` (needed for headless runs later), as soon as `HUMMINGBOT_CONFIG_PASSWORD` had a real value, every run -- including the very first one, before any password had ever been created -- skipped the creation flow and crashed trying to validate a file that couldn't exist yet.

**Decision:** For the one-time interactive bootstrap only, override the variable to empty on the command line: `docker compose run --rm -it -e CONFIG_PASSWORD= hummingbot`. This forces the real first-run flow. `docker-compose.yml` itself is unchanged and correct for every run after that.

**Impact:** `modules/module_a_cash_carry/README.md`'s setup instructions. Worth remembering for Module C or any other Hummingbot instance set up later in this project -- the same override will be needed for its first run too.

---

### 2026-08-17 — Module A's "paper trade" spans two different mechanisms

**Context:** Hummingbot's built-in paper-trade simulator (`connector/exchange/paper_trade/`) only wraps spot exchange connectors — there's no simulated equivalent under `connector/derivative/`. Module A's strategy needs a spot leg *and* a perpetual leg.

**Decision:** Spot leg uses Hummingbot's `binance_paper_trade` simulator (fully fake, no keys). Perpetual leg uses `binance_perpetual_testnet` — Binance's own dedicated Futures Testnet sandbox, a separate site and separate account from Spot Testnet, requiring its own API keys from testnet.binancefuture.com.

**Why this is worth logging:** both are zero-real-money, consistent with the project's dry-run-by-default rule, but they get there through genuinely different mechanisms (a local simulator vs. a real exchange's sandbox environment) — worth knowing explicitly rather than assuming "paper trade" is one uniform thing across every connector.

**Impact:** `modules/module_a_cash_carry/conf/strategies/conf_spot_perpetual_arbitrage_btc.yml` and `.env.example` (`BINANCE_FUTURES_TESTNET_API_KEY`, distinct from `BINANCE_API_KEY`).

---

### 2026-08-17 — Freqtrade and Hummingbot run via Docker, not pip

**Context:** Both frameworks depend on TA-Lib, a C library that must be compiled for the exact OS/chip/Python combination in use. In Phase 2, `pip install hummingbot` failed outright on this machine — a pinned Cython build dependency doesn't support Python 3.13.

**Decision:** Run both frameworks via their official Docker images instead of installing them into the project's Python environment. A Docker image bundles the entire working environment (OS, Python version, compiled libraries) pre-built and pre-tested by the framework's maintainers, so using it sidesteps the version-matching problem entirely rather than trying to solve it by hand.

**Why this matters beyond just "it installs":** the same container behaves identically in local development and on the Linux production server, removing an entire class of "works on my machine" bugs before they exist. It also isolates each module's dependencies from the others and from the lightweight Telegram daemon — valuable on a $0–5/month server where a remote dependency conflict is expensive to debug.

**Impact:** `modules/module_b_trend_following/docker-compose.yml` (Module A will follow the same pattern in Phase 5). Verified in Phase 4: TA-Lib indicators (EMA, ADX, ATR) worked immediately inside the container with zero setup — the same category of dependency that failed under pip.

---

### 2026-08-17 — Haiku JSON output needs defensive parsing

**Context:** The first live test of `haiku_sentiment.py` failed `json.loads` despite a system prompt explicitly saying "no markdown fences." Haiku wrapped the array in ` ```json ... ``` ` anyway.

**Decision:** Strip markdown code fences from Haiku's response before parsing, rather than trying to prompt-engineer away 100% of formatting variance. Applies to every future Haiku JSON-extraction call in this project (Telegram status formatting, etc.) — never assume a clean, fenceless response just because the prompt asked for one.

**Impact:** `data_ingestion/news_sentiment/haiku_sentiment.py` (`_strip_markdown_fences`). Any new Haiku JSON call site should reuse this pattern.

---

### 2026-08-17 — Repo name diverges from local folder name

**Context:** Spec required the GitHub repo name to exactly match the local project folder. Local folder is `Crypto investment AI Agent` (contains spaces), which is not a valid GitHub repository name.

**Decision:** GitHub repo created as `crypto-investment-ai-agent` (kebab-case). Local folder left as-is; git doesn't require the working directory name to match the remote repo name, so no local renaming was needed.

**Impact:** Cosmetic only. Noted here so it isn't mistaken for an inconsistency later.
