# Technical Decisions Log

Running log of non-obvious technical decisions, in chronological order. Each entry: what was decided, why, and what it affects.

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
