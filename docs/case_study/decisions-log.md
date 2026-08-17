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
