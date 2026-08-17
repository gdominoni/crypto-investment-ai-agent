# Phase 8: Telegram Orchestrator

**Goal:** build the human-facing control surface — status reporting via Haiku, and the live-mode confirmation flow the safety kernel (Phase 3) was built to gate — and prove both work against a real Telegram delivery, not just a passing test suite.

## The prompt

"continue" after Phase 7. The framing for this phase was set two messages earlier, when the human director was told what to expect next: "the Telegram orchestrator... This is also where the server-side cost model (Haiku-only, no Sonnet) gets exercised for real." That framing shaped the two things this phase treats as non-negotiable to verify concretely: the cost model, and the safety-critical confirmation flow.

## Decisions made in this phase

1. **One status-gathering function, shared by the CLI and the bot.** `orchestrator/status.py`'s `gather_system_status()` is the single source of truth for execution mode, module KPIs, allocation decision, and a live circuit-breaker check — both `run_allocation.py` (Phase 7's CLI) and the bot's `/status` command call the same function, so they can't silently drift into reporting different numbers for the same underlying state.
2. **The circuit-breaker check reuses Phase 2's cached OHLC data**, not Freqtrade's feather-format cache, specifically to avoid coupling the orchestrator to Freqtrade's internal data format just for a read — one more instance of the project's data-ingestion pipeline paying off in a later phase it wasn't originally built for (same pattern as Module C's VIX merge in Phase 6).
3. **Haiku's role is formatting only, with no open-ended chat fallback.** Full reasoning in [decisions-log.md](decisions-log.md) — the short version: giving Haiku any path to interpret free-form text into a system decision, even an unrelated "friendly chat" feature, is a standing invitation for that boundary to blur later. Easier to never build it than to police it.
4. **The live-mode confirmation check happens before Haiku is ever invoked**, against the raw incoming text, using the exact function (`safety.execution_mode.request_live_mode`) and exact phrase built in Phase 3. This phase didn't just reuse that function — it built the first real caller of it, and locked the behavior in with dedicated tests (`tests/test_telegram_bot.py`): exact match switches to live, wrong case doesn't, the phrase embedded in a longer sentence doesn't.
5. **Scheduled/periodic status checks (APScheduler) deferred to Phase 9.** Building a scheduler now would add a dependency I can't meaningfully verify without waiting hours for a cron tick to fire — it belongs with actually deploying this to run continuously on a VPS, not before.

## A real bug caught by the cost-model test itself

`test_orchestrator_never_references_sonnet_or_opus`, written to mirror Phase 3's `test_safety_isolation.py`, failed on its first run — not because of a real violation, but because `status_formatter.py`'s own module docstring explains the Haiku-only policy in prose, and a naive substring search for the bare words "sonnet"/"opus" doesn't distinguish *explaining* a policy from *violating* it. Fixed by narrowing the check to actual model-id prefixes (`claude-sonnet`, `claude-opus`). Small, but a reusable lesson for any future "scan the code for forbidden string X" test, logged in the decisions log.

## What got built and verified

- `orchestrator/status.py` — `gather_system_status()`, tested implicitly by every other verification step in this phase (no dedicated unit tests, since it's thin glue code over already-tested functions and real local files — the value is in the integration, which is what got exercised).
- `orchestrator/status_formatter.py` — Haiku-powered formatting, verified with a real API call against the real, current system status. Produced a correct, well-organized summary from real numbers (Module A at 84.6% win rate getting 60% allocation, Modules B/C correctly flagged as excluded for insufficient samples) without any hand-holding in the prompt beyond the JSON schema.
- `orchestrator/telegram_bot.py` — `/start`, `/status`, `/dry_run`, and the raw-text live-confirmation handler. 5 new tests locking in the safety-critical text-matching behavior.
- `tests/test_orchestrator_cost_model.py` — 2 tests enforcing the Haiku-only claim concretely.
- **Real Telegram delivery**: built the actual Haiku-formatted status message from the real system state and sent it through the real bot token to the human director's real chat — confirmed delivered (`message_id: 3`), not just "the code should work."

## Update: inbound testing surfaced a real credential leak

The human director tried `/status` right after this phase shipped and got no response — because the bot had only ever been exercised through one-off scripts, never actually run as a standing polling process. Starting it for real to diagnose that surfaced a more serious issue: `logging.basicConfig(level=logging.INFO)` also raised `httpx`'s logger to INFO, and httpx logs full request URLs — which, for the Telegram Bot API, embed the live bot token directly in the path. Every API call logged the token in plaintext, and that log was displayed while debugging, putting the token in the conversation transcript.

The first fix (silencing `httpx`'s logger entirely) traded one real problem for another: it also removed the only visibility into whether messages were being received at all, which surfaced immediately on the next verification attempt (an empty log, no way to tell "nothing arrived" from "arrived but now invisible"). The complete fix added explicit handler-level logging (chat_id, and message *length* rather than content, since free text is exactly the channel the live-mode confirmation phrase travels over) — restoring observability without the token ever appearing in a log again. Confirmed working immediately after: the next `/status` attempt logged `Received /status from chat_id=...` and `Replied to /status`.

Full incident writeup in [decisions-log.md](decisions-log.md). The human director was advised to rotate the bot token via BotFather as a precaution.

## Still pending

- Scheduled/periodic checks, and running this as an actual always-on daemon rather than an invoked script — Phase 9 territory.
- No news-sentiment integration into the bot yet, despite `data_ingestion/news_sentiment/` existing since Phase 2 — a natural addition once there's a concrete alerting use case for it, not added speculatively here.
