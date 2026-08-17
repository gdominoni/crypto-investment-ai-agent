# Orchestrator

Cross-module coordination layer: the Telegram bot (Haiku-powered NLP/status formatting), the dynamic capital allocator, and cross-module KPI ranking (Win Rate → Sortino Ratio → Net Profit) used for both rebalancing and monitoring.

Also owns the live/dry-run switch: the system always boots in dry-run, and only an explicit human-issued Telegram confirmation command can move any module to live execution.

Status: not yet built (Phases 7–8).
