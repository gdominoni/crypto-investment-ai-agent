"""Checks a host is actually ready to run the live daemon, BEFORE starting it.

Every check here exists because the failure it catches is silent or
slow-to-notice on a remote machine. A daemon with no `.env` crashes loudly and
you find out in seconds; a daemon whose market data is three months stale, or
whose production state was never migrated, starts perfectly, reports itself
healthy, sends a cheerful "Live daemon started", and quietly does nothing
useful. Those are the ones worth a script.

Reads only -- it never writes state, never opens a position, and the single
network call it makes (Telegram's getMe) has no side effect.

Run on the host, from the project root:
    python3 -m deploy.preflight
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# The live daemon needs none of these, and saying so is the point: freqtrade
# pulls in sklearn, sqlalchemy and a few hundred MB, and scipy another ~100MB,
# for code that only ever runs on a human's own laptop (the hyperopt
# cross-check) or offline (forecast/). Installing them on the host is the
# difference between a 1GB VPS being comfortable and being tight.
HOST_DOES_NOT_NEED = ["freqtrade", "scipy", "sklearn"]

REQUIRED_IMPORTS = ["pandas", "numpy", "pyarrow", "anthropic", "dotenv", "ccxt", "requests"]
REQUIRED_ENV = ["ANTHROPIC_API_KEY", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"]

# Hourly candles older than this mean the mechanical scan is reading a window
# with a hole in it. One missed hourly cycle is normal; a day is not.
MARKET_DATA_STALE_HOURS = 36

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))


def _python_version() -> None:
    v = sys.version_info
    check("Python >= 3.11", v >= (3, 11), f"found {v.major}.{v.minor}.{v.micro}")


def _imports() -> None:
    missing = []
    for mod in REQUIRED_IMPORTS:
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    check("Required packages installed", not missing,
          "all present" if not missing else f"MISSING: {', '.join(missing)}")

    present = []
    for mod in HOST_DOES_NOT_NEED:
        try:
            __import__(mod)
            present.append(mod)
        except ImportError:
            pass
    # Never a failure -- having them costs disk, not correctness.
    check("Host is not carrying local-only dependencies", True,
          "lean install" if not present
          else f"installed but unused here: {', '.join(present)} (safe to omit; see deploy/README.md)")


def _env() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    missing = [k for k in REQUIRED_ENV if not os.environ.get(k)]
    check(".env has every required key", not missing,
          "all set" if not missing else f"MISSING: {', '.join(missing)}")


def _telegram() -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        check("Telegram token works", False, "no token to test")
        return
    try:
        import requests
        r = requests.get(f"https://api.telegram.org/bot{token}/getMe", timeout=15)
        data = r.json()
        ok = bool(data.get("ok"))
        check("Telegram token works", ok,
              f"bot @{data['result']['username']}" if ok else f"rejected: {data.get('description', r.status_code)}")
    except Exception as e:
        check("Telegram token works", False, f"{type(e).__name__}: {e}")


def _market_data() -> None:
    """The check that would have caught the inert mechanical arm.

    Both frames matter and for different reasons: the daily series feeds the
    weekly battery, the hourly one feeds the mechanical scan -- and it was the
    hourly one that nothing refreshed, silently, for the entire life of
    production before it was found by running a real cycle.
    """
    try:
        import pandas as pd
        from candidates.run_battery import COINS
    except Exception as e:
        check("Market data present and fresh", False, f"could not load: {e}")
        return

    spot = ROOT / "data" / "market" / "binance" / "spot"
    now = datetime.now(timezone.utc)
    problems = []
    for tf, limit in (("1d", timedelta(days=3)), ("1h", timedelta(hours=MARKET_DATA_STALE_HOURS))):
        for coin in COINS:
            path = spot / f"{coin}_{tf}.parquet"
            if not path.exists():
                problems.append(f"{coin}_{tf} missing")
                continue
            try:
                last = pd.to_datetime(pd.read_parquet(path, columns=["timestamp"])["timestamp"], utc=True).max()
            except Exception as e:
                problems.append(f"{coin}_{tf} unreadable ({type(e).__name__})")
                continue
            if now - last > limit:
                problems.append(f"{coin}_{tf} last candle {last:%Y-%m-%d %H:%M}Z")
    check("Market data present and fresh", not problems,
          f"{len(COINS)} coins, daily + hourly, all current" if not problems
          else "; ".join(problems[:4]) + (" ..." if len(problems) > 4 else "")
          + "  -> run: python3 -m data_ingestion.market_data.binance_fetcher")


def _production_state() -> None:
    """A host that starts with empty state is not broken -- it is a fresh
    start, which is a legitimate choice. But it is almost never the INTENDED
    one when a completed replay exists to migrate, and starting empty is
    indistinguishable from a migration that silently did not happen."""
    try:
        from execution.live_test_state import load_open_trades, load_trade_log
        from execution.signal_store import load_battery_state
        from llm_pipeline.dynamic_candidates import registered_specs
    except Exception as e:
        check("Production state migrated", False, f"could not load: {e}")
        return

    tracked = len(registered_specs())
    accepted = len(((load_battery_state() or {}).get("candidates") or {}))
    tests = len(load_trade_log())
    still_open = len(load_open_trades())
    populated = tracked > 0 or tests > 0
    check("Production state migrated", populated,
          f"{tracked} conditions tracked, {accepted} accepted, "
          f"{tests} live tests ({still_open} still open)"
          if populated else
          "EMPTY -- if this host should inherit the replay, run "
          "scripts/migrate_replay_state_to_production.py and copy the state files across")


def _clock() -> None:
    """Every interval in the daemon, every timestamp in state, and the whole
    staleness logic is UTC-based, so the host's local timezone is irrelevant --
    but a wrong CLOCK is not. Skew shifts what "the last 24 hours" means."""
    try:
        import requests
        r = requests.get("https://api.telegram.org/", timeout=10)
        server_date = r.headers.get("Date")
        if not server_date:
            check("System clock", True, "could not verify (no Date header) -- assumed fine")
            return
        from email.utils import parsedate_to_datetime
        skew = abs((datetime.now(timezone.utc) - parsedate_to_datetime(server_date)).total_seconds())
        check("System clock", skew < 120, f"{skew:.0f}s from network time")
    except Exception as e:
        check("System clock", True, f"could not verify ({type(e).__name__}) -- assumed fine")


def _disk() -> None:
    import shutil
    free_gb = shutil.disk_usage(ROOT).free / (1024 ** 3)
    # Market data grows by a few MB a month; state by less. 2GB is generous
    # headroom, not a tight bound.
    check("Disk space", free_gb > 2, f"{free_gb:.1f} GB free")


def main() -> int:
    _python_version()
    _imports()
    _env()
    _telegram()
    _market_data()
    _production_state()
    _clock()
    _disk()

    width = max(len(name) for name, _, _ in results)
    print()
    for name, ok, detail in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name.ljust(width)}   {detail}")
    failed = [name for name, ok, _ in results if not ok]
    print()
    if failed:
        print(f"{len(failed)} check(s) failed: {', '.join(failed)}.")
        print("Fix these before starting the daemon -- see deploy/README.md.")
        return 1
    print("Ready. Start with:  sudo systemctl enable --now crypto-agent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
