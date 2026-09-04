"""Deliver any Telegram messages queued while the API was rate-limiting us.

    python3 -m telegram.flush            # send what can be sent, report the rest
    python3 -m telegram.flush --wait 6   # keep retrying for up to 6 hours

The replay flushes its own outbox when it finishes (see
replay/orchestrator.py::_finish), so this exists for the cases that bypasses:
a run killed part-way, a machine that slept through the flush, or a ban that
outlasted even the end-of-run wait. Safe to run at any time -- it sends only
what is already queued and does nothing when the queue is empty.
"""
from __future__ import annotations

import argparse

from telegram.bot import flush_outbox, outbox_pending


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wait", type=float, default=0.0,
                    help="hours to keep retrying while Telegram is rate-limiting (default: don't wait)")
    args = ap.parse_args()

    pending = outbox_pending()
    if not pending:
        print("Outbox is empty -- nothing to deliver.")
        return 0

    print(f"{pending} message(s) queued. Delivering...")
    left = flush_outbox(max_wait_s=args.wait * 3600)
    if left:
        print(f"{left} still queued -- Telegram is still rate-limiting. "
              f"Re-run later, or use --wait to sit it out.")
        return 1
    print("All queued messages delivered.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
