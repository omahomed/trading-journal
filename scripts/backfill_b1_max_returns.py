#!/usr/bin/env python3
"""CLI wrapper to backfill / reconcile trades_summary.b1_max_return_pct.

The reconcile engine lives in b1_reconcile.py (repo root) so the always-on
API can share it. This script is for manual / ad-hoc runs from a dev machine.

Two modes:
  - default (backfill): only rows where b1_max_return_pct IS NULL.
  - --reconcile: ALL open equity rows — recompute the close-basis peak and
    raise any stored value that has fallen behind. The monotonic guard only
    ever raises, never lowers, so this is safe and idempotent.

In production the API runs the same engine daily (see api/main.py), so you
normally don't need this — reach for it to force an immediate heal or to
eyeball the tier of every open position.

Usage:
    python scripts/backfill_b1_max_returns.py                 # backfill NULLs
    python scripts/backfill_b1_max_returns.py --reconcile     # heal all open rows
    python scripts/backfill_b1_max_returns.py --reconcile --dry-run
    python scripts/backfill_b1_max_returns.py --portfolio "CanSlim"
    python scripts/backfill_b1_max_returns.py --sleep 1.0
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from b1_reconcile import reconcile_open_positions  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("backfill_b1_max")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Report planned updates without writing.")
    parser.add_argument("--reconcile", action="store_true",
                        help="Recompute ALL open equity rows and raise any "
                             "stored peak that has fallen behind (default: "
                             "only fill NULL rows).")
    parser.add_argument("--portfolio", default=None,
                        help="Limit to one portfolio (default: all).")
    parser.add_argument("--sleep", type=float, default=0.5,
                        help="Seconds to sleep between yfinance calls (default 0.5).")
    args = parser.parse_args()

    scope = "ALL open equity rows" if args.reconcile else "open equity rows with NULL b1_max_return_pct"
    log.info("Reconciling candidates: %s ...", scope)

    summary = reconcile_open_positions(
        portfolio=args.portfolio,
        reconcile=args.reconcile,
        dry_run=args.dry_run,
        sleep=args.sleep,
    )

    rows = summary["rows"]
    if summary["total"] == 0:
        log.info("No candidates found.")
        return 0

    print(f"{'TICKER':<10} {'ENTRY DATE':<12} {'ENTRY $':>10} {'STORED %':>10} {'PEAK B1 %':>11} {'TIER':<5} {'ACTION':<9} {'TRADE_ID':<14} {'PORTFOLIO'}")
    print("-" * 112)
    for r in rows:
        stored_disp = f"{r['stored']:.2f}" if r["stored"] is not None else "—"
        pct_disp = f"{r['pct']:>10.2f}%" if r["pct"] is not None else f"{'—':>11}"
        print(f"{r['ticker']:<10} {r['entry_date']!s:<12} {r['entry_price']:>10.2f} "
              f"{stored_disp:>10} {pct_disp} {r['tier']:<5} {r['action']:<9} "
              f"{r['trade_id']:<14} {r['portfolio']}")

    c = summary["counters"]
    print("-" * 112)
    print(f"Summary: {c['raised']} raised · {c['unchanged']} already-current · "
          f"{c['skipped_no_b1']} skipped (no B1) · {c['skipped_no_data']} skipped (no yfinance data) · "
          f"{c['errors']} errors")
    if args.dry_run:
        print("DRY-RUN: no rows were modified. Re-run without --dry-run to persist.")
    return 0 if c["errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
