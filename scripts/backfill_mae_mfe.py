#!/usr/bin/env python3
"""CLI wrapper to backfill / reconcile trades_summary MAE/MFE columns.

The compute engine lives in api/mae_mfe_reconcile.py so the always-on
API can share it. This script is for manual / ad-hoc runs from a dev
machine (or a Railway shell) — e.g. the one-time backfill immediately
after migration 046 lands.

Scope (spec §3):
  * Default: OPEN equity positions only. This is the initial-backfill
    scope + the daily loop's ongoing scope.
  * --include-closed: extend to CLOSED equity campaigns too. Kept off
    by default per the spec ("Phase 2 — structure the script so
    pointing it at closed trades later is trivial, but do not run it
    now."). Off unless you explicitly pass the flag.

Idempotent by construction: atr21_entry_pct is only written when its
current value is NULL (frozen snapshot per spec); every other field is
recomputed from a fresh yfinance pull. Re-running against the same rows
does the same DB writes each time and produces the same values.

Usage:
    python scripts/backfill_mae_mfe.py                        # OPEN equity, all portfolios
    python scripts/backfill_mae_mfe.py --portfolio "CanSlim"  # one portfolio
    python scripts/backfill_mae_mfe.py --dry-run              # report only, no writes
    python scripts/backfill_mae_mfe.py --sleep 1.0            # slower yfinance cadence
    python scripts/backfill_mae_mfe.py --include-closed       # Phase 2: closed trades too
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from api.mae_mfe_reconcile import reconcile_open_positions  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("backfill_mae_mfe")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Report planned updates without writing.")
    parser.add_argument("--portfolio", default=None,
                        help="Limit to one portfolio (default: all).")
    parser.add_argument("--sleep", type=float, default=0.3,
                        help="Seconds to sleep between yfinance calls (default 0.3).")
    parser.add_argument("--include-closed", action="store_true",
                        help="Also process CLOSED equity campaigns (Phase 2 scope). "
                             "Default: OPEN only.")
    args = parser.parse_args()

    scope = "OPEN + CLOSED equity" if args.include_closed else "OPEN equity"
    log.info("Reconciling %s positions ...", scope)

    summary = reconcile_open_positions(
        portfolio=args.portfolio,
        include_closed=args.include_closed,
        dry_run=args.dry_run,
        sleep=args.sleep,
    )

    rows = summary["rows"]
    if summary["total"] == 0:
        log.info("No candidates found.")
        return 0

    # Header sizing chosen to fit the widest expected content — long
    # option tickers like "AMZN 261016 $240C" would blow the column,
    # but options are filtered upstream so equity tickers only.
    hdr = (
        f"{'TICKER':<8} {'ENTRY DATE':<12} {'ENTRY $':>10} "
        f"{'MAE %':>8} {'MFE %':>8} {'RETRACE %':>10} "
        f"{'ATR21 %':>9} {'D→MAE':>6} {'D→MFE':>6} "
        f"{'ACTION':<8} {'TRADE_ID':<14} {'PORTFOLIO'}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        # Rows can be "skip" / "err" (no excursion data) so guard every field.
        def _f(k: str, w: int, dp: int = 2, suffix: str = "") -> str:
            v = r.get(k)
            if v is None:
                return f"{'—':>{w}}"
            return f"{v:>{w-1}.{dp}f}{suffix}"
        atr = r.get("atr21_entry_pct")
        atr_disp = f"{atr:>8.2f}%" if atr is not None else f"{'—':>9}"
        print(
            f"{(r.get('ticker') or '')[:8]:<8} "
            f"{r.get('entry_date') or '—'!s:<12} "
            f"{_f('entry_price', 10):>10} "
            f"{_f('mae_pct', 8):>8} "
            f"{_f('mfe_pct', 8):>8} "
            f"{_f('max_retrace_pct', 10):>10} "
            f"{atr_disp:>9} "
            f"{r.get('days_to_mae', '—')!s:>6} "
            f"{r.get('days_to_mfe', '—')!s:>6} "
            f"{r.get('action', '—'):<8} "
            f"{str(r.get('trade_id') or '—'):<14} "
            f"{r.get('portfolio', '—')}"
        )

    c = summary["counters"]
    print("-" * len(hdr))
    print(
        f"Summary: {c['updated']} updated · "
        f"{c['atr_snapshotted']} ATR21 snapshotted · "
        f"{c['skipped_no_data']} skipped (no data) · "
        f"{c['skipped_bad_row']} skipped (bad row) · "
        f"{c['errors']} errors"
    )
    if args.dry_run:
        print("DRY-RUN: no rows were modified. Re-run without --dry-run to persist.")
    return 0 if c["errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
