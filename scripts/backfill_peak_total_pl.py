"""Backfill trades_summary.peak_total_pl for every cushion-qualified campaign.

For each campaign with b1_max_return_pct >= 50 (post-migration-065 audience):
  1. Load detail rows sorted by (date, trx_id).
  2. Fetch yfinance daily bars from B1 date through today (or closed_date).
  3. Walk day-by-day:
       - Apply any detail rows dated ON that day, updating shares / avg_cost /
         realized_bank in avg-cost accounting (method-invariant for total P&L).
       - Compute total_pl_at_high = realized_bank + shares × (day_high − avg_cost)
         using END-OF-DAY state (idealization matching MFE).
       - Track max across all bars.
  4. Write peak_total_pl via the same monotonic-up SQL guard update_b1_max_return_pct
     uses (never lowers the stored value; safe to re-run).

Dry-run by default; pass --commit to write. Filter to a single campaign with
--trade-id (e.g. --trade-id 202604-013 for DELL) to spot-check without waiting
on the full sweep. All 12 armed campaigns take ~5-10 minutes end-to-end on a
warm yfinance cache.

Runtime maintenance: b1_reconcile computes today's total_pl_at_high and passes
it to the same idempotent guard on every daily sweep — this script is the
historical seed, the reconcile keeps it ratcheted forward.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date, timedelta

sys.path.insert(0, "/Users/momacbookair/Developer/my_code")

import db_layer as db  # noqa: E402
from b1_reconcile import compute_peak_total_pl_since_b1  # noqa: E402

FOUNDER_UUID = "d7e8f9a0-1b2c-4d3e-8f4a-5b6c7d8e9f0a"
CUSHION_THRESHOLD = 50.0

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("backfill_peak_total_pl")


def fetch_armed_campaigns(cur, only_trade_id: str | None) -> list[dict]:
    """Every open+closed campaign with b1_max_return_pct >= 50 (or the one
    trade_id specified by --trade-id). We backfill both statuses so that a
    campaign that had a big peak, exited via SR7, and is now closed also
    gets its historical peak_total_pl on record (useful for lot-excursion
    studies + eventual post-mortems)."""
    where_extra = ""
    params: tuple = (CUSHION_THRESHOLD,)
    if only_trade_id:
        where_extra = " AND s.trade_id = %s"
        params = (CUSHION_THRESHOLD, only_trade_id)
    cur.execute(
        f"""
        SELECT p.name AS portfolio_name, s.trade_id, s.ticker,
               s.b1_max_return_pct, s.realized_pl, s.shares, s.avg_entry,
               s.peak_total_pl, s.open_date, s.closed_date, s.status
          FROM trades_summary s
          JOIN portfolios p ON p.id = s.portfolio_id
         WHERE s.deleted_at IS NULL
           AND s.b1_max_return_pct >= %s
           {where_extra}
         ORDER BY s.b1_max_return_pct DESC
        """,
        params,
    )
    return list(cur.fetchall())


def fetch_details(cur, portfolio_name: str, trade_id: str) -> list[dict]:
    cur.execute(
        """
        SELECT d.action, d.shares, d.amount, d.date, d.trx_id
          FROM trades_details d
          JOIN portfolios p ON p.id = d.portfolio_id
         WHERE p.name = %s AND d.trade_id = %s
           AND d.deleted_at IS NULL
         ORDER BY d.date, d.trx_id
        """,
        (portfolio_name, trade_id),
    )
    return list(cur.fetchall())


def update_peak_total_pl(cur, portfolio_name: str, trade_id: str, new_peak: float):
    """Monotonic-up guard, mirroring update_b1_max_return_pct's SQL shape."""
    cur.execute(
        "UPDATE trades_summary "
        "SET peak_total_pl = %s "
        "WHERE portfolio_id = (SELECT id FROM portfolios WHERE name = %s) "
        "  AND trade_id = %s "
        "  AND deleted_at IS NULL "
        "  AND (peak_total_pl IS NULL OR peak_total_pl < %s) "
        "RETURNING peak_total_pl",
        (new_peak, portfolio_name, trade_id, new_peak),
    )
    return cur.fetchone()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--commit", action="store_true",
                    help="Actually write; default is dry-run")
    ap.add_argument("--trade-id",
                    help="Filter to a single trade_id (spot-check mode)")
    ap.add_argument("--sleep", type=float, default=0.25,
                    help="Seconds to sleep between yfinance calls (default 0.25)")
    args = ap.parse_args()

    db.current_user_id.set(FOUNDER_UUID)
    from psycopg2.extras import RealDictCursor

    with db.get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            campaigns = fetch_armed_campaigns(cur, args.trade_id)
            if not campaigns:
                log.warning("No armed campaigns found.")
                return 0
            log.info("Backfilling %d armed campaign(s) [%s]",
                     len(campaigns), "COMMIT" if args.commit else "DRY-RUN")

            for c in campaigns:
                pf = c["portfolio_name"]
                tid = c["trade_id"]
                ticker = c["ticker"]
                details = fetch_details(cur, pf, tid)
                if not details:
                    log.warning("  %s/%s %s: no detail rows, skip", pf, tid, ticker)
                    continue
                first_day = min(
                    (d["date"].date() if hasattr(d["date"], "date") else d["date"])
                    for d in details
                )
                # Closed campaigns: end at closed_date; open: end today.
                if c["closed_date"]:
                    end_day = (
                        c["closed_date"].date()
                        if hasattr(c["closed_date"], "date")
                        else c["closed_date"]
                    )
                else:
                    end_day = date.today()

                time.sleep(args.sleep)
                peak_pl, peak_date = compute_peak_total_pl_since_b1(
                    ticker, details, first_day,
                )
                if peak_pl is None:
                    log.warning("  %s/%s %s: no yfinance data %s..%s, skip",
                                pf, tid, ticker, first_day, end_day)
                    continue
                stored = c["peak_total_pl"]
                stored_f = float(stored) if stored is not None else None
                would_raise = stored_f is None or stored_f < peak_pl
                arrow = "RAISE" if would_raise else "keep "
                log.info(
                    "  %s/%s %s: peak_total_pl $%s (peak_date=%s) — stored $%s %s",
                    pf, tid, ticker,
                    f"{peak_pl:,.2f}",
                    peak_date,
                    "None" if stored_f is None else f"{stored_f:,.2f}",
                    arrow,
                )
                if args.commit and would_raise:
                    result = update_peak_total_pl(cur, pf, tid, peak_pl)
                    if result:
                        log.info("    → wrote $%.2f", float(result["peak_total_pl"]))
            if args.commit:
                conn.commit()
                log.info("Committed.")
            else:
                log.info("Dry-run — no writes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
