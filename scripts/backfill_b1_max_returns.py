#!/usr/bin/env python3
"""Backfill / reconcile trades_summary.b1_max_return_pct (migration 036).

For every OPEN equity campaign, compute the historical peak B1 return %
using yfinance daily closes from the B1 entry date to today, and raise
the stored persistent peak toward it. Writes go through the monotonic
guard in db.update_b1_max_return_pct — a value is only ever RAISED, never
lowered — so a fresh run can promote a stale/NULL row to its true peak
but can never demote a legitimate SR8 core.

Why this exists: the Sell Rule tier is `max(current_b1_return, stored_peak)`.
The stored peak is populated by (a) the frontend's live auto-promote, which
only fires while the app is open, and (b) this job. A leader that peaked
above 50% while the app was closed — and was never covered by a backfill —
stores a stale-low (or NULL) peak, so it mis-tiers as SR11 on the pullback
even though its close-basis B1 return topped 50%. That mis-tier is
dangerous: SR11 (BE stop-out) and SR8 (RS-defended core) prescribe
opposite sell actions. Running this on a schedule closes the gap.

Two modes:
  - default (backfill): only rows where b1_max_return_pct IS NULL. Cheap;
    for first-time population.
  - --reconcile: ALL open equity rows. Recomputes the close-basis peak and
    raises any stored value that has fallen behind. This is the mode the
    daily GitHub Action runs (.github/workflows/reconcile-b1-max.yml) so the
    tier can't silently drift below the true peak. Idempotent and safe to
    re-run — the monotonic guard makes repeated runs no-ops once caught up.

Scope (both modes):
  - status='OPEN'
  - instrument_type='STOCK' (options skipped — trades_details.amount is
    premium-per-contract for options, not share price, so the % ladder
    doesn't translate)

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
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from db_layer import get_db_connection, update_b1_max_return_pct  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("backfill_b1_max")


def fetch_candidates(portfolio: str | None, reconcile: bool = False) -> list[dict]:
    """Find OPEN equity campaigns to (backfill / reconcile).

    reconcile=False (default): only rows with b1_max_return_pct IS NULL.
    reconcile=True: every open equity row, so a stored peak that has fallen
    behind the true close-basis peak can be raised. The monotonic guard in
    update_b1_max_return_pct ensures already-current rows are no-ops.

    Returns rows shaped:
        { trade_id, ticker, portfolio_name, b1_entry_date, b1_entry_price,
          stored_max_pct }
    Skips campaigns missing a B1 BUY row (data corruption / pre-app imports).
    """
    sql = """
        SELECT
            s.trade_id,
            s.ticker,
            p.name AS portfolio_name,
            s.b1_max_return_pct AS stored_max_pct,
            (SELECT d.date
             FROM trades_details d
             WHERE d.trade_id = s.trade_id
               AND d.portfolio_id = s.portfolio_id
               AND d.action = 'BUY'
               AND d.deleted_at IS NULL
             ORDER BY d.date ASC, d.id ASC
             LIMIT 1) AS b1_entry_date,
            (SELECT d.amount
             FROM trades_details d
             WHERE d.trade_id = s.trade_id
               AND d.portfolio_id = s.portfolio_id
               AND d.action = 'BUY'
               AND d.deleted_at IS NULL
             ORDER BY d.date ASC, d.id ASC
             LIMIT 1) AS b1_entry_price
        FROM trades_summary s
        JOIN portfolios p ON s.portfolio_id = p.id
        WHERE s.status = 'OPEN'
          AND s.deleted_at IS NULL
          AND COALESCE(s.instrument_type, 'STOCK') = 'STOCK'
    """
    if not reconcile:
        sql += " AND s.b1_max_return_pct IS NULL"
    params: list = []
    if portfolio:
        sql += " AND p.name = %s"
        params.append(portfolio)
    sql += " ORDER BY p.name, s.trade_id"

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            cols = [c[0] for c in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    return rows


def compute_max_b1_return(ticker: str, entry_date: date, entry_price: float) -> float | None:
    """Pull daily closes for [entry_date, today] and return the peak
    B1 return %. Returns None when yfinance has no data.
    """
    end = date.today() + timedelta(days=1)  # yfinance end is exclusive
    raw = yf.download(
        ticker,
        start=entry_date.isoformat(),
        end=end.isoformat(),
        progress=False,
        auto_adjust=False,
    )
    if raw is None or raw.empty:
        return None
    closes = raw["Close"]
    # yfinance can return a DataFrame (multi-ticker call) or a Series.
    # Normalize to a 1-D series.
    if isinstance(closes, pd.DataFrame):
        if closes.shape[1] != 1:
            return None
        closes = closes.iloc[:, 0]
    peak_close = float(closes.max())
    if not pd.notna(peak_close) or entry_price <= 0:
        return None
    return ((peak_close - entry_price) / entry_price) * 100.0


def classify_tier(pct: float) -> str:
    """Mirror frontend/src/lib/sell-rule.ts. Diagnostic only — backend
    is not the source of truth for the tier ladder."""
    if pct < 10:
        return "SR1"
    if pct < 50:
        return "SR11"
    return "SR8"


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
    log.info("Fetching candidates: %s ...", scope)
    candidates = fetch_candidates(args.portfolio, reconcile=args.reconcile)
    log.info("Found %d candidate(s)", len(candidates))
    if not candidates:
        return 0

    raised = 0
    unchanged = 0
    skipped_no_b1 = 0
    skipped_no_data = 0
    errors = 0

    print(f"{'TICKER':<10} {'ENTRY DATE':<12} {'ENTRY $':>10} {'STORED %':>10} {'PEAK B1 %':>11} {'TIER':<5} {'ACTION':<9} {'TRADE_ID':<14} {'PORTFOLIO'}")
    print("-" * 112)

    for row in candidates:
        ticker = (row.get("ticker") or "").strip().upper()
        trade_id = row.get("trade_id")
        portfolio = row.get("portfolio_name")
        entry_raw = row.get("b1_entry_date")
        entry_price_raw = row.get("b1_entry_price")
        stored_raw = row.get("stored_max_pct")
        stored = float(stored_raw) if stored_raw is not None else None
        stored_disp = f"{stored:.2f}" if stored is not None else "—"

        if not ticker or entry_raw is None or entry_price_raw is None:
            skipped_no_b1 += 1
            log.warning("Skipping %s/%s: missing B1 entry data", portfolio, trade_id)
            continue

        # b1_entry_date is a timestamp; normalize to date.
        if isinstance(entry_raw, datetime):
            entry_date = entry_raw.date()
        elif isinstance(entry_raw, date):
            entry_date = entry_raw
        else:
            entry_date = pd.to_datetime(entry_raw).date()
        entry_price = float(entry_price_raw)

        try:
            time.sleep(args.sleep)
            pct = compute_max_b1_return(ticker, entry_date, entry_price)
        except Exception as exc:
            errors += 1
            log.error("yfinance error for %s (%s): %s", ticker, trade_id, exc)
            continue

        if pct is None:
            skipped_no_data += 1
            log.warning("No yfinance data for %s (%s) from %s", ticker, trade_id, entry_date)
            print(f"{ticker:<10} {entry_date!s:<12} {entry_price:>10.2f} {stored_disp:>10} {'—':>11} {'—':<5} {'skip':<9} {trade_id:<14} {portfolio}")
            continue

        tier = classify_tier(pct)
        # Anticipate what the monotonic guard will do so the dry-run report
        # matches a live run: it writes only when stored is NULL or < pct.
        would_raise = stored is None or stored < pct

        if args.dry_run:
            action = "raise" if would_raise else "keep"
        elif not would_raise:
            action = "keep"
            unchanged += 1
        else:
            try:
                result = update_b1_max_return_pct(portfolio, trade_id, pct)
                if result is None:
                    errors += 1
                    action = "err"
                    log.error("Trade %s/%s not found on write", portfolio, trade_id)
                elif result.get("was_updated"):
                    raised += 1
                    action = "RAISED"
                else:
                    unchanged += 1
                    action = "keep"
            except Exception as exc:
                errors += 1
                action = "err"
                log.error("DB write failed for %s/%s: %s", portfolio, trade_id, exc)

        print(f"{ticker:<10} {entry_date!s:<12} {entry_price:>10.2f} {stored_disp:>10} {pct:>10.2f}% {tier:<5} {action:<9} {trade_id:<14} {portfolio}")

    print("-" * 112)
    print(f"Summary: {raised} raised · {unchanged} already-current · "
          f"{skipped_no_b1} skipped (no B1) · {skipped_no_data} skipped (no yfinance data) · "
          f"{errors} errors")
    if args.dry_run:
        print("DRY-RUN: no rows were modified. Re-run without --dry-run to persist.")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
