#!/usr/bin/env python3
"""One-time backfill of trades_summary.b1_max_return_pct (migration 036).

For every OPEN equity campaign with b1_max_return_pct IS NULL, compute the
historical peak B1 return % using yfinance daily closes from the B1 entry
date to today. Writes the result back to trades_summary.

Why backfill is necessary: the Sell Rule column's persistent tier was
introduced after positions were already open. Without backfill, leaders
that peaked above 50% but pulled back would mis-tier as SR11 until they
re-cross 50% — exactly the bug the persistent column is meant to fix.

Scope:
  - status='OPEN'
  - instrument_type='STOCK' (options skipped — trades_details.amount is
    premium-per-contract for options, not share price, so the % ladder
    doesn't translate)
  - b1_max_return_pct IS NULL (idempotent — re-runs skip filled rows)

Usage:
    python scripts/backfill_b1_max_returns.py
    python scripts/backfill_b1_max_returns.py --dry-run
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

from db_layer import get_db_connection  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("backfill_b1_max")


def fetch_candidates(portfolio: str | None) -> list[dict]:
    """Find every OPEN equity campaign that needs a backfill.

    Returns rows shaped:
        { trade_id, ticker, portfolio_name, b1_entry_date, b1_entry_price }
    Skips campaigns whose b1_max_return_pct is already populated and
    campaigns missing a B1 BUY row (data corruption / pre-app imports).
    """
    sql = """
        SELECT
            s.trade_id,
            s.ticker,
            p.name AS portfolio_name,
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
          AND s.b1_max_return_pct IS NULL
          AND COALESCE(s.instrument_type, 'STOCK') = 'STOCK'
    """
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


def write_back(portfolio: str, trade_id: str, value: float) -> None:
    sql = """
        UPDATE trades_summary
        SET b1_max_return_pct = %s
        FROM portfolios p
        WHERE p.id = trades_summary.portfolio_id
          AND p.name = %s
          AND trades_summary.trade_id = %s
          AND trades_summary.deleted_at IS NULL
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (value, portfolio, trade_id))
        conn.commit()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Report planned updates without writing.")
    parser.add_argument("--portfolio", default=None,
                        help="Limit to one portfolio (default: all).")
    parser.add_argument("--sleep", type=float, default=0.5,
                        help="Seconds to sleep between yfinance calls (default 0.5).")
    args = parser.parse_args()

    log.info("Fetching candidate OPEN equity campaigns with NULL b1_max_return_pct...")
    candidates = fetch_candidates(args.portfolio)
    log.info("Found %d candidate(s)", len(candidates))
    if not candidates:
        return 0

    updated = 0
    skipped_no_b1 = 0
    skipped_no_data = 0
    errors = 0

    print(f"{'TICKER':<10} {'ENTRY DATE':<12} {'ENTRY $':>10} {'PEAK B1 %':>11} {'TIER':<5} {'ACTION':<10} {'TRADE_ID':<14} {'PORTFOLIO'}")
    print("-" * 100)

    for row in candidates:
        ticker = (row.get("ticker") or "").strip().upper()
        trade_id = row.get("trade_id")
        portfolio = row.get("portfolio_name")
        entry_raw = row.get("b1_entry_date")
        entry_price_raw = row.get("b1_entry_price")

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
            print(f"{ticker:<10} {entry_date!s:<12} {entry_price:>10.2f} {'—':>11} {'—':<5} {'skip':<10} {trade_id:<14} {portfolio}")
            continue

        tier = classify_tier(pct)
        action = "DRY-RUN" if args.dry_run else "UPDATE"
        print(f"{ticker:<10} {entry_date!s:<12} {entry_price:>10.2f} {pct:>10.2f}% {tier:<5} {action:<10} {trade_id:<14} {portfolio}")

        if not args.dry_run:
            try:
                write_back(portfolio, trade_id, pct)
                updated += 1
            except Exception as exc:
                errors += 1
                log.error("DB write failed for %s/%s: %s", portfolio, trade_id, exc)

    print("-" * 100)
    print(f"Summary: {updated} updated · {skipped_no_b1} skipped (no B1) · "
          f"{skipped_no_data} skipped (no yfinance data) · {errors} errors")
    if args.dry_run:
        print("DRY-RUN: no rows were modified. Re-run without --dry-run to persist.")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
