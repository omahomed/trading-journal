#!/usr/bin/env python3
"""One-shot heal for OPTION trade rows after Migration 016.

Pre-016, the Log Buy / Log Sell pipeline saved option detail rows with
`value = shares * amount` — no contract multiplier — and the LIFO engine
rolled that up into summary.total_cost / realized_pl that were 100× too small
on every dollar amount. Some legacy rows had been hand-fixed with the ×100
already baked in (e.g. LUMN); the rest had not.

This script walks every OPTION campaign and:
  1. Rewrites trades_details.value to the canonical `shares × amount × multiplier`
     when it drifts from notional. Re-emits the matching cash_transactions
     row so the NLV ledger stays consistent.
  2. Calls _recompute_summary_matching() per campaign so trades_summary.total_cost
     and realized_pl reflect the multiplier-aware LIFO output.

Idempotent: safe to re-run. Rows already at the correct notional are skipped.

Usage:
    python scripts/heal_options_after_migration_016.py [--dry-run]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def get_database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if url:
        return url
    import tomllib
    with open(REPO_ROOT / ".streamlit" / "secrets.toml", "rb") as f:
        return tomllib.load(f)["database"]["url"]


# db_layer.get_db_config() only reads DATABASE_URL — it doesn't fall through
# to .streamlit/secrets.toml the way migrations/run.py does. Hydrate the env
# var before db_layer is imported so its first connection lands on the same
# Neon pooler we read the secret from.
os.environ.setdefault("DATABASE_URL", get_database_url())

import psycopg2  # noqa: E402

import db_layer as db  # noqa: E402

# All pre-auth rows are tagged with the founder UUID; db_layer's RLS
# filters by app.user_id, so any script writing through update_detail_row /
# save_summary_row needs to set this before opening a connection.
FOUNDER_UUID = "d7e8f9a0-1b2c-4d3e-8f4a-5b6c7d8e9f0a"


def heal(dry_run: bool) -> int:
    url = os.environ["DATABASE_URL"]
    # db_layer reads from a ContextVar; set it once for the whole run so
    # update_detail_row / save_summary_row see the founder's rows.
    db.current_user_id.set(FOUNDER_UUID)
    conn = psycopg2.connect(url)
    try:
        # Find every option detail row whose stored value drifts from
        # shares × amount × multiplier by more than a cent. The threshold
        # absorbs Decimal rounding on previously-correct rows.
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT d.id, d.portfolio_id, p.name AS portfolio,
                       d.trade_id, d.ticker, d.action, d.date,
                       d.shares, d.amount, d.value, d.multiplier,
                       d.rule, d.notes, d.stop_loss, d.trx_id
                FROM trades_details d
                JOIN portfolios p ON d.portfolio_id = p.id
                WHERE d.instrument_type = 'OPTION'
                  AND d.deleted_at IS NULL
                  AND ABS(d.value - d.shares * d.amount * d.multiplier) > 0.01
                ORDER BY d.trade_id, d.date
                """
            )
            drifted = cur.fetchall()

        print(f"Found {len(drifted)} option detail row(s) needing value rewrite.")
        for row in drifted:
            (did, portfolio_id, portfolio, trade_id, ticker, action, dt,
             shares, amount, value, mult, rule, notes, stop, trx_id) = row
            new_value = float(shares) * float(amount) * float(mult)
            print(f"  {trade_id} {ticker} {action} #{did}: "
                  f"${float(value):.2f} → ${new_value:.2f}")
            if dry_run:
                continue
            row_dict = {
                "Trade_ID": trade_id, "Ticker": ticker, "Action": action,
                "Date": dt, "Shares": float(shares), "Amount": float(amount),
                "Value": round(new_value, 2),
                "Rule": rule, "Notes": notes,
                "Stop_Loss": float(stop) if stop is not None else None,
                "Trx_ID": trx_id,
            }
            db.update_detail_row(portfolio, did, row_dict)

        # Recompute summary for every option campaign — picks up the new
        # detail.value plus the multiplier-aware LIFO from trade_calc.
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT s.trade_id, s.ticker, p.name AS portfolio
                FROM trades_summary s
                JOIN portfolios p ON s.portfolio_id = p.id
                WHERE s.instrument_type = 'OPTION'
                  AND s.deleted_at IS NULL
                ORDER BY s.trade_id
                """
            )
            campaigns = cur.fetchall()

        print(f"\nRecomputing {len(campaigns)} option campaign summary row(s)...")
        if dry_run:
            for c in campaigns:
                print(f"  (dry-run) would recompute {c[0]} {c[1]}")
            return 0

        # Lazy import — _recompute_summary_matching lives in api/main.py and pulls
        # in the FastAPI app, so we only load it for the live run.
        sys.path.insert(0, str(REPO_ROOT / "api"))
        from main import _recompute_summary_matching
        for trade_id, ticker, portfolio in campaigns:
            try:
                _recompute_summary_matching(portfolio, trade_id, ticker)
                print(f"  ✓ {trade_id} {ticker}")
            except Exception as e:
                print(f"  ✗ {trade_id} {ticker}: {e}")

        return 0
    finally:
        conn.close()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true",
                   help="Show what would change without writing.")
    args = p.parse_args()
    return heal(args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
