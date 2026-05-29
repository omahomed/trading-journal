#!/usr/bin/env python3
"""Backfill lot_closures for historical trades with reconciliation report.

Migration 017 added the lot_closures table; commits 5232c07 / 72fa009 / step-4
wired the live recompute path to populate it. Trades that were edited or had
new SELLs after deploy already have closures persisted. Trades untouched
since the deploy still have empty lot_closures rows — this script backfills
them, but defensively: it re-runs LIFO and only writes when the freshly-
computed Realized_PL matches the persisted summary value within 1¢.

Discrepancies are logged and SKIPPED. We don't auto-write new closures over
a stale-looking summary because either side could be wrong (the persisted
summary may have been written under a since-fixed bug, or the new computation
may have a regression we missed). Manual review required for each.

Per-trade transaction: each MATCH write (DELETE existing + INSERT new) runs
inside its own atomic_transaction. One trade's failure doesn't affect others.
The DELETE-then-INSERT is idempotent, so re-running --apply is safe.

Usage:
    # Dry-run (default) — reports matches/discrepancies without writing.
    python scripts/backfill_lot_closures.py --portfolio CanSlim

    # Actually persist closures for matched trades.
    python scripts/backfill_lot_closures.py --portfolio CanSlim --apply

    # --dry-run is allowed for explicitness; it's the default behavior.
    python scripts/backfill_lot_closures.py --portfolio CanSlim --dry-run

Exit codes:
    0  no per-trade processing errors (regardless of dry-run vs apply)
    1  one or more trades errored during processing
    2  argparse rejected the args
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

import pandas as pd  # noqa: E402

import db_layer as db  # noqa: E402
from trade_calc import compute_matching_summary, is_option_ticker  # noqa: E402

# All pre-auth rows are tagged with the founder UUID; db_layer's RLS filters
# by app.user_id, so any script writing through the layer needs to set this
# before opening a connection. Same constant heal_options uses.
FOUNDER_UUID = "d7e8f9a0-1b2c-4d3e-8f4a-5b6c7d8e9f0a"

# 1¢ reconciliation window + tiny epsilon to absorb float-comparison noise.
# IEEE 754 can round abs(-0.01) to 0.010000000000000009, which would trip a
# strict 0.01 comparison and falsely classify a clean 1¢ rounding delta as
# a discrepancy (caught on staging: FIGR 202601-004, delta=-0.01). The
# conceptual tolerance is still 1¢; the extra 0.001 is pure noise absorption.
TOLERANCE = 0.011

# Sub-half-cent → label as "exact" rather than "0¢ tolerance". Avoids the
# floating-point footguns of comparing delta directly to 0.0 (signed zero,
# accumulated 1e-16 noise, etc.).
EXACT_THRESHOLD = 0.005


# Columns load_details returns (TitleCase) → the snake_case shape
# compute_matching_summary expects. Mirror of the relevant slice of
# api.main._normalize_trades, inlined here so we don't have to import
# api.main (which would load the whole FastAPI app for a one-off script).
_LIFO_COLUMN_RENAME = {
    "Trade_ID": "trade_id",
    "Ticker": "ticker",
    "Action": "action",
    "Date": "date",
    "Shares": "shares",
    "Amount": "amount",
    "Trx_ID": "trx_id",
    "Multiplier": "multiplier",
    "Instrument_Type": "instrument_type",
}


def _normalize_for_lifo(df: pd.DataFrame) -> pd.DataFrame:
    """Rename load_details TitleCase columns to the snake_case shape
    compute_matching_summary expects."""
    return df.rename(columns={k: v for k, v in _LIFO_COLUMN_RENAME.items() if k in df.columns})


def resolve_multiplier(txns: pd.DataFrame, ticker: str) -> float:
    """Mirror of the multiplier resolution in api.main._recompute_summary_matching.

    Prefers the per-row multiplier column (Migration 016), falls back to
    ticker-pattern autodetect for any pre-016 row that still has the default
    1× multiplier. Inlined here so we don't need to import api.main.
    """
    multiplier = 1.0
    if not txns.empty and "multiplier" in txns.columns:
        mults = pd.to_numeric(txns["multiplier"], errors="coerce").dropna()
        if not mults.empty and float(mults.max()) > 1:
            multiplier = float(mults.max())
    if multiplier == 1.0 and is_option_ticker(ticker):
        multiplier = 100.0
    return multiplier


def reconcile_trade(trade_id: str, ticker: str, persisted_pl: float,
                    df_d_portfolio: pd.DataFrame) -> tuple[str, float, float, list[dict]]:
    """Run LIFO on this trade's details and compare to the persisted P&L.

    Returns (status, computed_pl, delta, closures) where:
      - status is 'MATCH' or 'DISCREPANCY'
      - delta = computed_pl - persisted_pl (signed)
      - closures is the list of LIFO pairings (empty for open-only trades)
    """
    txns = df_d_portfolio[df_d_portfolio["trade_id"] == trade_id]
    multiplier = resolve_multiplier(txns, ticker)
    result = compute_matching_summary(
        txns, trade_id, ticker, multiplier=multiplier, with_closures=True,
    )
    summary, closures = result
    computed_pl = float(summary["Realized_PL"]) if summary else 0.0
    delta = computed_pl - persisted_pl
    status = "MATCH" if abs(delta) <= TOLERANCE else "DISCREPANCY"
    return status, computed_pl, delta, closures


def write_closures(portfolio_name: str, trade_id: str, closures: list[dict]) -> None:
    """Replace the trade's lot_closures rows. Atomic per trade.

    DELETE existing + INSERT new in one atomic_transaction. Does NOT touch
    trades_summary — caller has already confirmed the persisted summary
    matches the freshly-computed one within tolerance.
    """
    with db.atomic_transaction() as (_conn, cur):
        cur.execute("SELECT id FROM portfolios WHERE name = %s", (portfolio_name,))
        result = cur.fetchone()
        if not result:
            raise ValueError(f"Portfolio '{portfolio_name}' not found")
        portfolio_id = result[0]

        cur.execute(
            "DELETE FROM lot_closures WHERE portfolio_id = %s AND trade_id = %s",
            (portfolio_id, trade_id),
        )

        if closures:
            cur.executemany(
                """
                INSERT INTO lot_closures (
                    portfolio_id, trade_id, sell_trx_id, buy_trx_id,
                    shares, buy_price, sell_price, multiplier,
                    realized_pl, closed_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                [
                    (
                        portfolio_id,
                        trade_id,
                        c["sell_trx_id"],
                        c["buy_trx_id"],
                        c["shares"],
                        c["buy_price"],
                        c["sell_price"],
                        c["multiplier"],
                        c["realized_pl"],
                        c["closed_at"],
                    )
                    for c in closures
                ],
            )


def _format_match_label(delta: float) -> str:
    """'exact' for sub-half-cent delta, otherwise '<n>¢ tolerance'."""
    if abs(delta) < EXACT_THRESHOLD:
        return "exact"
    return f"{round(abs(delta) * 100)}¢ tolerance"


def backfill(portfolio_name: str, apply_writes: bool) -> int:
    """Main loop. Returns shell exit code (0 ok, 1 if any trade errored)."""
    db.current_user_id.set(FOUNDER_UUID)

    print(f"Loading trades from portfolio '{portfolio_name}'...")
    df_s = db.load_summary(portfolio_name)
    if df_s.empty:
        print(f"No trades found in portfolio '{portfolio_name}'. Nothing to backfill.")
        return 0

    df_d_raw = db.load_details(portfolio_name)
    df_d = _normalize_for_lifo(df_d_raw) if not df_d_raw.empty else df_d_raw

    n = len(df_s)
    print(f"Found {n} trade(s).\n")

    matched: list[tuple[str, str, float, float, list[dict]]] = []
    discrepancies: list[tuple[str, str, float, float, float]] = []
    errors: list[tuple[str, str]] = []

    for i, row in enumerate(df_s.itertuples(index=False), start=1):
        trade_id = str(getattr(row, "Trade_ID", "") or "")
        ticker = str(getattr(row, "Ticker", "") or "")
        try:
            persisted_pl_raw = getattr(row, "Realized_PL", 0)
            persisted_pl = float(persisted_pl_raw) if persisted_pl_raw is not None else 0.0
        except (TypeError, ValueError):
            persisted_pl = 0.0

        try:
            status, computed_pl, delta, closures = reconcile_trade(
                trade_id, ticker, persisted_pl, df_d,
            )
        except Exception as e:
            errors.append((trade_id, str(e)))
            print(f"[{i}/{n}] Trade {trade_id} ({ticker}): ERROR — {e}")
            continue

        if status == "MATCH":
            label = _format_match_label(delta)
            print(f"[{i}/{n}] Trade {trade_id} ({ticker}): "
                  f"persisted={persisted_pl:.2f} computed={computed_pl:.2f} "
                  f"→ MATCH ({label})")
            matched.append((trade_id, ticker, persisted_pl, computed_pl, closures))
            if apply_writes:
                try:
                    write_closures(portfolio_name, trade_id, closures)
                except Exception as e:
                    # Demote from MATCH to ERROR — the reconciliation passed
                    # but the write failed, so we couldn't actually backfill.
                    errors.append((trade_id, f"write failed: {e}"))
                    matched.pop()
                    print(f"        ↳ write failed: {e}")
        else:
            print(f"[{i}/{n}] Trade {trade_id} ({ticker}): "
                  f"persisted={persisted_pl:.2f} computed={computed_pl:.2f} "
                  f"→ DISCREPANCY (delta: {delta:+.2f})")
            discrepancies.append((trade_id, ticker, persisted_pl, computed_pl, delta))

    # ─── Reconciliation report ───────────────────────────────────────────
    print()
    print("─" * 49)
    print("Reconciliation Report")
    print("─" * 49)
    print(f"Total trades processed:        {n}")
    print(f"Matched (≤1¢ delta):           {len(matched)}"
          f"{'  (auto-backfill candidates)' if not apply_writes else ''}")
    print(f"Discrepancies (>1¢ delta):     {len(discrepancies)}"
          f"  (manual review required)" if discrepancies else
          f"Discrepancies (>1¢ delta):     {len(discrepancies)}")
    print(f"Errors during processing:      {len(errors)}"
          f"{'  (logged below)' if errors else ''}")

    if discrepancies:
        print("\nDiscrepancies (manual review):")
        for trade_id, ticker, persisted_pl, computed_pl, delta in discrepancies:
            print(f"  {trade_id} ({ticker}): "
                  f"persisted={persisted_pl:.2f} computed={computed_pl:.2f} "
                  f"delta={delta:+.2f}")

    if errors:
        print("\nErrors:")
        for trade_id, message in errors:
            print(f"  {trade_id}: {message}")

    print()
    if apply_writes:
        print(f"Mode: APPLIED. Wrote closures for {len(matched)} trade(s). "
              f"Skipped {len(discrepancies)} discrepancy/discrepancies "
              f"and {len(errors)} error(s).")
    else:
        print("Mode: DRY-RUN (no writes performed). To apply matches, run with --apply.")

    return 1 if errors else 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--portfolio", required=True,
                    help="Portfolio name (e.g. 'CanSlim'). Required.")
    ap.add_argument("--apply", action="store_true",
                    help="Write closures for matched trades. "
                         "Without this flag, runs in dry-run mode.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Explicit dry-run (default behavior — included for clarity). "
                         "Mutually exclusive with --apply.")
    args = ap.parse_args()

    if args.apply and args.dry_run:
        print("ERROR: --apply and --dry-run are mutually exclusive.", file=sys.stderr)
        return 2

    return backfill(args.portfolio, apply_writes=args.apply)


if __name__ == "__main__":
    sys.exit(main())
