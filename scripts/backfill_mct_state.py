#!/usr/bin/env python3
"""Backfill trading_journal.market_cycle + mct_display_day_num for historical rows.

Background:
    Migration 015 added mct_display_day_num so the Daily Journal page can
    render the MCT badge straight from the row instead of replaying the V11
    engine on every visit. New Daily Routine saves stamp both fields, but
    existing rows pre-date the change. This script populates them.

Strategy:
    Run the V11 engine ONCE over full ^IXIC history (the slow part — ~1s).
    Build a date → (state, display_day_num) lookup from the resulting bar
    log, then UPDATE every journal row that's missing either field. The
    anchoring rules match api/main.py:_compute_mct_state_with_day_num and
    /api/journal/mct-state-by-date-range exactly.

Idempotent — only updates rows where market_cycle is NULL/empty OR
mct_display_day_num is NULL. Re-runs are no-ops once everything is filled.

Usage:
    python scripts/backfill_mct_state.py             # default: dry-run, all portfolios
    python scripts/backfill_mct_state.py --apply     # actually write
    python scripts/backfill_mct_state.py --apply --portfolio CanSlim
    python scripts/backfill_mct_state.py --apply --force   # overwrite even when populated
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from db_layer import get_db_connection  # noqa: E402
from api.mct_endpoint_adapter import run_engine  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("backfill_mct_state")


def build_lookup() -> dict:
    """Run the engine once and return {date_iso: (state_name, display_day_num)}."""
    log.info("Replaying V11 engine over ^IXIC history…")
    result = run_engine("^IXIC", as_of=None)
    if result.bars.empty:
        log.error("Engine returned no bars — market_data likely empty. Aborting.")
        sys.exit(1)

    bars = result.bars
    lookup: dict = {}
    for orig_idx, row in bars.iterrows():
        state_name = str(row["state"])
        cycle_start_idx = row.get("cycle_start_idx")
        pt_on_idx = row.get("pt_on_idx")
        rally_active = bool(row.get("rally_active"))

        cycle_day = 0
        if (rally_active and cycle_start_idx is not None
                and not pd.isna(cycle_start_idx)):
            cycle_day = int(orig_idx) - int(cycle_start_idx) + 1

        if state_name == "POWERTREND" and pt_on_idx is not None and not pd.isna(pt_on_idx):
            display_day_num = int(orig_idx) - int(pt_on_idx) + 1
        elif state_name in ("UPTREND", "RALLY MODE") and cycle_day > 0:
            display_day_num = cycle_day
        else:
            display_day_num = None

        td = row["trade_date"]
        date_iso = td.isoformat() if hasattr(td, "isoformat") else str(td)[:10]
        lookup[date_iso] = (state_name, display_day_num)

    log.info("Built lookup with %d trading days", len(lookup))
    return lookup


def fetch_journal_rows(conn, portfolio: str | None, force: bool):
    """Return rows that need backfill: id, day, market_cycle, mct_display_day_num."""
    where = []
    params: list = []
    if portfolio:
        where.append(
            "portfolio_id = (SELECT id FROM portfolios WHERE name = %s LIMIT 1)"
        )
        params.append(portfolio)
    if not force:
        where.append(
            "(market_cycle IS NULL OR market_cycle = '' "
            "OR mct_display_day_num IS NULL)"
        )
    where_sql = ("WHERE " + " AND ".join(where)) if where else ""
    sql = (
        "SELECT id, day, market_cycle, mct_display_day_num "
        f"FROM trading_journal {where_sql} ORDER BY day"
    )
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true",
                    help="Write changes (default is dry-run)")
    ap.add_argument("--portfolio", default=None,
                    help="Limit to one portfolio name (default: all portfolios)")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite rows even when market_cycle is already populated")
    args = ap.parse_args()

    lookup = build_lookup()

    with get_db_connection() as conn:
        rows = fetch_journal_rows(conn, args.portfolio, args.force)
        if not rows:
            log.info("Nothing to backfill — every row already has both fields.")
            return

        log.info("Considering %d journal rows", len(rows))

        updates: list[tuple] = []  # (mct_state, day_num, id)
        skipped_no_market_data = 0
        for row_id, day, current_cycle, current_day_num in rows:
            date_iso = day.isoformat() if hasattr(day, "isoformat") else str(day)[:10]
            if date_iso not in lookup:
                skipped_no_market_data += 1
                continue
            new_state, new_day_num = lookup[date_iso]

            # Preserve existing values unless --force; only fill nulls/empties.
            if not args.force:
                final_state = current_cycle if (current_cycle and current_cycle.strip()) else new_state
                final_day_num = current_day_num if current_day_num is not None else new_day_num
            else:
                final_state, final_day_num = new_state, new_day_num

            # Skip if nothing actually changes (avoids needless UPDATE traffic).
            if final_state == current_cycle and final_day_num == current_day_num:
                continue

            updates.append((final_state, final_day_num, row_id))

        log.info(
            "Plan: update %d rows · skipped %d (no matching market_data bar)",
            len(updates), skipped_no_market_data,
        )

        if not updates:
            return

        if not args.apply:
            log.warning("Dry-run — no writes performed. Re-run with --apply to commit.")
            for state, day_num, row_id in updates[:10]:
                log.info("  would update id=%s → %s D%s", row_id, state, day_num)
            if len(updates) > 10:
                log.info("  … and %d more", len(updates) - 10)
            return

        with conn.cursor() as cur:
            cur.executemany(
                "UPDATE trading_journal SET market_cycle = %s, "
                "mct_display_day_num = %s WHERE id = %s",
                updates,
            )
        conn.commit()
        log.info("Updated %d rows", len(updates))


if __name__ == "__main__":
    main()
