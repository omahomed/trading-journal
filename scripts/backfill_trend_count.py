#!/usr/bin/env python3
"""Backfill trading_journal.trend_count for historical rows.

Background:
    Migration 043 added trend_count so the Daily Journal page can render
    the signed Trend Count straight from the row instead of replaying the
    V11 engine on every visit. New Daily Routine saves stamp it (Half 2);
    this script populates existing rows.

Strategy:
    Same single-pass pattern as backfill_mct_state.py — run the V11 engine
    ONCE over full ^IXIC history, build a {date_iso: trend_count} lookup
    from the resulting bar log, then UPDATE every journal row whose date
    matches a bar. The value extraction mirrors
    api/mct_endpoint_adapter.py:to_rally_prefix_response exactly:
        span = orig_idx - trend_anchor_idx + 1
        trend_count = span * trend_sign
    where trend_anchor_idx and trend_sign are per-bar fields already
    exposed by MCTEngine._bar_record (Branch 1). NULL is emitted when
    trend_sign == 0 (pre-first-Step-4 in the replay), which is a
    first-class value distinct from 0-the-arm-bar.

Idempotent — only updates rows where trend_count is NULL (default). --force
overwrites even when populated (useful if the engine leg-logic changes).

Usage:
    python scripts/backfill_trend_count.py             # dry-run, all portfolios
    python scripts/backfill_trend_count.py --apply     # actually write
    python scripts/backfill_trend_count.py --apply --portfolio CanSlim
    python scripts/backfill_trend_count.py --apply --force   # overwrite non-nulls
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
log = logging.getLogger("backfill_trend_count")


def build_lookup() -> dict:
    """Run the engine once and return {date_iso: trend_count | None}.

    None is meaningful — it's the "no positive/negative leg yet" state,
    distinct from 0 (an arm bar). Callers should stamp NULL to preserve
    the distinction in the DB.
    """
    log.info("Replaying V11 engine over ^IXIC history…")
    result = run_engine("^IXIC", as_of=None)
    if result.bars.empty:
        log.error("Engine returned no bars — market_data likely empty. Aborting.")
        sys.exit(1)

    bars = result.bars
    lookup: dict = {}
    for orig_idx, row in bars.iterrows():
        ta_idx = row.get("trend_anchor_idx")
        sign = int(row.get("trend_sign") or 0)
        if sign == 0 or ta_idx is None or (isinstance(ta_idx, float) and pd.isna(ta_idx)):
            trend_count = None
        else:
            trend_count = (int(orig_idx) - int(ta_idx) + 1) * sign

        td = row["trade_date"]
        date_iso = td.isoformat() if hasattr(td, "isoformat") else str(td)[:10]
        lookup[date_iso] = trend_count

    log.info("Built lookup with %d trading days", len(lookup))
    return lookup


def fetch_journal_rows(conn, portfolio: str | None, force: bool):
    """Return rows that need backfill: id, day, trend_count.

    Always excludes soft-deleted rows — a tombstoned row's trend_count is
    irrelevant and stamping it would resurrect state onto a dead record if
    it later gets un-deleted. This differs from backfill_mct_state.py which
    inherited the older no-deleted_at pattern; today there are zero soft-
    deletes on trading_journal but the filter future-proofs the script.
    """
    where = ["deleted_at IS NULL"]
    params: list = []
    if portfolio:
        where.append(
            "portfolio_id = (SELECT id FROM portfolios WHERE name = %s LIMIT 1)"
        )
        params.append(portfolio)
    if not force:
        where.append("trend_count IS NULL")
    where_sql = "WHERE " + " AND ".join(where)
    sql = (
        "SELECT id, day, trend_count "
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
                    help="Overwrite rows even when trend_count is already populated")
    args = ap.parse_args()

    lookup = build_lookup()

    with get_db_connection() as conn:
        rows = fetch_journal_rows(conn, args.portfolio, args.force)
        if not rows:
            log.info("Nothing to backfill — every row already has trend_count.")
            return

        log.info("Considering %d journal rows", len(rows))

        updates: list[tuple] = []  # (trend_count, id)
        skipped_no_market_data = 0
        for row_id, day, current_tc in rows:
            date_iso = day.isoformat() if hasattr(day, "isoformat") else str(day)[:10]
            if date_iso not in lookup:
                skipped_no_market_data += 1
                continue
            new_tc = lookup[date_iso]

            # Preserve existing values unless --force; only fill nulls.
            if not args.force:
                final_tc = current_tc if current_tc is not None else new_tc
            else:
                final_tc = new_tc

            # Skip no-op writes.
            if final_tc == current_tc:
                continue

            updates.append((final_tc, row_id))

        log.info(
            "Plan: update %d rows · skipped %d (no matching market_data bar)",
            len(updates), skipped_no_market_data,
        )

        if not updates:
            return

        if not args.apply:
            log.warning("Dry-run — no writes performed. Re-run with --apply to commit.")
            for tc, row_id in updates[:15]:
                log.info("  would update id=%s → trend_count=%s", row_id, tc)
            if len(updates) > 15:
                log.info("  … and %d more", len(updates) - 15)
            return

        with conn.cursor() as cur:
            cur.executemany(
                "UPDATE trading_journal SET trend_count = %s WHERE id = %s",
                updates,
            )
        conn.commit()
        log.info("Updated %d rows", len(updates))


if __name__ == "__main__":
    main()
