#!/usr/bin/env python3
"""One-shot Daily Journal xlsx importer.

Reads a manually-reconstructed Daily Journal xlsx and writes its
rows into trading_journal for a target portfolio. Defaults to
dry-run; pass --commit to actually write.

Shape:
    The xlsx has a single sheet (default 'Sheet2') with an embedded
    header row above the data:

        | Date       | Beg NLV | Cash +/- | End NLV |
        | 2026-01-02 | 0       | NaN      | 0       |
        | ...                                       |
        | 2026-05-21 | 39506.79| 6500     | 46006.79|

Maps to trading_journal columns:
    Date     → day
    Beg NLV  → beg_nlv
    Cash +/- → cash_change (NaN → 0.0)
    End NLV  → end_nlv

Computed at import time (mirrors what the app would have written
had the user typed these entries through Daily Routine):
    daily_dollar_change = end_nlv - beg_nlv - cash_change
    daily_pct_change    = daily_dollar_change / beg_nlv
                          (NULL when beg_nlv == 0, undefined math)

Conflict strategy:
    ON CONFLICT (portfolio_id, day) DO NOTHING by default —
    historical backfill should never overwrite app-entered data.
    --overwrite switches to DO UPDATE on the 5 import-sourced
    columns only (beg_nlv, cash_change, end_nlv, daily_dollar_change,
    daily_pct_change) plus updated_at. Notes, daily_thoughts,
    highlights/lowlights/mistakes — all preserved.

user_id propagation:
    Same two-layer pattern as scripts/import_robinhood_csv.py:
      1. Explicit user_id in every INSERT (sourced from
         portfolios.user_id)
      2. SET LOCAL app.user_id = <uuid> at transaction start
         so RLS defaults and any future audit triggers see the
         correct binding.

Usage:
    # Dry-run against the real fixture:
    python scripts/import_daily_journal_xlsx.py \\
      --xlsx scripts/fixtures/long_term_growth_journal.xlsx \\
      --portfolio "Long-Term Growth"

    # Real commit, first-time import (table empty):
    python scripts/import_daily_journal_xlsx.py \\
      --xlsx scripts/fixtures/long_term_growth_journal.xlsx \\
      --portfolio "Long-Term Growth" --commit

    # Re-run after correcting xlsx, force overwrite of 5 columns:
    python scripts/import_daily_journal_xlsx.py \\
      --xlsx scripts/fixtures/long_term_growth_journal.xlsx \\
      --portfolio "Long-Term Growth" --overwrite --commit
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from db_layer import get_db_connection  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("journal_import")


# ─────────────────────────────────────────────────────────────────────────────
# Pure parsing utilities
# ─────────────────────────────────────────────────────────────────────────────


def _coerce_date(v: Any) -> date | None:
    """Accept datetime, date, string, or pandas Timestamp."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    if isinstance(v, pd.Timestamp):
        return v.date() if not pd.isna(v) else None
    s = str(v).strip()
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def _coerce_float(v: Any, default: float = 0.0) -> float:
    """Coerce NaN / blank / string-with-currency to float."""
    if v is None:
        return default
    if isinstance(v, float) and math.isnan(v):
        return default
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().replace("$", "").replace(",", "")
    if not s:
        return default
    try:
        return float(s)
    except ValueError:
        return default


def read_xlsx(path: str | Path, sheet: str = "Sheet2") -> list[dict]:
    """Read an xlsx, return list of {day, beg_nlv, cash_change, end_nlv}.

    The source file has an embedded header in row 1 ('Date', 'Beg NLV',
    'Cash +/-', 'End NLV'); pandas can't auto-detect because the column
    headers are 'Unnamed: 0..3'. We read with header=1 and explicitly
    rename. Rows whose date can't be parsed are dropped (defensive — the
    embedded header itself would otherwise sneak through with day=None).
    """
    df = pd.read_excel(
        path, sheet_name=sheet, header=1,
        names=["day", "beg_nlv", "cash_change", "end_nlv"],
    )
    out: list[dict] = []
    for _, row in df.iterrows():
        d = _coerce_date(row.get("day"))
        if d is None:
            continue
        out.append({
            "day": d,
            "beg_nlv": _coerce_float(row.get("beg_nlv")),
            "cash_change": _coerce_float(row.get("cash_change")),
            "end_nlv": _coerce_float(row.get("end_nlv")),
        })
    return out


def compute_derived(row: dict) -> dict:
    """Augment a parsed row with daily_dollar_change and daily_pct_change.

    Mirrors the canonical app write path in
    frontend/src/components/daily-routine.tsx:255-276 so a row written
    by the importer is indistinguishable from a row typed through
    Daily Routine.

    daily_dollar_change = end_nlv - beg_nlv - cash_change
    daily_pct_change    = (daily_dollar_change / (beg_nlv + cash_change)) * 100
                          (PERCENTAGE form: 3.39 means 3.39%.
                           None when adjusted_beg = beg_nlv + cash_change == 0;
                           division undefined. The DB column is NULL-allowed,
                           so we pass None and let psycopg2 map to SQL NULL.)

    Divisor uses the post-deposit baseline so a $1,000 deposit + $50
    gain on a $10k portfolio reports as 50/11000 = 0.455%, not the
    50/10000 = 0.5% that raw beg_nlv would give.
    """
    dollar = row["end_nlv"] - row["beg_nlv"] - row["cash_change"]
    adjusted_beg = row["beg_nlv"] + row["cash_change"]
    pct = (dollar / adjusted_beg) * 100 if adjusted_beg != 0 else None
    return {
        **row,
        "daily_dollar_change": dollar,
        "daily_pct_change": pct,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────


def detect_duplicates(rows: list[dict]) -> list[date]:
    """Return any duplicate day values in the source (sorted)."""
    counts = Counter(r["day"] for r in rows)
    return sorted([d for d, n in counts.items() if n > 1])


def detect_nlv_continuity_gaps(
    rows: list[dict], tolerance: float = 0.5,
) -> list[tuple]:
    """Return list of (day_n, end_nlv_n, day_n+1, beg_nlv_n+1, delta)
    where |end_nlv(N) − beg_nlv(N+1)| > tolerance.

    Informational only — the xlsx is the user's manually-reconstructed
    source of truth. Gaps surface as warnings in the dry-run report;
    they don't block the import.

    tolerance: dollars of slack; 50¢ is enough to absorb rounding noise
    in the xlsx without hiding real reconciliation deltas.
    """
    sorted_rows = sorted(rows, key=lambda r: r["day"])
    gaps: list[tuple] = []
    for i in range(len(sorted_rows) - 1):
        a, b = sorted_rows[i], sorted_rows[i + 1]
        delta = a["end_nlv"] - b["beg_nlv"]
        if abs(delta) > tolerance:
            gaps.append((a["day"], a["end_nlv"], b["day"], b["beg_nlv"], delta))
    return gaps


def filter_by_date(rows: list[dict], since: date) -> tuple[list[dict], int]:
    """Return (kept_rows, dropped_count). Rows missing a date have
    already been filtered by read_xlsx."""
    kept = [r for r in rows if r["day"] >= since]
    return kept, len(rows) - len(kept)


# ─────────────────────────────────────────────────────────────────────────────
# DB layer
# ─────────────────────────────────────────────────────────────────────────────


def _resolve_portfolio(cur, portfolio_name: str) -> tuple[int, str]:
    """Return (portfolio_id, user_id) for the named portfolio.

    Same helper as scripts/import_robinhood_csv.py — see that file's
    docstring for the user_id-propagation rationale.
    """
    cur.execute(
        "SELECT id, user_id FROM portfolios WHERE name = %s",
        (portfolio_name,),
    )
    row = cur.fetchone()
    if not row:
        raise ValueError(f"Portfolio '{portfolio_name}' not found")
    return row[0], str(row[1])


def check_existing_journal(cur, portfolio_id: int, since: date) -> int:
    """Count live trading_journal rows for this portfolio with
    day >= since. Non-zero = previous import or app-entered data
    exists; surfaces as a duplicate-import warning before --commit."""
    cur.execute(
        "SELECT COUNT(*) FROM trading_journal "
        "WHERE portfolio_id = %s AND day >= %s AND deleted_at IS NULL",
        (portfolio_id, since),
    )
    return cur.fetchone()[0]


_INSERT_BASE = """
    INSERT INTO trading_journal (
        portfolio_id, user_id, day,
        beg_nlv, cash_change, end_nlv,
        daily_dollar_change, daily_pct_change
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (portfolio_id, day) {conflict_clause}
    RETURNING (xmax = 0) AS inserted
"""

_CONFLICT_DO_NOTHING = "DO NOTHING"
_CONFLICT_DO_UPDATE = """DO UPDATE SET
    beg_nlv = EXCLUDED.beg_nlv,
    cash_change = EXCLUDED.cash_change,
    end_nlv = EXCLUDED.end_nlv,
    daily_dollar_change = EXCLUDED.daily_dollar_change,
    daily_pct_change = EXCLUDED.daily_pct_change,
    updated_at = CURRENT_TIMESTAMP"""


def write_rows(
    cur,
    portfolio_id: int,
    user_id: str,
    rows: list[dict],
    overwrite: bool,
) -> dict[str, int]:
    """INSERT (or UPSERT) every row.

    Returns {'inserted': N, 'updated': N, 'skipped': N}.
    `RETURNING (xmax = 0) AS inserted` lets us distinguish fresh
    inserts (xmax = 0) from conflict-driven updates (xmax != 0).
    DO NOTHING returns no row on conflict → counted as skipped.
    """
    conflict_clause = _CONFLICT_DO_UPDATE if overwrite else _CONFLICT_DO_NOTHING
    stmt = _INSERT_BASE.format(conflict_clause=conflict_clause)
    counts = {"inserted": 0, "updated": 0, "skipped": 0}
    for r in rows:
        cur.execute(stmt, (
            portfolio_id, user_id, r["day"],
            r["beg_nlv"], r["cash_change"], r["end_nlv"],
            r["daily_dollar_change"], r["daily_pct_change"],
        ))
        result = cur.fetchone()
        if result is None:
            # DO NOTHING conflict path: no row returned.
            counts["skipped"] += 1
        else:
            inserted = bool(result[0])
            counts["inserted" if inserted else "updated"] += 1
    return counts


# ─────────────────────────────────────────────────────────────────────────────
# Driver + verification report
# ─────────────────────────────────────────────────────────────────────────────


def _print_report(
    xlsx_path: Path,
    sheet: str,
    portfolio: str,
    since: date,
    commit: bool,
    overwrite: bool,
    source_count: int,
    after_filter_count: int,
    pre_cutoff_dropped: int,
    duplicates: list[date],
    gaps: list[tuple],
    existing_in_db: int,
    written: dict[str, int] | None,
) -> None:
    print("=" * 64)
    print("DAILY JOURNAL IMPORT — PORTFOLIO:", portfolio)
    print("=" * 64)
    print(f"Source xlsx: {xlsx_path}")
    print(f"Sheet:       {sheet}")
    print(f"Date filter: >= {since.isoformat()}")
    print(f"Mode:        {'COMMITTED' if commit else 'DRY-RUN'}")
    conflict_label = "DO UPDATE (--overwrite)" if overwrite else "DO NOTHING"
    print(f"Conflict:    {conflict_label}")
    print()

    print(f"Source rows:                    {source_count}")
    print(
        f"After date filter:              {after_filter_count}  "
        f"(dropped {pre_cutoff_dropped})"
    )
    print(
        f"After dedup:                    {after_filter_count - len(duplicates)}  "
        f"(dropped {len(duplicates)} duplicates)"
    )
    print()

    print(f"Existing rows in DB for date range: {existing_in_db}")
    if existing_in_db > 0 and not overwrite:
        print(
            "    Note: --overwrite is OFF, so existing rows will be "
            "preserved (DO NOTHING)."
        )
    print()

    if duplicates:
        print("Duplicate dates in source (last wins):")
        for d in duplicates[:10]:
            print(f"  - {d.isoformat()}")
        if len(duplicates) > 10:
            print(f"  ... and {len(duplicates) - 10} more")
        print()

    if gaps:
        print(
            f"Continuity warnings (informational — {len(gaps)} day-pairs "
            f"where end_nlv(N) ≠ beg_nlv(N+1)):"
        )
        for d_a, e, d_b, b, delta in gaps[:15]:
            print(
                f"  - {d_a.isoformat()} end ${e:.2f} vs "
                f"{d_b.isoformat()} beg ${b:.2f} (delta ${delta:+.2f})"
            )
        if len(gaps) > 15:
            print(f"  ... and {len(gaps) - 15} more")
        print()

    label = "Inserted (commit)" if commit else "Would insert (dry-run)"
    if written:
        print(f"{label}:")
        print(f"  trading_journal rows: {written.get('inserted', 0)}")
        print(f"  Skipped (conflict):   {written.get('skipped', 0)}")
        if overwrite:
            print(f"  Updated (overwrite):  {written.get('updated', 0)}")
    print("=" * 64)


def run_import(args: argparse.Namespace) -> int:
    xlsx_path = Path(args.xlsx).resolve()
    if not xlsx_path.exists():
        log.error("xlsx not found: %s", xlsx_path)
        return 2

    since = datetime.strptime(args.since, "%Y-%m-%d").date()

    raw_rows = read_xlsx(xlsx_path, sheet=args.sheet)
    source_count = len(raw_rows)
    filtered, dropped = filter_by_date(raw_rows, since)

    duplicates = detect_duplicates(filtered)
    if duplicates:
        # Dedupe: keep last occurrence (xlsx's natural order has newest
        # rows first → last-seen by sorted-by-day-then-original-index
        # would invert that. Simpler and safer: keep the last row in
        # source order, which is what the user last typed for that day).
        seen: dict[date, dict] = {}
        for r in filtered:
            seen[r["day"]] = r
        filtered = sorted(seen.values(), key=lambda r: r["day"])

    enriched = [compute_derived(r) for r in filtered]
    gaps = detect_nlv_continuity_gaps(enriched)

    written: dict[str, int] | None = None
    existing_in_db = 0

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            portfolio_id, user_id = _resolve_portfolio(cur, args.portfolio)
            cur.execute("SET LOCAL app.user_id = %s", (user_id,))
            existing_in_db = check_existing_journal(cur, portfolio_id, since)

            written = write_rows(
                cur, portfolio_id, user_id, enriched,
                overwrite=args.overwrite,
            )

            if args.commit:
                conn.commit()
                log.info(
                    "Committed: %d inserted, %d updated, %d skipped",
                    written["inserted"], written["updated"], written["skipped"],
                )
            else:
                conn.rollback()
                log.info("Dry-run: rolled back all writes")

    _print_report(
        xlsx_path=xlsx_path, sheet=args.sheet, portfolio=args.portfolio,
        since=since, commit=args.commit, overwrite=args.overwrite,
        source_count=source_count, after_filter_count=len(filtered),
        pre_cutoff_dropped=dropped, duplicates=duplicates, gaps=gaps,
        existing_in_db=existing_in_db, written=written,
    )
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Import a Daily Journal xlsx into trading_journal. "
                    "Dry-run by default; pass --commit to actually write."
    )
    p.add_argument("--xlsx", required=True,
                   help="Path to the Daily Journal xlsx")
    p.add_argument("--portfolio", required=True,
                   help="Target portfolio name (e.g. 'Long-Term Growth')")
    p.add_argument("--sheet", default="Sheet2",
                   help="Sheet name (default: 'Sheet2')")
    p.add_argument("--since", default="2026-01-01",
                   help="Filter rows with day >= this (YYYY-MM-DD)")
    p.add_argument("--overwrite", action="store_true",
                   help="ON CONFLICT DO UPDATE instead of DO NOTHING. "
                        "Touches only the 5 import-sourced columns + "
                        "updated_at; preserves user notes/daily_thoughts.")
    p.add_argument("--commit", action="store_true",
                   help="Actually write changes. Without this, the "
                        "script rolls back the transaction.")
    return p


def main() -> int:
    return run_import(build_arg_parser().parse_args())


if __name__ == "__main__":
    sys.exit(main())
