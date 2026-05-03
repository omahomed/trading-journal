#!/usr/bin/env python3
"""Rename duplicate trx_ids in trades_details so lot_closures backfill succeeds.

The lot_closures table has UNIQUE (portfolio_id, trade_id, sell_trx_id,
buy_trx_id) per migration 017. 16 trades on production failed the closure
backfill (script bd2f358 / 25dab2c) with that constraint because their
trades_details rows have multiple BUY or SELL rows sharing the same trx_id
within one trade — likely a mix of IBKR auto-import collisions and manual
entry mistakes. The LIFO walk produces a closure tuple per shares match,
so two SELLs both named 'SB1' against the same BUY produce two closures
with identical (sell_trx_id, buy_trx_id) — the constraint refuses the
second insert.

This script:
  1. Finds every (trade_id, trx_id) tuple appearing more than once in
     trades_details for a portfolio.
  2. Sorts duplicates by date (then id for ties), preserves the FIRST
     occurrence as-is.
  3. Renames subsequent siblings with a numeric suffix: -2, -3, -4, ...
     skipping any suffix already in use elsewhere in the trade.
  4. Re-runs _recompute_summary_lifo per affected trade so lot_closures
     get rewritten with the new (now-unique) sell_trx_id / buy_trx_id.

Per-trade flow is two-phase: an atomic_transaction commits all renames
for the trade, then the recompute fires in its own transactions (same
db.* helpers used by every other write path; they each open their own
connections, so genuine cross-statement atomicity isn't structurally
available without refactoring db_layer). If the recompute fails after
a successful rename, the trade is reported as a separate error category
("rename succeeded, recompute failed") so the operator can re-run --apply
or trigger recompute manually. lot_closures self-heal on the next edit
that touches the trade.

Idempotent: re-running --apply finds no remaining duplicates and exits
clean. Safe to re-run for the recompute-failed bucket — the renames are
already in place, so the script becomes a no-op for renames and just
re-tries the recompute.

Usage:
    # Dry-run (default) — reports the rename plan without writing.
    python scripts/dedupe_trx_ids.py --portfolio CanSlim

    # Actually rename + recompute.
    python scripts/dedupe_trx_ids.py --portfolio CanSlim --apply

    # --dry-run is allowed for explicitness; it's the default behavior.
    python scripts/dedupe_trx_ids.py --portfolio CanSlim --dry-run

Exit codes:
    0  no errors (regardless of dry-run vs apply)
    1  one or more rename or recompute errors occurred
    2  argparse rejected the args
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
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


# Hydrate DATABASE_URL before importing db_layer (its config reads env once).
os.environ.setdefault("DATABASE_URL", get_database_url())

import db_layer as db  # noqa: E402

# All pre-auth rows are tagged with the founder UUID; db_layer's RLS filters
# by app.user_id, so any script writing through the layer needs to set this
# before opening a connection. Same constant heal_options + backfill use.
FOUNDER_UUID = "d7e8f9a0-1b2c-4d3e-8f4a-5b6c7d8e9f0a"


def fetch_portfolio_id(cur, portfolio_name: str) -> int:
    cur.execute("SELECT id FROM portfolios WHERE name = %s", (portfolio_name,))
    result = cur.fetchone()
    if not result:
        raise ValueError(f"Portfolio '{portfolio_name}' not found")
    return result[0]


def find_duplicate_groups(cur, portfolio_id: int) -> dict[str, dict]:
    """Return {trade_id: {"ticker": str, "groups": [{trx_id, rows: [(detail_id, date)]}]}}.

    A 'group' is one (trade_id, trx_id) tuple with >1 occurrence. A trade may
    have multiple groups (different trx_ids each duplicated within the same
    trade). Excludes NULL/empty trx_ids — those would group together but
    suffixing an empty string is meaningless.
    """
    cur.execute(
        """
        SELECT d.trade_id, d.trx_id, d.id, d.date, s.ticker
        FROM trades_details d
        JOIN trades_summary s
          ON s.portfolio_id = d.portfolio_id
         AND s.trade_id = d.trade_id
        WHERE d.portfolio_id = %s
          AND d.deleted_at IS NULL
          AND s.deleted_at IS NULL
          AND (d.trade_id, d.trx_id) IN (
              SELECT trade_id, trx_id
              FROM trades_details
              WHERE portfolio_id = %s
                AND deleted_at IS NULL
                AND trx_id IS NOT NULL
                AND trx_id <> ''
              GROUP BY trade_id, trx_id
              HAVING COUNT(*) > 1
          )
        ORDER BY d.trade_id, d.trx_id, d.date, d.id
        """,
        (portfolio_id, portfolio_id),
    )
    rows = cur.fetchall()

    by_trade: dict[str, dict] = {}
    for trade_id, trx_id, detail_id, date, ticker in rows:
        entry = by_trade.setdefault(trade_id, {"ticker": ticker, "groups": defaultdict(list)})
        entry["groups"][trx_id].append((detail_id, date))

    # Convert defaultdict groups → ordered list for stable iteration.
    return {
        tid: {
            "ticker": payload["ticker"],
            "groups": [
                {"trx_id": trx_id, "rows": rows_list}
                for trx_id, rows_list in payload["groups"].items()
            ],
        }
        for tid, payload in by_trade.items()
    }


def fetch_existing_trx_ids(cur, portfolio_id: int, trade_id: str) -> set[str]:
    """Every trx_id currently in use anywhere in this trade. Used as the
    'taken' set for collision-checking new suffixes."""
    cur.execute(
        """
        SELECT DISTINCT trx_id
        FROM trades_details
        WHERE portfolio_id = %s
          AND trade_id = %s
          AND deleted_at IS NULL
          AND trx_id IS NOT NULL
        """,
        (portfolio_id, trade_id),
    )
    return {row[0] for row in cur.fetchall()}


def next_available_suffix(base_trx_id: str, taken: set[str]) -> str:
    """Smallest -N (N >= 2) suffix that isn't already in `taken`."""
    n = 2
    while f"{base_trx_id}-{n}" in taken:
        n += 1
    return f"{base_trx_id}-{n}"


def build_rename_plan(groups: list[dict], taken: set[str]) -> list[dict]:
    """For each duplicate group, keep the first row and rename the rest.

    Returns a flat list of plan entries: [{detail_id, date, old_trx_id,
    new_trx_id, action}] where action is 'keep' or 'rename'. Mutates `taken`
    so suffixes assigned within this run don't collide with each other.
    """
    plan: list[dict] = []
    for group in groups:
        trx_id = group["trx_id"]
        rows = group["rows"]  # already sorted (date, id) by the SQL
        # First occurrence: untouched.
        first_did, first_date = rows[0]
        plan.append({
            "detail_id": first_did, "date": first_date,
            "old_trx_id": trx_id, "new_trx_id": trx_id, "action": "keep",
        })
        # Subsequent occurrences: assign next available suffix.
        for did, dt in rows[1:]:
            new_trx_id = next_available_suffix(trx_id, taken)
            taken.add(new_trx_id)
            plan.append({
                "detail_id": did, "date": dt,
                "old_trx_id": trx_id, "new_trx_id": new_trx_id, "action": "rename",
            })
    return plan


def apply_renames(plan: list[dict]) -> int:
    """Run the actual UPDATEs for one trade in a single atomic_transaction.
    Returns the number of rows renamed (excludes the 'keep' rows)."""
    rename_entries = [e for e in plan if e["action"] == "rename"]
    if not rename_entries:
        return 0
    with db.atomic_transaction() as (_conn, cur):
        for entry in rename_entries:
            cur.execute(
                "UPDATE trades_details SET trx_id = %s WHERE id = %s",
                (entry["new_trx_id"], entry["detail_id"]),
            )
    return len(rename_entries)


def count_closures(portfolio_id: int, trade_id: str) -> int:
    """Read lot_closures count for a trade — used for the 'N closures written'
    line in the apply-mode per-trade output."""
    with db.atomic_transaction() as (_conn, cur):
        cur.execute(
            "SELECT COUNT(*) FROM lot_closures WHERE portfolio_id = %s AND trade_id = %s",
            (portfolio_id, trade_id),
        )
        return int(cur.fetchone()[0])


def _format_group_summary(groups: list[dict]) -> str:
    """e.g. '3 duplicate rows for SB1, 2 duplicate rows for B1-Auto'."""
    parts = [f"{len(g['rows'])} duplicate rows for {g['trx_id']}" for g in groups]
    return ", ".join(parts)


def _format_date(dt) -> str:
    """Compact date format for the rename log lines."""
    if dt is None:
        return "—"
    try:
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(dt)


def dedupe(portfolio_name: str, apply_writes: bool) -> int:
    """Main loop. Returns shell exit code (0 ok, 1 on any rename/recompute error)."""
    db.current_user_id.set(FOUNDER_UUID)

    print(f"Loading duplicates from portfolio '{portfolio_name}'...")

    # Discovery query in its own transaction so we don't hold a connection
    # during the per-trade work below.
    with db.atomic_transaction() as (_conn, cur):
        portfolio_id = fetch_portfolio_id(cur, portfolio_name)
        by_trade = find_duplicate_groups(cur, portfolio_id)

    if not by_trade:
        print(f"No duplicate trx_ids found in portfolio '{portfolio_name}'. Nothing to do.")
        return 0

    n_trades = len(by_trade)
    n_groups = sum(len(payload["groups"]) for payload in by_trade.values())
    print(f"Found {n_groups} duplicate (trx_id, trade_id) group(s) "
          f"affecting {n_trades} trade(s).\n")

    # Lazy import — only needed in --apply mode. Keeps dry-run fast and avoids
    # loading FastAPI when we don't need it.
    recompute_fn = None
    if apply_writes:
        sys.path.insert(0, str(REPO_ROOT / "api"))
        from main import _recompute_summary_lifo as recompute_fn  # noqa: E402

    rename_planned_total = 0
    rename_succeeded_total = 0
    rename_failures: list[tuple[str, str]] = []          # (trade_id, message)
    recompute_failures: list[tuple[str, str]] = []       # (trade_id, message)
    fully_succeeded = 0

    for i, (trade_id, payload) in enumerate(sorted(by_trade.items()), start=1):
        ticker = payload["ticker"]
        groups = payload["groups"]

        # Build the rename plan. Need the trade's currently-used trx_ids to
        # avoid colliding with names already manually assigned elsewhere.
        with db.atomic_transaction() as (_conn, cur):
            taken = fetch_existing_trx_ids(cur, portfolio_id, trade_id)
        plan = build_rename_plan(groups, taken)
        rows_in_plan = sum(1 for e in plan if e["action"] == "rename")
        rename_planned_total += rows_in_plan

        print(f"[{i}/{n_trades}] Trade {trade_id} ({ticker}): "
              f"{_format_group_summary(groups)}")
        for entry in plan:
            verb = "keep as" if entry["action"] == "keep" else (
                "renamed to" if apply_writes else "rename to"
            )
            print(f"  {entry['old_trx_id']} @ {_format_date(entry['date'])} "
                  f"→ {verb} {entry['new_trx_id']}")

        if not apply_writes:
            continue

        # Phase 1: rename atomically.
        try:
            renamed = apply_renames(plan)
            rename_succeeded_total += renamed
        except Exception as e:
            rename_failures.append((trade_id, str(e)))
            print(f"  ✗ rename failed: {e}")
            continue

        # Phase 2: clear caches (raw SQL bypassed them) and recompute.
        # Recompute opens its own connections — outside the rename's atomic
        # transaction by necessity (see the script's docstring rationale).
        db.load_details.clear()
        db.load_summary.clear()
        try:
            recompute_fn(portfolio_name, trade_id, ticker)
            closure_count = count_closures(portfolio_id, trade_id)
            print(f"  ✓ Recompute fired: {closure_count} closures written")
            fully_succeeded += 1
        except Exception as e:
            # Rename committed; closures may now be stale (referencing the
            # old trx_ids). Reported as a distinct error category so the
            # operator can re-run --apply (rename is a no-op, recompute
            # retries) or trigger recompute manually.
            recompute_failures.append((trade_id, str(e)))
            print(f"  ✗ Recompute failed: {e}")

    # ─── Report ──────────────────────────────────────────────────────────
    print()
    print("─" * 49)
    print("Cleanup Plan" if not apply_writes else "Cleanup Report")
    print("─" * 49)
    print(f"Total trades with duplicates:   {n_trades}")
    if apply_writes:
        print(f"Total rows renamed:             {rename_succeeded_total}")
        print(f"Trades recomputed successfully: {fully_succeeded}")
        print(f"Renames failed:                 {len(rename_failures)}")
        print(f"Rename OK but recompute failed: {len(recompute_failures)}")
    else:
        print(f"Total rows to rename:           {rename_planned_total}")
        print(f"Total recomputes triggered:     0 (dry-run — no recomputes fired)")

    if rename_failures:
        print("\nRename failures (atomic rollback — no changes for these trades):")
        for trade_id, message in rename_failures:
            print(f"  {trade_id}: {message}")

    if recompute_failures:
        print("\nRename succeeded, recompute failed (lot_closures may be stale "
              "until next edit or re-run with --apply):")
        for trade_id, message in recompute_failures:
            print(f"  {trade_id}: {message}")

    print()
    if apply_writes:
        print(f"Mode: APPLIED. Renamed {rename_succeeded_total} row(s) across "
              f"{fully_succeeded + len(recompute_failures)} trade(s); "
              f"{fully_succeeded} fully recomputed, "
              f"{len(rename_failures)} rename failure(s), "
              f"{len(recompute_failures)} recompute failure(s).")
    else:
        print("Mode: DRY-RUN (no writes performed). To apply renames + recomputes, "
              "run with --apply.")

    return 1 if (rename_failures or recompute_failures) else 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--portfolio", required=True,
                    help="Portfolio name (e.g. 'CanSlim'). Required.")
    ap.add_argument("--apply", action="store_true",
                    help="Apply renames + recomputes. "
                         "Without this flag, runs in dry-run mode.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Explicit dry-run (default behavior — included for clarity). "
                         "Mutually exclusive with --apply.")
    args = ap.parse_args()

    if args.apply and args.dry_run:
        print("ERROR: --apply and --dry-run are mutually exclusive.", file=sys.stderr)
        return 2

    return dedupe(args.portfolio, apply_writes=args.apply)


if __name__ == "__main__":
    sys.exit(main())
