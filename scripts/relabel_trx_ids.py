#!/usr/bin/env python3
"""Phase 1b: forensic relabel of pre-df6141a trx_id scrambles.

The trx_id assignment helper (db_layer.generate_unique_trx_id) was
fixed in commit df6141a (2026-05-03). Rows inserted before that
commit can carry numbering produced by the buggy "count + 1" path
that produced 24 duplicate trx_ids in production and the off-by-N
sequences documented in the NBIS-024 audit. The strict-mode commits
on fix/trx-id-strict-by-default-hardening (merged 6b97a8f) close the
remaining bypass paths so no NEW scramble can occur. This script
cleans up the historical fossils.

Per the audit, four categories are repaired:
  GAP           live rows within a prefix have a non-contiguous sequence
  START_HIGH    live rows within a prefix start higher than 1
  DEDUPE_SUFFIX -N or -Auto suffix variants folded back into the
                canonical sequence
  FREEFORM      trx_id doesn't match TRX_ID_PATTERN. Two sub-buckets:
                  sub_1: {base}-Auto-N — machine-noise (text discarded)
                  sub_2: free-form user text (text preserved into notes)

DUPLICATE (5th audit category) cannot occur post-migration-018 and is
asserted-zero by the post-run verification.

Architecture mirrors scripts/dedupe_trx_ids.py:
  - Discovery in its own short transaction.
  - Per-trade Phase A (notes-lift) + Phase B (rename) in ONE atomic
    transaction so rename rollback also rolls back the notes-lift.
  - Phase C runs OUTSIDE the atomic transaction: clear the @ttl_cache'd
    load_details / load_summary, call _recompute_summary_matching to rewrite
    lot_closures with the new trx_ids, emit a RELABEL_TRADE audit row.
    A recompute failure does not roll back the rename — the rename is
    the high-stakes write, lot_closures self-heal on the next edit.

Idempotent: re-running --apply against an already-clean trade short-
circuits at the "current == planned" check before any UPDATE runs.

Usage:
    # Dry-run (default) — prints plan + writes snapshot CSV.
    python scripts/relabel_trx_ids.py --portfolio CanSlim

    # Apply renames + recomputes.
    python scripts/relabel_trx_ids.py --portfolio CanSlim --apply

    # Single-trade surgical fix.
    python scripts/relabel_trx_ids.py --trade-id 202604-024 --apply

Exit codes:
    0  clean (no errors)
    1  one or more per-trade errors during apply
    2  CLI argument problem
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def get_database_url() -> str:
    """Resolve DATABASE_URL from environment or .env. Called from main()
    only — keeping it out of module-level execution means importing this
    script from tests doesn't trigger an env-var side effect that
    unintentionally enables DB-dependent tests."""
    url = os.environ.get("DATABASE_URL")
    if url:
        return url
    env_path = REPO_ROOT / ".env"
    for line in env_path.read_text().splitlines():
        if line.strip().startswith("DATABASE_URL="):
            v = line.split("=", 1)[1].strip()
            return v[1:-1] if v.startswith('"') and v.endswith('"') else v
    raise RuntimeError("DATABASE_URL not found (env or .env)")


# db_layer reads DATABASE_URL lazily inside get_db_config() — so importing
# the module here is safe even when the env var isn't set. The CLI's main()
# hydrates the env before any DB call runs.
import db_layer as db  # noqa: E402

# All pre-auth rows are tagged with the founder UUID; db_layer's RLS filters
# by app.user_id, so any script writing through the layer needs to set this
# before opening a connection. Same constant the dedupe / heal scripts use.
FOUNDER_UUID = "d7e8f9a0-1b2c-4d3e-8f4a-5b6c7d8e9f0a"

# Longest-prefix-first ordering. The classifier walks this list and stops at
# the first match, so 'SB1' classifies as SB (not S + B1).
PREFIXES: tuple[str, ...] = ("SB", "SA", "B", "A", "S")

# A canonical trx_id including the dedupe / auto suffix variants. Captures the
# prefix (group 1) and the integer suffix (group 2). The trailing optional
# group covers `-Auto` and `-N` suffix markers we'll fold back.
CANONICAL_RE = re.compile(r"^(SB|SA|B|A|S)([1-9]\d*)(?:-Auto|-(\d+))?$")

# Sub-bucket 1 of FREEFORM — machine-generated noise from running the dedupe
# script on top of `-Auto` rows. Text is discarded (not preserved to notes).
DEDUPE_DOUBLE_RE = re.compile(r"^(SB|SA|B|A|S)[1-9]\d*-Auto-\d+$")


# ────────────────────────────────────────────────────────────────────────
# CLASSIFIER
# ────────────────────────────────────────────────────────────────────────


def classify(trx_id: str | None) -> tuple[str, str | None, bool, bool]:
    """Return (category, prefix_or_None, has_dedupe_or_auto, is_freeform_sub_1).

    category is one of:
      CANONICAL  — matches CANONICAL_RE (possibly with -Auto/-N suffix)
      B0-Pre     — the literal
      EMPTY      — NULL or empty string (pre-trx_id legacy rows)
      FREEFORM   — anything else
    """
    if trx_id is None or trx_id == "":
        return ("EMPTY", None, False, False)
    if trx_id == "B0-Pre":
        return ("B0-Pre", None, False, False)
    m = CANONICAL_RE.match(trx_id)
    if m:
        has_auto = trx_id.endswith("-Auto")
        has_n = m.group(3) is not None
        return ("CANONICAL", m.group(1), has_auto or has_n, False)
    if DEDUPE_DOUBLE_RE.match(trx_id):
        return ("FREEFORM", None, False, True)
    return ("FREEFORM", None, False, False)


def infer_prefix_from_action(
    row: dict, prior_b_count: int
) -> str:
    """For FREEFORM rows where the trx_id doesn't expose a prefix, derive it
    from action. Production confirmation (NBIS-024 audit + sweep): every
    FREEFORM Sub-bucket-2 row is a SELL → S prefix. Kept general for
    Sub-bucket-1 and future cases.

    Convention: first BUY in a trade → B; subsequent BUY → A; SELL → S.
    Legacy SA / SB are never produced by current code paths.
    """
    action = (row.get("action") or "").upper()
    if action == "SELL":
        return "S"
    if action == "BUY":
        return "B" if prior_b_count == 0 else "A"
    return ""  # unknown — caller treats as non-relabelable


# ────────────────────────────────────────────────────────────────────────
# DISCOVERY
# ────────────────────────────────────────────────────────────────────────


def fetch_portfolio_id(cur, portfolio_name: str) -> int:
    cur.execute("SELECT id FROM portfolios WHERE name = %s", (portfolio_name,))
    result = cur.fetchone()
    if not result:
        raise ValueError(f"Portfolio '{portfolio_name}' not found")
    return result[0]


def load_active_rows(
    cur, portfolio_name: str | None, trade_id_filter: str | None
) -> list[dict]:
    """Pull all active rows + per-trade context, ordered by the same
    `(created_at NULLS LAST, id)` key the script renumbers under."""
    where = ["d.deleted_at IS NULL", "d.trade_id IS NOT NULL"]
    args: list[Any] = []
    if portfolio_name is not None:
        where.append("p.name = %s")
        args.append(portfolio_name)
    if trade_id_filter is not None:
        where.append("d.trade_id = %s")
        args.append(trade_id_filter)
    sql = f"""
        SELECT
          p.name      AS portfolio,
          d.portfolio_id,
          d.trade_id,
          d.id        AS detail_id,
          d.trx_id,
          d.action,
          d.shares,
          d.amount    AS price,
          d.date      AS user_date,
          d.created_at,
          d.notes,
          d.rule,
          s.ticker
        FROM trades_details d
        JOIN portfolios p     ON p.id = d.portfolio_id
        LEFT JOIN trades_summary s
          ON s.portfolio_id = d.portfolio_id
         AND s.trade_id     = d.trade_id
         AND s.deleted_at IS NULL
        WHERE {" AND ".join(where)}
        ORDER BY d.portfolio_id, d.trade_id,
                 d.created_at ASC NULLS LAST,
                 d.id ASC
    """
    cur.execute(sql, args)
    cols = [c.name for c in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


# ────────────────────────────────────────────────────────────────────────
# PLAN
# ────────────────────────────────────────────────────────────────────────


def _append_notes(prev: str | None, addition: str) -> str:
    """NULL-safe notes append with " / " separator. Idempotency belt: if
    `addition` is already a substring of `prev`, return prev unchanged."""
    if prev is None or prev == "":
        return addition
    if addition in prev:
        return prev
    return f"{prev} / {addition}"


def detect_duplicate_pairs(rows_in_trade: list[dict]) -> list[dict]:
    """Find every (trx_id) value held by 2+ active rows within this trade.
    These violate migration-018's partial UNIQUE index. The script cannot
    relabel them — running dedupe_trx_ids.py first is required so each
    sibling row carries a distinct trx_id (e.g. B1 + B1-2 + B1-3).

    Returns a list of {"trx_id": str, "detail_ids": [int, ...]} entries,
    sorted by trx_id for determinism in the dry-run report.
    """
    by_trx: dict[str, list[int]] = defaultdict(list)
    for r in rows_in_trade:
        if r["trx_id"]:
            by_trx[r["trx_id"]].append(r["detail_id"])
    return [
        {"trx_id": txid, "detail_ids": sorted(ids)}
        for txid, ids in sorted(by_trx.items())
        if len(ids) > 1
    ]


def plan_for_trade(rows_in_trade: list[dict]) -> dict[str, Any]:
    """Build the rename + notes-lift plan for one trade. Returns:

      {
        "trade_id":         str,
        "portfolio":        str,
        "portfolio_id":     int,
        "ticker":           str | None,
        "categories":       set of strings (GAP / START_HIGH / DEDUPE_SUFFIX
                                            / FREEFORM / DUPLICATE)
        "duplicate_pairs":  [{trx_id, detail_ids}, ...] — non-empty iff
                            DUPLICATE in categories
        "notes_updates":    [{detail_id, prev_notes, new_notes, reason}],
        "rename_updates":   [{detail_id, prev_trx_id, new_trx_id}],
      }

    The `categories` set is informational — used by the report. Apply
    paths additionally check for DUPLICATE and skip those trades before
    opening the per-trade atomic_transaction; their notes_updates and
    rename_updates lists are returned empty so a stale call to
    apply_one_trade would be a no-op.
    """
    portfolio = rows_in_trade[0]["portfolio"]
    portfolio_id = rows_in_trade[0]["portfolio_id"]
    trade_id = rows_in_trade[0]["trade_id"]
    ticker = rows_in_trade[0].get("ticker")

    categories: set[str] = set()
    notes_updates: list[dict] = []
    rename_updates: list[dict] = []

    # DUPLICATE detection comes first. The two failure modes the discovery
    # missed: (a) the classifier never computed this category, and (b) any
    # of these trades sent into the rename loop would violate the partial
    # UNIQUE constraint mid-batch anyway. Short-circuit with an empty plan
    # so the apply path's DUPLICATE check is the canonical skip point.
    duplicate_pairs = detect_duplicate_pairs(rows_in_trade)
    if duplicate_pairs:
        categories.add("DUPLICATE")
        return {
            "trade_id": trade_id,
            "portfolio": portfolio,
            "portfolio_id": portfolio_id,
            "ticker": ticker,
            "categories": categories,
            "duplicate_pairs": duplicate_pairs,
            "notes_updates": [],
            "rename_updates": [],
        }

    # Per-prefix buckets keyed by the FOLDED prefix (i.e. dedupe-survivor and
    # -Auto siblings collapse into their base prefix).
    by_prefix: dict[str, list[dict]] = defaultdict(list)

    # Snapshot of notes mutations within this plan so subsequent rows in the
    # same trade see the in-plan running state (idempotency belt).
    pending_notes: dict[int, str | None] = {}

    # FREEFORM rows that need action-based prefix inference.
    freeform_unparsed: list[dict] = []

    # Walk rows in canonical order (already ordered by the SQL).
    for row in rows_in_trade:
        category, prefix, has_extra, is_sub_1 = classify(row["trx_id"])
        if category == "B0-Pre":
            # Literal — never renumbered, never bucketed.
            continue
        if category == "EMPTY":
            # Legacy row predating the column — leave alone.
            continue
        if category == "CANONICAL":
            assert prefix is not None
            by_prefix[prefix].append(row)
            # Provenance: -Auto rows get "(was {old})" appended to notes.
            if row["trx_id"].endswith("-Auto"):
                addition = f"(was {row['trx_id']})"
                prev_notes = pending_notes.get(row["detail_id"], row["notes"])
                new_notes = _append_notes(prev_notes, addition)
                if new_notes != prev_notes:
                    notes_updates.append({
                        "detail_id": row["detail_id"],
                        "prev_notes": prev_notes,
                        "new_notes": new_notes,
                        "reason": "auto_provenance",
                    })
                    pending_notes[row["detail_id"]] = new_notes
                categories.add("DEDUPE_SUFFIX")
            elif has_extra:
                # -N dedupe survivor; no notes write per locked policy.
                categories.add("DEDUPE_SUFFIX")
            continue
        # FREEFORM
        if is_sub_1:
            # Sub-bucket 1: machine noise (e.g. S2-Auto-2). Drop the text.
            # Parse the underlying base prefix from the regex.
            m = DEDUPE_DOUBLE_RE.match(row["trx_id"])
            assert m is not None  # is_sub_1 implies the regex matched
            base_prefix = m.group(1)
            by_prefix[base_prefix].append(row)
            categories.add("FREEFORM")
        else:
            # Sub-bucket 2: user free-form text. Lift to notes, then bucket
            # by action-inferred prefix.
            freeform_unparsed.append(row)
            categories.add("FREEFORM")

    # Process Sub-bucket 2 rows: notes-lift + action-based bucketing.
    # `prior_b_count` here is per-trade — counted from CANONICAL B rows we've
    # already collected, then incremented as we assign more.
    prior_b_count = len(by_prefix["B"])
    for row in freeform_unparsed:
        prefix = infer_prefix_from_action(row, prior_b_count)
        if not prefix:
            continue
        # Lift the freeform trx_id text into notes.
        prev_notes = pending_notes.get(row["detail_id"], row["notes"])
        new_notes = _append_notes(prev_notes, row["trx_id"])
        if new_notes != prev_notes:
            notes_updates.append({
                "detail_id": row["detail_id"],
                "prev_notes": prev_notes,
                "new_notes": new_notes,
                "reason": "freeform_sub2",
            })
            pending_notes[row["detail_id"]] = new_notes
        by_prefix[prefix].append(row)
        if prefix == "B":
            prior_b_count += 1

    # Per-prefix renumber. Detect GAP / START_HIGH on the canonical rows for
    # the report.
    for prefix, group in by_prefix.items():
        # Identify GAP / START_HIGH from the CURRENT numbering (canonical
        # rows only — freeform rows are being inserted, not gap-checked).
        canonical_nums: list[int] = []
        for r in group:
            if r["trx_id"] is None:
                continue
            m = CANONICAL_RE.match(r["trx_id"])
            if m:
                canonical_nums.append(int(m.group(2)))
        if canonical_nums:
            nums_set = set(canonical_nums)
            min_n, max_n = min(nums_set), max(nums_set)
            if min_n > 1:
                categories.add("START_HIGH")
            if (max_n - min_n + 1) > len(nums_set):
                categories.add("GAP")

        # Group is already sorted by the SQL's (created_at NULLS LAST, id)
        # because rows_in_trade preserves that order and we append in that
        # order. Defensive re-sort to make the contract explicit.
        group_sorted = sorted(
            group,
            key=lambda r: (r["created_at"] is None, r["created_at"], r["detail_id"]),
        )
        for idx, row in enumerate(group_sorted, start=1):
            new_trx_id = f"{prefix}{idx}"
            if row["trx_id"] != new_trx_id:
                rename_updates.append({
                    "detail_id": row["detail_id"],
                    "prev_trx_id": row["trx_id"],
                    "new_trx_id": new_trx_id,
                })

    return {
        "trade_id": trade_id,
        "portfolio": portfolio,
        "portfolio_id": portfolio_id,
        "ticker": ticker,
        "categories": categories,
        "duplicate_pairs": [],
        "notes_updates": notes_updates,
        "rename_updates": rename_updates,
    }


def build_all_plans(rows: list[dict]) -> list[dict]:
    by_trade: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        by_trade[(r["portfolio"], r["trade_id"])].append(r)
    plans = []
    for key in sorted(by_trade.keys()):
        plan = plan_for_trade(by_trade[key])
        # Keep DUPLICATE plans even with empty update lists so they show up
        # in the report and the apply loop sees them to skip.
        if plan["notes_updates"] or plan["rename_updates"] \
                or "DUPLICATE" in plan["categories"]:
            plans.append(plan)
    return plans


# ────────────────────────────────────────────────────────────────────────
# SNAPSHOT
# ────────────────────────────────────────────────────────────────────────


def write_snapshot(plans: list[dict], rows: list[dict], snapshot_dir: Path) -> Path:
    """Dump the affected rows + plan to a single CSV under snapshot_dir.
    Returns the CSV path."""
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    csv_path = snapshot_dir / "affected_rows.csv"

    # Index plans by (portfolio, trade_id) for per-row lookup.
    plan_by_trade: dict[tuple[str, str], dict] = {
        (p["portfolio"], p["trade_id"]): p for p in plans
    }
    rename_by_id: dict[int, str] = {}
    notes_by_id: dict[int, tuple[str | None, str | None, str]] = {}
    for p in plans:
        for u in p["rename_updates"]:
            rename_by_id[u["detail_id"]] = u["new_trx_id"]
        for u in p["notes_updates"]:
            notes_by_id[u["detail_id"]] = (u["prev_notes"], u["new_notes"], u["reason"])

    affected_keys = set(plan_by_trade.keys())

    headers = [
        "portfolio", "trade_id", "detail_id", "current_trx_id", "planned_trx_id",
        "action", "shares", "price", "user_date", "created_at",
        "freeform_subbucket", "notes_before", "notes_after",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for r in rows:
            key = (r["portfolio"], r["trade_id"])
            if key not in affected_keys:
                continue
            cat, _, _, is_sub_1 = classify(r["trx_id"])
            if cat == "FREEFORM":
                subbucket = "sub_1" if is_sub_1 else "sub_2"
            elif cat == "CANONICAL" and r["trx_id"] and r["trx_id"].endswith("-Auto"):
                subbucket = "auto_provenance"
            else:
                subbucket = "none"
            planned = rename_by_id.get(r["detail_id"], r["trx_id"])
            notes_pair = notes_by_id.get(r["detail_id"])
            notes_before = (notes_pair[0] if notes_pair else r["notes"]) or ""
            notes_after = (notes_pair[1] if notes_pair else r["notes"]) or ""
            writer.writerow([
                r["portfolio"], r["trade_id"], r["detail_id"],
                r["trx_id"] or "", planned,
                r["action"] or "", r["shares"], r["price"],
                str(r["user_date"]), str(r["created_at"]),
                subbucket, notes_before, notes_after,
            ])
    return csv_path


# ────────────────────────────────────────────────────────────────────────
# APPLY
# ────────────────────────────────────────────────────────────────────────


def _tmp_marker(detail_id: int) -> str:
    """Per-row unique marker used during Phase B-1 to dodge the partial
    UNIQUE constraint. Underscore prefix guarantees it never matches
    TRX_ID_PATTERN, so an accidental commit would still fail strict
    validation downstream. detail_id is the table PK → markers are unique
    across the transaction."""
    return f"_RELABEL_TMP_{detail_id}"


def apply_one_trade(plan: dict) -> int:
    """Phase A (notes-lift) + Phase B (two-phase rename) inside a SINGLE
    atomic_transaction. Returns count of LOGICAL operations applied
    (notes-lifts + final renames). The B-1 temp-marker UPDATEs are
    infrastructure and not counted.

    Two-phase rename rationale: the partial UNIQUE index
    `unique_trx_id_per_trade` is checked per-statement, not deferred to
    commit. A direct A2→A1 UPDATE collides with the still-present A1 row
    even when that A1 row is itself queued for rename. Phase B-1 stages
    every renaming row to a unique `_RELABEL_TMP_{detail_id}` marker
    (markers unique because detail_id is unique); Phase B-2 then writes
    the final canonical values (canonical values unique within the trade
    by construction). Both phases share one transaction so a failure in
    either rolls back the temp markers."""
    n = 0
    notes_updates = plan["notes_updates"]
    rename_updates = plan["rename_updates"]
    if not notes_updates and not rename_updates:
        return 0
    with db.atomic_transaction() as (_conn, cur):
        # Phase A — notes-lift (idempotency belt above already filtered noops)
        for u in notes_updates:
            cur.execute(
                "UPDATE trades_details SET notes = %s WHERE id = %s",
                (u["new_notes"], u["detail_id"]),
            )
            n += 1
        # Phase B-1 — stage every renaming row to a unique temp marker.
        for u in rename_updates:
            cur.execute(
                "UPDATE trades_details SET trx_id = %s WHERE id = %s",
                (_tmp_marker(u["detail_id"]), u["detail_id"]),
            )
        # Phase B-2 — temp → final canonical value.
        for u in rename_updates:
            cur.execute(
                "UPDATE trades_details SET trx_id = %s WHERE id = %s",
                (u["new_trx_id"], u["detail_id"]),
            )
            n += 1
    return n


def cache_clear_and_recompute(
    plan: dict, recompute_fn, log_audit_fn
) -> None:
    """Phase C — outside the atomic_transaction. Clears the @ttl_cache'd
    loaders so the recompute reads the post-rename state, then triggers
    the lot_closures rewrite, then emits an explicit audit_trail row.

    Per the locked spec, this is the SINGLE most failure-prone gap.
    `db.load_details` and `db.load_summary` are decorated with @ttl_cache;
    skipping the clear lets the recompute see stale cached rows and
    rewrite the summary without the new trx_ids — a silent failure mode."""
    db.load_details.clear()
    db.load_summary.clear()
    recompute_fn(plan["portfolio"], plan["trade_id"], plan["ticker"])
    # Compact summary string for the audit message.
    renames = plan["rename_updates"]
    summary_parts = [f"{u['prev_trx_id']}→{u['new_trx_id']}" for u in renames[:5]]
    if len(renames) > 5:
        summary_parts.append(f"+{len(renames) - 5} more")
    summary_str = (
        f"Renamed {len(renames)} trx_id(s), "
        f"lifted {len(plan['notes_updates'])} notes field(s): "
        f"{', '.join(summary_parts)}"
    )
    log_audit_fn(
        plan["portfolio"], "RELABEL_TRADE", plan["trade_id"],
        plan["ticker"] or "", summary_str, username="phase1b",
    )


# ────────────────────────────────────────────────────────────────────────
# VERIFICATION
# ────────────────────────────────────────────────────────────────────────


def verify_clean(
    portfolio_name: str | None, trade_id_filter: str | None
) -> dict[str, int]:
    """Re-run the scan after --apply and report any residual issues.
    Returns a dict of category → count. All canonical categories should be
    zero; FREEFORM should also be zero (sub_1 overwritten, sub_2 lifted)."""
    with db.atomic_transaction() as (_conn, cur):
        rows = load_active_rows(cur, portfolio_name, trade_id_filter)
    plans = build_all_plans(rows)
    counts: dict[str, int] = defaultdict(int)
    for p in plans:
        for cat in p["categories"]:
            counts[cat] += 1
    return dict(counts)


# ────────────────────────────────────────────────────────────────────────
# REPORT
# ────────────────────────────────────────────────────────────────────────


def _sample_plans(plans: list[dict], n: int = 5) -> Iterable[str]:
    for p in sorted(plans, key=lambda p: (p["portfolio"], p["trade_id"]))[:n]:
        renames_disp = ", ".join(
            f"{u['prev_trx_id']}→{u['new_trx_id']}"
            for u in p["rename_updates"][:6]
        )
        yield (
            f"  {p['portfolio']:<14} {p['trade_id']:<14} {p['ticker'] or '—':<8} "
            f"cats={sorted(p['categories'])}  "
            f"+{len(p['notes_updates'])} notes-lift, "
            f"{len(p['rename_updates'])} rename: {renames_disp}"
        )


def relabel(
    portfolio_name: str | None,
    trade_id_filter: str | None,
    apply_writes: bool,
    snapshot_dir: Path,
) -> int:
    db.current_user_id.set(FOUNDER_UUID)

    print("─" * 80)
    print("Phase 1b — trx_id relabel runner")
    print("─" * 80)
    print(f"Portfolio filter: {portfolio_name or '(all)'}")
    print(f"Trade-ID filter:  {trade_id_filter or '(all)'}")
    print(f"Mode:             {'APPLY' if apply_writes else 'DRY-RUN'}")
    print(f"Snapshot dir:     {snapshot_dir}")
    print()

    # Discovery
    print("Loading affected rows…")
    with db.atomic_transaction() as (_conn, cur):
        rows = load_active_rows(cur, portfolio_name, trade_id_filter)
    print(f"  fetched {len(rows)} active row(s)")

    plans = build_all_plans(rows)
    if not plans:
        print("\nNo affected trades — data already clean.")
        return 0

    # Snapshot (always, for both dry-run and apply audit trail)
    csv_path = write_snapshot(plans, rows, snapshot_dir)
    print(f"  snapshot CSV: {csv_path}")
    print()

    # Per-category counts
    category_counts: dict[str, int] = defaultdict(int)
    for p in plans:
        for cat in p["categories"]:
            category_counts[cat] += 1

    duplicate_plans = [p for p in plans if "DUPLICATE" in p["categories"]]
    work_plans = [p for p in plans if "DUPLICATE" not in p["categories"]]

    print("─" * 80)
    print("PLAN SUMMARY")
    print("─" * 80)
    print(f"  Total trades in plan:         {len(plans)}")
    print(f"    Will be relabeled:          {len(work_plans)}")
    print(f"    Will be SKIPPED (DUPLICATE):{len(duplicate_plans):>4}")
    total_notes = sum(len(p["notes_updates"]) for p in work_plans)
    total_renames = sum(len(p["rename_updates"]) for p in work_plans)
    print(f"  Phase A notes-lift UPDATEs:   {total_notes}")
    print(f"  Phase B rename UPDATEs:       {total_renames}")
    print(f"  Total logical UPDATEs:        {total_notes + total_renames}")
    print()
    print("  Per-category trade counts (a trade can hit multiple categories):")
    for cat in ("START_HIGH", "GAP", "DEDUPE_SUFFIX", "FREEFORM", "DUPLICATE"):
        print(f"    {cat:<16} {category_counts.get(cat, 0)}")
    print()
    if duplicate_plans:
        print("  DUPLICATE-flagged trades (will be SKIPPED in --apply mode):")
        for p in duplicate_plans:
            pairs_disp = ", ".join(
                f"{q['trx_id']}({len(q['detail_ids'])}×)"
                for q in p["duplicate_pairs"]
            )
            print(f"    {p['portfolio']:<14} {p['trade_id']:<14} {p['ticker'] or '—':<8} "
                  f"{pairs_disp}")
        print()
    print(f"  Sample plans (first 5 non-DUPLICATE by portfolio+trade):")
    for line in _sample_plans(work_plans):
        print(line)
    print()

    if not apply_writes:
        print("─" * 80)
        print("DRY-RUN — no DB writes performed.")
        print("Review the snapshot CSV before re-running with --apply.")
        print("─" * 80)
        return 0

    # ── Apply path ────────────────────────────────────────────────────────
    print("─" * 80)
    print("APPLYING — per-trade atomic transactions")
    print("─" * 80)

    sys.path.insert(0, str(REPO_ROOT / "api"))
    from main import _recompute_summary_matching as recompute_fn  # noqa: E402

    rename_failures: list[tuple[str, str]] = []
    recompute_failures: list[tuple[str, str]] = []
    duplicate_skips: list[tuple[str, str]] = []
    fully_succeeded = 0
    no_op_count = 0
    n_updates_applied = 0

    for i, plan in enumerate(plans, start=1):
        trade_id = plan["trade_id"]
        portfolio = plan["portfolio"]
        ticker = plan.get("ticker") or "?"

        # DUPLICATE skip BEFORE opening the atomic_transaction. Pre-existing
        # duplicate (trade_id, trx_id) pairs require scripts/dedupe_trx_ids
        # .py to run first; this script cannot collapse them.
        if "DUPLICATE" in plan["categories"]:
            pairs_disp = ", ".join(
                f"{p['trx_id']}({len(p['detail_ids'])}×)"
                for p in plan["duplicate_pairs"]
            )
            duplicate_skips.append((trade_id, pairs_disp))
            print(f"  [{i}/{len(plans)}] {portfolio} {trade_id} ({ticker}): "
                  f"SKIP — DUPLICATE pairs present: {pairs_disp}. "
                  f"Run scripts/dedupe_trx_ids.py --apply first.")
            continue

        # No-op short-circuit (idempotent re-runs against already-canonical
        # trades). Reported separately from real work.
        if not plan["notes_updates"] and not plan["rename_updates"]:
            no_op_count += 1
            continue

        # Phase A + B (two-phase inside one atomic_transaction)
        try:
            n = apply_one_trade(plan)
            n_updates_applied += n
        except Exception as e:
            rename_failures.append((trade_id, str(e)))
            print(f"  [{i}/{len(plans)}] {portfolio} {trade_id}: rename FAILED — {e}")
            continue
        # Phase C
        try:
            cache_clear_and_recompute(plan, recompute_fn, db.log_audit)
            fully_succeeded += 1
            print(f"  [{i}/{len(plans)}] {portfolio} {trade_id}: "
                  f"{len(plan['rename_updates'])} rename + "
                  f"{len(plan['notes_updates'])} notes-lift OK, recompute OK")
        except Exception as e:
            recompute_failures.append((trade_id, str(e)))
            print(f"  [{i}/{len(plans)}] {portfolio} {trade_id}: "
                  f"rename committed but recompute FAILED — {e}")

    # Post-run verification
    print()
    print("─" * 80)
    print("POST-RUN VERIFICATION")
    print("─" * 80)
    residual = verify_clean(portfolio_name, trade_id_filter)
    if not residual:
        print("  ✓ No residual canonical issues (script idempotent).")
    else:
        print("  ✗ Residual issues detected:")
        for cat, n in residual.items():
            print(f"      {cat}: {n}")

    # Report
    print()
    print("─" * 80)
    print("APPLY REPORT")
    print("─" * 80)
    print(f"  Trades attempted:        {len(plans)}")
    print(f"  Fully succeeded:         {fully_succeeded}")
    print(f"  No-op (already clean):   {no_op_count}")
    print(f"  DUPLICATE skipped:       {len(duplicate_skips)}")
    print(f"  Rename failed:           {len(rename_failures)}")
    print(f"  Rename OK, recompute X:  {len(recompute_failures)}")
    print(f"  Total UPDATEs applied:   {n_updates_applied}")
    print(f"  Snapshot CSV:            {csv_path}")
    if duplicate_skips:
        print("\n  DUPLICATE trades skipped (run dedupe_trx_ids.py first):")
        for t, pairs in duplicate_skips:
            print(f"    {t}: {pairs}")
    if rename_failures:
        print("\n  Rename failures (atomic rollback — no changes for these trades):")
        for t, m in rename_failures:
            print(f"    {t}: {m}")
    if recompute_failures:
        print("\n  Rename succeeded, recompute failed (lot_closures may be stale "
              "until next edit or re-run):")
        for t, m in recompute_failures:
            print(f"    {t}: {m}")
    return 1 if (rename_failures or recompute_failures or residual) else 0


# ────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────


def _default_snapshot_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return REPO_ROOT / "snapshots" / f"relabel_{stamp}"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(__doc__ or "").split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--portfolio", default=None,
                    help="Restrict to one portfolio (e.g. 'CanSlim').")
    ap.add_argument("--trade-id", default=None,
                    help="Restrict to a single trade_id (e.g. '202604-024').")
    ap.add_argument("--apply", action="store_true",
                    help="Apply renames + recomputes. "
                         "Without this flag, runs in dry-run mode.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Explicit dry-run (default behavior). Mutually "
                         "exclusive with --apply.")
    ap.add_argument("--snapshot-dir", type=Path, default=None,
                    help="Directory to write the affected_rows.csv snapshot. "
                         "Default: ./snapshots/relabel_<ISO_TIMESTAMP>/")
    args = ap.parse_args()

    if args.apply and args.dry_run:
        print("ERROR: --apply and --dry-run are mutually exclusive.", file=sys.stderr)
        return 2

    # Hydrate DATABASE_URL here (not at module load) so test imports stay
    # side-effect-free. db_layer reads the var lazily on first connection.
    os.environ.setdefault("DATABASE_URL", get_database_url())

    snapshot_dir = args.snapshot_dir or _default_snapshot_dir()
    return relabel(
        portfolio_name=args.portfolio,
        trade_id_filter=args.trade_id,
        apply_writes=args.apply,
        snapshot_dir=snapshot_dir,
    )


if __name__ == "__main__":
    sys.exit(main())
