"""
Drift checks — codified consistency invariants for the trades / journal /
lot_closures schema. Phase 2 Commit 8 of the data integrity sweep.

Each entry in DRIFT_CHECKS is a SQL query that returns one row per drift
violation. The runner wraps each query to compute (a) a total violation
count and (b) up to N sample rows for inspection.

Why a registry of SQL strings rather than per-check Python functions:
  - The checks ARE SQL — the entire point is to look at DB state, not
    compute anything in Python. A registry makes each check a single
    self-contained tuple a reader can eyeball.
  - Adding a new check is one tuple, no plumbing.
  - The runner enforces uniform behaviour (statement_timeout, portfolio
    filter, sample limit, duration tracking) across all checks.

SQL conventions every check follows:
  - Single SELECT (no top-level WITH) so the runner can wrap it in
    `SELECT COUNT(*) FROM ({sql}) v` and `SELECT * FROM ({sql}) v LIMIT N`
    without rewriting it.
  - Portfolio filter: `(s.portfolio_id = %(portfolio_id)s OR %(portfolio_id)s IS NULL)`
    so a single SQL string handles both scoped and unscoped runs.
  - First three columns are always (trade_id, ticker, portfolio) when
    available — gives the frontend a stable canonical sample shape.
  - Extra check-specific columns appear after — frontend renders them
    generically as a wide row.

Tripwire checks (severity='error', should ALWAYS return 0 post-Migration 022):
  - risk_budget_null_or_zero_post_021: Migration 021 should have eliminated.
  - string_nan_in_prose: chk_summary_*_no_sentinel CHECKs prevent.
  - open_with_closed_date: chk_summary_closed_date_consistency prevents.
A non-zero count means a CHECK constraint was disabled / the audit trigger
was bypassed / someone wrote directly via a path that skipped Migration 022.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Literal

# Default and ceiling for the per-check sample row limit. The endpoint
# accepts ?limit_samples=N and clamps to [1, MAX]. 10 is enough for human
# inspection; 50 is a soft ceiling so a misconfigured client can't stream
# every offending row through the gate.
SAMPLE_LIMIT_DEFAULT = 10
SAMPLE_LIMIT_MAX = 50

# Timeout per individual check, in milliseconds. Postgres aborts the
# statement after this; the runner converts the abort into a check-level
# error and the scan continues to the next check.
CHECK_STATEMENT_TIMEOUT_MS = 30000


@dataclass(frozen=True)
class DriftCheck:
    check_id: str
    description: str
    severity: Literal["warning", "error"]
    sql: str
    remediation: str


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
# Order is the canonical UI display order: easier-to-fix first, structural
# LIFO checks last. Severity is "error" for invariants whose violation
# means user-visible incorrectness; "warning" for soft drift that's worth
# investigating but doesn't break the journal.

DRIFT_CHECKS: list[DriftCheck] = [
    DriftCheck(
        check_id="summary_detail_rule_mismatch",
        description=(
            "trades_summary.rule differs from the rule on the earliest BUY "
            "detail (the buy that opened the campaign)."
        ),
        severity="warning",
        # Earliest BUY = ORDER BY date ASC, id ASC. Schema: trades_details.action
        # is 'BUY' or 'SELL' (line 119); rule is VARCHAR(100). LATERAL join
        # is per-row; one earliest BUY per (portfolio_id, trade_id).
        sql="""
            SELECT
                s.trade_id,
                s.ticker,
                p.name AS portfolio,
                s.rule AS summary_rule,
                d.rule AS detail_rule
            FROM trades_summary s
            JOIN portfolios p ON p.id = s.portfolio_id
            JOIN LATERAL (
                SELECT rule
                  FROM trades_details
                 WHERE portfolio_id = s.portfolio_id
                   AND trade_id     = s.trade_id
                   AND action       = 'BUY'
                   AND deleted_at IS NULL
                 ORDER BY date ASC, id ASC
                 LIMIT 1
            ) d ON TRUE
            WHERE (s.portfolio_id = %(portfolio_id)s OR %(portfolio_id)s IS NULL)
              AND s.deleted_at IS NULL
              AND s.rule IS DISTINCT FROM d.rule
        """,
        remediation=(
            "Recompute the campaign via Trade Manager → Database Health, or "
            "edit the summary's rule field to match the earliest BUY detail."
        ),
    ),
    DriftCheck(
        check_id="risk_budget_null_or_zero_post_021",
        description=(
            "Open trades opened on/after 2026-01-01 with NULL or 0 risk_budget. "
            "Migration 021 backfilled these from sizing_mode × prior-day NLV; "
            "a non-zero count here means the migration didn't run or new "
            "writes are bypassing the calc."
        ),
        severity="error",  # Tripwire — Migration 021 should have eliminated.
        sql="""
            SELECT
                s.trade_id,
                s.ticker,
                p.name AS portfolio,
                s.open_date,
                s.risk_budget
            FROM trades_summary s
            JOIN portfolios p ON p.id = s.portfolio_id
            WHERE (s.portfolio_id = %(portfolio_id)s OR %(portfolio_id)s IS NULL)
              AND s.deleted_at IS NULL
              AND s.status = 'OPEN'
              AND s.open_date >= TIMESTAMP '2026-01-01'
              AND (s.risk_budget IS NULL OR s.risk_budget = 0)
        """,
        remediation=(
            "Re-run migration 021. If the trade was opened post-migration, "
            "investigate the write path that produced a 0/NULL risk_budget."
        ),
    ),
    DriftCheck(
        check_id="string_nan_in_prose",
        description=(
            "trades_summary prose columns (rule, buy_notes, sell_rule, "
            "sell_notes, notes) containing the literal strings 'nan', "
            "'none', or 'null' (case-insensitive). Migration 022 added "
            "CHECK constraints that prevent new rows; a non-zero count "
            "here means the constraint was disabled."
        ),
        severity="error",  # Tripwire — chk_summary_*_no_sentinel prevents.
        # UNION ALL across the 5 prose columns. Filter applied inside each
        # branch (where s.portfolio_id is in scope) rather than wrapping
        # the whole UNION — keeps the SQL composable in the runner.
        sql="""
            SELECT
                s.trade_id,
                s.ticker,
                p.name AS portfolio,
                'rule' AS column_name,
                s.rule AS bad_value
              FROM trades_summary s
              JOIN portfolios p ON p.id = s.portfolio_id
             WHERE (s.portfolio_id = %(portfolio_id)s OR %(portfolio_id)s IS NULL)
               AND s.deleted_at IS NULL
               AND s.rule IS NOT NULL
               AND LOWER(TRIM(s.rule)) IN ('nan', 'none', 'null')
            UNION ALL
            SELECT
                s.trade_id, s.ticker, p.name,
                'buy_notes', s.buy_notes
              FROM trades_summary s
              JOIN portfolios p ON p.id = s.portfolio_id
             WHERE (s.portfolio_id = %(portfolio_id)s OR %(portfolio_id)s IS NULL)
               AND s.deleted_at IS NULL
               AND s.buy_notes IS NOT NULL
               AND LOWER(TRIM(s.buy_notes)) IN ('nan', 'none', 'null')
            UNION ALL
            SELECT
                s.trade_id, s.ticker, p.name,
                'sell_rule', s.sell_rule
              FROM trades_summary s
              JOIN portfolios p ON p.id = s.portfolio_id
             WHERE (s.portfolio_id = %(portfolio_id)s OR %(portfolio_id)s IS NULL)
               AND s.deleted_at IS NULL
               AND s.sell_rule IS NOT NULL
               AND LOWER(TRIM(s.sell_rule)) IN ('nan', 'none', 'null')
            UNION ALL
            SELECT
                s.trade_id, s.ticker, p.name,
                'sell_notes', s.sell_notes
              FROM trades_summary s
              JOIN portfolios p ON p.id = s.portfolio_id
             WHERE (s.portfolio_id = %(portfolio_id)s OR %(portfolio_id)s IS NULL)
               AND s.deleted_at IS NULL
               AND s.sell_notes IS NOT NULL
               AND LOWER(TRIM(s.sell_notes)) IN ('nan', 'none', 'null')
            UNION ALL
            SELECT
                s.trade_id, s.ticker, p.name,
                'notes', s.notes
              FROM trades_summary s
              JOIN portfolios p ON p.id = s.portfolio_id
             WHERE (s.portfolio_id = %(portfolio_id)s OR %(portfolio_id)s IS NULL)
               AND s.deleted_at IS NULL
               AND s.notes IS NOT NULL
               AND LOWER(TRIM(s.notes)) IN ('nan', 'none', 'null')
        """,
        remediation=(
            "Set the offending column to NULL or a real value via Trade "
            "Manager. Re-verify Migration 022 CHECK constraints are present: "
            "SELECT conname FROM pg_constraint WHERE conrelid = "
            "'trades_summary'::regclass AND contype = 'c'."
        ),
    ),
    DriftCheck(
        check_id="open_with_closed_date",
        description=(
            "trades_summary rows with status='OPEN' but a non-NULL "
            "closed_date. Migration 022's chk_summary_closed_date_consistency "
            "prevents this; a non-zero count means the constraint was "
            "disabled."
        ),
        severity="error",  # Tripwire — chk_summary_closed_date_consistency.
        sql="""
            SELECT
                s.trade_id,
                s.ticker,
                p.name AS portfolio,
                s.status,
                s.closed_date
            FROM trades_summary s
            JOIN portfolios p ON p.id = s.portfolio_id
            WHERE (s.portfolio_id = %(portfolio_id)s OR %(portfolio_id)s IS NULL)
              AND s.deleted_at IS NULL
              AND s.status = 'OPEN'
              AND s.closed_date IS NOT NULL
        """,
        remediation=(
            "Either flip the trade to CLOSED (if the lot is fully sold) or "
            "NULL out closed_date (if the trade was reopened). Re-verify "
            "the chk_summary_closed_date_consistency constraint exists."
        ),
    ),
    DriftCheck(
        check_id="invalid_journal_source_values",
        description=(
            "trading_journal entries where nlv_source or holdings_source is "
            "outside the allowed set (manual / ibkr_auto / ibkr_override). "
            "schema.sql lines 197-200 enforce this; drift here means a row "
            "was inserted before the CHECK landed."
        ),
        severity="warning",
        # Allowed values pinned to the schema-level CHECK constraint
        # (schema.sql:197-200). If the allowed set ever expands, update both.
        sql="""
            SELECT
                p.name           AS portfolio,
                j.day,
                j.nlv_source,
                j.holdings_source
            FROM trading_journal j
            JOIN portfolios p ON p.id = j.portfolio_id
            WHERE (j.portfolio_id = %(portfolio_id)s OR %(portfolio_id)s IS NULL)
              AND j.deleted_at IS NULL
              AND ( j.nlv_source      NOT IN ('manual', 'ibkr_auto', 'ibkr_override')
                 OR j.holdings_source NOT IN ('manual', 'ibkr_auto', 'ibkr_override') )
        """,
        remediation=(
            "Edit the journal entry's nlv_source / holdings_source to one of "
            "the allowed values via Daily Routine."
        ),
    ),
    DriftCheck(
        check_id="open_summary_no_open_buys",
        description=(
            "trades_summary rows with status='OPEN' that have NO open BUY "
            "lots (every BUY's shares are fully closed by lot_closures). "
            "Orphan summaries — LIFO inconsistency. Excludes trades with "
            "any empty/NULL trx_id (those are flagged separately by "
            "lot_closures_empty_trx_id; recompute LIFO on those first, "
            "then re-run this check)."
        ),
        severity="error",
        # Single SELECT shape (no top-level WITH) so the runner can wrap
        # it in COUNT(*) / SELECT *. open_remaining is the per-trade sum
        # of GREATEST(buy.shares - SUM(closed.shares for that buy), 0).
        # Trades with no BUY details at all → tol.open_remaining is NULL,
        # COALESCE'd to 0, flagged as drift.
        #
        # Empty-trx-id exclusion: the LIFO subquery joins lc.buy_trx_id =
        # d.trx_id; if a trade has any empty='' on either side, the join
        # would falsely pair every empty closure with every empty BUY,
        # producing nonsense remainders. Exclude such trades up-front via
        # NOT EXISTS — they're surfaced by check 'lot_closures_empty_trx_id'
        # and rejoin the eligible set once recompute rewrites their trx_ids.
        sql="""
            SELECT
                os.trade_id,
                os.ticker,
                os.portfolio,
                COALESCE(tol.open_remaining, 0) AS open_remaining
            FROM (
                SELECT s.portfolio_id, s.trade_id, s.ticker, p.name AS portfolio
                  FROM trades_summary s
                  JOIN portfolios p ON p.id = s.portfolio_id
                 WHERE s.status = 'OPEN'
                   AND s.deleted_at IS NULL
                   AND (s.portfolio_id = %(portfolio_id)s OR %(portfolio_id)s IS NULL)
            ) os
            LEFT JOIN (
                SELECT br.portfolio_id, br.trade_id,
                       SUM(GREATEST(br.remaining, 0)) AS open_remaining
                  FROM (
                    SELECT
                        d.portfolio_id, d.trade_id, d.trx_id,
                        d.shares - COALESCE(SUM(lc.shares), 0) AS remaining
                      FROM trades_details d
                      LEFT JOIN lot_closures lc
                        ON  lc.portfolio_id = d.portfolio_id
                        AND lc.trade_id     = d.trade_id
                        AND lc.buy_trx_id   = d.trx_id
                     WHERE d.action = 'BUY'
                       AND d.deleted_at IS NULL
                     GROUP BY d.portfolio_id, d.trade_id, d.trx_id, d.shares
                  ) br
                 GROUP BY br.portfolio_id, br.trade_id
            ) tol
              ON tol.portfolio_id = os.portfolio_id
             AND tol.trade_id     = os.trade_id
            WHERE COALESCE(tol.open_remaining, 0) <= 0
              AND NOT EXISTS (
                  SELECT 1 FROM trades_details d_check
                   WHERE d_check.portfolio_id = os.portfolio_id
                     AND d_check.trade_id     = os.trade_id
                     AND d_check.action       = 'BUY'
                     AND d_check.deleted_at IS NULL
                     AND (d_check.trx_id IS NULL OR d_check.trx_id = '')
              )
              AND NOT EXISTS (
                  SELECT 1 FROM lot_closures lc_check
                   WHERE lc_check.portfolio_id = os.portfolio_id
                     AND lc_check.trade_id     = os.trade_id
                     AND (lc_check.buy_trx_id  IS NULL OR lc_check.buy_trx_id  = ''
                       OR lc_check.sell_trx_id IS NULL OR lc_check.sell_trx_id = '')
              )
        """,
        remediation=(
            "Recompute LIFO for the trade. If genuinely closed, flip status "
            "to CLOSED with closed_date set. If a SELL was logged in error, "
            "delete the offending detail and let recompute reopen the trade. "
            "If the trade has empty trx_ids (check #10), recompute LIFO "
            "first — that rewrites lot_closures with proper trx_ids and "
            "this check will then evaluate the trade correctly."
        ),
    ),
    DriftCheck(
        check_id="closed_summary_with_open_buys",
        description=(
            "trades_summary rows with status='CLOSED' that still have "
            "open BUY lots (BUY shares > SUM of matching lot_closures). "
            "Inverse of open_summary_no_open_buys — closed when shouldn't "
            "be. Excludes trades with any empty/NULL trx_id (those are "
            "flagged separately by lot_closures_empty_trx_id; recompute "
            "LIFO on those first, then re-run this check)."
        ),
        severity="error",
        # See open_summary_no_open_buys for the empty-trx-id rationale —
        # the LIFO subquery is structurally identical and shares the same
        # join hazard. NOT EXISTS in the outer WHERE excludes ambiguous
        # trades up-front.
        sql="""
            SELECT
                cs.trade_id,
                cs.ticker,
                cs.portfolio,
                tol.open_remaining
            FROM (
                SELECT s.portfolio_id, s.trade_id, s.ticker, p.name AS portfolio
                  FROM trades_summary s
                  JOIN portfolios p ON p.id = s.portfolio_id
                 WHERE s.status = 'CLOSED'
                   AND s.deleted_at IS NULL
                   AND (s.portfolio_id = %(portfolio_id)s OR %(portfolio_id)s IS NULL)
            ) cs
            JOIN (
                SELECT br.portfolio_id, br.trade_id,
                       SUM(GREATEST(br.remaining, 0)) AS open_remaining
                  FROM (
                    SELECT
                        d.portfolio_id, d.trade_id, d.trx_id,
                        d.shares - COALESCE(SUM(lc.shares), 0) AS remaining
                      FROM trades_details d
                      LEFT JOIN lot_closures lc
                        ON  lc.portfolio_id = d.portfolio_id
                        AND lc.trade_id     = d.trade_id
                        AND lc.buy_trx_id   = d.trx_id
                     WHERE d.action = 'BUY'
                       AND d.deleted_at IS NULL
                     GROUP BY d.portfolio_id, d.trade_id, d.trx_id, d.shares
                  ) br
                 GROUP BY br.portfolio_id, br.trade_id
            ) tol
              ON tol.portfolio_id = cs.portfolio_id
             AND tol.trade_id     = cs.trade_id
            WHERE tol.open_remaining > 0
              AND NOT EXISTS (
                  SELECT 1 FROM trades_details d_check
                   WHERE d_check.portfolio_id = cs.portfolio_id
                     AND d_check.trade_id     = cs.trade_id
                     AND d_check.action       = 'BUY'
                     AND d_check.deleted_at IS NULL
                     AND (d_check.trx_id IS NULL OR d_check.trx_id = '')
              )
              AND NOT EXISTS (
                  SELECT 1 FROM lot_closures lc_check
                   WHERE lc_check.portfolio_id = cs.portfolio_id
                     AND lc_check.trade_id     = cs.trade_id
                     AND (lc_check.buy_trx_id  IS NULL OR lc_check.buy_trx_id  = ''
                       OR lc_check.sell_trx_id IS NULL OR lc_check.sell_trx_id = '')
              )
        """,
        remediation=(
            "Flip status to OPEN and clear closed_date, then recompute LIFO. "
            "If the SELLs that closed the trade were duplicated, delete the "
            "duplicates first. If the trade has empty trx_ids (check #10), "
            "recompute LIFO first — that rewrites lot_closures with proper "
            "trx_ids and this check will then evaluate the trade correctly."
        ),
    ),
    DriftCheck(
        check_id="lot_closures_empty_trx_id",
        description=(
            "lot_closures rows with empty-string buy_trx_id or sell_trx_id. "
            "Repro of the 252-trade bug from 5/4/2026 where recompute wrote "
            "rows before trx_ids were assigned. Empty trx_ids break the "
            "Trade Journal's per-row P&L lookup."
        ),
        severity="error",
        sql="""
            SELECT
                lc.trade_id,
                COALESCE(s.ticker, '?') AS ticker,
                p.name AS portfolio,
                lc.sell_trx_id,
                lc.buy_trx_id,
                lc.id AS closure_id
            FROM lot_closures lc
            JOIN portfolios p ON p.id = lc.portfolio_id
            LEFT JOIN trades_summary s
              ON  s.portfolio_id = lc.portfolio_id
              AND s.trade_id     = lc.trade_id
            WHERE (lc.portfolio_id = %(portfolio_id)s OR %(portfolio_id)s IS NULL)
              AND (lc.buy_trx_id = '' OR lc.sell_trx_id = '')
        """,
        remediation=(
            "Recompute LIFO for the affected trade — the recompute now "
            "writes trx_ids correctly. Old empty rows will be replaced by "
            "the delete-then-insert path in _recompute_summary_matching."
        ),
    ),
    DriftCheck(
        check_id="summary_realized_pl_vs_lot_closures_sum",
        description=(
            "trades_summary.realized_pl ≠ SUM of lot_closures.realized_pl "
            "for the same trade (penny tolerance). Only checked on trades "
            "that have at least one SELL detail — trades with no SELLs "
            "should have realized_pl=0 and zero lot_closures, which match "
            "trivially and aren't drift."
        ),
        severity="warning",
        # 0.01 tolerance: NUMERIC(15,2) for both columns means exact rounding,
        # but lot_closures math might accumulate fractional cents in older
        # data from before the persisted-LIFO migration. Loud enough to
        # catch real drift, quiet enough to ignore one-cent rounding.
        sql="""
            SELECT
                s.trade_id,
                s.ticker,
                p.name AS portfolio,
                s.realized_pl                  AS summary_realized_pl,
                COALESCE(lc.sum_pl, 0)          AS lot_closures_sum,
                s.realized_pl - COALESCE(lc.sum_pl, 0) AS diff
            FROM trades_summary s
            JOIN portfolios p ON p.id = s.portfolio_id
            LEFT JOIN (
                SELECT portfolio_id, trade_id, SUM(realized_pl) AS sum_pl
                  FROM lot_closures
                 GROUP BY portfolio_id, trade_id
            ) lc
              ON lc.portfolio_id = s.portfolio_id
             AND lc.trade_id     = s.trade_id
            WHERE (s.portfolio_id = %(portfolio_id)s OR %(portfolio_id)s IS NULL)
              AND s.deleted_at IS NULL
              AND EXISTS (
                  SELECT 1 FROM trades_details d
                   WHERE d.portfolio_id = s.portfolio_id
                     AND d.trade_id     = s.trade_id
                     AND d.action       = 'SELL'
                     AND d.deleted_at IS NULL
              )
              AND ABS(s.realized_pl - COALESCE(lc.sum_pl, 0)) > 0.01
        """,
        remediation=(
            "Recompute LIFO for the campaign — _recompute_summary_matching "
            "rewrites lot_closures and recomputes summary.realized_pl in a "
            "single transaction, so they should match after."
        ),
    ),
    DriftCheck(
        check_id="summary_shares_vs_open_buy_remaining",
        description=(
            "trades_summary.shares ≠ SUM of open BUY remaining shares "
            "(buy.shares - SUM(matching lot_closures.shares)). Only "
            "checked on OPEN trades; for CLOSED trades, summary.shares "
            "carries total_buy_shs by design (campaign face-card metric), "
            "not current remaining. Tolerance is fractional-share (0.0001). "
            "Excludes trades with any empty/NULL trx_id (those are flagged "
            "separately by lot_closures_empty_trx_id; recompute LIFO on "
            "those first, then re-run this check)."
        ),
        severity="error",
        # NUMERIC(12,4) for shares — 0.0001 tolerance to catch real drift
        # without flagging quarter-share rounding.
        #
        # OPEN-only scope: trades_summary.shares is dual-semantic. For
        # OPEN trades it carries the LIFO post-sell remaining inventory
        # (what this check compares against). For CLOSED trades it
        # carries the lifetime sum of BUY shares (campaign face-card
        # metric — see compute_matching_summary in trade_calc.py:177 and
        # the matching write in api/main.py:3142). Comparing those to
        # `SUM(open_remaining)=0` for closed trades over-flags every
        # closed campaign that ever bought shares — pure false positive.
        #
        # Empty-trx-id exclusion: same hazard as #8/#9 — `closed.buy_trx_id
        # = d.trx_id` would falsely match every empty closure to every
        # empty BUY. Exclude such trades up-front via NOT EXISTS in the
        # outer WHERE; they're surfaced by lot_closures_empty_trx_id.
        sql="""
            SELECT
                s.trade_id,
                s.ticker,
                p.name AS portfolio,
                s.shares                          AS summary_shares,
                COALESCE(br.open_remaining, 0)    AS detail_remaining,
                s.shares - COALESCE(br.open_remaining, 0) AS diff
            FROM trades_summary s
            JOIN portfolios p ON p.id = s.portfolio_id
            LEFT JOIN (
                SELECT
                    d.portfolio_id, d.trade_id,
                    SUM(GREATEST(d.shares - COALESCE(closed.closed_shares, 0), 0)) AS open_remaining
                  FROM trades_details d
                  LEFT JOIN (
                      SELECT portfolio_id, trade_id, buy_trx_id, SUM(shares) AS closed_shares
                        FROM lot_closures
                       GROUP BY portfolio_id, trade_id, buy_trx_id
                  ) closed
                    ON  closed.portfolio_id = d.portfolio_id
                    AND closed.trade_id     = d.trade_id
                    AND closed.buy_trx_id   = d.trx_id
                 WHERE d.action = 'BUY'
                   AND d.deleted_at IS NULL
                 GROUP BY d.portfolio_id, d.trade_id
            ) br
              ON br.portfolio_id = s.portfolio_id
             AND br.trade_id     = s.trade_id
            WHERE (s.portfolio_id = %(portfolio_id)s OR %(portfolio_id)s IS NULL)
              AND s.deleted_at IS NULL
              AND s.status = 'OPEN'
              AND ABS(s.shares - COALESCE(br.open_remaining, 0)) > 0.0001
              AND NOT EXISTS (
                  SELECT 1 FROM trades_details d_check
                   WHERE d_check.portfolio_id = s.portfolio_id
                     AND d_check.trade_id     = s.trade_id
                     AND d_check.action       = 'BUY'
                     AND d_check.deleted_at IS NULL
                     AND (d_check.trx_id IS NULL OR d_check.trx_id = '')
              )
              AND NOT EXISTS (
                  SELECT 1 FROM lot_closures lc_check
                   WHERE lc_check.portfolio_id = s.portfolio_id
                     AND lc_check.trade_id     = s.trade_id
                     AND (lc_check.buy_trx_id  IS NULL OR lc_check.buy_trx_id  = ''
                       OR lc_check.sell_trx_id IS NULL OR lc_check.sell_trx_id = '')
              )
        """,
        remediation=(
            "Recompute LIFO for the OPEN campaign — both summary.shares "
            "and the per-buy remaining are derived from the same "
            "trades_details rows; recompute makes them agree by "
            "construction. If the trade has empty trx_ids (check #10), "
            "recompute LIFO first — that rewrites lot_closures with "
            "proper trx_ids and this check will then evaluate the trade "
            "correctly. CLOSED trades are intentionally out of scope; "
            "their summary.shares carries lifetime total_buy_shs by "
            "design and should not be compared against open_remaining."
        ),
    ),
]


# Map of check_id -> DriftCheck for O(1) lookup by the endpoint when
# ?check_id= is provided.
DRIFT_CHECKS_BY_ID: dict[str, DriftCheck] = {c.check_id: c for c in DRIFT_CHECKS}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _coerce_value(v: Any) -> Any:
    """JSON-friendly value coercion for sample rows.

    psycopg2 returns Decimal for NUMERIC columns and datetime objects for
    TIMESTAMP/DATE — neither is JSON-serialisable by default. Coerce to
    float/str so the FastAPI auto-encoder doesn't choke. Pure values
    (str/int/None/bool) pass through untouched.
    """
    if v is None:
        return None
    if isinstance(v, Decimal):
        return float(v)
    if isinstance(v, datetime):
        return v.isoformat()
    if hasattr(v, "isoformat"):  # date / time / Timestamp
        return v.isoformat()
    return v


def run_check(
    conn,
    check: DriftCheck,
    portfolio_id: int | None,
    sample_limit: int,
) -> dict[str, Any]:
    """Execute one check.

    Returns a dict with violation_count / samples / duration_ms. On
    statement timeout (or any psycopg2 error) returns the same shape
    with severity="error" and a remediation hint pointing at the query.
    The orchestrator keeps going regardless — one slow check shouldn't
    abort the whole scan.
    """
    started = time.monotonic()
    params = {"portfolio_id": portfolio_id, "limit": int(sample_limit)}
    try:
        with conn.cursor() as cur:
            # SET LOCAL only takes effect inside a transaction. psycopg2
            # opens an implicit txn on the first execute(); SET LOCAL has
            # to come AFTER that to be inside the same txn. So we issue a
            # noop SELECT first to materialise the txn, then SET LOCAL.
            cur.execute("SELECT 1")
            cur.fetchall()
            cur.execute(
                f"SET LOCAL statement_timeout = {CHECK_STATEMENT_TIMEOUT_MS}"
            )

            count_sql = f"SELECT COUNT(*) FROM ({check.sql}) AS v"
            cur.execute(count_sql, params)
            count_row = cur.fetchone()
            violation_count = int(count_row[0]) if count_row else 0

            sample_sql = (
                f"SELECT * FROM ({check.sql}) AS v LIMIT %(limit)s"
            )
            cur.execute(sample_sql, params)
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()
            samples = [
                {col: _coerce_value(val) for col, val in zip(cols, row)}
                for row in rows
            ]
        # Read-only scan — rollback the implicit transaction so we don't
        # hold a snapshot longer than necessary.
        conn.rollback()
        duration_ms = int((time.monotonic() - started) * 1000)
        return {
            "violation_count": violation_count,
            "samples": samples,
            "duration_ms": duration_ms,
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 — runner must not propagate
        try:
            conn.rollback()
        except Exception:
            pass
        duration_ms = int((time.monotonic() - started) * 1000)
        return {
            "violation_count": 0,
            "samples": [],
            "duration_ms": duration_ms,
            "error": str(exc) or exc.__class__.__name__,
        }


def run_drift_scan(
    conn,
    *,
    portfolio_id: int | None = None,
    portfolio_name: str | None = None,
    check_id: str | None = None,
    sample_limit: int = SAMPLE_LIMIT_DEFAULT,
) -> dict[str, Any]:
    """Orchestrate a scan and return the response payload.

    portfolio_id is the resolved int FK; portfolio_name is echoed on the
    response so the caller doesn't have to look it up again. Pass exactly
    one of (portfolio_id, portfolio_name) or neither (scan all).

    check_id selects a single check; unknown id raises KeyError so the
    endpoint can return a 400.
    """
    if check_id is not None:
        if check_id not in DRIFT_CHECKS_BY_ID:
            raise KeyError(check_id)
        checks = [DRIFT_CHECKS_BY_ID[check_id]]
    else:
        checks = list(DRIFT_CHECKS)

    sample_limit = max(1, min(int(sample_limit), SAMPLE_LIMIT_MAX))

    results = []
    summary_passed = 0
    summary_warnings = 0
    summary_errors = 0
    for c in checks:
        outcome = run_check(conn, c, portfolio_id, sample_limit)
        # A check that errored at the SQL level is treated as an "error"
        # bucket regardless of its declared severity, so timeouts don't
        # silently turn into warnings on a "warning"-typed check.
        effective_severity = "error" if outcome["error"] else c.severity
        if outcome["error"]:
            remediation = (
                f"Check failed to run ({outcome['error']}) — investigate "
                "query plan or DB connectivity."
            )
        else:
            remediation = c.remediation

        if outcome["violation_count"] == 0 and not outcome["error"]:
            summary_passed += 1
        elif effective_severity == "error":
            summary_errors += 1
        else:
            summary_warnings += 1

        results.append({
            "check_id": c.check_id,
            "description": c.description,
            "severity": effective_severity,
            "violation_count": outcome["violation_count"],
            "samples": outcome["samples"],
            "remediation": remediation,
            "duration_ms": outcome["duration_ms"],
            "error": outcome["error"],
        })

    return {
        "scanned_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "portfolio_filter": portfolio_name,
        "check_filter": check_id,
        "sample_limit": sample_limit,
        "checks": results,
        "summary": {
            "total_checks": len(results),
            "passed":   summary_passed,
            "warnings": summary_warnings,
            "errors":   summary_errors,
        },
    }
