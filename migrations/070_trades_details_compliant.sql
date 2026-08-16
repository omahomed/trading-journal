-- ============================================================================
-- Migration 070: trades_details.compliant — per-decision rule-adherence flag
-- ============================================================================
-- Adds a per-row binary flag to trades_details for the Weekly Ledger's
-- "compliance %" tile. Per-trade psychology says the actionable signal
-- is "did I follow my process on this decision?" — cheaper to record
-- than a 1-5 subjective grade and directly maps to the weekly
-- compliance score the operator wants to track.
--
-- NULL   = ungraded (default; nothing enforced)
-- TRUE   = followed process
-- FALSE  = broke rule
--
-- Not repurposing the existing `exec_grade` TEXT column: it holds
-- freeform letter grades ("A (Perfect)", "F (Impulse)") from an
-- older UI on ~15 rows. Keeping it untouched preserves that history;
-- the new boolean is a clean semantic that can't be confused with
-- letter grades.
--
-- Weekly compliance % is computed on the fly from
-- COUNT(compliant=TRUE) / COUNT(compliant IS NOT NULL) — ungraded
-- rows don't drag the denominator.
--
-- No RLS or CHECK constraint needed — trades_details is already
-- tenant-scoped via user_id + the RLS policy from migration 003.
--
-- The migration runner wraps this file in a transaction; no BEGIN/COMMIT.
-- ============================================================================

ALTER TABLE trades_details
    ADD COLUMN IF NOT EXISTS compliant BOOLEAN;

COMMENT ON COLUMN trades_details.compliant IS
    'Per-decision rule adherence flag. NULL = ungraded, TRUE = followed '
    'process, FALSE = broke rule. Drives the Weekly Ledger compliance %% '
    'tile — a compliance score treats process consistency, not P&L, as '
    'the signal.';

-- Partial index for the "recent graded" lookup pattern the Weekly Ledger
-- runs on every visit (WHERE date range AND compliant IS NOT NULL). Skips
-- ungraded rows so the index stays small.
CREATE INDEX IF NOT EXISTS idx_trades_details_compliant_graded
    ON trades_details (portfolio_id, date)
    WHERE compliant IS NOT NULL AND deleted_at IS NULL;


-- ============================================================================
-- Verification (manual, post-COMMIT)
-- ============================================================================
--   \d trades_details
--   SELECT compliant, COUNT(*) FROM trades_details GROUP BY compliant;
