-- ============================================================================
-- Migration 066: drop trades_summary.sr12_floor_pct (retire the B1 anchor)
-- ============================================================================
-- Migration 064 introduced sr12_floor_pct as the SR12 anchor. Migration 065
-- rewrote SR12 to use peak_total_pl (correct anchor for scaled-in positions).
-- The 064 column was left in place through the cutover so mid-deploy reads
-- wouldn't 500; that grace period is over.
--
-- Retirement: DROP the column. This migration ships alongside a commit that
-- removes every remaining read/write of sr12_floor_pct (db_layer helper,
-- b1_reconcile hook, load_summary SELECT, _normalize_trades COL_MAP,
-- frontend positions row-shape field, trade-rules glossary text).
--
-- Verification: after this migration commits, the sr12_floor_pct column
-- no longer exists on trades_summary. Any stray reader still expecting
-- it will fail loudly instead of silently mis-anchoring an SR12 nudge.
--
-- The migration runner wraps this file in a transaction; no BEGIN/COMMIT.
-- ============================================================================

ALTER TABLE trades_summary
    DROP COLUMN IF EXISTS sr12_floor_pct;


-- ============================================================================
-- Verification queries (manual, after COMMIT)
-- ============================================================================
--   SELECT column_name
--     FROM information_schema.columns
--    WHERE table_name = 'trades_summary' AND column_name = 'sr12_floor_pct';
--   → 0 rows.
