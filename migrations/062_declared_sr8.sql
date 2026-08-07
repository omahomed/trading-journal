-- ============================================================================
-- Migration 062: is_declared_sr8 — split cushion-qualified from declared SR8
-- ============================================================================
-- Doctrine v8.1 §1 governs a Two-Hold intent (max 2 declared SR8 monster-holds
-- book-wide, promotion requires demotion, cluster-uncorrelated). The current
-- app auto-tags every campaign whose peak b1_return crossed +50% as SR8 —
-- there is no user opt-in and no cap. The 2026-07 drawdown had multiple
-- correlated names running SR8 semantics at high leverage; that failure mode
-- was the trigger for this split.
--
-- New model (per the 2026-08-07 canonical handoff):
--   * cushion_qualified (DERIVED) = b1_max_return_pct >= 50   — same test today
--   * is_declared_sr8  (PERSISTED) = user-controlled boolean, off by default
--
-- Cushion-qualified but undeclared names display / behave as SR7 (21 EMA +
-- cushion cascade); only declared names run the SR8 weekly MO RS funnel
-- ladder and are exempt from SR2 floors, etc.
--
-- No hard cap enforced in the DB; the user retains full flexibility on how
-- many to declare. An informational counter chip lives in ACS instead.
--
-- Backfill: exactly one row across every portfolio currently displays as
-- SR8 today (CanSlim / DELL / 202604-013 / peak +166.47%). Everything else
-- flips FALSE. This preserves the current DELL SR8 display through the
-- migration and forces every future +50% crossing to be an explicit
-- declaration.
--
-- The migration runner wraps this file in a transaction; no BEGIN/COMMIT.
-- ============================================================================

ALTER TABLE trades_summary
    ADD COLUMN IF NOT EXISTS is_declared_sr8 BOOLEAN NOT NULL DEFAULT FALSE;

CREATE INDEX IF NOT EXISTS idx_trades_summary_declared_sr8
    ON trades_summary (user_id, is_declared_sr8)
    WHERE is_declared_sr8 = TRUE;

COMMENT ON COLUMN trades_summary.is_declared_sr8 IS
    'User-declared SR8 monster-hold flag. FALSE by default. TRUE requires '
    'explicit right-click promotion in ACS after the campaign is cushion-'
    'qualified (b1_max_return_pct >= 50). Cushion-qualified but not '
    'declared = SR7 tier. See migration 062 for the doctrine context.';

-- Preserve the current DELL display as SR8 through the migration. Any
-- other portfolio's DELL is untouched — the WHERE clause pins to CanSlim.
-- Idempotent: repeat runs are no-ops because is_declared_sr8 already true.
UPDATE trades_summary
   SET is_declared_sr8 = TRUE
 WHERE trade_id = '202604-013'
   AND ticker = 'DELL'
   AND portfolio_id = (SELECT id FROM portfolios WHERE name = 'CanSlim')
   AND deleted_at IS NULL
   AND is_declared_sr8 = FALSE;


-- ============================================================================
-- Verification queries (manual, after COMMIT)
-- ============================================================================
--   SELECT column_name, data_type, is_nullable, column_default
--     FROM information_schema.columns
--    WHERE table_name = 'trades_summary' AND column_name = 'is_declared_sr8';
--   → boolean, NO, false
--
--   SELECT p.name, s.trade_id, s.ticker, s.b1_max_return_pct,
--          s.is_declared_sr8
--     FROM trades_summary s
--     JOIN portfolios p ON p.id = s.portfolio_id
--    WHERE s.deleted_at IS NULL
--      AND (s.b1_max_return_pct >= 50 OR s.is_declared_sr8 = TRUE)
--   ORDER BY s.b1_max_return_pct DESC;
--   → DELL 202604-013 CanSlim should be the only is_declared_sr8 = TRUE row.
