-- ============================================================================
-- Migration 030: Close the Week — Phase 4.6
-- ============================================================================
-- Replace the single Overall Week Grade with 3 axis grades (Execution /
-- Process / P&L) and add a persistent "reviewed" state. The existing
-- week_grade column stays as the canonical OVERALL letter; it is now either
-- derived from the 3 axes (default) or overridden by the user (when
-- overall_override = TRUE).
--
-- Legacy rows: pre-Phase-4.6 retros have week_grade set and all 3 axis
-- columns NULL. The frontend treats this as "Overall set, axes never
-- entered" and lets the user keep the existing overall or grade the axes
-- and let the derived value take over.
--
-- Carry-forward / action items: dropped from Phase 4.6 scope per user
-- direction ("if there is a new rule it will be added to the rules set.
-- no need for any carry forward."). No new table.
--
-- The migration runner wraps this file in a transaction; no BEGIN/COMMIT
-- statements here.
-- ============================================================================

-- ---------------------------------------------------------------------------
-- 1. Axis grades + override flag + reviewed_at on weekly_retros
-- ---------------------------------------------------------------------------
-- All axis columns nullable: legacy rows + in-progress rows where the user
-- hasn't graded all 3 axes yet are valid states.
-- overall_override NOT NULL DEFAULT FALSE: Postgres 11+ stores a constant
-- default in catalog metadata so this is a metadata-only ALTER for existing
-- rows (no table rewrite).
-- reviewed_at nullable: NULL = not reviewed yet.
ALTER TABLE weekly_retros
    ADD COLUMN execution_grade   VARCHAR(3),
    ADD COLUMN process_grade     VARCHAR(3),
    ADD COLUMN pnl_grade         VARCHAR(3),
    ADD COLUMN overall_override  BOOLEAN     NOT NULL DEFAULT FALSE,
    ADD COLUMN reviewed_at       TIMESTAMPTZ;

-- Vocab CHECK per axis — same closed set as the existing week_grade vocab.
-- Mirrors weekly_retros_week_grade_vocab from migration 025.
ALTER TABLE weekly_retros
    ADD CONSTRAINT weekly_retros_execution_grade_vocab CHECK (
        execution_grade IS NULL OR execution_grade IN
        ('A+','A','A-','B+','B','B-','C+','C','C-','D','F')
    ),
    ADD CONSTRAINT weekly_retros_process_grade_vocab CHECK (
        process_grade IS NULL OR process_grade IN
        ('A+','A','A-','B+','B','B-','C+','C','C-','D','F')
    ),
    ADD CONSTRAINT weekly_retros_pnl_grade_vocab CHECK (
        pnl_grade IS NULL OR pnl_grade IN
        ('A+','A','A-','B+','B','B-','C+','C','C-','D','F')
    );


-- ============================================================================
-- Verification queries (manual, after COMMIT)
-- ============================================================================
-- Expect: 5 new columns on weekly_retros.
--   SELECT column_name, data_type, is_nullable, column_default
--   FROM information_schema.columns
--   WHERE table_name = 'weekly_retros'
--     AND column_name IN ('execution_grade','process_grade','pnl_grade',
--                         'overall_override','reviewed_at')
--   ORDER BY column_name;
--
-- Expect: bad axis grade rejected by the new CHECK.
--   UPDATE weekly_retros SET execution_grade = 'Z' WHERE id = 1;
--   → ERROR:  new row violates check constraint "weekly_retros_execution_grade_vocab"
--
-- Expect: legacy rows unchanged (all axes NULL, override FALSE, reviewed_at NULL).
--   SELECT count(*) FROM weekly_retros
--    WHERE execution_grade IS NULL AND overall_override = FALSE AND reviewed_at IS NULL;
