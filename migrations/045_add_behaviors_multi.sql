-- ============================================================================
-- Migration 045: add behaviors JSONB column to weekly_retro_ticker_grades
-- ============================================================================
-- Trader Mindset Phase 1: convert the single-value `behavior` VARCHAR(40)
-- column to a multi-value store so the Per-Ticker retro can capture the
-- full emotional mix of a trade (e.g. FOMO Entry + Overconfidence +
-- Panic Sell together), instead of forcing one label.
--
-- Rollback-friendly split (not an in-place type change):
--   • Add `behaviors JSONB NOT NULL DEFAULT '[]'::jsonb`
--   • Backfill: every existing row's single `behavior` becomes a
--     single-element array in `behaviors` (or `[]` if NULL/empty).
--   • `behavior` column stays alongside for one release window so a
--     rollback path exists. Backend writes both columns during the
--     transition — `behavior` = behaviors[0] (or '' when the array is
--     empty), `behaviors` = the full array. A follow-up migration will
--     drop `behavior` once the frontend has been on the array shape
--     for a stable window.
--
-- Storage shape (behaviors):
--   ["FOMO Entry", "Sized Too Big"]
--   ["Followed Plan"]
--   []
-- ============================================================================

ALTER TABLE weekly_retro_ticker_grades
    ADD COLUMN IF NOT EXISTS behaviors JSONB NOT NULL DEFAULT '[]'::jsonb;

-- Backfill: coerce every existing single-value `behavior` into a
-- one-element JSONB array. NULL / empty strings become '[]'.
UPDATE weekly_retro_ticker_grades
   SET behaviors = jsonb_build_array(behavior)
 WHERE behavior IS NOT NULL
   AND behavior <> ''
   AND behaviors = '[]'::jsonb;

-- Index for the aggregation endpoint (/api/mindset/traps) — GIN handles
-- the "which rows contain this tag?" query efficiently.
CREATE INDEX IF NOT EXISTS idx_retro_grades_behaviors
    ON weekly_retro_ticker_grades USING gin (behaviors);
